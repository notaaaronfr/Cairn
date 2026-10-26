//! Native Google Gemini adapter (Gemini API and Vertex AI).
//!
//! Gemini attaches *thought signatures* to parts of its answers; models that
//! think (Gemini 2.5 and later) require them back on the next turn,
//! especially on function calls. Cairn keeps each signature as a
//! `Content::Thinking` block tagged `{"provider": "gemini", "attach": ...}`
//! right before the part it belongs to, and puts it back on that part when
//! the conversation is sent to Gemini again. Other providers drop them.

use std::collections::HashMap;
use std::sync::Arc;

use async_trait::async_trait;
use cairn_runtime::{
    Content, Model, ModelError, ModelRequest, ModelResponse, ModelStream, Pricing, Role, StopReason, StreamEvent, Usage,
};
use serde_json::{Value, json};

use crate::openai::TokenProvider;
use crate::sse::{self, SseEvent};
use crate::{result_string, send_json};

const PROVIDER: &str = "gemini";

#[derive(Clone)]
enum Auth {
    ApiKey(String),
    Token(TokenProvider),
}

pub struct GeminiModel {
    model: String,
    /// URL up to and including `models/<model>`.
    model_url: String,
    auth: Auth,
    pricing: Option<Pricing>,
    client: reqwest::Client,
}

impl GeminiModel {
    /// Gemini API (Google AI Studio key).
    pub fn new(model: impl Into<String>, api_key: impl Into<String>) -> Self {
        let model = model.into();
        Self {
            model_url: format!("https://generativelanguage.googleapis.com/v1beta/models/{model}"),
            model,
            auth: Auth::ApiKey(api_key.into()),
            pricing: None,
            client: reqwest::Client::new(),
        }
    }

    /// Gemini on Vertex AI. `token` returns a cached OAuth access token
    /// (scope `https://www.googleapis.com/auth/cloud-platform`).
    pub fn vertex(
        project: &str,
        location: &str,
        model: impl Into<String>,
        token: impl Fn() -> Result<String, String> + Send + Sync + 'static,
    ) -> Self {
        let model = model.into();
        let host = if location == "global" {
            "aiplatform.googleapis.com".to_owned()
        } else {
            format!("{location}-aiplatform.googleapis.com")
        };
        Self {
            model_url: format!(
                "https://{host}/v1/projects/{project}/locations/{location}/publishers/google/models/{model}"
            ),
            model,
            auth: Auth::Token(Arc::new(token)),
            pricing: None,
            client: reqwest::Client::new(),
        }
    }

    /// Point at another base URL (tests, proxies). `url` replaces everything
    /// before `/models/<model>`.
    pub fn with_base_url(mut self, url: &str) -> Self {
        self.model_url = format!("{}/models/{}", url.trim_end_matches('/'), self.model);
        self
    }

    pub fn with_pricing(mut self, pricing: Pricing) -> Self {
        self.pricing = Some(pricing);
        self
    }

    fn post(&self, method: &str) -> Result<reqwest::RequestBuilder, ModelError> {
        let http = self.client.post(format!("{}:{method}", self.model_url));
        Ok(match &self.auth {
            Auth::ApiKey(key) => http.header("x-goog-api-key", key),
            Auth::Token(provider) => http.bearer_auth(provider().map_err(ModelError::Other)?),
        })
    }

    pub(crate) fn build_body(&self, req: &ModelRequest) -> Value {
        // functionResponse needs the function's name; results only carry the call id.
        let names: HashMap<&str, &str> = req
            .messages
            .iter()
            .flat_map(|m| m.content.iter())
            .filter_map(|c| match c {
                Content::ToolUse { id, name, .. } => Some((id.as_str(), name.as_str())),
                _ => None,
            })
            .collect();

        let mut contents = Vec::new();
        for m in &req.messages {
            let mut parts = Vec::new();
            let mut pending_sig: HashMap<String, Value> = HashMap::new();
            for c in &m.content {
                match c {
                    Content::Thinking { provider_data: Some(d), .. } if d["provider"] == PROVIDER => {
                        if let (Some(attach), Some(sig)) = (d["attach"].as_str(), d.get("signature")) {
                            pending_sig.insert(attach.to_owned(), sig.clone());
                        }
                    }
                    Content::Thinking { .. } => {}
                    Content::Text { text } => {
                        let mut part = json!({ "text": text });
                        if let Some(sig) = pending_sig.remove("text") {
                            part["thoughtSignature"] = sig;
                        }
                        parts.push(part);
                    }
                    Content::ToolUse { id, name, input } => {
                        let mut call = json!({ "name": name, "args": input });
                        if !id.starts_with(GENERATED_ID) {
                            call["id"] = json!(id);
                        }
                        let mut part = json!({ "functionCall": call });
                        if let Some(sig) = pending_sig.remove(&format!("call:{id}")) {
                            part["thoughtSignature"] = sig;
                        }
                        parts.push(part);
                    }
                    Content::ToolResult { tool_use_id, content, is_error } => {
                        let response = match (content, is_error) {
                            (_, true) => json!({ "error": result_string(content) }),
                            (Value::Object(_), false) => content.clone(),
                            (other, false) => json!({ "result": other }),
                        };
                        let mut fr = json!({
                            "name": names.get(tool_use_id.as_str()).copied().unwrap_or("unknown"),
                            "response": response,
                        });
                        if !tool_use_id.starts_with(GENERATED_ID) {
                            fr["id"] = json!(tool_use_id);
                        }
                        parts.push(json!({ "functionResponse": fr }));
                    }
                }
            }
            if !parts.is_empty() {
                let role = if m.role == Role::User { "user" } else { "model" };
                contents.push(json!({ "role": role, "parts": parts }));
            }
        }

        let mut body = json!({ "contents": contents });
        if let Some(system) = &req.system {
            body["systemInstruction"] = json!({ "parts": [{ "text": system }] });
        }
        if !req.tools.is_empty() {
            let decls: Vec<Value> = req
                .tools
                .iter()
                .map(
                    |t| json!({ "name": t.name, "description": t.description, "parametersJsonSchema": t.input_schema }),
                )
                .collect();
            body["tools"] = json!([{ "functionDeclarations": decls }]);
        }
        let mut config = json!({ "maxOutputTokens": req.max_tokens });
        if let Some(t) = req.temperature {
            config["temperature"] = json!(t);
        }
        body["generationConfig"] = config;
        if let Some(extra) = req.provider_options.get(PROVIDER) {
            deep_merge(&mut body, extra);
        }
        body
    }
}

/// Prefix for tool-call ids Cairn generates when Gemini gives none; such ids
/// are not sent back to Gemini.
const GENERATED_ID: &str = "gc-";

fn deep_merge(target: &mut Value, extra: &Value) {
    match (target, extra) {
        (Value::Object(t), Value::Object(e)) => {
            for (k, v) in e {
                deep_merge(t.entry(k.clone()).or_insert(Value::Null), v);
            }
        }
        (t, e) => *t = e.clone(),
    }
}

/// Accumulates Gemini response parts (one response, or many stream chunks).
#[derive(Default)]
pub(crate) struct Acc {
    model: String,
    content: Vec<Content>,
    finish: Option<String>,
    usage: Usage,
    tools: usize,
}

impl Acc {
    fn new(model: &str) -> Self {
        Self { model: model.to_owned(), ..Default::default() }
    }

    /// Fold in one response or stream chunk, returning stream deltas.
    pub(crate) fn add(&mut self, v: &Value) -> Result<Vec<StreamEvent>, ModelError> {
        if let Some(err) = v.get("error") {
            let status = err["code"].as_u64().unwrap_or(500) as u16;
            return Err(ModelError::Api { status, body: err.to_string() });
        }
        if let Some(m) = v["modelVersion"].as_str() {
            self.model = m.to_owned();
        }
        let u = &v["usageMetadata"];
        if u.is_object() {
            let n = |k: &str| u[k].as_u64().unwrap_or(0);
            let cached = n("cachedContentTokenCount");
            self.usage = Usage {
                input_tokens: n("promptTokenCount").saturating_sub(cached),
                output_tokens: n("candidatesTokenCount") + n("thoughtsTokenCount"),
                cache_read_tokens: cached,
                cache_write_tokens: 0,
            };
        }
        let cand = &v["candidates"][0];
        if cand.is_null() {
            if let Some(reason) = v["promptFeedback"]["blockReason"].as_str() {
                return Err(ModelError::Api { status: 400, body: format!("prompt blocked by Gemini: {reason}") });
            }
        }
        if let Some(f) = cand["finishReason"].as_str() {
            self.finish = Some(f.to_owned());
        }
        let mut deltas = Vec::new();
        for part in cand["content"]["parts"].as_array().into_iter().flatten() {
            let sig = part.get("thoughtSignature").cloned();
            let signature = |attach: String| Content::Thinking {
                text: String::new(),
                provider_data: Some(json!({ "provider": PROVIDER, "attach": attach, "signature": sig })),
            };
            if part["thought"] == true {
                let text = part["text"].as_str().unwrap_or_default().to_owned();
                deltas.push(StreamEvent::ThinkingDelta(text.clone()));
                match self.content.last_mut() {
                    Some(Content::Thinking { text: t, provider_data: Some(d) })
                        if d["attach"] == "thought" && sig.is_none() =>
                    {
                        t.push_str(&text)
                    }
                    _ => self.content.push(Content::Thinking {
                        text,
                        provider_data: Some(json!({ "provider": PROVIDER, "attach": "thought", "signature": sig })),
                    }),
                }
            } else if let Some(fc) = part.get("functionCall") {
                let id =
                    fc["id"].as_str().map_or_else(|| format!("{GENERATED_ID}{}", uuid::Uuid::now_v7()), str::to_owned);
                if sig.is_some() {
                    self.content.push(signature(format!("call:{id}")));
                }
                let name = fc["name"].as_str().unwrap_or_default().to_owned();
                let args = if fc["args"].is_object() { fc["args"].clone() } else { json!({}) };
                deltas.push(StreamEvent::ToolCallStart { index: self.tools, id: id.clone(), name: name.clone() });
                deltas.push(StreamEvent::ToolInputDelta { index: self.tools, partial_json: args.to_string() });
                self.tools += 1;
                self.content.push(Content::ToolUse { id, name, input: args });
            } else if let Some(text) = part["text"].as_str() {
                if sig.is_some() {
                    self.content.push(signature("text".into()));
                }
                deltas.push(StreamEvent::TextDelta(text.to_owned()));
                match self.content.last_mut() {
                    Some(Content::Text { text: t }) => t.push_str(text),
                    _ => self.content.push(Content::Text { text: text.to_owned() }),
                }
            }
        }
        Ok(deltas)
    }

    pub(crate) fn finish(&mut self) -> ModelResponse {
        let stop_reason = match self.finish.as_deref() {
            _ if self.tools > 0 => StopReason::ToolUse,
            Some("STOP") | None => StopReason::EndTurn,
            Some("MAX_TOKENS") => StopReason::MaxTokens,
            Some(other) => StopReason::Other(other.to_lowercase()),
        };
        ModelResponse {
            model: self.model.clone(),
            content: std::mem::take(&mut self.content),
            stop_reason,
            usage: self.usage,
            ..Default::default()
        }
    }

    fn step(&mut self, ev: SseEvent) -> Result<Vec<StreamEvent>, ModelError> {
        let v: Value = serde_json::from_str(&ev.data).map_err(|e| ModelError::Decode(e.to_string()))?;
        self.add(&v)
    }

    fn end(&mut self) -> Result<Vec<StreamEvent>, ModelError> {
        Ok(vec![StreamEvent::Completed(self.finish())])
    }
}

#[async_trait]
impl Model for GeminiModel {
    fn id(&self) -> String {
        format!("{PROVIDER}:{}", self.model)
    }

    async fn complete(&self, request: &ModelRequest) -> Result<ModelResponse, ModelError> {
        let v = send_json(self.post("generateContent")?, &self.build_body(request)).await?;
        let mut acc = Acc::new(&self.model);
        acc.add(&v)?;
        Ok(acc.finish())
    }

    async fn stream(&self, request: &ModelRequest) -> Result<ModelStream, ModelError> {
        let resp = self
            .post("streamGenerateContent")?
            .query(&[("alt", "sse")])
            .json(&self.build_body(request))
            .send()
            .await
            .map_err(|e| ModelError::Transport(e.to_string()))?;
        let resp = sse::check_status(resp).await?;
        Ok(sse::translate(sse::events(resp), Acc::new(&self.model), Acc::step, Acc::end))
    }

    fn pricing(&self) -> Option<Pricing> {
        self.pricing
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn vertex_urls() {
        let m = GeminiModel::vertex("proj", "us-central1", "gemini-x", || Ok("t".into()));
        assert_eq!(
            m.model_url,
            "https://us-central1-aiplatform.googleapis.com/v1/projects/proj/locations/us-central1/publishers/google/models/gemini-x"
        );
        let g = GeminiModel::vertex("proj", "global", "gemini-x", || Ok("t".into()));
        assert!(g.model_url.starts_with("https://aiplatform.googleapis.com/v1/projects/proj/locations/global/"));
    }

    #[test]
    fn options_deep_merge_into_generation_config() {
        let m = GeminiModel::new("gemini-x", "k");
        let req = ModelRequest {
            max_tokens: 50,
            provider_options: [(
                "gemini".into(),
                json!({"generationConfig": {"thinkingConfig": {"thinkingBudget": 128}}}),
            )]
            .into(),
            ..Default::default()
        };
        let body = m.build_body(&req);
        assert_eq!(body["generationConfig"]["maxOutputTokens"], 50, "kept");
        assert_eq!(body["generationConfig"]["thinkingConfig"]["thinkingBudget"], 128, "merged");
    }
}
