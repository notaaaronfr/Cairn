use async_trait::async_trait;
use cairn_runtime::{Content, Model, ModelError, ModelRequest, ModelResponse, Role, StopReason, Usage};
use serde_json::{Value, json};

use crate::{Pricing, merge_options, result_string, send_json};

/// Chat Completions adapter for any OpenAI-compatible endpoint.
pub struct OpenAiCompatModel {
    provider: String,
    model: String,
    base_url: String,
    api_key: Option<String>,
    max_tokens_field: &'static str,
    pricing: Option<Pricing>,
    client: reqwest::Client,
}

impl OpenAiCompatModel {
    /// Generic endpoint. `provider` names the key used for
    /// `provider_options` and the prefix of [`Model::id`].
    pub fn new(
        provider: impl Into<String>,
        model: impl Into<String>,
        base_url: impl Into<String>,
        api_key: Option<String>,
    ) -> Self {
        Self {
            provider: provider.into(),
            model: model.into(),
            base_url: base_url.into().trim_end_matches('/').to_owned(),
            api_key,
            max_tokens_field: "max_tokens",
            pricing: None,
            client: reqwest::Client::new(),
        }
    }

    pub fn openai(model: impl Into<String>, api_key: impl Into<String>) -> Self {
        let mut m = Self::new("openai", model, "https://api.openai.com/v1", Some(api_key.into()));
        m.max_tokens_field = "max_completion_tokens";
        m
    }

    pub fn gemini(model: impl Into<String>, api_key: impl Into<String>) -> Self {
        Self::new("gemini", model, "https://generativelanguage.googleapis.com/v1beta/openai", Some(api_key.into()))
    }

    pub fn openrouter(model: impl Into<String>, api_key: impl Into<String>) -> Self {
        Self::new("openrouter", model, "https://openrouter.ai/api/v1", Some(api_key.into()))
    }

    pub fn with_pricing(mut self, pricing: Pricing) -> Self {
        self.pricing = Some(pricing);
        self
    }

    pub(crate) fn build_body(&self, req: &ModelRequest) -> Value {
        let mut messages = Vec::new();
        if let Some(system) = &req.system {
            messages.push(json!({ "role": "system", "content": system }));
        }
        for m in &req.messages {
            match m.role {
                Role::User => {
                    let mut text = String::new();
                    for c in &m.content {
                        match c {
                            Content::ToolResult { tool_use_id, content, .. } => messages.push(json!({
                                "role": "tool",
                                "tool_call_id": tool_use_id,
                                "content": result_string(content),
                            })),
                            Content::Text { text: t } => text.push_str(t),
                            _ => {}
                        }
                    }
                    if !text.is_empty() {
                        messages.push(json!({ "role": "user", "content": text }));
                    }
                }
                Role::Assistant => {
                    let text = cairn_runtime::message::content_text(&m.content);
                    let calls: Vec<Value> = m
                        .content
                        .iter()
                        .filter_map(|c| match c {
                            Content::ToolUse { id, name, input } => Some(json!({
                                "id": id,
                                "type": "function",
                                "function": { "name": name, "arguments": input.to_string() },
                            })),
                            _ => None,
                        })
                        .collect();
                    let mut msg = json!({
                        "role": "assistant",
                        "content": if text.is_empty() { Value::Null } else { json!(text) },
                    });
                    if !calls.is_empty() {
                        msg["tool_calls"] = json!(calls);
                    }
                    messages.push(msg);
                }
            }
        }

        let mut body = json!({ "model": self.model, "messages": messages });
        body[self.max_tokens_field] = json!(req.max_tokens);
        if !req.tools.is_empty() {
            body["tools"] = json!(
                req.tools
                    .iter()
                    .map(|t| json!({
                        "type": "function",
                        "function": { "name": t.name, "description": t.description, "parameters": t.input_schema },
                    }))
                    .collect::<Vec<_>>()
            );
        }
        if let Some(t) = req.temperature {
            body["temperature"] = json!(t);
        }
        merge_options(&mut body, req.provider_options.get(&self.provider));
        body
    }

    pub(crate) fn parse_response(&self, v: &Value) -> Result<ModelResponse, ModelError> {
        let choice = &v["choices"][0];
        let msg = &choice["message"];
        if msg.is_null() {
            return Err(ModelError::Decode("missing `choices[0].message`".into()));
        }

        let mut content = Vec::new();
        if let Some(r) = msg["reasoning_content"].as_str().filter(|s| !s.is_empty()) {
            content.push(Content::Thinking { text: r.to_owned(), provider_data: Some(json!({ "provider": self.provider })) });
        }
        if let Some(t) = msg["content"].as_str().filter(|s| !s.is_empty()) {
            content.push(Content::Text { text: t.to_owned() });
        }
        for call in msg["tool_calls"].as_array().into_iter().flatten() {
            let args = call["function"]["arguments"].as_str().unwrap_or("{}");
            content.push(Content::ToolUse {
                id: call["id"].as_str().unwrap_or_default().to_owned(),
                name: call["function"]["name"].as_str().unwrap_or_default().to_owned(),
                input: serde_json::from_str(args).unwrap_or_else(|_| json!({ "_unparsed_arguments": args })),
            });
        }

        let stop_reason = match choice["finish_reason"].as_str() {
            Some("stop") => StopReason::EndTurn,
            Some("tool_calls") | Some("function_call") => StopReason::ToolUse,
            Some("length") => StopReason::MaxTokens,
            Some(other) => StopReason::Other(other.to_owned()),
            None => StopReason::Other("unknown".into()),
        };
        let u = &v["usage"];
        let prompt = u["prompt_tokens"].as_u64().unwrap_or(0);
        let cached = u["prompt_tokens_details"]["cached_tokens"].as_u64().unwrap_or(0);
        Ok(ModelResponse {
            model: v["model"].as_str().unwrap_or(&self.model).to_owned(),
            content,
            stop_reason,
            usage: Usage {
                input_tokens: prompt.saturating_sub(cached),
                output_tokens: u["completion_tokens"].as_u64().unwrap_or(0),
                cache_read_tokens: cached,
                cache_write_tokens: 0,
            },
        })
    }
}

#[async_trait]
impl Model for OpenAiCompatModel {
    fn id(&self) -> String {
        format!("{}:{}", self.provider, self.model)
    }

    async fn complete(&self, request: &ModelRequest) -> Result<ModelResponse, ModelError> {
        let body = self.build_body(request);
        let mut http = self.client.post(format!("{}/chat/completions", self.base_url));
        if let Some(key) = &self.api_key {
            http = http.bearer_auth(key);
        }
        let v = send_json(http, &body).await?;
        self.parse_response(&v)
    }

    fn cost_usd(&self, response: &ModelResponse) -> Option<f64> {
        self.pricing.map(|p| p.cost(&response.usage))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use cairn_runtime::Message;

    #[test]
    fn round_trips_tool_calls() {
        let m = OpenAiCompatModel::openai("gpt-x", "k");
        let req = ModelRequest {
            system: Some("sys".into()),
            messages: vec![
                Message::user("hi"),
                Message::assistant(vec![
                    Content::Thinking { text: "x".into(), provider_data: Some(json!({"provider": "anthropic"})) },
                    Content::ToolUse { id: "c1".into(), name: "f".into(), input: json!({"a": 1}) },
                ]),
                Message {
                    role: Role::User,
                    content: vec![Content::ToolResult { tool_use_id: "c1".into(), content: json!("ok"), is_error: false }],
                },
            ],
            max_tokens: 50,
            ..Default::default()
        };
        let body = m.build_body(&req);
        assert_eq!(body["max_completion_tokens"], 50);
        let msgs = body["messages"].as_array().unwrap();
        assert_eq!(msgs.len(), 4);
        assert_eq!(msgs[2]["tool_calls"][0]["function"]["arguments"], r#"{"a":1}"#);
        assert!(msgs[2]["content"].is_null());
        assert_eq!(msgs[3]["role"], "tool");

        let resp = m
            .parse_response(&json!({
                "model": "gpt-x",
                "choices": [{"finish_reason": "tool_calls", "message": {"content": null, "tool_calls": [
                    {"id": "c2", "type": "function", "function": {"name": "f", "arguments": "{\"a\":2}"}}
                ]}}],
                "usage": {"prompt_tokens": 100, "completion_tokens": 7, "prompt_tokens_details": {"cached_tokens": 60}}
            }))
            .unwrap();
        assert_eq!(resp.stop_reason, StopReason::ToolUse);
        assert_eq!(resp.content[0], Content::ToolUse { id: "c2".into(), name: "f".into(), input: json!({"a": 2}) });
        assert_eq!(resp.usage.input_tokens, 40);
        assert_eq!(resp.usage.cache_read_tokens, 60);
    }
}
