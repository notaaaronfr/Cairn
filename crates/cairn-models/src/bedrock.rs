//! Amazon Bedrock adapter (Converse and ConverseStream APIs).
//!
//! Requests are signed with AWS Signature Version 4, or sent with a Bedrock
//! API key. No AWS SDK is needed; to use profiles, SSO or instance roles,
//! pass a [`BedrockAuth::Provider`] that returns current credentials (for
//! example from `aws-config`).

use std::collections::BTreeMap;
use std::sync::Arc;

use async_trait::async_trait;
use cairn_runtime::{
    Content, Model, ModelError, ModelRequest, ModelResponse, ModelStream, Pricing, Role, StopReason, StreamEvent, Usage,
};
use futures::StreamExt;
use serde_json::{Value, json};

use crate::eventstream::{self, Frame};
use crate::result_string;
use crate::sigv4::{self, AwsCredentials};

const PROVIDER: &str = "bedrock";

/// How requests to Bedrock are authenticated.
#[derive(Clone)]
pub enum BedrockAuth {
    /// SigV4 with fixed credentials.
    Credentials(AwsCredentials),
    /// SigV4 with credentials fetched per request (refreshing sources).
    Provider(Arc<dyn Fn() -> Result<AwsCredentials, String> + Send + Sync>),
    /// A Bedrock API key, sent as a bearer token.
    ApiKey(String),
}

pub struct BedrockModel {
    model_id: String,
    region: String,
    endpoint: String,
    auth: BedrockAuth,
    prompt_caching: bool,
    pricing: Option<Pricing>,
    client: reqwest::Client,
}

impl BedrockModel {
    pub fn new(model_id: impl Into<String>, region: impl Into<String>, auth: BedrockAuth) -> Self {
        let region = region.into();
        Self {
            model_id: model_id.into(),
            endpoint: format!("https://bedrock-runtime.{region}.amazonaws.com"),
            region,
            auth,
            prompt_caching: false,
            pricing: None,
            client: reqwest::Client::new(),
        }
    }

    /// Region from `AWS_REGION` / `AWS_DEFAULT_REGION`; auth from
    /// `AWS_BEARER_TOKEN_BEDROCK`, else `AWS_ACCESS_KEY_ID` /
    /// `AWS_SECRET_ACCESS_KEY` / `AWS_SESSION_TOKEN`.
    pub fn from_env(model_id: impl Into<String>) -> Result<Self, ModelError> {
        let get = |k: &str| std::env::var(k).ok().filter(|v| !v.is_empty());
        let region = get("AWS_REGION")
            .or_else(|| get("AWS_DEFAULT_REGION"))
            .ok_or_else(|| ModelError::Other("AWS_REGION is not set".into()))?;
        let auth = match get("AWS_BEARER_TOKEN_BEDROCK") {
            Some(key) => BedrockAuth::ApiKey(key),
            None => BedrockAuth::Credentials(AwsCredentials::from_env().ok_or_else(|| {
                ModelError::Other("set AWS_BEARER_TOKEN_BEDROCK or AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY".into())
            })?),
        };
        Ok(Self::new(model_id, region, auth))
    }

    /// Override the endpoint (VPC endpoints, FIPS endpoints, tests).
    pub fn with_endpoint(mut self, endpoint: impl Into<String>) -> Self {
        self.endpoint = endpoint.into().trim_end_matches('/').to_owned();
        self
    }

    /// Add cache points after the system prompt and tool definitions, for
    /// models that support Bedrock prompt caching.
    pub fn with_prompt_caching(mut self, on: bool) -> Self {
        self.prompt_caching = on;
        self
    }

    pub fn with_pricing(mut self, pricing: Pricing) -> Self {
        self.pricing = Some(pricing);
        self
    }

    pub(crate) fn build_body(&self, req: &ModelRequest) -> Value {
        let messages: Vec<Value> = req
            .messages
            .iter()
            .filter_map(|m| {
                let content: Vec<Value> = m.content.iter().filter_map(block_to_json).collect();
                let role = if m.role == Role::User { "user" } else { "assistant" };
                (!content.is_empty()).then(|| json!({ "role": role, "content": content }))
            })
            .collect();
        let mut body = json!({ "messages": messages });
        let mut config = json!({ "maxTokens": req.max_tokens });
        if let Some(t) = req.temperature {
            config["temperature"] = json!(t);
        }
        body["inferenceConfig"] = config;
        if let Some(system) = &req.system {
            let mut sys = vec![json!({ "text": system })];
            if self.prompt_caching {
                sys.push(json!({ "cachePoint": { "type": "default" } }));
            }
            body["system"] = json!(sys);
        }
        if !req.tools.is_empty() {
            let mut tools: Vec<Value> = req
                .tools
                .iter()
                .map(|t| {
                    json!({ "toolSpec": { "name": t.name, "description": t.description, "inputSchema": { "json": t.input_schema } } })
                })
                .collect();
            if self.prompt_caching {
                tools.push(json!({ "cachePoint": { "type": "default" } }));
            }
            body["toolConfig"] = json!({ "tools": tools });
        }
        if let Some(Value::Object(extra)) = req.provider_options.get(PROVIDER) {
            for (k, v) in extra {
                body[k] = v.clone();
            }
        }
        body
    }

    fn path(&self, action: &str) -> String {
        format!("/model/{}/{action}", sigv4::uri_encode(&self.model_id))
    }

    async fn send(&self, action: &str, body: &Value) -> Result<reqwest::Response, ModelError> {
        let path = self.path(action);
        let bytes = serde_json::to_vec(body).map_err(|e| ModelError::Other(e.to_string()))?;
        let mut req = self.client.post(format!("{}{path}", self.endpoint)).header("content-type", "application/json");
        let creds = match &self.auth {
            BedrockAuth::ApiKey(key) => {
                req = req.bearer_auth(key);
                None
            }
            BedrockAuth::Credentials(c) => Some(c.clone()),
            BedrockAuth::Provider(p) => Some(p().map_err(ModelError::Other)?),
        };
        if let Some(creds) = creds {
            let host = self.endpoint.split("://").nth(1).unwrap_or(&self.endpoint).to_owned();
            for (k, v) in sigv4::sign_post(&creds, &self.region, "bedrock", &host, &path, &bytes, chrono::Utc::now()) {
                req = req.header(k, v);
            }
        }
        let resp = req.body(bytes).send().await.map_err(|e| ModelError::Transport(e.to_string()))?;
        let status = resp.status();
        if !status.is_success() {
            let text = resp.text().await.unwrap_or_default();
            return Err(ModelError::Api { status: status.as_u16(), body: text });
        }
        Ok(resp)
    }

    pub(crate) fn parse_response(&self, v: &Value) -> Result<ModelResponse, ModelError> {
        let blocks = v["output"]["message"]["content"]
            .as_array()
            .ok_or_else(|| ModelError::Decode("missing output.message.content".into()))?;
        let content = blocks.iter().filter_map(block_from_json).collect();
        Ok(ModelResponse {
            model: self.model_id.clone(),
            content,
            stop_reason: stop_reason(v["stopReason"].as_str()),
            usage: usage(&v["usage"]),
            ..Default::default()
        })
    }
}

fn block_to_json(c: &Content) -> Option<Value> {
    Some(match c {
        Content::Text { text } => json!({ "text": text }),
        Content::ToolUse { id, name, input } => json!({ "toolUse": { "toolUseId": id, "name": name, "input": input } }),
        Content::ToolResult { tool_use_id, content, is_error } => {
            let inner = match content {
                Value::Object(_) | Value::Array(_) if !is_error => json!({ "json": content }),
                other => json!({ "text": result_string(other) }),
            };
            json!({ "toolResult": {
                "toolUseId": tool_use_id,
                "content": [inner],
                "status": if *is_error { "error" } else { "success" },
            }})
        }
        Content::Thinking { text, provider_data } => {
            let d = provider_data.as_ref()?;
            if d["provider"] != PROVIDER {
                return None;
            }
            if let Some(redacted) = d.get("redacted") {
                json!({ "reasoningContent": { "redactedContent": redacted } })
            } else {
                json!({ "reasoningContent": { "reasoningText": { "text": text, "signature": d["signature"] } } })
            }
        }
    })
}

fn block_from_json(b: &Value) -> Option<Content> {
    if let Some(text) = b["text"].as_str() {
        return Some(Content::Text { text: text.to_owned() });
    }
    if let Some(t) = b.get("toolUse") {
        return Some(Content::ToolUse {
            id: t["toolUseId"].as_str()?.to_owned(),
            name: t["name"].as_str()?.to_owned(),
            input: t["input"].clone(),
        });
    }
    let r = b.get("reasoningContent")?;
    if let Some(redacted) = r.get("redactedContent") {
        return Some(Content::Thinking {
            text: String::new(),
            provider_data: Some(json!({ "provider": PROVIDER, "redacted": redacted })),
        });
    }
    Some(Content::Thinking {
        text: r["reasoningText"]["text"].as_str().unwrap_or_default().to_owned(),
        provider_data: Some(json!({ "provider": PROVIDER, "signature": r["reasoningText"]["signature"] })),
    })
}

fn stop_reason(r: Option<&str>) -> StopReason {
    match r {
        Some("end_turn") | Some("stop_sequence") => StopReason::EndTurn,
        Some("tool_use") => StopReason::ToolUse,
        Some("max_tokens") => StopReason::MaxTokens,
        Some(other) => StopReason::Other(other.to_owned()),
        None => StopReason::Other("unknown".into()),
    }
}

fn usage(u: &Value) -> Usage {
    let n = |k: &str| u[k].as_u64().unwrap_or(0);
    Usage {
        input_tokens: n("inputTokens"),
        output_tokens: n("outputTokens"),
        cache_read_tokens: n("cacheReadInputTokens"),
        cache_write_tokens: n("cacheWriteInputTokens"),
    }
}

/// ConverseStream accumulator.
#[derive(Default)]
pub(crate) struct StreamAcc {
    model: String,
    /// Content blocks by contentBlockIndex: (kind, text/json, id, name, signature).
    blocks: BTreeMap<u64, Block>,
    tools: usize,
    stop: Option<String>,
    usage: Usage,
    done: bool,
}

#[derive(Default)]
struct Block {
    kind: &'static str,
    text: String,
    id: String,
    name: String,
    signature: String,
    tool_index: usize,
}

impl StreamAcc {
    pub(crate) fn step(&mut self, frame: Frame) -> Result<Vec<StreamEvent>, ModelError> {
        let kind = frame.headers.get(":message-type").map(String::as_str).unwrap_or("event");
        if kind != "event" {
            let name = frame.headers.get(":exception-type").cloned().unwrap_or_else(|| "error".into());
            let status = match name.as_str() {
                "throttlingException" => 429,
                "validationException" => 400,
                "serviceUnavailableException" | "modelNotReadyException" => 503,
                _ => 500,
            };
            return Err(ModelError::Api {
                status,
                body: format!("{name}: {}", String::from_utf8_lossy(&frame.payload)),
            });
        }
        let v: Value = serde_json::from_slice(&frame.payload).map_err(|e| ModelError::Decode(e.to_string()))?;
        let index = v["contentBlockIndex"].as_u64().unwrap_or(0);
        let mut out = Vec::new();
        match frame.headers.get(":event-type").map(String::as_str) {
            Some("contentBlockStart") => {
                if let Some(t) = v["start"].get("toolUse") {
                    let block = Block {
                        kind: "tool",
                        id: t["toolUseId"].as_str().unwrap_or_default().to_owned(),
                        name: t["name"].as_str().unwrap_or_default().to_owned(),
                        tool_index: self.tools,
                        ..Default::default()
                    };
                    out.push(StreamEvent::ToolCallStart {
                        index: self.tools,
                        id: block.id.clone(),
                        name: block.name.clone(),
                    });
                    self.tools += 1;
                    self.blocks.insert(index, block);
                }
            }
            Some("contentBlockDelta") => {
                let d = &v["delta"];
                let block = self.blocks.entry(index).or_default();
                if let Some(text) = d["text"].as_str() {
                    block.kind = "text";
                    block.text.push_str(text);
                    out.push(StreamEvent::TextDelta(text.to_owned()));
                } else if let Some(input) = d["toolUse"]["input"].as_str() {
                    block.text.push_str(input);
                    out.push(StreamEvent::ToolInputDelta { index: block.tool_index, partial_json: input.to_owned() });
                } else if let Some(r) = d.get("reasoningContent") {
                    block.kind = "reasoning";
                    if let Some(text) = r["text"].as_str() {
                        block.text.push_str(text);
                        out.push(StreamEvent::ThinkingDelta(text.to_owned()));
                    }
                    if let Some(sig) = r["signature"].as_str() {
                        block.signature.push_str(sig);
                        out.push(StreamEvent::ThinkingSignature(sig.to_owned()));
                    }
                }
            }
            Some("messageStop") => self.stop = v["stopReason"].as_str().map(str::to_owned),
            Some("metadata") => {
                self.usage = usage(&v["usage"]);
                out.extend(self.complete());
            }
            _ => {}
        }
        Ok(out)
    }

    fn complete(&mut self) -> Option<StreamEvent> {
        if self.done {
            return None;
        }
        self.done = true;
        let content = std::mem::take(&mut self.blocks)
            .into_values()
            .filter_map(|b| match b.kind {
                "tool" => Some(Content::ToolUse {
                    id: b.id,
                    name: b.name,
                    input: if b.text.trim().is_empty() {
                        json!({})
                    } else {
                        serde_json::from_str(&b.text).unwrap_or(json!({}))
                    },
                }),
                "reasoning" => Some(Content::Thinking {
                    text: b.text,
                    provider_data: Some(json!({ "provider": PROVIDER, "signature": b.signature })),
                }),
                "text" => Some(Content::Text { text: b.text }),
                _ => None,
            })
            .collect();
        Some(StreamEvent::Completed(ModelResponse {
            model: self.model.clone(),
            content,
            stop_reason: stop_reason(self.stop.as_deref()),
            usage: self.usage,
            ..Default::default()
        }))
    }

    /// The stream ended; complete if the provider stopped without metadata.
    pub(crate) fn end(&mut self) -> Result<Vec<StreamEvent>, ModelError> {
        if self.done {
            return Ok(Vec::new());
        }
        if self.stop.is_some() {
            return Ok(self.complete().into_iter().collect());
        }
        Err(ModelError::Transport("Bedrock stream ended before messageStop".into()))
    }
}

#[async_trait]
impl Model for BedrockModel {
    fn id(&self) -> String {
        format!("{PROVIDER}:{}", self.model_id)
    }

    async fn complete(&self, request: &ModelRequest) -> Result<ModelResponse, ModelError> {
        let resp = self.send("converse", &self.build_body(request)).await?;
        let v: Value = resp.json().await.map_err(|e| ModelError::Decode(e.to_string()))?;
        self.parse_response(&v)
    }

    async fn stream(&self, request: &ModelRequest) -> Result<ModelStream, ModelError> {
        let resp = self.send("converse-stream", &self.build_body(request)).await?;
        let acc = StreamAcc { model: self.model_id.clone(), ..Default::default() };
        let state = (eventstream::frames(resp), acc, std::collections::VecDeque::<StreamEvent>::new(), false);
        Ok(Box::pin(futures::stream::unfold(state, |(mut frames, mut acc, mut queue, mut ended)| async move {
            loop {
                if let Some(ev) = queue.pop_front() {
                    return Some((Ok(ev), (frames, acc, queue, ended)));
                }
                if ended {
                    return None;
                }
                match frames.next().await {
                    Some(Ok(frame)) => match acc.step(frame) {
                        Ok(evs) => queue.extend(evs),
                        Err(e) => return Some((Err(e), (frames, acc, queue, true))),
                    },
                    Some(Err(e)) => return Some((Err(e), (frames, acc, queue, true))),
                    None => {
                        ended = true;
                        match acc.end() {
                            Ok(evs) => queue.extend(evs),
                            Err(e) => return Some((Err(e), (frames, acc, queue, true))),
                        }
                    }
                }
            }
        })))
    }

    fn pricing(&self) -> Option<Pricing> {
        self.pricing
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::eventstream::{encode, parse_frame};
    use cairn_runtime::{Effect, Message, ToolSpec};

    fn model() -> BedrockModel {
        BedrockModel::new("anthropic.claude-test-v1:0", "us-east-1", BedrockAuth::ApiKey("k".into()))
    }

    #[test]
    fn converse_round_trip() {
        let req = ModelRequest {
            system: Some("be brief".into()),
            messages: vec![
                Message::user("refund 7"),
                Message::assistant(vec![
                    Content::Thinking {
                        text: "check".into(),
                        provider_data: Some(json!({"provider": "bedrock", "signature": "s1"})),
                    },
                    Content::Thinking { text: "x".into(), provider_data: Some(json!({"provider": "anthropic"})) },
                    Content::ToolUse { id: "t1".into(), name: "refund".into(), input: json!({"id": 7}) },
                ]),
                Message {
                    role: Role::User,
                    content: vec![Content::ToolResult {
                        tool_use_id: "t1".into(),
                        content: json!({"ok": true}),
                        is_error: false,
                    }],
                },
            ],
            tools: vec![ToolSpec {
                name: "refund".into(),
                description: "d".into(),
                input_schema: json!({"type": "object"}),
                effect: Effect::Write,
            }],
            max_tokens: 100,
            provider_options: [(
                "bedrock".into(),
                json!({"additionalModelRequestFields": {"thinking": {"type": "enabled"}}}),
            )]
            .into(),
            ..Default::default()
        };
        let body = model().with_prompt_caching(true).build_body(&req);
        assert_eq!(body["system"][1]["cachePoint"]["type"], "default");
        assert_eq!(body["toolConfig"]["tools"][0]["toolSpec"]["inputSchema"]["json"]["type"], "object");
        let assistant = &body["messages"][1]["content"];
        assert_eq!(assistant.as_array().unwrap().len(), 2, "foreign thinking dropped");
        assert_eq!(assistant[0]["reasoningContent"]["reasoningText"]["signature"], "s1");
        assert_eq!(body["messages"][2]["content"][0]["toolResult"]["content"][0]["json"]["ok"], true);
        assert_eq!(body["additionalModelRequestFields"]["thinking"]["type"], "enabled");
        assert_eq!(model().path("converse"), "/model/anthropic.claude-test-v1%3A0/converse");

        let resp = model()
            .parse_response(&json!({
                "output": {"message": {"role": "assistant", "content": [
                    {"reasoningContent": {"reasoningText": {"text": "hmm", "signature": "s2"}}},
                    {"text": "Calling"},
                    {"toolUse": {"toolUseId": "t2", "name": "refund", "input": {"id": 8}}}
                ]}},
                "stopReason": "tool_use",
                "usage": {"inputTokens": 10, "outputTokens": 5, "cacheReadInputTokens": 3}
            }))
            .unwrap();
        assert_eq!(resp.stop_reason, StopReason::ToolUse);
        assert_eq!(
            resp.content[2],
            Content::ToolUse { id: "t2".into(), name: "refund".into(), input: json!({"id": 8}) }
        );
        assert_eq!(resp.usage.cache_read_tokens, 3);
    }

    fn ev(kind: &str, payload: Value) -> Frame {
        parse_frame(&encode(&[(":event-type", kind), (":message-type", "event")], payload.to_string().as_bytes()))
            .unwrap()
    }

    #[test]
    fn converse_stream_accumulates() {
        let mut acc = StreamAcc { model: "m".into(), ..Default::default() };
        let frames = vec![
            ev("messageStart", json!({"role": "assistant"})),
            ev("contentBlockDelta", json!({"contentBlockIndex": 0, "delta": {"text": "Hel"}})),
            ev("contentBlockDelta", json!({"contentBlockIndex": 0, "delta": {"text": "lo"}})),
            ev(
                "contentBlockStart",
                json!({"contentBlockIndex": 1, "start": {"toolUse": {"toolUseId": "t1", "name": "f"}}}),
            ),
            ev("contentBlockDelta", json!({"contentBlockIndex": 1, "delta": {"toolUse": {"input": "{\"a\":"}}})),
            ev("contentBlockDelta", json!({"contentBlockIndex": 1, "delta": {"toolUse": {"input": "1}"}}})),
            ev("messageStop", json!({"stopReason": "tool_use"})),
            ev("metadata", json!({"usage": {"inputTokens": 4, "outputTokens": 6}})),
        ];
        let events: Vec<StreamEvent> = frames.into_iter().flat_map(|f| acc.step(f).unwrap()).collect();
        assert_eq!(events[0], StreamEvent::TextDelta("Hel".into()));
        assert!(events.contains(&StreamEvent::ToolCallStart { index: 0, id: "t1".into(), name: "f".into() }));
        let StreamEvent::Completed(r) = events.last().unwrap() else { panic!() };
        assert_eq!(r.content[0], Content::Text { text: "Hello".into() });
        assert_eq!(r.content[1], Content::ToolUse { id: "t1".into(), name: "f".into(), input: json!({"a": 1}) });
        assert_eq!(r.usage.output_tokens, 6);
        assert!(acc.end().unwrap().is_empty(), "no duplicate completion");

        let throttled =
            parse_frame(&encode(&[(":message-type", "exception"), (":exception-type", "throttlingException")], b"{}"))
                .unwrap();
        assert!(StreamAcc::default().step(throttled).unwrap_err().retryable());
    }
}
