use async_trait::async_trait;
use cairn_runtime::{
    Content, Message, Model, ModelError, ModelRequest, ModelResponse, ModelStream, Role, StopReason, StreamEvent, Usage,
};
use serde_json::{Value, json};

use crate::sse::{self, SseEvent};
use crate::{Pricing, merge_options, result_string, send_json};

const PROVIDER: &str = "anthropic";

/// Native Anthropic Messages API adapter.
///
/// Signed thinking blocks are kept and replayed to Anthropic on later turns,
/// and dropped automatically when the conversation is sent to another provider.
pub struct AnthropicModel {
    model: String,
    api_key: String,
    base_url: String,
    version: String,
    prompt_caching: bool,
    pricing: Option<Pricing>,
    client: reqwest::Client,
}

impl AnthropicModel {
    pub fn new(model: impl Into<String>, api_key: impl Into<String>) -> Self {
        Self {
            model: model.into(),
            api_key: api_key.into(),
            base_url: "https://api.anthropic.com".into(),
            version: "2023-06-01".into(),
            prompt_caching: true,
            pricing: None,
            client: reqwest::Client::new(),
        }
    }

    pub fn with_base_url(mut self, url: impl Into<String>) -> Self {
        self.base_url = url.into();
        self
    }

    /// Cache the system prompt and tool definitions (on by default). Agent
    /// loops resend them every step, so this cuts cost and latency sharply.
    pub fn with_prompt_caching(mut self, on: bool) -> Self {
        self.prompt_caching = on;
        self
    }

    pub fn with_pricing(mut self, pricing: Pricing) -> Self {
        self.pricing = Some(pricing);
        self
    }

    pub(crate) fn build_body(&self, req: &ModelRequest) -> Value {
        let cache = json!({ "type": "ephemeral" });
        let messages: Vec<Value> = req
            .messages
            .iter()
            .filter_map(|m| {
                let content: Vec<Value> = m.content.iter().filter_map(block_to_json).collect();
                (!content.is_empty()).then(|| json!({ "role": role(m), "content": content }))
            })
            .collect();

        let mut body = json!({
            "model": self.model,
            "max_tokens": req.max_tokens,
            "messages": messages,
        });

        if let Some(system) = &req.system {
            body["system"] = if self.prompt_caching {
                json!([{ "type": "text", "text": system, "cache_control": cache }])
            } else {
                json!(system)
            };
        }
        if !req.tools.is_empty() {
            let mut tools: Vec<Value> = req
                .tools
                .iter()
                .map(|t| json!({ "name": t.name, "description": t.description, "input_schema": t.input_schema }))
                .collect();
            if self.prompt_caching {
                if let Some(last) = tools.last_mut() {
                    last["cache_control"] = cache.clone();
                }
            }
            body["tools"] = json!(tools);
        }
        if let Some(t) = req.temperature {
            body["temperature"] = json!(t);
        }
        merge_options(&mut body, req.provider_options.get(PROVIDER));
        body
    }

    pub(crate) fn parse_response(v: &Value) -> Result<ModelResponse, ModelError> {
        let blocks = v["content"].as_array().ok_or_else(|| ModelError::Decode("missing `content`".into()))?;
        let content = blocks
            .iter()
            .filter_map(|b| match b["type"].as_str()? {
                "text" => Some(Content::Text { text: b["text"].as_str()?.to_owned() }),
                "tool_use" => Some(Content::ToolUse {
                    id: b["id"].as_str()?.to_owned(),
                    name: b["name"].as_str()?.to_owned(),
                    input: b["input"].clone(),
                }),
                "thinking" => Some(Content::Thinking {
                    text: b["thinking"].as_str().unwrap_or_default().to_owned(),
                    provider_data: Some(json!({ "provider": PROVIDER, "signature": b["signature"] })),
                }),
                "redacted_thinking" => Some(Content::Thinking {
                    text: String::new(),
                    provider_data: Some(json!({ "provider": PROVIDER, "redacted_data": b["data"] })),
                }),
                _ => None,
            })
            .collect();

        let stop_reason = match v["stop_reason"].as_str() {
            Some("end_turn") | Some("stop_sequence") => StopReason::EndTurn,
            Some("tool_use") => StopReason::ToolUse,
            Some("max_tokens") => StopReason::MaxTokens,
            Some(other) => StopReason::Other(other.to_owned()),
            None => StopReason::Other("unknown".into()),
        };
        let u = &v["usage"];
        let n = |k: &str| u[k].as_u64().unwrap_or(0);
        Ok(ModelResponse {
            model: v["model"].as_str().unwrap_or_default().to_owned(),
            content,
            stop_reason,
            usage: Usage {
                input_tokens: n("input_tokens"),
                output_tokens: n("output_tokens"),
                cache_read_tokens: n("cache_read_input_tokens"),
                cache_write_tokens: n("cache_creation_input_tokens"),
            },
            ..Default::default()
        })
    }
}

fn role(m: &Message) -> &'static str {
    match m.role {
        Role::User => "user",
        Role::Assistant => "assistant",
    }
}

fn block_to_json(c: &Content) -> Option<Value> {
    Some(match c {
        Content::Text { text } => json!({ "type": "text", "text": text }),
        Content::ToolUse { id, name, input } => json!({ "type": "tool_use", "id": id, "name": name, "input": input }),
        Content::ToolResult { tool_use_id, content, is_error } => json!({
            "type": "tool_result",
            "tool_use_id": tool_use_id,
            "content": result_string(content),
            "is_error": is_error,
        }),
        Content::Thinking { text, provider_data } => {
            let data = provider_data.as_ref()?;
            if data["provider"] != PROVIDER {
                return None;
            }
            if !data["redacted_data"].is_null() {
                json!({ "type": "redacted_thinking", "data": data["redacted_data"] })
            } else {
                json!({ "type": "thinking", "thinking": text, "signature": data["signature"] })
            }
        }
    })
}

#[async_trait]
impl Model for AnthropicModel {
    fn id(&self) -> String {
        format!("{PROVIDER}:{}", self.model)
    }

    async fn complete(&self, request: &ModelRequest) -> Result<ModelResponse, ModelError> {
        let body = self.build_body(request);
        let http = self
            .client
            .post(format!("{}/v1/messages", self.base_url))
            .header("x-api-key", &self.api_key)
            .header("anthropic-version", &self.version);
        let v = send_json(http, &body).await?;
        Self::parse_response(&v)
    }

    async fn stream(&self, request: &ModelRequest) -> Result<ModelStream, ModelError> {
        let mut body = self.build_body(request);
        body["stream"] = json!(true);
        let resp = self
            .client
            .post(format!("{}/v1/messages", self.base_url))
            .header("x-api-key", &self.api_key)
            .header("anthropic-version", &self.version)
            .json(&body)
            .send()
            .await
            .map_err(|e| ModelError::Transport(e.to_string()))?;
        let resp = sse::check_status(resp).await?;
        Ok(sse::translate(sse::events(resp), StreamAcc::default(), StreamAcc::step, StreamAcc::end))
    }

    fn pricing(&self) -> Option<Pricing> {
        self.pricing
    }
}

enum Block {
    Text(String),
    Thinking { text: String, signature: String },
    Redacted(Value),
    Tool { id: String, name: String, json: String, index: usize },
}

#[derive(Default)]
pub(crate) struct StreamAcc {
    model: String,
    blocks: Vec<Block>,
    tools: usize,
    usage: Usage,
    stop_reason: Option<String>,
}

impl StreamAcc {
    pub(crate) fn step(&mut self, ev: SseEvent) -> Result<Vec<StreamEvent>, ModelError> {
        let v: Value = serde_json::from_str(&ev.data).map_err(|e| ModelError::Decode(e.to_string()))?;
        let mut out = Vec::new();
        match v["type"].as_str().unwrap_or_default() {
            "message_start" => {
                let m = &v["message"];
                self.model = m["model"].as_str().unwrap_or_default().to_owned();
                self.read_usage(&m["usage"]);
            }
            "content_block_start" => {
                let b = &v["content_block"];
                self.blocks.push(match b["type"].as_str().unwrap_or_default() {
                    "tool_use" => {
                        let (id, name) = (b["id"].as_str().unwrap_or_default(), b["name"].as_str().unwrap_or_default());
                        out.push(StreamEvent::ToolCallStart { index: self.tools, id: id.into(), name: name.into() });
                        self.tools += 1;
                        Block::Tool { id: id.into(), name: name.into(), json: String::new(), index: self.tools - 1 }
                    }
                    "thinking" => Block::Thinking { text: String::new(), signature: String::new() },
                    "redacted_thinking" => Block::Redacted(b["data"].clone()),
                    _ => Block::Text(String::new()),
                });
            }
            "content_block_delta" => {
                let d = &v["delta"];
                let piece = |k: &str| d[k].as_str().unwrap_or_default().to_owned();
                match (self.blocks.last_mut(), d["type"].as_str().unwrap_or_default()) {
                    (Some(Block::Text(t)), "text_delta") => {
                        t.push_str(&piece("text"));
                        out.push(StreamEvent::TextDelta(piece("text")));
                    }
                    (Some(Block::Thinking { text, .. }), "thinking_delta") => {
                        text.push_str(&piece("thinking"));
                        out.push(StreamEvent::ThinkingDelta(piece("thinking")));
                    }
                    (Some(Block::Thinking { signature, .. }), "signature_delta") => {
                        signature.push_str(&piece("signature"));
                        out.push(StreamEvent::ThinkingSignature(piece("signature")));
                    }
                    (Some(Block::Tool { json, index, .. }), "input_json_delta") => {
                        json.push_str(&piece("partial_json"));
                        out.push(StreamEvent::ToolInputDelta { index: *index, partial_json: piece("partial_json") });
                    }
                    _ => {}
                }
            }
            "message_delta" => {
                if let Some(r) = v["delta"]["stop_reason"].as_str() {
                    self.stop_reason = Some(r.to_owned());
                }
                self.read_usage(&v["usage"]);
            }
            "message_stop" => out.push(StreamEvent::Completed(self.finish()?)),
            "error" => {
                let kind = v["error"]["type"].as_str().unwrap_or_default();
                let status = match kind {
                    "overloaded_error" => 529,
                    "rate_limit_error" => 429,
                    _ => 500,
                };
                return Err(ModelError::Api { status, body: v["error"].to_string() });
            }
            _ => {}
        }
        Ok(out)
    }

    pub(crate) fn end(&mut self) -> Result<Vec<StreamEvent>, ModelError> {
        Err(ModelError::Transport("stream closed before message_stop".into()))
    }

    fn read_usage(&mut self, u: &Value) {
        let set = |field: &mut u64, k: &str| {
            if let Some(n) = u[k].as_u64() {
                *field = n;
            }
        };
        set(&mut self.usage.input_tokens, "input_tokens");
        set(&mut self.usage.output_tokens, "output_tokens");
        set(&mut self.usage.cache_read_tokens, "cache_read_input_tokens");
        set(&mut self.usage.cache_write_tokens, "cache_creation_input_tokens");
    }

    fn finish(&mut self) -> Result<ModelResponse, ModelError> {
        let content = std::mem::take(&mut self.blocks)
            .into_iter()
            .map(|b| match b {
                Block::Text(text) => Content::Text { text },
                Block::Thinking { text, signature } => Content::Thinking {
                    text,
                    provider_data: Some(json!({ "provider": PROVIDER, "signature": signature })),
                },
                Block::Redacted(data) => Content::Thinking {
                    text: String::new(),
                    provider_data: Some(json!({ "provider": PROVIDER, "redacted_data": data })),
                },
                Block::Tool { id, name, json, .. } => Content::ToolUse {
                    id,
                    name,
                    input: if json.trim().is_empty() { json!({}) } else { serde_json::from_str(&json).unwrap_or(json!({})) },
                },
            })
            .collect();
        let response = Self::parse_stop(self.stop_reason.as_deref());
        Ok(ModelResponse { model: self.model.clone(), content, stop_reason: response, usage: self.usage, ..Default::default() })
    }

    fn parse_stop(r: Option<&str>) -> StopReason {
        match r {
            Some("end_turn") | Some("stop_sequence") => StopReason::EndTurn,
            Some("tool_use") => StopReason::ToolUse,
            Some("max_tokens") => StopReason::MaxTokens,
            Some(other) => StopReason::Other(other.to_owned()),
            None => StopReason::Other("unknown".into()),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use cairn_runtime::{Effect, ToolSpec};

    fn request() -> ModelRequest {
        ModelRequest {
            system: Some("be terse".into()),
            messages: vec![
                Message::user("refund order 7"),
                Message::assistant(vec![
                    Content::Thinking {
                        text: "need order".into(),
                        provider_data: Some(json!({"provider": "anthropic", "signature": "sig"})),
                    },
                    Content::Thinking { text: "foreign".into(), provider_data: Some(json!({"provider": "openai"})) },
                    Content::ToolUse { id: "t1".into(), name: "get_order".into(), input: json!({"id": 7}) },
                ]),
                Message {
                    role: Role::User,
                    content: vec![Content::ToolResult {
                        tool_use_id: "t1".into(),
                        content: json!({"total": 10}),
                        is_error: false,
                    }],
                },
            ],
            tools: vec![ToolSpec {
                name: "get_order".into(),
                description: "d".into(),
                input_schema: json!({"type": "object"}),
                effect: Effect::Read,
            }],
            max_tokens: 100,
            provider_options: [("anthropic".to_owned(), json!({"thinking": {"type": "enabled", "budget_tokens": 1024}}))]
                .into(),
            ..Default::default()
        }
    }

    #[test]
    fn builds_body_with_caching_thinking_and_options() {
        let body = AnthropicModel::new("claude-x", "k").build_body(&request());
        assert_eq!(body["system"][0]["cache_control"]["type"], "ephemeral");
        assert_eq!(body["tools"][0]["cache_control"]["type"], "ephemeral");
        assert_eq!(body["thinking"]["budget_tokens"], 1024);
        let assistant = &body["messages"][1]["content"];
        assert_eq!(assistant.as_array().unwrap().len(), 2, "foreign thinking dropped");
        assert_eq!(assistant[0]["signature"], "sig");
        assert_eq!(body["messages"][2]["content"][0]["content"], r#"{"total":10}"#);
    }

    #[test]
    fn parses_response() {
        let v = json!({
            "model": "claude-x",
            "stop_reason": "tool_use",
            "content": [
                {"type": "thinking", "thinking": "hmm", "signature": "s"},
                {"type": "text", "text": "checking"},
                {"type": "tool_use", "id": "t9", "name": "get_order", "input": {"id": 1}}
            ],
            "usage": {"input_tokens": 10, "output_tokens": 5, "cache_read_input_tokens": 90}
        });
        let r = AnthropicModel::parse_response(&v).unwrap();
        assert_eq!(r.stop_reason, StopReason::ToolUse);
        assert_eq!(r.content.len(), 3);
        assert_eq!(r.usage.cache_read_tokens, 90);
    }

    #[test]
    fn accumulates_stream() {
        let raw = r#"event: message_start
data: {"type":"message_start","message":{"model":"claude-x","usage":{"input_tokens":12}}}

event: content_block_start
data: {"type":"content_block_start","index":0,"content_block":{"type":"thinking","thinking":""}}

event: content_block_delta
data: {"type":"content_block_delta","index":0,"delta":{"type":"thinking_delta","thinking":"hmm"}}

event: content_block_delta
data: {"type":"content_block_delta","index":0,"delta":{"type":"signature_delta","signature":"sig"}}

event: content_block_start
data: {"type":"content_block_start","index":1,"content_block":{"type":"text","text":""}}

event: ping
data: {"type":"ping"}

event: content_block_delta
data: {"type":"content_block_delta","index":1,"delta":{"type":"text_delta","text":"Look"}}

event: content_block_delta
data: {"type":"content_block_delta","index":1,"delta":{"type":"text_delta","text":"ing"}}

event: content_block_start
data: {"type":"content_block_start","index":2,"content_block":{"type":"tool_use","id":"tu1","name":"get_order","input":{}}}

event: content_block_delta
data: {"type":"content_block_delta","index":2,"delta":{"type":"input_json_delta","partial_json":"{\"id\":"}}

event: content_block_delta
data: {"type":"content_block_delta","index":2,"delta":{"type":"input_json_delta","partial_json":"7}"}}

event: message_delta
data: {"type":"message_delta","delta":{"stop_reason":"tool_use"},"usage":{"output_tokens":30}}

event: message_stop
data: {"type":"message_stop"}
"#;
        let mut acc = StreamAcc::default();
        let events: Vec<StreamEvent> =
            sse::parse_text(raw).into_iter().flat_map(|e| acc.step(e).unwrap()).collect();
        assert_eq!(events[0], StreamEvent::ThinkingDelta("hmm".into()));
        assert!(events.contains(&StreamEvent::TextDelta("Look".into())));
        assert!(events.contains(&StreamEvent::ToolCallStart { index: 0, id: "tu1".into(), name: "get_order".into() }));
        let StreamEvent::Completed(r) = events.last().unwrap() else { panic!("no completion") };
        assert_eq!(r.stop_reason, StopReason::ToolUse);
        assert_eq!(r.usage.input_tokens, 12);
        assert_eq!(r.usage.output_tokens, 30);
        assert_eq!(r.content[1], Content::Text { text: "Looking".into() });
        assert_eq!(r.content[2], Content::ToolUse { id: "tu1".into(), name: "get_order".into(), input: json!({"id": 7}) });
        assert!(matches!(&r.content[0], Content::Thinking { provider_data: Some(d), .. } if d["signature"] == "sig"));
    }

    #[test]
    fn stream_error_event_is_retryable_when_overloaded() {
        let mut acc = StreamAcc::default();
        let ev = SseEvent { event: Some("error".into()), data: r#"{"type":"error","error":{"type":"overloaded_error","message":"busy"}}"#.into() };
        assert!(acc.step(ev).unwrap_err().retryable());
    }
}
