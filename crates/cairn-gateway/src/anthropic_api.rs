//! Anthropic Messages wire format <-> Cairn's neutral types.

use cairn_runtime::{Content, Effect, Message, ModelRequest, ModelResponse, Role, StopReason, StreamEvent, ToolSpec};
use serde_json::{Map, Value, json};

use crate::{GatewayError, SseRenderer};

const KNOWN: &[&str] = &["model", "messages", "system", "tools", "max_tokens", "temperature", "stream"];

pub fn parse_request(body: &Value) -> Result<(String, ModelRequest, bool), GatewayError> {
    let model = body["model"].as_str().ok_or_else(|| GatewayError::bad_request("`model` is required"))?.to_owned();
    let max_tokens =
        body["max_tokens"].as_u64().ok_or_else(|| GatewayError::bad_request("`max_tokens` is required"))? as u32;

    let system = match &body["system"] {
        Value::Null => None,
        Value::String(s) => Some(s.clone()),
        Value::Array(blocks) => Some(blocks.iter().filter_map(|b| b["text"].as_str()).collect::<Vec<_>>().join("\n\n")),
        _ => return Err(GatewayError::bad_request("invalid `system`")),
    };

    let mut messages = Vec::new();
    for m in body["messages"].as_array().ok_or_else(|| GatewayError::bad_request("`messages` must be an array"))? {
        let role = match m["role"].as_str() {
            Some("user") => Role::User,
            Some("assistant") => Role::Assistant,
            _ => return Err(GatewayError::bad_request("message role must be `user` or `assistant`")),
        };
        let content = match &m["content"] {
            Value::String(s) => vec![Content::Text { text: s.clone() }],
            Value::Array(blocks) => blocks.iter().map(block).collect::<Result<Vec<_>, _>>()?,
            _ => return Err(GatewayError::bad_request("invalid message content")),
        };
        messages.push(Message { role, content });
    }

    let mut tools = Vec::new();
    for t in body["tools"].as_array().into_iter().flatten() {
        if let Some(kind) = t["type"].as_str().filter(|k| *k != "custom") {
            return Err(GatewayError::bad_request(format!("server tool `{kind}` is not supported by the gateway yet")));
        }
        tools.push(ToolSpec {
            name: t["name"].as_str().unwrap_or_default().to_owned(),
            description: t["description"].as_str().unwrap_or_default().to_owned(),
            input_schema: t["input_schema"].clone(),
            effect: Effect::Write,
        });
    }

    let extras: Map<String, Value> = body
        .as_object()
        .into_iter()
        .flatten()
        .filter(|(k, _)| !KNOWN.contains(&k.as_str()))
        .map(|(k, v)| (k.clone(), v.clone()))
        .collect();

    let request = ModelRequest {
        system,
        messages,
        tools,
        max_tokens,
        temperature: body["temperature"].as_f64().map(|t| t as f32),
        provider_options: [("anthropic".to_owned(), Value::Object(extras))].into(),
    };
    Ok((model, request, body["stream"].as_bool().unwrap_or(false)))
}

fn block(b: &Value) -> Result<Content, GatewayError> {
    Ok(match b["type"].as_str().unwrap_or_default() {
        "text" => Content::Text { text: b["text"].as_str().unwrap_or_default().to_owned() },
        "tool_use" => Content::ToolUse {
            id: b["id"].as_str().unwrap_or_default().to_owned(),
            name: b["name"].as_str().unwrap_or_default().to_owned(),
            input: b["input"].clone(),
        },
        "tool_result" => Content::ToolResult {
            tool_use_id: b["tool_use_id"].as_str().unwrap_or_default().to_owned(),
            content: match &b["content"] {
                Value::Array(parts) => {
                    json!(parts.iter().filter_map(|p| p["text"].as_str()).collect::<Vec<_>>().join(""))
                }
                other => other.clone(),
            },
            is_error: b["is_error"].as_bool().unwrap_or(false),
        },
        "thinking" => Content::Thinking {
            text: b["thinking"].as_str().unwrap_or_default().to_owned(),
            provider_data: Some(json!({"provider": "anthropic", "signature": b["signature"]})),
        },
        "redacted_thinking" => Content::Thinking {
            text: String::new(),
            provider_data: Some(json!({"provider": "anthropic", "redacted_data": b["data"]})),
        },
        other => return Err(GatewayError::bad_request(format!("content block `{other}` is not supported yet"))),
    })
}

fn stop_reason(r: &StopReason) -> &str {
    match r {
        StopReason::EndTurn => "end_turn",
        StopReason::ToolUse => "tool_use",
        StopReason::MaxTokens => "max_tokens",
        StopReason::Other(s) => s,
    }
}

fn blocks(resp: &ModelResponse) -> Vec<Value> {
    resp.content
        .iter()
        .filter_map(|c| match c {
            Content::Text { text } => Some(json!({"type": "text", "text": text})),
            Content::ToolUse { id, name, input } => {
                Some(json!({"type": "tool_use", "id": id, "name": name, "input": input}))
            }
            // Only Anthropic-signed reasoning can be handed to Anthropic clients.
            Content::Thinking { text, provider_data: Some(d) } if d["provider"] == "anthropic" => {
                Some(if d["redacted_data"].is_null() {
                    json!({"type": "thinking", "thinking": text, "signature": d["signature"]})
                } else {
                    json!({"type": "redacted_thinking", "data": d["redacted_data"]})
                })
            }
            _ => None,
        })
        .collect()
}

fn usage(resp: &ModelResponse) -> Value {
    let u = &resp.usage;
    json!({
        "input_tokens": u.input_tokens,
        "output_tokens": u.output_tokens,
        "cache_read_input_tokens": u.cache_read_tokens,
        "cache_creation_input_tokens": u.cache_write_tokens,
    })
}

pub fn render_response(id: &str, model: &str, resp: &ModelResponse) -> Value {
    json!({
        "id": id,
        "type": "message",
        "role": "assistant",
        "model": model,
        "content": blocks(resp),
        "stop_reason": stop_reason(&resp.stop_reason),
        "stop_sequence": null,
        "usage": usage(resp),
    })
}

#[derive(Clone, Copy, PartialEq)]
enum Block {
    Text,
    Thinking,
    Tool,
}

/// Incremental Messages API SSE writer.
pub struct SseOut {
    id: String,
    model: String,
    /// Thinking can only be forwarded to Anthropic clients when Anthropic
    /// produced (and signed) it.
    thinking_ok: bool,
    open: Option<Block>,
    index: usize,
}

impl SseOut {
    pub fn boxed(id: &str, model: &str, served_by_anthropic: bool) -> Box<dyn SseRenderer> {
        Box::new(Self { id: id.into(), model: model.into(), thinking_ok: served_by_anthropic, open: None, index: 0 })
    }

    fn ev(name: &str, data: Value) -> String {
        format!("event: {name}\ndata: {data}\n\n")
    }

    fn close(&mut self) -> String {
        match self.open.take() {
            Some(_) => {
                let out = Self::ev("content_block_stop", json!({"type": "content_block_stop", "index": self.index}));
                self.index += 1;
                out
            }
            None => String::new(),
        }
    }

    fn open_block(&mut self, kind: Block, block: Value) -> String {
        let mut out = self.close();
        out += &Self::ev(
            "content_block_start",
            json!({"type": "content_block_start", "index": self.index, "content_block": block}),
        );
        self.open = Some(kind);
        out
    }

    fn block_delta(&self, delta: Value) -> String {
        Self::ev("content_block_delta", json!({"type": "content_block_delta", "index": self.index, "delta": delta}))
    }
}

impl SseRenderer for SseOut {
    fn start(&mut self) -> String {
        Self::ev(
            "message_start",
            json!({"type": "message_start", "message": {
                "id": self.id, "type": "message", "role": "assistant", "model": self.model, "content": [],
                "stop_reason": null, "stop_sequence": null, "usage": {"input_tokens": 0, "output_tokens": 0},
            }}),
        )
    }

    fn delta(&mut self, ev: &StreamEvent) -> String {
        match ev {
            StreamEvent::TextDelta(t) => {
                let mut out = String::new();
                if self.open != Some(Block::Text) {
                    out += &self.open_block(Block::Text, json!({"type": "text", "text": ""}));
                }
                out + &self.block_delta(json!({"type": "text_delta", "text": t}))
            }
            StreamEvent::ThinkingDelta(t) if self.thinking_ok => {
                let mut out = String::new();
                if self.open != Some(Block::Thinking) {
                    out += &self.open_block(Block::Thinking, json!({"type": "thinking", "thinking": ""}));
                }
                out + &self.block_delta(json!({"type": "thinking_delta", "thinking": t}))
            }
            StreamEvent::ThinkingSignature(sig) if self.thinking_ok && self.open == Some(Block::Thinking) => {
                self.block_delta(json!({"type": "signature_delta", "signature": sig}))
            }
            StreamEvent::ToolCallStart { id, name, .. } => {
                self.open_block(Block::Tool, json!({"type": "tool_use", "id": id, "name": name, "input": {}}))
            }
            StreamEvent::ToolInputDelta { partial_json, .. } if self.open == Some(Block::Tool) => {
                self.block_delta(json!({"type": "input_json_delta", "partial_json": partial_json}))
            }
            _ => String::new(),
        }
    }

    fn finish(&mut self, resp: &ModelResponse) -> String {
        let mut out = self.close();
        out += &Self::ev(
            "message_delta",
            json!({"type": "message_delta",
                   "delta": {"stop_reason": stop_reason(&resp.stop_reason), "stop_sequence": null},
                   "usage": usage(resp)}),
        );
        out + &Self::ev("message_stop", json!({"type": "message_stop"}))
    }

    fn error(&mut self, err: &GatewayError) -> String {
        Self::ev("error", render_error(err))
    }
}

pub fn render_error(err: &GatewayError) -> Value {
    json!({"type": "error", "error": {"type": err.kind, "message": err.message}})
}
