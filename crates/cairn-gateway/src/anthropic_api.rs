//! Anthropic Messages wire format <-> Cairn's neutral types.

use cairn_runtime::{Content, Effect, Message, ModelRequest, ModelResponse, Role, StopReason, ToolSpec};
use serde_json::{Map, Value, json};

use crate::GatewayError;

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

    let extras: Map<String, Value> =
        body.as_object().into_iter().flatten().filter(|(k, _)| !KNOWN.contains(&k.as_str())).map(|(k, v)| (k.clone(), v.clone())).collect();

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
                Value::Array(parts) => json!(parts.iter().filter_map(|p| p["text"].as_str()).collect::<Vec<_>>().join("")),
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
            Content::ToolUse { id, name, input } => Some(json!({"type": "tool_use", "id": id, "name": name, "input": input})),
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

/// Server-sent events for a completed response (buffered streaming).
pub fn render_stream(id: &str, model: &str, resp: &ModelResponse) -> String {
    let ev = |name: &str, data: Value| format!("event: {name}\ndata: {data}\n\n");
    let mut start = render_response(id, model, resp);
    start["content"] = json!([]);
    start["stop_reason"] = Value::Null;
    let mut out = ev("message_start", json!({"type": "message_start", "message": start}));

    for (i, b) in blocks(resp).into_iter().enumerate() {
        let (empty, delta) = match b["type"].as_str().unwrap_or_default() {
            "text" => (json!({"type": "text", "text": ""}), json!({"type": "text_delta", "text": b["text"]})),
            "tool_use" => (
                json!({"type": "tool_use", "id": b["id"], "name": b["name"], "input": {}}),
                json!({"type": "input_json_delta", "partial_json": b["input"].to_string()}),
            ),
            "thinking" => (
                json!({"type": "thinking", "thinking": ""}),
                json!({"type": "thinking_delta", "thinking": b["thinking"]}),
            ),
            _ => (b.clone(), Value::Null),
        };
        out += &ev("content_block_start", json!({"type": "content_block_start", "index": i, "content_block": empty}));
        if !delta.is_null() {
            out += &ev("content_block_delta", json!({"type": "content_block_delta", "index": i, "delta": delta}));
        }
        if b["type"] == "thinking" {
            let sig = json!({"type": "signature_delta", "signature": b["signature"]});
            out += &ev("content_block_delta", json!({"type": "content_block_delta", "index": i, "delta": sig}));
        }
        out += &ev("content_block_stop", json!({"type": "content_block_stop", "index": i}));
    }
    out += &ev(
        "message_delta",
        json!({"type": "message_delta", "delta": {"stop_reason": stop_reason(&resp.stop_reason), "stop_sequence": null},
               "usage": {"output_tokens": resp.usage.output_tokens}}),
    );
    out += &ev("message_stop", json!({"type": "message_stop"}));
    out
}

pub fn render_error(err: &GatewayError) -> Value {
    json!({"type": "error", "error": {"type": err.kind, "message": err.message}})
}
