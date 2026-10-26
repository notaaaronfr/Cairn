//! OpenAI Chat Completions wire format <-> Cairn's neutral types.

use cairn_runtime::{Content, Effect, Message, ModelRequest, ModelResponse, Role, StopReason, StreamEvent, ToolSpec};
use serde_json::{Map, Value, json};

use crate::{GatewayError, SseRenderer};

const KNOWN: &[&str] =
    &["model", "messages", "tools", "max_tokens", "max_completion_tokens", "temperature", "stream", "stream_options"];
const OPENAI_FAMILY: &[&str] = &["openai", "gemini", "openrouter", "ollama"];

pub fn parse_request(body: &Value) -> Result<(String, ModelRequest, bool), GatewayError> {
    let model = body["model"].as_str().ok_or_else(|| GatewayError::bad_request("`model` is required"))?.to_owned();
    let raw = body["messages"].as_array().ok_or_else(|| GatewayError::bad_request("`messages` must be an array"))?;

    let mut system = Vec::new();
    let mut messages: Vec<Message> = Vec::new();
    for m in raw {
        match m["role"].as_str().unwrap_or_default() {
            "system" | "developer" => system.push(text_of(&m["content"])?),
            "user" => push(&mut messages, Role::User, vec![Content::Text { text: text_of(&m["content"])? }]),
            "assistant" => {
                let mut content = Vec::new();
                let text = if m["content"].is_null() { String::new() } else { text_of(&m["content"])? };
                if !text.is_empty() {
                    content.push(Content::Text { text });
                }
                for call in m["tool_calls"].as_array().into_iter().flatten() {
                    let args = call["function"]["arguments"].as_str().unwrap_or("{}");
                    content.push(Content::ToolUse {
                        id: call["id"].as_str().unwrap_or_default().to_owned(),
                        name: call["function"]["name"].as_str().unwrap_or_default().to_owned(),
                        input: serde_json::from_str(args).unwrap_or_else(|_| json!({})),
                    });
                }
                push(&mut messages, Role::Assistant, content);
            }
            "tool" => push(
                &mut messages,
                Role::User,
                vec![Content::ToolResult {
                    tool_use_id: m["tool_call_id"].as_str().unwrap_or_default().to_owned(),
                    content: json!(text_of(&m["content"])?),
                    is_error: false,
                }],
            ),
            other => return Err(GatewayError::bad_request(format!("unsupported role `{other}`"))),
        }
    }

    let tools = body["tools"]
        .as_array()
        .into_iter()
        .flatten()
        .map(|t| ToolSpec {
            name: t["function"]["name"].as_str().unwrap_or_default().to_owned(),
            description: t["function"]["description"].as_str().unwrap_or_default().to_owned(),
            input_schema: if t["function"]["parameters"].is_null() {
                json!({"type": "object", "properties": {}})
            } else {
                t["function"]["parameters"].clone()
            },
            effect: Effect::Write,
        })
        .collect();

    let max_tokens = body["max_completion_tokens"].as_u64().or(body["max_tokens"].as_u64()).unwrap_or(4096) as u32;
    let extras: Map<String, Value> = body
        .as_object()
        .into_iter()
        .flatten()
        .filter(|(k, _)| !KNOWN.contains(&k.as_str()))
        .map(|(k, v)| (k.clone(), v.clone()))
        .collect();

    let request = ModelRequest {
        system: (!system.is_empty()).then(|| system.join("\n\n")),
        messages,
        tools,
        max_tokens,
        temperature: body["temperature"].as_f64().map(|t| t as f32),
        provider_options: OPENAI_FAMILY.iter().map(|p| ((*p).to_owned(), Value::Object(extras.clone()))).collect(),
    };
    Ok((model, request, body["stream"].as_bool().unwrap_or(false)))
}

/// Merge consecutive same-role messages (e.g. several `tool` results).
fn push(messages: &mut Vec<Message>, role: Role, content: Vec<Content>) {
    match messages.last_mut() {
        Some(last) if last.role == role => last.content.extend(content),
        _ => messages.push(Message { role, content }),
    }
}

fn text_of(v: &Value) -> Result<String, GatewayError> {
    match v {
        Value::String(s) => Ok(s.clone()),
        Value::Null => Ok(String::new()),
        Value::Array(parts) => parts
            .iter()
            .map(|p| match p["type"].as_str() {
                Some("text") => Ok(p["text"].as_str().unwrap_or_default().to_owned()),
                Some(other) => Err(GatewayError::bad_request(format!("content part `{other}` is not supported yet"))),
                None => Err(GatewayError::bad_request("content part missing `type`")),
            })
            .collect::<Result<Vec<_>, _>>()
            .map(|v| v.join("")),
        _ => Err(GatewayError::bad_request("invalid message content")),
    }
}

fn finish_reason(r: &StopReason) -> &'static str {
    match r {
        StopReason::EndTurn => "stop",
        StopReason::ToolUse => "tool_calls",
        StopReason::MaxTokens => "length",
        StopReason::Other(_) => "stop",
    }
}

fn parts(resp: &ModelResponse) -> (Option<String>, Vec<Value>, Option<String>) {
    let text = cairn_runtime::message::content_text(&resp.content);
    let calls: Vec<Value> = resp
        .content
        .iter()
        .filter_map(|c| match c {
            Content::ToolUse { id, name, input } => Some(json!({
                "id": id, "type": "function", "function": {"name": name, "arguments": input.to_string()}
            })),
            _ => None,
        })
        .collect();
    let reasoning: String = resp
        .content
        .iter()
        .filter_map(|c| match c {
            Content::Thinking { text, .. } if !text.is_empty() => Some(text.as_str()),
            _ => None,
        })
        .collect();
    ((!text.is_empty()).then_some(text), calls, (!reasoning.is_empty()).then_some(reasoning))
}

fn usage(resp: &ModelResponse) -> Value {
    let u = &resp.usage;
    let prompt = u.input_tokens + u.cache_read_tokens + u.cache_write_tokens;
    json!({
        "prompt_tokens": prompt,
        "completion_tokens": u.output_tokens,
        "total_tokens": prompt + u.output_tokens,
        "prompt_tokens_details": {"cached_tokens": u.cache_read_tokens},
    })
}

pub fn render_response(id: &str, model: &str, resp: &ModelResponse) -> Value {
    let (text, calls, reasoning) = parts(resp);
    let mut message = json!({"role": "assistant", "content": text});
    if !calls.is_empty() {
        message["tool_calls"] = json!(calls);
    }
    if let Some(r) = reasoning {
        message["reasoning_content"] = json!(r);
    }
    json!({
        "id": id,
        "object": "chat.completion",
        "created": chrono::Utc::now().timestamp(),
        "model": model,
        "choices": [{"index": 0, "message": message, "finish_reason": finish_reason(&resp.stop_reason)}],
        "usage": usage(resp),
    })
}

/// Incremental Chat Completions SSE writer.
pub struct SseOut {
    id: String,
    model: String,
    created: i64,
}

impl SseOut {
    pub fn boxed(id: &str, model: &str, _served_by_anthropic: bool) -> Box<dyn SseRenderer> {
        Box::new(Self { id: id.into(), model: model.into(), created: chrono::Utc::now().timestamp() })
    }

    fn chunk(&self, delta: Value, finish: Value, usage: Option<Value>) -> String {
        let mut c = json!({
            "id": self.id, "object": "chat.completion.chunk", "created": self.created, "model": self.model,
            "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
        });
        if let Some(u) = usage {
            c["usage"] = u;
        }
        format!("data: {c}\n\n")
    }
}

impl SseRenderer for SseOut {
    fn start(&mut self) -> String {
        self.chunk(json!({"role": "assistant", "content": ""}), Value::Null, None)
    }

    fn delta(&mut self, ev: &StreamEvent) -> String {
        match ev {
            StreamEvent::TextDelta(t) => self.chunk(json!({"content": t}), Value::Null, None),
            StreamEvent::ThinkingDelta(t) => self.chunk(json!({"reasoning_content": t}), Value::Null, None),
            StreamEvent::ToolCallStart { index, id, name } => self.chunk(
                json!({"tool_calls": [{"index": index, "id": id, "type": "function", "function": {"name": name, "arguments": ""}}]}),
                Value::Null,
                None,
            ),
            StreamEvent::ToolInputDelta { index, partial_json } => self.chunk(
                json!({"tool_calls": [{"index": index, "function": {"arguments": partial_json}}]}),
                Value::Null,
                None,
            ),
            StreamEvent::ThinkingSignature(_) | StreamEvent::Completed(_) => String::new(),
        }
    }

    fn finish(&mut self, resp: &ModelResponse) -> String {
        self.chunk(json!({}), json!(finish_reason(&resp.stop_reason)), Some(usage(resp))) + "data: [DONE]\n\n"
    }

    fn error(&mut self, err: &GatewayError) -> String {
        format!("data: {}\n\n", render_error(err))
    }
}

pub fn render_error(err: &GatewayError) -> Value {
    json!({"error": {"message": err.message, "type": err.kind, "code": err.status}})
}
