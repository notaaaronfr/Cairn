use async_trait::async_trait;
use cairn_runtime::{
    Content, Model, ModelError, ModelRequest, ModelResponse, ModelStream, Role, StopReason, StreamEvent, Usage,
};
use serde_json::{Value, json};

use crate::sse::{self, SseEvent};
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

    /// Reasoning this adapter's own provider produced, as opposed to another
    /// provider's earlier in a fallback chain.
    fn own_reasoning<'a>(&self, c: &'a Content) -> Option<&'a str> {
        match c {
            Content::Thinking { text, provider_data } if provider_data.as_ref().and_then(|d| d["provider"].as_str()) == Some(self.provider.as_str()) => {
                Some(text.as_str())
            }
            _ => None,
        }
    }

    pub(crate) fn build_body(&self, req: &ModelRequest) -> Value {
        // Reasoning models (IFM's K2, DeepSeek and others) return
        // `reasoning_content` and refuse a later turn unless every assistant
        // message carries it back verbatim. Once this provider has returned
        // any, replay it on every assistant message: its own text where there
        // is some, "" where another model wrote the turn.
        let replay_reasoning = req.messages.iter().flat_map(|m| &m.content).any(|c| self.own_reasoning(c).is_some());
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
                    if replay_reasoning {
                        let r: String = m.content.iter().filter_map(|c| self.own_reasoning(c)).collect();
                        msg["reasoning_content"] = json!(r);
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
        // Kept even when empty: "" is a value such providers expect back.
        if let Some(r) = msg["reasoning_content"].as_str().or_else(|| msg["reasoning"].as_str()) {
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
            ..Default::default()
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

    async fn stream(&self, request: &ModelRequest) -> Result<ModelStream, ModelError> {
        let mut body = self.build_body(request);
        body["stream"] = json!(true);
        body["stream_options"] = json!({ "include_usage": true });
        let mut http = self.client.post(format!("{}/chat/completions", self.base_url));
        if let Some(key) = &self.api_key {
            http = http.bearer_auth(key);
        }
        let resp = http.json(&body).send().await.map_err(|e| ModelError::Transport(e.to_string()))?;
        let resp = sse::check_status(resp).await?;
        let acc = StreamAcc { provider: self.provider.clone(), model: self.model.clone(), ..Default::default() };
        Ok(sse::translate(sse::events(resp), acc, StreamAcc::step, StreamAcc::end))
    }

    fn pricing(&self) -> Option<Pricing> {
        self.pricing
    }
}

#[derive(Default)]
pub(crate) struct StreamAcc {
    provider: String,
    model: String,
    text: String,
    reasoning: String,
    /// Whether the stream carried a reasoning field at all, even an empty one.
    saw_reasoning: bool,
    /// (id, name, arguments) per tool-call index.
    tools: Vec<(String, String, String)>,
    finish: Option<String>,
    usage: Usage,
    done: bool,
}

impl StreamAcc {
    pub(crate) fn step(&mut self, ev: SseEvent) -> Result<Vec<StreamEvent>, ModelError> {
        if ev.data.trim() == "[DONE]" {
            return self.complete();
        }
        let v: Value = serde_json::from_str(&ev.data).map_err(|e| ModelError::Decode(e.to_string()))?;
        if !v["error"].is_null() {
            let status = v["error"]["code"].as_u64().filter(|c| (400..600).contains(c)).unwrap_or(500) as u16;
            return Err(ModelError::Api { status, body: v["error"].to_string() });
        }
        if let Some(m) = v["model"].as_str() {
            self.model = m.to_owned();
        }
        if v["usage"].is_object() {
            let u = &v["usage"];
            let prompt = u["prompt_tokens"].as_u64().unwrap_or(0);
            let cached = u["prompt_tokens_details"]["cached_tokens"].as_u64().unwrap_or(0);
            self.usage = Usage {
                input_tokens: prompt.saturating_sub(cached),
                output_tokens: u["completion_tokens"].as_u64().unwrap_or(0),
                cache_read_tokens: cached,
                cache_write_tokens: 0,
            };
        }

        let mut out = Vec::new();
        let choice = &v["choices"][0];
        let delta = &choice["delta"];
        if let Some(r) = delta["reasoning_content"].as_str().or_else(|| delta["reasoning"].as_str()) {
            self.saw_reasoning = true;
            if !r.is_empty() {
                self.reasoning.push_str(r);
                out.push(StreamEvent::ThinkingDelta(r.to_owned()));
            }
        }
        if let Some(t) = delta["content"].as_str().filter(|s| !s.is_empty()) {
            self.text.push_str(t);
            out.push(StreamEvent::TextDelta(t.to_owned()));
        }
        for call in delta["tool_calls"].as_array().into_iter().flatten() {
            let index = call["index"].as_u64().unwrap_or(self.tools.len() as u64) as usize;
            if index >= self.tools.len() {
                self.tools.resize(index + 1, Default::default());
            }
            let slot = &mut self.tools[index];
            if let Some(id) = call["id"].as_str().filter(|s| !s.is_empty()) {
                slot.0 = id.to_owned();
                slot.1 = call["function"]["name"].as_str().unwrap_or_default().to_owned();
                out.push(StreamEvent::ToolCallStart { index, id: slot.0.clone(), name: slot.1.clone() });
            }
            if let Some(args) = call["function"]["arguments"].as_str().filter(|s| !s.is_empty()) {
                slot.2.push_str(args);
                out.push(StreamEvent::ToolInputDelta { index, partial_json: args.to_owned() });
            }
        }
        if let Some(f) = choice["finish_reason"].as_str() {
            self.finish = Some(f.to_owned());
        }
        Ok(out)
    }

    /// Some servers close the stream without `[DONE]`; complete if they
    /// at least sent a finish reason.
    pub(crate) fn end(&mut self) -> Result<Vec<StreamEvent>, ModelError> {
        if self.done {
            Ok(Vec::new())
        } else if self.finish.is_some() {
            self.complete()
        } else {
            Err(ModelError::Transport("stream closed before completion".into()))
        }
    }

    fn complete(&mut self) -> Result<Vec<StreamEvent>, ModelError> {
        if self.done {
            return Ok(Vec::new());
        }
        self.done = true;
        let mut content = Vec::new();
        if self.saw_reasoning || !self.reasoning.is_empty() {
            content.push(Content::Thinking {
                text: std::mem::take(&mut self.reasoning),
                provider_data: Some(json!({ "provider": self.provider })),
            });
        }
        if !self.text.is_empty() {
            content.push(Content::Text { text: std::mem::take(&mut self.text) });
        }
        for (id, name, args) in std::mem::take(&mut self.tools) {
            let input = if args.trim().is_empty() {
                json!({})
            } else {
                serde_json::from_str(&args).unwrap_or_else(|_| json!({ "_unparsed_arguments": args }))
            };
            content.push(Content::ToolUse { id, name, input });
        }
        let stop_reason = match self.finish.as_deref() {
            Some("stop") => StopReason::EndTurn,
            Some("tool_calls") | Some("function_call") => StopReason::ToolUse,
            Some("length") => StopReason::MaxTokens,
            Some(other) => StopReason::Other(other.to_owned()),
            None => StopReason::Other("unknown".into()),
        };
        Ok(vec![StreamEvent::Completed(ModelResponse {
            model: self.model.clone(),
            content,
            stop_reason,
            usage: self.usage,
            ..Default::default()
        })])
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use cairn_runtime::Message;

    #[test]
    fn reasoning_is_replayed_verbatim_once_the_provider_returns_it() {
        let m = OpenAiCompatModel::new("k2", "IFM/K2", "https://example.invalid/v1", None::<String>);
        let first = m
            .parse_response(&json!({"model": "IFM/K2", "choices": [{"finish_reason": "stop",
                "message": {"content": "Hello", "reasoning_content": "think think"}}]}))
            .unwrap();
        let empty = m
            .parse_response(&json!({"model": "IFM/K2", "choices": [{"finish_reason": "stop",
                "message": {"content": "Again", "reasoning_content": ""}}]}))
            .unwrap();
        let req = ModelRequest {
            messages: vec![
                Message::user("a"),
                Message::assistant(first.content),
                Message::user("b"),
                // A turn another model wrote, after a fallback.
                Message::assistant(vec![Content::Text { text: "from haiku".into() }]),
                Message::user("c"),
                Message::assistant(empty.content),
                Message::user("d"),
            ],
            max_tokens: 10,
            ..Default::default()
        };
        let msgs = m.build_body(&req)["messages"].as_array().unwrap().clone();
        assert_eq!(msgs[1]["reasoning_content"], "think think");
        assert_eq!(msgs[1]["content"], "Hello", "reasoning never leaks into the content");
        assert_eq!(msgs[3]["reasoning_content"], "", "another model's turn carries an empty one");
        assert_eq!(msgs[5]["reasoning_content"], "", "an empty one is kept, not dropped");
    }

    #[test]
    fn providers_without_reasoning_get_no_reasoning_field() {
        let m = OpenAiCompatModel::openai("gpt-x", "k");
        let req = ModelRequest {
            messages: vec![
                Message::user("a"),
                Message::assistant(vec![
                    // Another provider's thinking is not this one's to replay.
                    Content::Thinking { text: "x".into(), provider_data: Some(json!({"provider": "anthropic"})) },
                    Content::Text { text: "hi".into() },
                ]),
                Message::user("b"),
            ],
            max_tokens: 10,
            ..Default::default()
        };
        let msgs = m.build_body(&req)["messages"].as_array().unwrap().clone();
        assert!(msgs[1].get("reasoning_content").is_none());
    }

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

    #[test]
    fn accumulates_stream_with_parallel_tool_calls() {
        let raw = r#"data: {"model":"gpt-x","choices":[{"index":0,"delta":{"role":"assistant","content":""}}]}

data: {"choices":[{"index":0,"delta":{"content":"Check"}}]}

data: {"choices":[{"index":0,"delta":{"content":"ing"}}]}

data: {"choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"id":"c1","type":"function","function":{"name":"a","arguments":""}}]}}]}

data: {"choices":[{"index":0,"delta":{"tool_calls":[{"index":1,"id":"c2","type":"function","function":{"name":"b","arguments":"{}"}}]}}]}

data: {"choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"function":{"arguments":"{\"x\":"}}]}}]}

data: {"choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"function":{"arguments":"1}"}}]}}]}

data: {"choices":[{"index":0,"delta":{},"finish_reason":"tool_calls"}]}

data: {"choices":[],"usage":{"prompt_tokens":50,"completion_tokens":9,"prompt_tokens_details":{"cached_tokens":20}}}

data: [DONE]
"#;
        let mut acc = StreamAcc { provider: "openai".into(), ..Default::default() };
        let events: Vec<StreamEvent> =
            sse::parse_text(raw).into_iter().flat_map(|e| acc.step(e).unwrap()).collect();
        assert_eq!(events[0], StreamEvent::TextDelta("Check".into()));
        assert!(events.contains(&StreamEvent::ToolCallStart { index: 1, id: "c2".into(), name: "b".into() }));
        let StreamEvent::Completed(r) = events.last().unwrap() else { panic!("no completion") };
        assert_eq!(r.stop_reason, StopReason::ToolUse);
        assert_eq!(r.content[0], Content::Text { text: "Checking".into() });
        assert_eq!(r.content[1], Content::ToolUse { id: "c1".into(), name: "a".into(), input: json!({"x": 1}) });
        assert_eq!(r.content[2], Content::ToolUse { id: "c2".into(), name: "b".into(), input: json!({}) });
        assert_eq!(r.usage.input_tokens, 30);
        assert_eq!(r.usage.cache_read_tokens, 20);
        assert!(acc.end().unwrap().is_empty(), "no duplicate completion");
    }
}
