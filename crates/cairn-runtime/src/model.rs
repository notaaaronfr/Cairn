use std::collections::BTreeMap;
use std::sync::Arc;

use async_trait::async_trait;
use futures::stream::{self, BoxStream};
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::message::{Content, Message};
use crate::tool::ToolSpec;

#[derive(Debug, Clone, Default)]
pub struct ModelRequest {
    pub system: Option<String>,
    pub messages: Vec<Message>,
    pub tools: Vec<ToolSpec>,
    pub max_tokens: u32,
    pub temperature: Option<f32>,
    /// Provider-specific options, keyed by provider id (e.g. `"anthropic"`,
    /// `"openai"`). Adapters merge their own entry into the request body and
    /// ignore the rest, so one agent definition can carry tuned options for
    /// every provider it may fall back to.
    pub provider_options: BTreeMap<String, Value>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum StopReason {
    EndTurn,
    ToolUse,
    MaxTokens,
    Other(String),
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct Usage {
    pub input_tokens: u64,
    pub output_tokens: u64,
    #[serde(default)]
    pub cache_read_tokens: u64,
    #[serde(default)]
    pub cache_write_tokens: u64,
}

impl Usage {
    pub fn total(&self) -> u64 {
        self.input_tokens + self.output_tokens + self.cache_read_tokens + self.cache_write_tokens
    }

    pub fn add(&mut self, other: &Usage) {
        self.input_tokens += other.input_tokens;
        self.output_tokens += other.output_tokens;
        self.cache_read_tokens += other.cache_read_tokens;
        self.cache_write_tokens += other.cache_write_tokens;
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ModelResponse {
    /// Concrete model that produced the response (after any fallback).
    pub model: String,
    pub content: Vec<Content>,
    pub stop_reason: StopReason,
    pub usage: Usage,
}

#[derive(Debug, Clone, thiserror::Error)]
pub enum ModelError {
    #[error("transport error: {0}")]
    Transport(String),
    #[error("provider returned {status}: {body}")]
    Api { status: u16, body: String },
    #[error("could not decode provider response: {0}")]
    Decode(String),
    #[error("{0}")]
    Other(String),
}

impl ModelError {
    /// Whether another attempt (or another provider) may succeed.
    pub fn retryable(&self) -> bool {
        match self {
            ModelError::Transport(_) => true,
            ModelError::Api { status, .. } => *status == 408 || *status == 429 || *status >= 500,
            _ => false,
        }
    }
}

#[async_trait]
pub trait Model: Send + Sync {
    /// Identifier such as `anthropic:claude-opus-5-5`.
    fn id(&self) -> String;

    async fn complete(&self, request: &ModelRequest) -> Result<ModelResponse, ModelError>;

    /// Stream the response as it is generated. The stream always ends with
    /// [`StreamEvent::Completed`] carrying the full response. The default
    /// implementation calls [`Model::complete`] and replays it as deltas.
    async fn stream(&self, request: &ModelRequest) -> Result<ModelStream, ModelError> {
        let response = self.complete(request).await?;
        Ok(Box::pin(stream::iter(synthesize_stream(response).into_iter().map(Ok))))
    }

    /// Cost of a response in USD, if pricing is configured.
    fn cost_usd(&self, _response: &ModelResponse) -> Option<f64> {
        None
    }
}

/// Incremental output from a streaming model call.
#[derive(Debug, Clone, PartialEq)]
pub enum StreamEvent {
    TextDelta(String),
    ThinkingDelta(String),
    /// Signature for the current thinking block (Anthropic).
    ThinkingSignature(String),
    /// A new tool call; `index` counts tool calls within this response.
    ToolCallStart { index: usize, id: String, name: String },
    ToolInputDelta { index: usize, partial_json: String },
    /// Final event of every successful stream.
    Completed(ModelResponse),
}

pub type ModelStream = BoxStream<'static, Result<StreamEvent, ModelError>>;

/// Express a complete response as the deltas a streaming call would yield.
pub fn synthesize_stream(response: ModelResponse) -> Vec<StreamEvent> {
    let mut events = Vec::new();
    let mut tool_index = 0;
    for c in &response.content {
        match c {
            Content::Text { text } => events.push(StreamEvent::TextDelta(text.clone())),
            Content::Thinking { text, provider_data } => {
                events.push(StreamEvent::ThinkingDelta(text.clone()));
                if let Some(sig) = provider_data.as_ref().and_then(|d| d["signature"].as_str()) {
                    events.push(StreamEvent::ThinkingSignature(sig.to_owned()));
                }
            }
            Content::ToolUse { id, name, input } => {
                events.push(StreamEvent::ToolCallStart { index: tool_index, id: id.clone(), name: name.clone() });
                events.push(StreamEvent::ToolInputDelta { index: tool_index, partial_json: input.to_string() });
                tool_index += 1;
            }
            Content::ToolResult { .. } => {}
        }
    }
    events.push(StreamEvent::Completed(response));
    events
}

/// Tries each model in order, moving on only for retryable errors.
pub struct FallbackModel {
    models: Vec<Arc<dyn Model>>,
}

impl FallbackModel {
    pub fn new(models: Vec<Arc<dyn Model>>) -> Self {
        assert!(!models.is_empty(), "FallbackModel needs at least one model");
        Self { models }
    }
}

#[async_trait]
impl Model for FallbackModel {
    fn id(&self) -> String {
        let ids: Vec<_> = self.models.iter().map(|m| m.id()).collect();
        format!("fallback({})", ids.join(","))
    }

    async fn complete(&self, request: &ModelRequest) -> Result<ModelResponse, ModelError> {
        let mut last_err = None;
        for model in &self.models {
            match model.complete(request).await {
                Ok(resp) => return Ok(resp),
                Err(e) if e.retryable() => {
                    tracing::warn!(model = %model.id(), error = %e, "model failed, falling back");
                    last_err = Some(e);
                }
                Err(e) => return Err(e),
            }
        }
        Err(last_err.expect("at least one model"))
    }

    /// Fails over only while opening the stream; once tokens flow, errors
    /// are surfaced to the caller.
    async fn stream(&self, request: &ModelRequest) -> Result<ModelStream, ModelError> {
        let mut last_err = None;
        for model in &self.models {
            match model.stream(request).await {
                Ok(s) => return Ok(s),
                Err(e) if e.retryable() => {
                    tracing::warn!(model = %model.id(), error = %e, "model stream failed, falling back");
                    last_err = Some(e);
                }
                Err(e) => return Err(e),
            }
        }
        Err(last_err.expect("at least one model"))
    }

    fn cost_usd(&self, response: &ModelResponse) -> Option<f64> {
        self.models.iter().find(|m| m.id().ends_with(&response.model)).and_then(|m| m.cost_usd(response))
    }
}

#[async_trait]
impl<M: Model + ?Sized> Model for Arc<M> {
    fn id(&self) -> String {
        (**self).id()
    }

    async fn complete(&self, request: &ModelRequest) -> Result<ModelResponse, ModelError> {
        (**self).complete(request).await
    }

    async fn stream(&self, request: &ModelRequest) -> Result<ModelStream, ModelError> {
        (**self).stream(request).await
    }

    fn cost_usd(&self, response: &ModelResponse) -> Option<f64> {
        (**self).cost_usd(response)
    }
}
