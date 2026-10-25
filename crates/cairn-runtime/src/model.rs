use std::collections::BTreeMap;
use std::sync::Arc;

use async_trait::async_trait;
use futures::stream::{self, BoxStream};
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::message::{Content, Message};
use crate::pricing::Pricing;
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

#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum StopReason {
    #[default]
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

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct ModelResponse {
    /// Model name as reported by the provider.
    pub model: String,
    pub content: Vec<Content>,
    pub stop_reason: StopReason,
    pub usage: Usage,
    /// [`Model::id`] of the model that served this response, when a
    /// composite model (e.g. [`FallbackModel`]) chose among several.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub served_by: Option<String>,
    /// Earlier answers in this call that were discarded as unusable. Their
    /// tokens were still paid for and count toward budgets.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub rejected: Vec<RejectedAttempt>,
}

/// A model answer that was received, paid for, and discarded.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct RejectedAttempt {
    pub model: String,
    pub reason: String,
    pub usage: Usage,
    pub cost_usd: Option<f64>,
}

impl ModelResponse {
    /// Tokens for this response plus any rejected attempts.
    pub fn total_usage(&self) -> Usage {
        let mut u = self.usage;
        for r in &self.rejected {
            u.add(&r.usage);
        }
        u
    }
}

#[derive(Debug, Clone, thiserror::Error)]
pub enum ModelError {
    #[error("transport error: {0}")]
    Transport(String),
    #[error("provider returned {status}: {body}")]
    Api { status: u16, body: String },
    #[error("could not decode provider response: {0}")]
    Decode(String),
    /// Every model answered, but no answer passed the usability check.
    #[error("no usable response: {reason}")]
    Unusable { reason: String, attempts: Vec<RejectedAttempt> },
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

    /// Tokens and USD already spent before this error (e.g. on answers that
    /// were rejected as unusable). `cost` is `None` if any attempt was unpriced.
    pub fn spent(&self) -> (Usage, Option<f64>) {
        match self {
            ModelError::Unusable { attempts, .. } => {
                let mut usage = Usage::default();
                let mut cost = Some(0.0);
                for a in attempts {
                    usage.add(&a.usage);
                    cost = cost.zip(a.cost_usd).map(|(x, y)| x + y);
                }
                (usage, cost)
            }
            _ => (Usage::default(), Some(0.0)),
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

    /// Prices for this model, if configured. Required for USD budgets.
    fn pricing(&self) -> Option<Pricing> {
        None
    }

    /// Cost of a response in USD, if pricing is configured.
    fn cost_usd(&self, response: &ModelResponse) -> Option<f64> {
        self.pricing().map(|p| p.cost(&response.usage))
    }

    /// Upper bound in USD for one call with this many input and output
    /// tokens, counting every attempt a composite model might make. Input is
    /// priced at the higher of the input and cache-write rates.
    fn max_call_cost(&self, input_tokens: u64, output_tokens: u64) -> Option<f64> {
        self.pricing().map(|p| {
            (input_tokens as f64 * p.input.max(p.cache_write) + output_tokens as f64 * p.output) / 1_000_000.0
        })
    }

    /// Ids of models (including any inside a composite) that have no
    /// pricing. A USD budget cannot be enforced while this is non-empty.
    fn unpriced_models(&self) -> Vec<String> {
        if self.pricing().is_none() { vec![self.id()] } else { Vec::new() }
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

type UsableFn = Arc<dyn Fn(&ModelResponse) -> Result<(), String> + Send + Sync>;

/// Tries each model in order. Moves on when a request fails with a
/// retryable error **or** when the answer is unusable:
///
/// * cut off by the token limit (unless [`accept_truncated`](Self::accept_truncated)),
/// * empty — no text and no tool calls (unless [`accept_empty`](Self::accept_empty)),
/// * a final answer that is not JSON with the required fields
///   ([`require_json_fields`](Self::require_json_fields)),
/// * rejected by a custom check ([`require`](Self::require)).
///
/// Rejected answers are recorded in [`ModelResponse::rejected`] and their
/// cost still counts toward budgets. With usability checks enabled,
/// streaming is buffered per attempt so a rejected answer never reaches
/// the caller.
pub struct FallbackModel {
    models: Vec<Arc<dyn Model>>,
    allow_truncated: bool,
    allow_empty: bool,
    json_fields: Option<Vec<String>>,
    custom: Option<UsableFn>,
}

impl FallbackModel {
    pub fn new(models: Vec<Arc<dyn Model>>) -> Self {
        assert!(!models.is_empty(), "FallbackModel needs at least one model");
        Self { models, allow_truncated: false, allow_empty: false, json_fields: None, custom: None }
    }

    /// Keep answers that hit the token limit.
    pub fn accept_truncated(mut self) -> Self {
        self.allow_truncated = true;
        self
    }

    /// Keep answers with no text and no tool calls.
    pub fn accept_empty(mut self) -> Self {
        self.allow_empty = true;
        self
    }

    /// Final answers (no tool calls) must contain a JSON object with these
    /// top-level fields, e.g. `["summary", "work", "risks", "confidence"]`.
    pub fn require_json_fields(mut self, fields: &[&str]) -> Self {
        self.json_fields = Some(fields.iter().map(|f| (*f).to_owned()).collect());
        self
    }

    /// Custom usability check; return `Err(reason)` to try the next model.
    pub fn require(mut self, check: impl Fn(&ModelResponse) -> Result<(), String> + Send + Sync + 'static) -> Self {
        self.custom = Some(Arc::new(check));
        self
    }

    fn has_checks(&self) -> bool {
        !self.allow_truncated || !self.allow_empty || self.json_fields.is_some() || self.custom.is_some()
    }

    /// Why a response is unusable, if it is.
    pub fn check(&self, r: &ModelResponse) -> Result<(), String> {
        if !self.allow_truncated && r.stop_reason == StopReason::MaxTokens {
            return Err("reply was cut off at the token limit".into());
        }
        let has_tools = r.content.iter().any(|c| matches!(c, Content::ToolUse { .. }));
        let text = crate::message::content_text(&r.content);
        if !self.allow_empty && !has_tools && text.trim().is_empty() {
            return Err("reply is empty".into());
        }
        if let (Some(fields), false) = (&self.json_fields, has_tools) {
            let Some(serde_json::Value::Object(obj)) = crate::output::extract_json(&text) else {
                return Err("reply is not a JSON object".into());
            };
            let missing: Vec<&str> = fields.iter().filter(|f| !obj.contains_key(*f)).map(String::as_str).collect();
            if !missing.is_empty() {
                return Err(format!("JSON is missing required field(s): {}", missing.join(", ")));
            }
        }
        if let Some(check) = &self.custom {
            check(r)?;
        }
        Ok(())
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
        let mut rejected: Vec<RejectedAttempt> = Vec::new();
        for model in &self.models {
            match model.complete(request).await {
                Ok(mut resp) => {
                    resp.served_by.get_or_insert_with(|| model.id());
                    match self.check(&resp) {
                        Ok(()) => {
                            rejected.append(&mut resp.rejected);
                            resp.rejected = rejected;
                            return Ok(resp);
                        }
                        Err(reason) => {
                            tracing::warn!(model = %model.id(), %reason, "unusable answer, falling back");
                            rejected.append(&mut resp.rejected);
                            rejected.push(RejectedAttempt {
                                model: model.id(),
                                reason,
                                usage: resp.usage,
                                cost_usd: model.cost_usd(&resp),
                            });
                        }
                    }
                }
                Err(e) if e.retryable() => {
                    tracing::warn!(model = %model.id(), error = %e, "model failed, falling back");
                    last_err = Some(e);
                }
                Err(ModelError::Unusable { reason, attempts }) => {
                    rejected.extend(attempts);
                    last_err = Some(ModelError::Other(reason));
                }
                Err(e) if rejected.is_empty() => return Err(e),
                Err(e) => {
                    last_err = Some(e);
                    break;
                }
            }
        }
        if rejected.is_empty() {
            return Err(last_err.expect("at least one model"));
        }
        // Keep the paid-for attempts on the error so budgets still see them.
        let mut reason = rejected.iter().map(|r| format!("{}: {}", r.model, r.reason)).collect::<Vec<_>>().join("; ");
        if let Some(e) = last_err {
            reason = format!("{reason}; last error: {e}");
        }
        Err(ModelError::Unusable { reason, attempts: rejected })
    }

    /// Fails over only while opening the stream; once tokens flow, errors
    /// are surfaced to the caller.
    async fn stream(&self, request: &ModelRequest) -> Result<ModelStream, ModelError> {
        if self.has_checks() {
            // An answer must be validated before any of it is released.
            let response = self.complete(request).await?;
            return Ok(Box::pin(stream::iter(synthesize_stream(response).into_iter().map(Ok))));
        }
        let mut last_err = None;
        for model in &self.models {
            match model.stream(request).await {
                Ok(s) => {
                    let id = model.id();
                    return Ok(Box::pin(futures::StreamExt::map(s, move |ev| match ev {
                        Ok(StreamEvent::Completed(mut resp)) => {
                            resp.served_by.get_or_insert_with(|| id.clone());
                            Ok(StreamEvent::Completed(resp))
                        }
                        other => other,
                    })));
                }
                Err(e) if e.retryable() => {
                    tracing::warn!(model = %model.id(), error = %e, "model stream failed, falling back");
                    last_err = Some(e);
                }
                Err(e) => return Err(e),
            }
        }
        Err(last_err.expect("at least one model"))
    }

    /// Conservative (field-wise maximum) pricing across all models, or
    /// `None` if any model is unpriced.
    fn pricing(&self) -> Option<Pricing> {
        self.models.iter().map(|m| m.pricing()).try_fold(Pricing::default(), |acc, p| p.map(|p| acc.max(p)))
    }

    /// Prices the response with the model that actually served it; falls
    /// back to the conservative maximum rather than ever under-counting.
    /// Cost of the answer (priced on the model that served it, else the
    /// conservative maximum) plus every rejected attempt.
    fn cost_usd(&self, response: &ModelResponse) -> Option<f64> {
        let served = response.served_by.as_deref();
        let answer = match self.models.iter().find(|m| Some(m.id().as_str()) == served) {
            Some(m) => m.cost_usd(&ModelResponse { rejected: Vec::new(), ..response.clone() }),
            None => self.pricing().map(|p| p.cost(&response.usage)),
        }?;
        let conservative = self.pricing();
        response.rejected.iter().try_fold(answer, |acc, r| {
            r.cost_usd.or_else(|| conservative.map(|p| p.cost(&r.usage))).map(|c| acc + c)
        })
    }

    fn unpriced_models(&self) -> Vec<String> {
        self.models.iter().flat_map(|m| m.unpriced_models()).collect()
    }

    /// Every model may be tried (and paid for) in one call.
    fn max_call_cost(&self, input_tokens: u64, output_tokens: u64) -> Option<f64> {
        self.models.iter().map(|m| m.max_call_cost(input_tokens, output_tokens)).sum()
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

    fn pricing(&self) -> Option<Pricing> {
        (**self).pricing()
    }

    fn cost_usd(&self, response: &ModelResponse) -> Option<f64> {
        (**self).cost_usd(response)
    }

    fn unpriced_models(&self) -> Vec<String> {
        (**self).unpriced_models()
    }

    fn max_call_cost(&self, input_tokens: u64, output_tokens: u64) -> Option<f64> {
        (**self).max_call_cost(input_tokens, output_tokens)
    }
}
