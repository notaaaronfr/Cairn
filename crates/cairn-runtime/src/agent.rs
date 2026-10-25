use std::collections::BTreeMap;
use std::sync::Arc;
use std::time::{Duration, Instant};

use chrono::Utc;
use serde_json::{Value, json};

use crate::budget::Budget;
use crate::event::{Event, EventKind};
use crate::journal::{InMemoryJournal, Journal, JournalError};
use crate::message::{Content, Message, content_text};
use futures::StreamExt;

use crate::model::{Model, ModelError, ModelRequest, ModelResponse, StreamEvent, Usage};
use crate::policy::{Decision, EffectPolicy, Policy};
use crate::state::{PendingCall, RunState, Status};
use crate::tool::{Tool, ToolCtx};

#[derive(Debug, thiserror::Error)]
pub enum AgentError {
    #[error(transparent)]
    Journal(#[from] JournalError),
    #[error("no run with id `{0}`")]
    UnknownRun(String),
    #[error("call `{call_id}` in run `{run_id}` is not awaiting approval")]
    NotAwaitingApproval { run_id: String, call_id: String },
    #[error("invalid agent configuration: {0}")]
    Config(String),
}

/// Result of driving a run until it finishes or needs a human.
#[derive(Debug, Clone)]
pub struct RunOutcome {
    pub run_id: String,
    pub status: Status,
    pub steps: u32,
    pub usage: Usage,
    pub cost_usd: f64,
    pub awaiting_approval: Vec<PendingCall>,
    /// The validated final answer, when the agent has a finish schema.
    pub structured: Option<Value>,
}

impl RunOutcome {
    pub fn output(&self) -> Option<&str> {
        match &self.status {
            Status::Completed { output } => Some(output),
            _ => None,
        }
    }

    fn from_state(state: &RunState) -> Self {
        Self {
            run_id: state.run_id.clone(),
            status: state.status.clone(),
            steps: state.steps,
            usage: state.usage,
            cost_usd: state.cost_usd,
            awaiting_approval: state.awaiting_approval(),
            structured: state.structured_output.clone(),
        }
    }

    /// Deserialize the structured final answer into `T`.
    pub fn output_as<T: serde::de::DeserializeOwned>(&self) -> Option<Result<T, serde_json::Error>> {
        self.structured.clone().map(serde_json::from_value)
    }
}

pub struct Agent {
    name: String,
    model: Arc<dyn Model>,
    system: Option<String>,
    tools: Vec<Arc<dyn Tool>>,
    budget: Budget,
    policy: Arc<dyn Policy>,
    journal: Arc<dyn Journal>,
    max_tokens: u32,
    temperature: Option<f32>,
    provider_options: BTreeMap<String, Value>,
    on_delta: Option<DeltaFn>,
    tool_timeout: Option<Duration>,
    finish: Option<FinishSchema>,
}

/// A required shape for the run's final answer.
struct FinishSchema {
    schema: Value,
    validator: jsonschema::Validator,
    retries: u32,
}

impl FinishSchema {
    /// The validated JSON, or why the answer does not satisfy the schema.
    fn check(&self, text: &str) -> Result<Value, String> {
        let value = crate::output::extract_json(text).ok_or_else(|| "the answer is not valid JSON".to_owned())?;
        let errors: Vec<String> = self
            .validator
            .iter_errors(&value)
            .take(5)
            .map(|e| {
                let path = e.instance_path().to_string();
                if path.is_empty() { e.to_string() } else { format!("{path}: {e}") }
            })
            .collect();
        if errors.is_empty() { Ok(value) } else { Err(errors.join("; ")) }
    }
}

/// Default limit for a single tool call.
pub const DEFAULT_TOOL_TIMEOUT: Duration = Duration::from_secs(120);

type DeltaFn = Arc<dyn Fn(&StreamEvent) + Send + Sync>;

impl Agent {
    pub fn builder(name: impl Into<String>) -> AgentBuilder {
        AgentBuilder {
            name: name.into(),
            model: None,
            system: None,
            tools: Vec::new(),
            budget: Budget::default(),
            policy: Arc::new(EffectPolicy::default()),
            journal: Arc::new(InMemoryJournal::new()),
            max_tokens: 4096,
            temperature: None,
            provider_options: BTreeMap::new(),
            on_delta: None,
            tool_timeout: Some(DEFAULT_TOOL_TIMEOUT),
            finish_schema: None,
            finish_retries: 1,
        }
    }

    pub fn name(&self) -> &str {
        &self.name
    }

    pub fn journal(&self) -> &Arc<dyn Journal> {
        &self.journal
    }

    /// Start a new run with a user message and drive it.
    pub async fn run(&self, input: impl Into<String>) -> Result<RunOutcome, AgentError> {
        self.run_message(Message::user(input)).await
    }

    pub async fn run_message(&self, input: Message) -> Result<RunOutcome, AgentError> {
        let run_id = uuid::Uuid::now_v7().to_string();
        let first = Event {
            run_id: run_id.clone(),
            seq: 0,
            at: Utc::now(),
            kind: EventKind::RunStarted { agent: self.name.clone(), input },
        };
        self.journal.append(&first).await?;
        let state = RunState::from_events(std::slice::from_ref(&first)).expect("one event");
        tracing::info!(run_id = %run_id, agent = %self.name, "run started");
        self.drive(state).await
    }

    /// Continue a run from its journal, e.g. after a crash or an approval.
    /// Completed tool calls are never re-executed; a call that was in flight
    /// during a crash runs again with the same idempotency key.
    pub async fn resume(&self, run_id: &str) -> Result<RunOutcome, AgentError> {
        let state = self.state(run_id).await?;
        self.drive(state).await
    }

    /// Current state of a run, rebuilt from its journal.
    pub async fn state(&self, run_id: &str) -> Result<RunState, AgentError> {
        let events = self.journal.load(run_id).await?;
        RunState::from_events(&events).ok_or_else(|| AgentError::UnknownRun(run_id.to_owned()))
    }

    /// Record a human decision for a call that is awaiting approval.
    /// Call [`Agent::resume`] afterwards to continue the run.
    pub async fn decide(
        &self,
        run_id: &str,
        call_id: &str,
        approved: bool,
        by: Option<&str>,
        note: Option<&str>,
    ) -> Result<(), AgentError> {
        let mut state = self.state(run_id).await?;
        if !state.pending.iter().any(|p| p.id == call_id && p.awaiting_approval) {
            return Err(AgentError::NotAwaitingApproval { run_id: run_id.to_owned(), call_id: call_id.to_owned() });
        }
        self.emit(
            &mut state,
            EventKind::ApprovalDecided {
                call_id: call_id.to_owned(),
                approved,
                by: by.map(str::to_owned),
                note: note.map(str::to_owned),
            },
        )
        .await
    }

    async fn emit(&self, state: &mut RunState, kind: EventKind) -> Result<(), AgentError> {
        let event = Event { run_id: state.run_id.clone(), seq: state.next_seq, at: Utc::now(), kind };
        tracing::debug!(run_id = %event.run_id, seq = event.seq, event = event.kind.name(), "append");
        self.journal.append(&event).await?;
        state.apply(&event);
        Ok(())
    }

    fn find_tool(&self, name: &str) -> Option<&Arc<dyn Tool>> {
        self.tools.iter().find(|t| t.spec().name == name)
    }

    async fn drive(&self, mut state: RunState) -> Result<RunOutcome, AgentError> {
        loop {
            if state.status.is_terminal() {
                return Ok(RunOutcome::from_state(&state));
            }

            if !state.pending.is_empty() {
                self.handle_tool_calls(&mut state).await?;
                if state.status == Status::AwaitingApproval {
                    return Ok(RunOutcome::from_state(&state));
                }
                continue;
            }

            if let Some(reason) = self.budget.exceeded(state.steps, state.usage.total(), state.cost_usd) {
                self.emit(&mut state, EventKind::BudgetExceeded { reason }).await?;
                continue;
            }

            self.model_turn(&mut state).await?;
        }
    }

    async fn model_turn(&self, state: &mut RunState) -> Result<(), AgentError> {
        let mut request = ModelRequest {
            system: self.system_prompt(),
            messages: state.messages.clone(),
            tools: self.tools.iter().map(|t| t.spec().clone()).collect(),
            max_tokens: self.max_tokens,
            temperature: self.temperature,
            provider_options: self.provider_options.clone(),
        };
        // Shrink this step's answer limit to what the remaining budget can
        // pay for, so one call cannot overshoot it.
        let est_input = estimate_input_tokens(&request);
        match self.budget.output_cap(self.max_tokens, state.usage.total(), state.cost_usd, est_input, |i, o| {
            self.model.max_call_cost(i, o)
        }) {
            Ok(cap) => request.max_tokens = cap,
            Err(reason) => return self.emit(state, EventKind::BudgetExceeded { reason }).await,
        }
        let step = state.steps + 1;
        let started = Instant::now();

        let result = match &self.on_delta {
            Some(on_delta) => self.stream_turn(&request, on_delta).await,
            None => self.model.complete(&request).await,
        };
        match result {
            Ok(response) => {
                let cost_usd = self.model.cost_usd(&response);
                let wants_tools = response.content.iter().any(|c| matches!(c, Content::ToolUse { .. }));
                let text = content_text(&response.content);
                let duration_ms = started.elapsed().as_millis() as u64;
                self.emit(state, EventKind::ModelCalled { step, response, cost_usd, duration_ms }).await?;
                if !wants_tools {
                    self.finish(state, step, text).await?;
                }
            }
            Err(e) => {
                let error = e.to_string();
                let (usage, spent) = e.spent();
                // Unpriced spend can only happen without a USD budget.
                let cost_usd = spent.or_else(|| self.model.pricing().map(|p| p.cost(&usage)));
                self.emit(state, EventKind::ModelFailed { step, error: error.clone(), usage, cost_usd }).await?;
                self.emit(state, EventKind::RunFailed { error }).await?;
            }
        }
        Ok(())
    }

    /// The configured system prompt, plus — when a finish schema is set —
    /// the instruction stating the required final-answer format. Both are
    /// part of the request and visible in traces.
    fn system_prompt(&self) -> Option<String> {
        let Some(finish) = &self.finish else { return self.system.clone() };
        let rule = format!(
            "When you have finished, reply with only a JSON object (no other text) that matches this JSON Schema:\n{}",
            finish.schema
        );
        Some(match &self.system {
            Some(s) => format!("{s}\n\n{rule}"),
            None => rule,
        })
    }

    /// Complete the run, enforcing the finish schema if there is one.
    async fn finish(&self, state: &mut RunState, step: u32, text: String) -> Result<(), AgentError> {
        let Some(finish) = &self.finish else {
            return self.emit(state, EventKind::RunCompleted { output: text, structured: None }).await;
        };
        match finish.check(&text) {
            Ok(value) => self.emit(state, EventKind::RunCompleted { output: text, structured: Some(value) }).await,
            Err(reason) if state.finish_rejections < finish.retries => {
                self.emit(state, EventKind::FinishRejected { step, reason }).await
            }
            Err(reason) => {
                let error = format!(
                    "final answer did not match the finish schema after {} retr{}: {reason}",
                    finish.retries,
                    if finish.retries == 1 { "y" } else { "ies" }
                );
                self.emit(state, EventKind::RunFailed { error }).await
            }
        }
    }

    /// Streams deltas to the observer. Only the completed response is
    /// journaled, so replay and resume are identical with or without streaming.
    async fn stream_turn(&self, request: &ModelRequest, on_delta: &DeltaFn) -> Result<ModelResponse, ModelError> {
        let mut stream = self.model.stream(request).await?;
        while let Some(event) = stream.next().await {
            match event? {
                StreamEvent::Completed(response) => return Ok(response),
                delta => on_delta(&delta),
            }
        }
        Err(ModelError::Decode("stream ended before the response completed".into()))
    }

    async fn handle_tool_calls(&self, state: &mut RunState) -> Result<(), AgentError> {
        for call in state.pending.clone() {
            if call.awaiting_approval {
                continue;
            }
            if let Some(&approved) = state.approvals.get(&call.id) {
                if approved {
                    self.execute(state, &call).await?;
                } else {
                    let kind = EventKind::ToolDenied {
                        call_id: call.id.clone(),
                        tool: call.name.clone(),
                        reason: "rejected by approver".into(),
                    };
                    self.emit(state, kind).await?;
                }
                continue;
            }
            let Some(tool) = self.find_tool(&call.name) else {
                let kind = EventKind::ToolCompleted {
                    call_id: call.id.clone(),
                    tool: call.name.clone(),
                    output: json!({ "error": format!("unknown tool `{}`", call.name) }),
                    is_error: true,
                    duration_ms: 0,
                };
                self.emit(state, kind).await?;
                continue;
            };
            let kind = match self.policy.decide(tool.spec(), &call.input) {
                Decision::Allow => {
                    self.execute(state, &call).await?;
                    continue;
                }
                Decision::Deny { reason } => {
                    EventKind::ToolDenied { call_id: call.id.clone(), tool: call.name.clone(), reason }
                }
                Decision::RequireApproval { reason } => EventKind::ApprovalRequested {
                    call_id: call.id.clone(),
                    tool: call.name.clone(),
                    input: call.input.clone(),
                    reason,
                },
            };
            self.emit(state, kind).await?;
        }
        Ok(())
    }

    async fn execute(&self, state: &mut RunState, call: &PendingCall) -> Result<(), AgentError> {
        let tool = self.find_tool(&call.name).cloned();
        let ctx = ToolCtx::new(&state.run_id, &call.id);
        let started = Instant::now();
        let (output, is_error) = match tool {
            Some(tool) => {
                let limit = tool.timeout().or(self.tool_timeout);
                let fut = tool.call(call.input.clone(), &ctx);
                let result = match limit {
                    Some(limit) => tokio::time::timeout(limit, fut).await.unwrap_or_else(|_| {
                        tracing::warn!(tool = %call.name, ?limit, "tool timed out");
                        Err(crate::tool::ToolError(format!(
                            "tool `{}` timed out after {:.1}s and was cancelled",
                            call.name,
                            limit.as_secs_f64()
                        )))
                    }),
                    None => fut.await,
                };
                match result {
                    Ok(v) => (v, false),
                    Err(e) => (json!({ "error": e.0 }), true),
                }
            }
            None => (json!({ "error": format!("unknown tool `{}`", call.name) }), true),
        };
        let kind = EventKind::ToolCompleted {
            call_id: call.id.clone(),
            tool: call.name.clone(),
            output,
            is_error,
            duration_ms: started.elapsed().as_millis() as u64,
        };
        self.emit(state, kind).await
    }
}

/// Conservative input-size estimate: about 3 characters per token over the
/// serialized system prompt, messages and tool definitions.
fn estimate_input_tokens(request: &ModelRequest) -> u64 {
    let chars = request.system.as_deref().map_or(0, str::len)
        + serde_json::to_string(&request.messages).map_or(0, |s| s.len())
        + serde_json::to_string(&request.tools).map_or(0, |s| s.len());
    (chars as u64).div_ceil(3)
}

pub struct AgentBuilder {
    name: String,
    model: Option<Arc<dyn Model>>,
    system: Option<String>,
    tools: Vec<Arc<dyn Tool>>,
    budget: Budget,
    policy: Arc<dyn Policy>,
    journal: Arc<dyn Journal>,
    max_tokens: u32,
    temperature: Option<f32>,
    provider_options: BTreeMap<String, Value>,
    on_delta: Option<DeltaFn>,
    tool_timeout: Option<Duration>,
    finish_schema: Option<Value>,
    finish_retries: u32,
}

impl AgentBuilder {
    pub fn model(mut self, model: impl Model + 'static) -> Self {
        self.model = Some(Arc::new(model));
        self
    }

    pub fn system(mut self, prompt: impl Into<String>) -> Self {
        self.system = Some(prompt.into());
        self
    }

    pub fn tool(mut self, tool: impl Tool + 'static) -> Self {
        self.tools.push(Arc::new(tool));
        self
    }

    /// Add several tools at once (e.g. all tools from an MCP server).
    pub fn tools<T: Tool + 'static>(mut self, tools: impl IntoIterator<Item = T>) -> Self {
        self.tools.extend(tools.into_iter().map(|t| Arc::new(t) as Arc<dyn Tool>));
        self
    }

    pub fn budget(mut self, budget: Budget) -> Self {
        self.budget = budget;
        self
    }

    pub fn policy(mut self, policy: impl Policy + 'static) -> Self {
        self.policy = Arc::new(policy);
        self
    }

    pub fn journal(mut self, journal: impl Journal + 'static) -> Self {
        self.journal = Arc::new(journal);
        self
    }

    pub fn shared_journal(mut self, journal: Arc<dyn Journal>) -> Self {
        self.journal = journal;
        self
    }

    pub fn max_tokens(mut self, n: u32) -> Self {
        self.max_tokens = n;
        self
    }

    pub fn temperature(mut self, t: f32) -> Self {
        self.temperature = Some(t);
        self
    }

    /// Provider-specific request options, e.g.
    /// `.provider_option("anthropic", json!({"thinking": {"type": "enabled", "budget_tokens": 4096}}))`.
    pub fn provider_option(mut self, provider: &str, options: Value) -> Self {
        self.provider_options.insert(provider.to_owned(), options);
        self
    }

    /// Stream model output as it is generated (text, thinking, tool-call
    /// arguments). Enables streaming calls to the model.
    pub fn on_delta(mut self, f: impl Fn(&StreamEvent) + Send + Sync + 'static) -> Self {
        self.on_delta = Some(Arc::new(f));
        self
    }

    /// Default limit for each tool call (120 s unless changed). Tools can
    /// override it with `with_timeout`. A timed-out call is cancelled and
    /// reported to the model as an error; the run continues.
    pub fn tool_timeout(mut self, limit: Duration) -> Self {
        self.tool_timeout = Some(limit);
        self
    }

    /// Require the run's final answer to be JSON matching `schema` (JSON
    /// Schema). The schema is added to the system prompt; an answer that does
    /// not match gets [`finish_retries`](Self::finish_retries) corrective
    /// retries (default 1), after which the run fails. The validated value
    /// is available as [`RunOutcome::structured`].
    pub fn finish_schema(mut self, schema: Value) -> Self {
        self.finish_schema = Some(schema);
        self
    }

    /// [`finish_schema`](Self::finish_schema) derived from a Rust type.
    pub fn finish_as<T: schemars::JsonSchema>(self) -> Self {
        let mut schema = serde_json::to_value(schemars::schema_for!(T)).unwrap_or(Value::Null);
        if let Value::Object(map) = &mut schema {
            map.remove("$schema");
        }
        self.finish_schema(schema)
    }

    /// Corrective retries allowed for a final answer that fails the schema.
    pub fn finish_retries(mut self, n: u32) -> Self {
        self.finish_retries = n;
        self
    }

    /// Remove the default tool timeout (per-tool timeouts still apply).
    pub fn without_tool_timeout(mut self) -> Self {
        self.tool_timeout = None;
        self
    }

    pub fn build(self) -> Result<Agent, AgentError> {
        let model = self.model.ok_or_else(|| AgentError::Config("a model is required".into()))?;
        if self.budget.max_usd.is_some() {
            let unpriced = model.unpriced_models();
            if !unpriced.is_empty() {
                return Err(AgentError::Config(format!(
                    "max_usd is set but these models have no pricing, so spend could not be counted: {}. \
                     Set pricing (e.g. Priced::new(model, Pricing::new(input, output))); \
                     use Pricing::new(0.0, 0.0) for a model that is genuinely free",
                    unpriced.join(", ")
                )));
            }
        }
        let finish = match self.finish_schema {
            Some(schema) => {
                let validator = jsonschema::validator_for(&schema)
                    .map_err(|e| AgentError::Config(format!("invalid finish schema: {e}")))?;
                Some(FinishSchema { schema, validator, retries: self.finish_retries })
            }
            None => None,
        };
        let mut seen = std::collections::HashSet::new();
        for t in &self.tools {
            if !seen.insert(t.spec().name.clone()) {
                return Err(AgentError::Config(format!("duplicate tool name `{}`", t.spec().name)));
            }
        }
        Ok(Agent {
            name: self.name,
            model,
            system: self.system,
            tools: self.tools,
            budget: self.budget,
            policy: self.policy,
            journal: self.journal,
            max_tokens: self.max_tokens,
            temperature: self.temperature,
            provider_options: self.provider_options,
            on_delta: self.on_delta,
            tool_timeout: self.tool_timeout,
            finish,
        })
    }
}
