use std::collections::BTreeMap;
use std::sync::Arc;
use std::time::{Duration, Instant};

use chrono::Utc;
use serde_json::{Value, json};

use crate::budget::Budget;
use crate::context::ContextPolicy;
use crate::event::{Event, EventKind};
use crate::journal::{InMemoryJournal, Journal, JournalError};
use crate::lease::Lease;
use crate::message::{Content, Message, content_text};
use futures::StreamExt;
use tracing::Instrument;

use crate::middleware::{HookCtx, Middleware};
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
    #[error("a run with id `{0}` already exists")]
    RunExists(String),
    #[error("call `{call_id}` in run `{run_id}` is not awaiting approval")]
    NotAwaitingApproval { run_id: String, call_id: String },
    #[error("invalid agent configuration: {0}")]
    Config(String),
    #[error("run `{run_id}` is being driven by `{owner}` (lease until {expires_at})")]
    Leased { run_id: String, owner: String, expires_at: chrono::DateTime<chrono::Utc> },
    #[error("lost the lease on run `{run_id}`; another worker may have taken over")]
    LeaseLost { run_id: String },
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
    worker_id: String,
    lease_ttl: Duration,
    middleware: Vec<Arc<dyn Middleware>>,
    context: ContextPolicy,
}

/// Default lease length; renewed every third of it while a run is driven.
pub const DEFAULT_LEASE_TTL: Duration = Duration::from_secs(60);

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
            worker_id: None,
            lease_ttl: DEFAULT_LEASE_TTL,
            middleware: Vec::new(),
            context: ContextPolicy::default(),
        }
    }

    pub fn name(&self) -> &str {
        &self.name
    }

    /// A builder pre-filled with this agent's configuration, for deriving
    /// variants (another model, journal or tool set).
    pub fn to_builder(&self) -> AgentBuilder {
        AgentBuilder {
            name: self.name.clone(),
            model: Some(self.model.clone()),
            system: self.system.clone(),
            tools: self.tools.clone(),
            budget: self.budget.clone(),
            policy: self.policy.clone(),
            journal: self.journal.clone(),
            max_tokens: self.max_tokens,
            temperature: self.temperature,
            provider_options: self.provider_options.clone(),
            on_delta: self.on_delta.clone(),
            tool_timeout: self.tool_timeout,
            finish_schema: self.finish.as_ref().map(|f| f.schema.clone()),
            finish_retries: self.finish.as_ref().map_or(1, |f| f.retries),
            worker_id: Some(self.worker_id.clone()),
            lease_ttl: self.lease_ttl,
            middleware: self.middleware.clone(),
            context: self.context.clone(),
        }
    }

    pub fn worker_id(&self) -> &str {
        &self.worker_id
    }

    pub fn journal(&self) -> &Arc<dyn Journal> {
        &self.journal
    }

    /// Start a new run with a user message and drive it.
    pub async fn run(&self, input: impl Into<String>) -> Result<RunOutcome, AgentError> {
        self.run_message(Message::user(input)).await
    }

    pub async fn run_message(&self, input: Message) -> Result<RunOutcome, AgentError> {
        self.start(uuid::Uuid::now_v7().to_string(), input).await
    }

    /// Start a new run under an id the caller chooses, and drive it.
    ///
    /// For a worker serving an outside job queue: derive the id from the job
    /// (e.g. `"task-42:1"`), and after a crash the run can be found and
    /// [`resume`](Self::resume)d without having stored a generated id
    /// anywhere first. Fails with [`AgentError::RunExists`] if the id is
    /// already in the journal.
    pub async fn run_with_id(&self, run_id: &str, input: impl Into<String>) -> Result<RunOutcome, AgentError> {
        // The id names a file in the JSONL journal, so it must be a plain name:
        // nothing that could climb out of the directory or trip up a filesystem.
        if !crate::lease::is_plain_run_id(run_id) {
            return Err(AgentError::Config(format!(
                "run id `{run_id}` must be 1–128 characters of A–Z, a–z, 0–9, '-', '_' or '.', not starting with '.'"
            )));
        }
        if !self.journal.load(run_id).await?.is_empty() {
            return Err(AgentError::RunExists(run_id.to_owned()));
        }
        self.start(run_id.to_owned(), Message::user(input)).await
    }

    async fn start(&self, run_id: String, input: Message) -> Result<RunOutcome, AgentError> {
        let first = Event {
            run_id: run_id.clone(),
            seq: 0,
            at: Utc::now(),
            kind: EventKind::RunStarted { agent: self.name.clone(), input },
        };
        // Two workers racing to start the same chosen id: the journal's
        // sequence check lets exactly one in.
        match self.journal.append(&first).await {
            Ok(()) => {}
            Err(JournalError::Conflict { .. }) => return Err(AgentError::RunExists(run_id)),
            Err(e) => return Err(e.into()),
        }
        let state = RunState::from_events(std::slice::from_ref(&first)).expect("one event");
        tracing::info!(run_id = %run_id, agent = %self.name, "run started");
        self.drive_leased(state).await
    }

    /// Continue a run from its journal, e.g. after a crash or an approval.
    /// Completed tool calls are never re-executed; a call that was in flight
    /// during a crash runs again with the same idempotency key.
    ///
    /// Takes the run's lease first, so two workers never drive the same run;
    /// the loser gets [`AgentError::Leased`].
    pub async fn resume(&self, run_id: &str) -> Result<RunOutcome, AgentError> {
        let lease = self.take_lease(run_id).await?;
        // Load after taking the lease so we see everything the previous holder wrote.
        let result = match self.state(run_id).await {
            Ok(state) => self.drive(state, &lease).instrument(self.run_span(run_id)).await,
            Err(e) => Err(e),
        };
        self.end_lease(run_id, lease).await;
        result
    }

    async fn drive_leased(&self, state: RunState) -> Result<RunOutcome, AgentError> {
        let run_id = state.run_id.clone();
        let lease = self.take_lease(&run_id).await?;
        let span = self.run_span(&run_id);
        let result = self.drive(state, &lease).instrument(span).await;
        self.end_lease(&run_id, lease).await;
        result
    }

    /// `invoke_agent` span (OpenTelemetry GenAI conventions).
    fn run_span(&self, run_id: &str) -> tracing::Span {
        tracing::info_span!(
            "invoke_agent",
            otel.name = %format!("invoke_agent {}", self.name),
            gen_ai.operation.name = "invoke_agent",
            gen_ai.agent.name = %self.name,
            cairn.run_id = %run_id,
        )
    }

    async fn take_lease(&self, run_id: &str) -> Result<Lease, AgentError> {
        Lease::acquire(&self.journal, run_id, &self.worker_id, self.lease_ttl).await?.map_err(|holder| {
            AgentError::Leased { run_id: run_id.to_owned(), owner: holder.owner, expires_at: holder.expires_at }
        })
    }

    async fn end_lease(&self, _run_id: &str, lease: Lease) {
        lease.release().await;
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

    async fn drive(&self, mut state: RunState, lease: &Lease) -> Result<RunOutcome, AgentError> {
        loop {
            if state.status.is_terminal() {
                return Ok(RunOutcome::from_state(&state));
            }
            if lease.lost() {
                return Err(AgentError::LeaseLost { run_id: state.run_id.clone() });
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

    fn build_request(&self, state: &RunState) -> ModelRequest {
        let mut request = ModelRequest {
            system: self.system_prompt(),
            messages: state.messages.clone(),
            tools: self.tools.iter().map(|t| t.spec().clone()).collect(),
            max_tokens: self.max_tokens,
            temperature: self.temperature,
            provider_options: self.provider_options.clone(),
        };
        self.context.trim(&mut request);
        request
    }

    /// Replace older turns with a journaled summary if the request is too
    /// large. Returns `Ok(false)` when the run must stop (budget).
    async fn compact_if_needed(&self, state: &mut RunState, request: &ModelRequest) -> Result<bool, AgentError> {
        if !self.context.should_compact(estimate_input_tokens(request)) {
            return Ok(true);
        }
        let Some(cut) = self.context.compaction_cut(&state.messages) else {
            tracing::warn!(run_id = %state.run_id, "context is over the compaction threshold but nothing can be compacted");
            return Ok(true);
        };
        let model: &dyn Model = match self.context.summarizer() {
            Some(m) => m.as_ref(),
            None => self.model.as_ref(),
        };
        let older = &state.messages[1..1 + cut];
        let est = serde_json::to_string(older).map_or(0, |s| s.len() as u64).div_ceil(3);
        let max_tokens = match self
            .budget
            .output_cap(2048, state.usage.total(), state.cost_usd, est, |i, o| model.max_call_cost(i, o))
        {
            Ok(cap) => cap,
            Err(reason) => {
                self.emit(state, EventKind::BudgetExceeded { reason }).await?;
                return Ok(false);
            }
        };
        match crate::context::summarize(model, older, max_tokens).await {
            Ok((summary, usage, cost_usd)) => {
                let kind = EventKind::ContextCompacted { replaced: cut, summary, usage, cost_usd };
                self.emit(state, kind).await?;
            }
            Err(e) => tracing::warn!(run_id = %state.run_id, error = %e, "context compaction failed; continuing"),
        }
        Ok(true)
    }

    async fn model_turn(&self, state: &mut RunState) -> Result<(), AgentError> {
        let mut request = self.build_request(state);
        if !self.compact_if_needed(state, &request).await? {
            return Ok(());
        }
        if state.messages.len() != request.messages.len() {
            request = self.build_request(state);
        }
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
        let hook = HookCtx { run_id: state.run_id.clone(), agent: self.name.clone(), step };
        for m in &self.middleware {
            if let Err(reason) = m.before_model(&hook, &mut request).await {
                return self
                    .fail_step(state, step, format!("blocked by middleware before the model call: {reason}"))
                    .await;
            }
        }
        let model_id = self.model.id();
        let span = tracing::info_span!(
            "chat",
            otel.name = %format!("chat {model_id}"),
            gen_ai.operation.name = "chat",
            gen_ai.provider.name = %model_id.split(':').next().unwrap_or_default(),
            gen_ai.request.model = %model_id,
            gen_ai.request.max_tokens = request.max_tokens,
            gen_ai.response.model = tracing::field::Empty,
            gen_ai.response.finish_reasons = tracing::field::Empty,
            gen_ai.usage.input_tokens = tracing::field::Empty,
            gen_ai.usage.output_tokens = tracing::field::Empty,
            cairn.cost_usd = tracing::field::Empty,
            cairn.step = step,
            error.type = tracing::field::Empty,
        );
        let started = Instant::now();

        let mut result = match &self.on_delta {
            Some(on_delta) => self.stream_turn(&request, on_delta).instrument(span.clone()).await,
            None => self.model.complete(&request).instrument(span.clone()).await,
        };
        if let Ok(response) = &mut result {
            for m in &self.middleware {
                if let Err(reason) = m.after_model(&hook, response).await {
                    span.record("error.type", "middleware");
                    return self
                        .fail_step(state, step, format!("blocked by middleware after the model call: {reason}"))
                        .await;
                }
            }
        }
        match result {
            Ok(response) => {
                let cost_usd = self.model.cost_usd(&response);
                let usage = response.total_usage();
                span.record("gen_ai.response.model", response.model.as_str());
                span.record("gen_ai.response.finish_reasons", format!("{:?}", response.stop_reason).to_lowercase());
                span.record(
                    "gen_ai.usage.input_tokens",
                    usage.input_tokens + usage.cache_read_tokens + usage.cache_write_tokens,
                );
                span.record("gen_ai.usage.output_tokens", usage.output_tokens);
                if let Some(c) = cost_usd {
                    span.record("cairn.cost_usd", c);
                }
                let wants_tools = response.content.iter().any(|c| matches!(c, Content::ToolUse { .. }));
                let text = content_text(&response.content);
                let duration_ms = started.elapsed().as_millis() as u64;
                self.emit(state, EventKind::ModelCalled { step, response, cost_usd, duration_ms }).await?;
                if !wants_tools {
                    self.finish(state, step, text).await?;
                }
            }
            Err(e) => {
                span.record("error.type", "model_error");
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

    /// Fail the run at this step (journaled as ModelFailed + RunFailed).
    async fn fail_step(&self, state: &mut RunState, step: u32, error: String) -> Result<(), AgentError> {
        let kind = EventKind::ModelFailed { step, error: error.clone(), usage: Usage::default(), cost_usd: None };
        self.emit(state, kind).await?;
        self.emit(state, EventKind::RunFailed { error }).await
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
                    usage: Usage::default(),
                    cost_usd: None,
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
        let mut ctx = ToolCtx::new(&state.run_id, &call.id);
        ctx.remaining_usd = self.budget.max_usd.map(|max| (max - state.cost_usd).max(0.0));
        let started = Instant::now();
        let hook = HookCtx { run_id: state.run_id.clone(), agent: self.name.clone(), step: state.steps };
        let mut input = call.input.clone();
        if let Some(tool) = &tool {
            for m in &self.middleware {
                if let Err(reason) = m.before_tool(&hook, tool.spec(), &mut input).await {
                    let kind = EventKind::ToolDenied {
                        call_id: call.id.clone(),
                        tool: call.name.clone(),
                        reason: format!("blocked by middleware: {reason}"),
                    };
                    return self.emit(state, kind).await;
                }
            }
        }
        let span = tracing::info_span!(
            "execute_tool",
            otel.name = %format!("execute_tool {}", call.name),
            gen_ai.operation.name = "execute_tool",
            gen_ai.tool.name = %call.name,
            gen_ai.tool.call.id = %call.id,
            gen_ai.tool.type = "function",
            error.type = tracing::field::Empty,
        );
        let (mut output, mut is_error) = match &tool {
            Some(tool) => {
                let limit = tool.timeout().or(self.tool_timeout);
                let fut = tool.call(input, &ctx).instrument(span.clone());
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
        if let Some(tool) = &tool {
            for m in &self.middleware {
                if let Err(reason) = m.after_tool(&hook, tool.spec(), &mut output, &mut is_error).await {
                    output = json!({ "error": format!("blocked by middleware: {reason}") });
                    is_error = true;
                }
            }
        }
        if is_error {
            span.record("error.type", "tool_error");
        }
        let (usage, cost) = ctx.charged();
        let kind = EventKind::ToolCompleted {
            call_id: call.id.clone(),
            tool: call.name.clone(),
            output,
            is_error,
            duration_ms: started.elapsed().as_millis() as u64,
            usage,
            cost_usd: (cost > 0.0).then_some(cost),
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
    worker_id: Option<String>,
    lease_ttl: Duration,
    middleware: Vec<Arc<dyn Middleware>>,
    context: ContextPolicy,
}

impl AgentBuilder {
    pub fn model(mut self, model: impl Model + 'static) -> Self {
        self.model = Some(Arc::new(model));
        self
    }

    /// Replace the tool set.
    pub fn replace_tools(mut self, tools: Vec<Arc<dyn Tool>>) -> Self {
        self.tools = tools;
        self
    }

    pub(crate) fn tool_list(&self) -> &[Arc<dyn Tool>] {
        &self.tools
    }

    /// Stop streaming deltas (e.g. for background replays).
    pub fn without_on_delta(mut self) -> Self {
        self.on_delta = None;
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

    /// How long runs stay within the context window: trimming of old tool
    /// results and journaled compaction of older turns.
    pub fn context(mut self, policy: ContextPolicy) -> Self {
        self.context = policy;
        self
    }

    /// Add a hook around every model and tool call (run in the order added).
    pub fn middleware(mut self, m: impl Middleware + 'static) -> Self {
        self.middleware.push(Arc::new(m));
        self
    }

    /// Lower the USD budget to at most `usd` (keeping a smaller one).
    pub fn cap_usd(mut self, usd: f64) -> Self {
        self.budget.max_usd = Some(self.budget.max_usd.map_or(usd, |m| m.min(usd)));
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

    /// Name this worker in run leases (default: a random `worker-<uuid>`).
    pub fn worker_id(mut self, id: impl Into<String>) -> Self {
        self.worker_id = Some(id.into());
        self
    }

    /// How long a run lease lasts without renewal (default 60 s). A crashed
    /// worker's runs can be taken over after this long.
    pub fn lease_ttl(mut self, ttl: Duration) -> Self {
        self.lease_ttl = ttl;
        self
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
            let mut unpriced = model.unpriced_models();
            if let Some(summarizer) = self.context.summarizer() {
                unpriced.extend(summarizer.unpriced_models());
            }
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
            worker_id: self.worker_id.unwrap_or_else(|| format!("worker-{}", uuid::Uuid::now_v7())),
            lease_ttl: self.lease_ttl,
            middleware: self.middleware,
            context: self.context,
        })
    }
}
