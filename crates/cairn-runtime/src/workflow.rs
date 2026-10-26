//! Durable workflows and multi-agent orchestration.
//!
//! A workflow is ordinary async Rust. Each [`WorkflowCtx::step`] and
//! [`WorkflowCtx::agent`] result is journaled, so after a crash, a deploy or
//! a human approval, [`Workflow::resume`] re-runs the function and every
//! finished step returns its recorded result instead of running again. A
//! child agent that was mid-run is resumed, not restarted; a child waiting
//! for approval suspends the whole workflow until it is decided.
//!
//! ```ignore
//! let wf = Workflow::new("triage", journal, |ctx: WorkflowCtx, ticket: String| async move {
//!     let facts = ctx.agent("research", &researcher, &ticket).await?.into_text()?;
//!     let [legal, tone] = ctx.concurrent("review", &[&legal_reviewer, &tone_reviewer], &facts).await?
//!         .try_into().unwrap();
//!     ctx.agent("reply", &writer, format!("{facts}\n{}\n{}", legal.text(), tone.text())).await?.into_text()
//! });
//! let outcome = wf.run_with_id("ticket-42", "Customer was double charged".into()).await?;
//! ```
//!
//! **Determinism:** code between steps must make the same step calls, in
//! the same order, on every replay. Put anything non-deterministic (clocks,
//! randomness, I/O) inside a step. Concurrent steps get their ids when the
//! step is created, so `join!`/`join_all` over steps created in a fixed
//! order is deterministic.

use std::collections::HashMap;
use std::future::Future;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use chrono::Utc;
use futures::future::BoxFuture;
use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

use crate::agent::{Agent, AgentError, DEFAULT_LEASE_TTL};
use crate::event::{Event, EventKind};
use crate::journal::{Journal, JournalError};
use crate::lease::{Lease, child_run_id, is_plain_run_id};
use crate::model::Usage;
use crate::state::Status;
use crate::tool::{Effect, ToolCtx, ToolError, tool};

#[derive(Debug, Clone, thiserror::Error)]
pub enum WorkflowError {
    #[error("step `{step}` failed: {error}")]
    Step { step: String, error: String },
    /// A child run is waiting for a human decision. Returned by step
    /// futures; [`Workflow`] turns it into [`WorkflowStatus::Suspended`].
    #[error("waiting for approval in child run `{0}`")]
    Suspended(String),
    #[error("workflow budget exhausted: {0}")]
    Budget(String),
    #[error("run `{0}` already exists")]
    RunExists(String),
    #[error("no workflow run `{0}`")]
    UnknownRun(String),
    #[error("{0}")]
    Leased(String),
    #[error("journal error: {0}")]
    Journal(String),
    #[error("{0}")]
    Other(String),
}

impl From<JournalError> for WorkflowError {
    fn from(e: JournalError) -> Self {
        WorkflowError::Journal(e.to_string())
    }
}

impl WorkflowError {
    pub fn other(msg: impl Into<String>) -> Self {
        WorkflowError::Other(msg.into())
    }
}

/// What a child agent run produced.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AgentResult {
    pub agent: String,
    pub run_id: String,
    pub status: Status,
    pub output: Option<String>,
    pub structured: Option<Value>,
    pub steps: u32,
    pub usage: Usage,
    pub cost_usd: f64,
}

impl AgentResult {
    pub fn completed(&self) -> bool {
        matches!(self.status, Status::Completed { .. })
    }

    /// The answer as text: structured output as JSON if present, else the
    /// final message, else an empty string.
    pub fn text(&self) -> String {
        self.structured.as_ref().map(Value::to_string).or_else(|| self.output.clone()).unwrap_or_default()
    }

    /// The answer, or an error if the child did not complete.
    pub fn into_text(self) -> Result<String, WorkflowError> {
        if self.completed() {
            Ok(self.text())
        } else {
            Err(WorkflowError::Step { step: self.run_id.clone(), error: format!("{:?}", self.status) })
        }
    }

    /// Deserialize the structured answer.
    pub fn output_as<T: DeserializeOwned>(&self) -> Result<T, WorkflowError> {
        let value = self.structured.clone().ok_or_else(|| WorkflowError::other("child has no structured output"))?;
        serde_json::from_value(value).map_err(|e| WorkflowError::other(e.to_string()))
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum WorkflowStatus<O> {
    Completed(O),
    /// Waiting on these child runs' approvals; decide them, then resume.
    Suspended(Vec<String>),
    Failed(String),
}

#[derive(Debug, Clone)]
pub struct WorkflowOutcome<O> {
    pub run_id: String,
    pub status: WorkflowStatus<O>,
    pub usage: Usage,
    pub cost_usd: f64,
}

impl<O> WorkflowOutcome<O> {
    pub fn output(&self) -> Option<&O> {
        match &self.status {
            WorkflowStatus::Completed(o) => Some(o),
            _ => None,
        }
    }
}

#[derive(Debug, Clone)]
struct Record {
    output: Value,
    is_error: bool,
}

struct Inner {
    run_id: String,
    journal: Arc<dyn Journal>,
    next_seq: tokio::sync::Mutex<u64>,
    recorded: HashMap<String, Record>,
    counters: Mutex<HashMap<String, u32>>,
    spent: Mutex<(Usage, f64)>,
    max_usd: Option<f64>,
    suspended: Mutex<Vec<String>>,
    lease_lost: Arc<AtomicBool>,
}

/// Handle a workflow function uses to run durable steps.
#[derive(Clone)]
pub struct WorkflowCtx {
    inner: Arc<Inner>,
}

impl WorkflowCtx {
    pub fn run_id(&self) -> &str {
        &self.inner.run_id
    }

    /// USD still available to this workflow, if it has a budget.
    pub fn remaining_usd(&self) -> Option<f64> {
        self.inner.max_usd.map(|max| (max - self.inner.spent.lock().unwrap().1).max(0.0))
    }

    /// Allocate a stable id for the next use of `name` (eagerly, so that
    /// concurrently created steps get ids in creation order).
    fn step_id(&self, name: &str) -> String {
        let mut counters = self.inner.counters.lock().unwrap();
        let n = counters.entry(name.to_owned()).or_insert(0);
        *n += 1;
        if *n == 1 { name.to_owned() } else { format!("{name}#{n}") }
    }

    fn recorded(&self, id: &str) -> Option<Record> {
        self.inner.recorded.get(id).cloned()
    }

    async fn record(
        &self,
        step: &str,
        output: Value,
        is_error: bool,
        child_run: Option<String>,
        usage: Usage,
        cost_usd: Option<f64>,
    ) -> Result<(), WorkflowError> {
        if self.inner.lease_lost.load(Ordering::SeqCst) {
            return Err(WorkflowError::Leased(format!("lost the lease on workflow `{}`", self.inner.run_id)));
        }
        let mut seq = self.inner.next_seq.lock().await;
        let event = Event {
            run_id: self.inner.run_id.clone(),
            seq: *seq,
            at: Utc::now(),
            kind: EventKind::StepCompleted { step: step.to_owned(), output, is_error, child_run, usage, cost_usd },
        };
        self.inner.journal.append(&event).await?;
        *seq += 1;
        let mut spent = self.inner.spent.lock().unwrap();
        spent.0.add(&usage);
        spent.1 += cost_usd.unwrap_or(0.0);
        Ok(())
    }

    /// Run `f` once and journal its result. On replay the recorded result is
    /// returned and `f` is not called. Errors are recorded too, so a replay
    /// takes the same path; retry inside `f` if a failure may be transient.
    pub fn step<T, E, F, Fut>(
        &self,
        name: &str,
        f: F,
    ) -> impl Future<Output = Result<T, WorkflowError>> + Send + 'static + use<T, E, F, Fut>
    where
        T: Serialize + DeserializeOwned + Send + 'static,
        E: std::fmt::Display,
        F: FnOnce() -> Fut + Send + 'static,
        Fut: Future<Output = Result<T, E>> + Send + 'static,
    {
        let id = self.step_id(name);
        let ctx = self.clone();
        async move {
            if let Some(r) = ctx.recorded(&id) {
                return replay(&id, r);
            }
            match f().await.map_err(|e| e.to_string()) {
                Ok(v) => {
                    let value = serde_json::to_value(&v).map_err(|e| WorkflowError::other(e.to_string()))?;
                    ctx.record(&id, value, false, None, Usage::default(), None).await?;
                    Ok(v)
                }
                Err(error) => {
                    ctx.record(&id, json!(error), true, None, Usage::default(), None).await?;
                    Err(WorkflowError::Step { step: id, error })
                }
            }
        }
    }

    /// Run `agent` on `input` as a durable child run (id
    /// `<workflow run>.<step>`). The child's spend counts toward the
    /// workflow budget, and its own USD budget is capped to what is left.
    pub fn agent<S: Into<String>>(
        &self,
        name: &str,
        agent: &Agent,
        input: S,
    ) -> impl Future<Output = Result<AgentResult, WorkflowError>> + Send + 'static + use<S> {
        let id = self.step_id(name);
        let child = child_run_id(&self.inner.run_id, &id);
        let builder = agent.to_builder();
        let input = input.into();
        let ctx = self.clone();
        async move {
            if let Some(r) = ctx.recorded(&id) {
                return replay(&id, r);
            }
            let mut builder = builder;
            if let Some(remaining) = ctx.remaining_usd() {
                if remaining <= 0.0 {
                    return Err(WorkflowError::Budget(format!("nothing left for step `{id}`")));
                }
                builder = builder.cap_usd(remaining);
            }
            let agent = builder.build().map_err(|e| WorkflowError::other(e.to_string()))?;
            let outcome = match agent.run_with_id(&child, input).await {
                Err(AgentError::RunExists(_)) => agent.resume(&child).await,
                other => other,
            }
            .map_err(|e| match e {
                AgentError::Leased { .. } => WorkflowError::Leased(e.to_string()),
                e => WorkflowError::other(e.to_string()),
            })?;
            if outcome.status == Status::AwaitingApproval {
                ctx.inner.suspended.lock().unwrap().push(child.clone());
                return Err(WorkflowError::Suspended(child));
            }
            let result = AgentResult {
                agent: agent.name().to_owned(),
                run_id: child.clone(),
                output: outcome.output().map(str::to_owned),
                status: outcome.status,
                structured: outcome.structured,
                steps: outcome.steps,
                usage: outcome.usage,
                cost_usd: outcome.cost_usd,
            };
            let value = serde_json::to_value(&result).map_err(|e| WorkflowError::other(e.to_string()))?;
            ctx.record(&id, value, false, Some(child), result.usage, Some(result.cost_usd)).await?;
            Ok(result)
        }
    }

    /// Run agents one after another, each receiving the previous answer.
    /// Stops with an error if an agent does not complete.
    pub async fn sequential(&self, name: &str, agents: &[&Agent], input: &str) -> Result<AgentResult, WorkflowError> {
        let mut text = input.to_owned();
        let mut last = None;
        for (i, agent) in agents.iter().enumerate() {
            let result = self.agent(&format!("{name}.{i}.{}", agent.name()), agent, text).await?;
            if !result.completed() {
                return Err(WorkflowError::Step { step: result.run_id.clone(), error: format!("{:?}", result.status) });
            }
            text = result.text();
            last = Some(result);
        }
        last.ok_or_else(|| WorkflowError::other("sequential needs at least one agent"))
    }

    /// Run agents in parallel on the same input; results in agent order.
    pub async fn concurrent(
        &self,
        name: &str,
        agents: &[&Agent],
        input: &str,
    ) -> Result<Vec<AgentResult>, WorkflowError> {
        let futures: Vec<_> = agents
            .iter()
            .enumerate()
            .map(|(i, a)| self.agent(&format!("{name}.{i}.{}", a.name()), a, input.to_owned()))
            .collect();
        let results = futures::future::join_all(futures).await;
        // Suspension wins over other errors so the workflow can be resumed.
        if let Some(Err(WorkflowError::Suspended(id))) =
            results.iter().find(|r| matches!(r, Err(WorkflowError::Suspended(_))))
        {
            return Err(WorkflowError::Suspended(id.clone()));
        }
        results.into_iter().collect()
    }

    /// Let agents pass control to each other. Each agent gets one
    /// `handoff_to_<name>` tool per other agent; when it uses one, the target
    /// runs next with the original task, the note and the previous answer.
    /// Ends when an agent finishes without handing off, or after
    /// `max_handoffs`.
    pub async fn handoff(
        &self,
        name: &str,
        agents: &[&Agent],
        start: &str,
        input: &str,
        max_handoffs: u32,
    ) -> Result<AgentResult, WorkflowError> {
        let names: Vec<String> = agents.iter().map(|a| a.name().to_owned()).collect();
        let index = |n: &str| names.iter().position(|x| x == n);
        let mut current = index(start).ok_or_else(|| WorkflowError::other(format!("no agent named `{start}`")))?;
        let mut task = input.to_owned();
        for hop in 0..=max_handoffs {
            let agent = with_handoff_tools(agents[current], &names)?;
            let step = format!("{name}.{hop}.{}", names[current]);
            let result = self.agent(&step, &agent, task.clone()).await?;
            let target = self.step(&format!("{step}.route"), {
                // The child's history lives in its agent's journal.
                let (journal, run) = (agent.journal().clone(), result.run_id.clone());
                move || async move { last_handoff(journal.as_ref(), &run).await }
            });
            let Some((to, note)) = target.await? else { return Ok(result) };
            let Some(next) = index(&to) else { return Ok(result) };
            if hop == max_handoffs {
                return Ok(result);
            }
            task = format!(
                "{input}\n\n---\nHanded off from `{}`: {note}\nTheir answer so far:\n{}",
                names[current],
                result.text()
            );
            current = next;
        }
        unreachable!("loop returns")
    }

    /// Agents take turns on a shared transcript. With a `manager`, the
    /// manager picks each speaker and decides when the task is done;
    /// otherwise speakers rotate and the chat ends when an answer contains
    /// `done_marker` or after `max_rounds`.
    pub async fn group_chat(
        &self,
        name: &str,
        chat: GroupChat<'_>,
        task: &str,
    ) -> Result<GroupChatResult, WorkflowError> {
        let names: Vec<String> = chat.participants.iter().map(|a| a.name().to_owned()).collect();
        let mut transcript: Vec<Turn> = Vec::new();
        let manager = match chat.manager {
            Some(m) => Some(
                m.to_builder()
                    .finish_schema(json!({
                        "type": "object",
                        "properties": {
                            "next": {"type": "string", "enum": names},
                            "done": {"type": "boolean"},
                            "answer": {"type": "string"}
                        },
                        "required": ["done"]
                    }))
                    .build()
                    .map_err(|e| WorkflowError::other(e.to_string()))?,
            ),
            None => None,
        };
        for round in 0..chat.max_rounds {
            let speaker = match &manager {
                Some(m) => {
                    let prompt = format!(
                        "Task:\n{task}\n\nParticipants: {}\n\nTranscript so far:\n{}\n\n\
                         Choose who speaks next (`next`), or set `done` to true with the final `answer`.",
                        names.join(", "),
                        render(&transcript)
                    );
                    let decision = self.agent(&format!("{name}.{round}.manager"), m, prompt).await?;
                    let d = decision.structured.clone().unwrap_or(Value::Null);
                    if d["done"] == true {
                        let answer = d["answer"].as_str().map(str::to_owned);
                        return Ok(GroupChatResult { transcript, answer, rounds: round });
                    }
                    let next = d["next"].as_str().unwrap_or_default();
                    names
                        .iter()
                        .position(|n| n == next)
                        .ok_or_else(|| WorkflowError::other(format!("manager chose unknown participant `{next}`")))?
                }
                None => round as usize % names.len(),
            };
            let prompt = format!(
                "Task:\n{task}\n\nTranscript so far:\n{}\n\nYou are `{}`. Add your contribution.",
                render(&transcript),
                names[speaker]
            );
            let result =
                self.agent(&format!("{name}.{round}.{}", names[speaker]), chat.participants[speaker], prompt).await?;
            let text = result.text();
            transcript.push(Turn { speaker: names[speaker].clone(), text: text.clone() });
            if manager.is_none() && chat.done_marker.is_some_and(|m| text.contains(m)) {
                return Ok(GroupChatResult { transcript, answer: Some(text), rounds: round + 1 });
            }
        }
        Ok(GroupChatResult { answer: None, rounds: chat.max_rounds, transcript })
    }
}

fn replay<T: DeserializeOwned>(id: &str, r: Record) -> Result<T, WorkflowError> {
    if r.is_error {
        return Err(WorkflowError::Step {
            step: id.to_owned(),
            error: r.output.as_str().map(str::to_owned).unwrap_or_else(|| r.output.to_string()),
        });
    }
    serde_json::from_value(r.output)
        .map_err(|e| WorkflowError::other(format!("recorded result of `{id}` no longer fits its type: {e}")))
}

/// Settings for [`WorkflowCtx::group_chat`].
pub struct GroupChat<'a> {
    pub participants: &'a [&'a Agent],
    pub manager: Option<&'a Agent>,
    pub max_rounds: u32,
    pub done_marker: Option<&'a str>,
}

impl<'a> GroupChat<'a> {
    pub fn round_robin(participants: &'a [&'a Agent], max_rounds: u32) -> Self {
        Self { participants, manager: None, max_rounds, done_marker: Some("TERMINATE") }
    }

    pub fn managed(participants: &'a [&'a Agent], manager: &'a Agent, max_rounds: u32) -> Self {
        Self { participants, manager: Some(manager), max_rounds, done_marker: None }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Turn {
    pub speaker: String,
    pub text: String,
}

#[derive(Debug, Clone, PartialEq)]
pub struct GroupChatResult {
    pub transcript: Vec<Turn>,
    /// Final answer: the manager's, or the turn that said the done marker.
    pub answer: Option<String>,
    pub rounds: u32,
}

fn render(transcript: &[Turn]) -> String {
    if transcript.is_empty() {
        return "(empty)".into();
    }
    transcript.iter().map(|t| format!("[{}] {}", t.speaker, t.text)).collect::<Vec<_>>().join("\n")
}

fn handoff_tool_name(agent: &str) -> String {
    let clean: String = agent.chars().map(|c| if c.is_ascii_alphanumeric() || c == '_' { c } else { '_' }).collect();
    format!("handoff_to_{clean}").chars().take(64).collect()
}

/// Shapes the handoff tool's input schema; the note is read back from the
/// journal by `last_handoff`, not from this struct.
#[derive(Deserialize, schemars::JsonSchema)]
#[allow(dead_code)]
struct HandoffInput {
    /// What the next agent needs to know and do.
    note: String,
}

fn with_handoff_tools(agent: &Agent, names: &[String]) -> Result<Agent, WorkflowError> {
    let mut builder = agent.to_builder();
    for target in names.iter().filter(|n| n.as_str() != agent.name()) {
        let t = tool(
            &handoff_tool_name(target),
            &format!("Hand the task to `{target}`. Use when `{target}` is better suited for the rest of it."),
            Effect::Read,
            |_: HandoffInput, _ctx: ToolCtx| async move {
                Ok::<_, ToolError>(json!(
                    "Handoff accepted. End your turn now with a brief summary for the next agent."
                ))
            },
        );
        builder = builder.tool(t);
    }
    builder.build().map_err(|e| WorkflowError::other(e.to_string()))
}

/// The last handoff a child run made: (target agent, note).
async fn last_handoff(journal: &dyn Journal, run_id: &str) -> Result<Option<(String, String)>, JournalError> {
    let events = journal.load(run_id).await?;
    let mut calls: HashMap<String, (String, Value)> = HashMap::new();
    let mut last = None;
    for e in &events {
        match &e.kind {
            EventKind::ModelCalled { response, .. } => {
                for c in &response.content {
                    if let crate::message::Content::ToolUse { id, name, input } = c {
                        calls.insert(id.clone(), (name.clone(), input.clone()));
                    }
                }
            }
            EventKind::ToolCompleted { call_id, is_error: false, .. } => {
                if let Some((name, input)) = calls.get(call_id) {
                    if let Some(target) = name.strip_prefix("handoff_to_") {
                        last = Some((target.to_owned(), input["note"].as_str().unwrap_or_default().to_owned()));
                    }
                }
            }
            _ => {}
        }
    }
    Ok(last)
}

type WorkflowFn<I, O> = Arc<dyn Fn(WorkflowCtx, I) -> BoxFuture<'static, Result<O, WorkflowError>> + Send + Sync>;

/// A named, durable workflow.
pub struct Workflow<I, O> {
    name: String,
    journal: Arc<dyn Journal>,
    worker_id: String,
    lease_ttl: Duration,
    max_usd: Option<f64>,
    f: WorkflowFn<I, O>,
}

impl<I, O> Workflow<I, O>
where
    I: Serialize + DeserializeOwned + Send + 'static,
    O: Serialize + DeserializeOwned + Send + 'static,
{
    pub fn new<F, Fut>(name: impl Into<String>, journal: Arc<dyn Journal>, f: F) -> Self
    where
        F: Fn(WorkflowCtx, I) -> Fut + Send + Sync + 'static,
        Fut: Future<Output = Result<O, WorkflowError>> + Send + 'static,
    {
        Self {
            name: name.into(),
            journal,
            worker_id: format!("worker-{}", uuid::Uuid::now_v7()),
            lease_ttl: DEFAULT_LEASE_TTL,
            max_usd: None,
            f: Arc::new(move |ctx, input| Box::pin(f(ctx, input))),
        }
    }

    /// Total USD the workflow's child agents may spend. Each child's own
    /// USD budget is capped to what is left when it starts.
    pub fn max_usd(mut self, usd: f64) -> Self {
        self.max_usd = Some(usd);
        self
    }

    pub fn worker_id(mut self, id: impl Into<String>) -> Self {
        self.worker_id = id.into();
        self
    }

    pub fn lease_ttl(mut self, ttl: Duration) -> Self {
        self.lease_ttl = ttl;
        self
    }

    pub async fn run(&self, input: I) -> Result<WorkflowOutcome<O>, WorkflowError> {
        self.run_with_id(&uuid::Uuid::now_v7().to_string(), input).await
    }

    /// Start under a caller-chosen id (see [`Agent::run_with_id`]).
    pub async fn run_with_id(&self, run_id: &str, input: I) -> Result<WorkflowOutcome<O>, WorkflowError> {
        if !is_plain_run_id(run_id) {
            return Err(WorkflowError::other(format!("`{run_id}` is not a valid run id")));
        }
        let value = serde_json::to_value(&input).map_err(|e| WorkflowError::other(e.to_string()))?;
        let first = Event {
            run_id: run_id.to_owned(),
            seq: 0,
            at: Utc::now(),
            kind: EventKind::WorkflowStarted { workflow: self.name.clone(), input: value },
        };
        match self.journal.append(&first).await {
            Ok(()) => {}
            Err(JournalError::Conflict { .. }) => return Err(WorkflowError::RunExists(run_id.to_owned())),
            Err(e) => return Err(e.into()),
        }
        self.drive(run_id, input).await
    }

    /// Continue a workflow after a crash, a deploy or an approval.
    pub async fn resume(&self, run_id: &str) -> Result<WorkflowOutcome<O>, WorkflowError> {
        let events = self.journal.load(run_id).await?;
        let Some(EventKind::WorkflowStarted { input, .. }) = events.first().map(|e| &e.kind) else {
            return Err(WorkflowError::UnknownRun(run_id.to_owned()));
        };
        if let Some(done) = finished(run_id, &events)? {
            return Ok(done);
        }
        let input: I = serde_json::from_value(input.clone()).map_err(|e| WorkflowError::other(e.to_string()))?;
        self.drive(run_id, input).await
    }

    async fn drive(&self, run_id: &str, input: I) -> Result<WorkflowOutcome<O>, WorkflowError> {
        let lease = Lease::acquire(&self.journal, run_id, &self.worker_id, self.lease_ttl).await?.map_err(|h| {
            WorkflowError::Leased(format!("workflow `{run_id}` is driven by `{}` until {}", h.owner, h.expires_at))
        })?;
        // Load after taking the lease, to see everything the last holder wrote.
        let events = self.journal.load(run_id).await?;
        if let Some(done) = finished(run_id, &events)? {
            lease.release().await;
            return Ok(done);
        }
        let mut recorded = HashMap::new();
        let mut spent = (Usage::default(), 0.0);
        for e in &events {
            if let EventKind::StepCompleted { step, output, is_error, usage, cost_usd, .. } = &e.kind {
                recorded.insert(step.clone(), Record { output: output.clone(), is_error: *is_error });
                spent.0.add(usage);
                spent.1 += cost_usd.unwrap_or(0.0);
            }
        }
        let ctx = WorkflowCtx {
            inner: Arc::new(Inner {
                run_id: run_id.to_owned(),
                journal: self.journal.clone(),
                next_seq: tokio::sync::Mutex::new(events.len() as u64),
                recorded,
                counters: Mutex::default(),
                spent: Mutex::new(spent),
                max_usd: self.max_usd,
                suspended: Mutex::default(),
                lease_lost: lease.lost_flag(),
            }),
        };

        let result = (self.f)(ctx.clone(), input).await;
        let suspended = std::mem::take(&mut *ctx.inner.suspended.lock().unwrap());
        let (usage, cost_usd) = *ctx.inner.spent.lock().unwrap();
        let outcome = |status| WorkflowOutcome { run_id: run_id.to_owned(), status, usage, cost_usd };

        let terminal = match result {
            Err(WorkflowError::Suspended(_)) => None,
            _ if !suspended.is_empty() => None,
            Err(e @ (WorkflowError::Leased(_) | WorkflowError::Journal(_))) => {
                lease.release().await;
                return Err(e);
            }
            Ok(o) => {
                let value = serde_json::to_value(&o).map_err(|e| WorkflowError::other(e.to_string()))?;
                Some((EventKind::WorkflowCompleted { output: value }, WorkflowStatus::Completed(o)))
            }
            Err(e) => Some((EventKind::WorkflowFailed { error: e.to_string() }, WorkflowStatus::Failed(e.to_string()))),
        };
        let status = match terminal {
            None => WorkflowStatus::Suspended(suspended),
            Some((kind, status)) => {
                let mut seq = ctx.inner.next_seq.lock().await;
                let event = Event { run_id: run_id.to_owned(), seq: *seq, at: Utc::now(), kind };
                self.journal.append(&event).await?;
                *seq += 1;
                status
            }
        };
        lease.release().await;
        Ok(outcome(status))
    }
}

/// The recorded outcome of a finished workflow run, if it finished.
fn finished<O: DeserializeOwned>(run_id: &str, events: &[Event]) -> Result<Option<WorkflowOutcome<O>>, WorkflowError> {
    let state = crate::state::RunState::from_events(events).ok_or_else(|| WorkflowError::UnknownRun(run_id.into()))?;
    let status = match events.last().map(|e| &e.kind) {
        Some(EventKind::WorkflowCompleted { output }) => WorkflowStatus::Completed(
            serde_json::from_value(output.clone()).map_err(|e| WorkflowError::other(e.to_string()))?,
        ),
        Some(EventKind::WorkflowFailed { error }) => WorkflowStatus::Failed(error.clone()),
        _ => return Ok(None),
    };
    Ok(Some(WorkflowOutcome { run_id: run_id.to_owned(), status, usage: state.usage, cost_usd: state.cost_usd }))
}
