use std::convert::Infallible;
use std::sync::Arc;
use std::time::Duration;

use axum::extract::State;
use axum::http::{HeaderMap, StatusCode, header};
use axum::response::sse::{Event as SseEvent, Sse};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use cairn_runtime::{
    Agent, AgentError, Event, EventKind, Message as CairnMessage, RunOutcome, RunState, Status, StreamEvent,
};
use chrono::Utc;
use futures::stream::{BoxStream, StreamExt};
use serde::de::DeserializeOwned;
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use tokio::sync::mpsc;

use crate::types::*;

/// Serves one Cairn agent over A2A 1.0 (JSON-RPC binding).
///
/// * An A2A **context** is a durable Cairn run; each **task** is one turn
///   of it (`<run id>:<turn>`). Follow-up messages with a `contextId`
///   continue the run with the whole conversation, via
///   [`Agent::continue_run`].
/// * The first message of a conversation is deduplicated by `messageId`:
///   a client that retries gets the same task, not a second run.
/// * Tool calls that need approval show up as `TASK_STATE_WORKING` with a
///   status message, until an operator decides (e.g. in `cairn-server`).
///   With [`client_approvals`](Self::client_approvals) the A2A client
///   decides instead: the task is `TASK_STATE_INPUT_REQUIRED` and the reply
///   "approve" or "deny" resumes it.
/// * `CancelTask` stops tasks that are not being worked on right now (e.g.
///   waiting for approval).
/// * Push notifications and the extended card are not supported.
pub struct A2aServer {
    agent: Arc<Agent>,
    card: AgentCard,
    rpc_path: String,
    key_hashes: Vec<String>,
    client_approvals: bool,
}

#[derive(Clone)]
struct App {
    agent: Arc<Agent>,
    card: AgentCard,
    key_hashes: Vec<String>,
    client_approvals: bool,
}

fn sha256_hex(s: &str) -> String {
    hex::encode(Sha256::digest(s.as_bytes()))
}

impl A2aServer {
    /// `url` is the public address of the JSON-RPC endpoint, as clients
    /// will reach it (e.g. `https://agents.example.com/a2a/support`). Its
    /// path is where [`router`](Self::router) mounts the endpoint; the card
    /// is served at `/.well-known/agent-card.json`.
    pub fn new(agent: impl Into<Arc<Agent>>, url: &str) -> Self {
        let agent = agent.into();
        let rpc_path = reqwest::Url::parse(url).map(|u| u.path().to_owned()).unwrap_or_else(|_| "/".into());
        let name = agent.name().to_owned();
        let modes = vec!["text/plain".to_owned(), "application/json".to_owned()];
        let card = AgentCard {
            description: format!("The `{name}` agent, served by Cairn."),
            name,
            supported_interfaces: vec![AgentInterface {
                url: url.to_owned(),
                protocol_binding: JSONRPC_BINDING.into(),
                tenant: None,
                protocol_version: PROTOCOL_VERSION.into(),
            }],
            provider: None,
            version: "1.0.0".into(),
            documentation_url: None,
            capabilities: AgentCapabilities {
                streaming: Some(true),
                push_notifications: Some(false),
                extensions: Vec::new(),
                extended_agent_card: Some(false),
            },
            security_schemes: Default::default(),
            security_requirements: Vec::new(),
            default_input_modes: modes.clone(),
            default_output_modes: modes,
            skills: Vec::new(),
            icon_url: None,
        };
        Self { agent, card, rpc_path, key_hashes: Vec::new(), client_approvals: false }
    }

    pub fn description(mut self, description: &str) -> Self {
        self.card.description = description.to_owned();
        self
    }

    /// The agent's own version (not the protocol's).
    pub fn version(mut self, version: &str) -> Self {
        self.card.version = version.to_owned();
        self
    }

    pub fn provider(mut self, organization: &str, url: &str) -> Self {
        self.card.provider = Some(AgentProvider { organization: organization.to_owned(), url: url.to_owned() });
        self
    }

    pub fn documentation_url(mut self, url: &str) -> Self {
        self.card.documentation_url = Some(url.to_owned());
        self
    }

    /// Advertise a skill on the card. Without any, the card lists one
    /// skill named after the agent.
    pub fn skill(mut self, skill: AgentSkill) -> Self {
        self.card.skills.push(skill);
        self
    }

    /// Require `Authorization: Bearer <key>` on the JSON-RPC endpoint. May
    /// be called several times; any listed key is accepted. The card stays
    /// public and declares the requirement.
    pub fn api_key(mut self, key: impl AsRef<str>) -> Self {
        self.key_hashes.push(sha256_hex(key.as_ref()));
        self.card.security_schemes.insert("bearer".into(), json!({"httpAuthSecurityScheme": {"scheme": "Bearer"}}));
        self.card.security_requirements = vec![json!({"schemes": {"bearer": {"list": []}}})];
        self
    }

    /// Let the A2A client approve or deny tool calls that need approval.
    /// Only for clients you would trust with that decision.
    pub fn client_approvals(mut self, enabled: bool) -> Self {
        self.client_approvals = enabled;
        self
    }

    pub fn card(&self) -> &AgentCard {
        &self.card
    }

    pub fn router(mut self) -> Router {
        if self.card.skills.is_empty() {
            let name = self.card.name.clone();
            self.card.skills.push(AgentSkill::new(&name, &name, &self.card.description).tags(["cairn"]));
        }
        let app = Arc::new(App {
            agent: self.agent,
            card: self.card,
            key_hashes: self.key_hashes,
            client_approvals: self.client_approvals,
        });
        Router::new()
            .route(AGENT_CARD_PATH, get(|State(app): State<Arc<App>>| async move { Json(app.card.clone()) }))
            .route(&self.rpc_path, post(rpc))
            .with_state(app)
    }

    pub async fn serve(self, addr: &str) -> std::io::Result<()> {
        let listener = tokio::net::TcpListener::bind(addr).await?;
        tracing::info!("A2A agent `{}` listening on http://{addr}", self.card.name);
        axum::serve(listener, self.router()).await
    }
}

/// A JSON-RPC error on its way to the client.
#[derive(Debug)]
struct RpcError {
    code: i64,
    message: String,
}

impl RpcError {
    fn new(code: i64, message: impl Into<String>) -> Self {
        Self { code, message: message.into() }
    }

    fn task_not_found(id: &str) -> Self {
        Self::new(codes::TASK_NOT_FOUND, format!("task `{id}` not found"))
    }

    fn unsupported(message: impl Into<String>) -> Self {
        Self::new(codes::UNSUPPORTED_OPERATION, message)
    }

    fn internal(e: impl std::fmt::Display) -> Self {
        Self::new(codes::INTERNAL_ERROR, e.to_string())
    }
}

fn agent_error(e: AgentError) -> RpcError {
    match e {
        AgentError::UnknownRun(id) => RpcError::task_not_found(&id),
        AgentError::Config(msg) => RpcError::new(codes::INVALID_PARAMS, msg),
        AgentError::Leased { .. } | AgentError::NotFinished(_) | AgentError::RunExists(_) => {
            RpcError::unsupported(format!("{e}"))
        }
        other => RpcError::internal(other),
    }
}

fn rpc_response(id: &Value, result: Result<Value, RpcError>) -> Response {
    let body = match result {
        Ok(result) => json!({"jsonrpc": "2.0", "id": id, "result": result}),
        Err(e) => json!({"jsonrpc": "2.0", "id": id, "error": {"code": e.code, "message": e.message}}),
    };
    Json(body).into_response()
}

fn params<T: DeserializeOwned>(body: &Value) -> Result<T, RpcError> {
    serde_json::from_value(body.get("params").cloned().unwrap_or(json!({})))
        .map_err(|e| RpcError::new(codes::INVALID_PARAMS, format!("invalid params: {e}")))
}

fn to_value(v: impl serde::Serialize) -> Result<Value, RpcError> {
    serde_json::to_value(v).map_err(RpcError::internal)
}

async fn rpc(State(app): State<Arc<App>>, headers: HeaderMap, body: axum::body::Bytes) -> Response {
    if !app.key_hashes.is_empty() {
        let presented =
            headers.get(header::AUTHORIZATION).and_then(|v| v.to_str().ok()).and_then(|v| v.strip_prefix("Bearer "));
        if !presented.is_some_and(|key| app.key_hashes.contains(&sha256_hex(key))) {
            return (
                StatusCode::UNAUTHORIZED,
                [(header::WWW_AUTHENTICATE, "Bearer")],
                Json(json!({"error": "invalid or missing API key"})),
            )
                .into_response();
        }
    }
    let Ok(body) = serde_json::from_slice::<Value>(&body) else {
        return rpc_response(&Value::Null, Err(RpcError::new(codes::PARSE_ERROR, "Invalid JSON payload")));
    };
    let id = body.get("id").cloned().unwrap_or(Value::Null);
    let method = match (body["jsonrpc"].as_str(), body["method"].as_str()) {
        (Some("2.0"), Some(m)) => m.to_owned(),
        _ => return rpc_response(&id, Err(RpcError::new(codes::INVALID_REQUEST, "not a JSON-RPC 2.0 request"))),
    };
    let version = headers.get("a2a-version").and_then(|v| v.to_str().ok()).unwrap_or("").trim();
    if !version.is_empty() && version != "1" && !version.starts_with("1.") {
        let msg = format!("A2A version {version} is not supported; this agent speaks {PROTOCOL_VERSION}");
        return rpc_response(&id, Err(RpcError::new(codes::VERSION_NOT_SUPPORTED, msg)));
    }

    match method.as_str() {
        "SendStreamingMessage" => match params::<SendMessageRequest>(&body) {
            Ok(req) => match app.stream_message(req).await {
                Ok(stream) => sse(id, stream),
                Err(e) => rpc_response(&id, Err(e)),
            },
            Err(e) => rpc_response(&id, Err(e)),
        },
        "SubscribeToTask" => match params::<TaskIdRequest>(&body) {
            Ok(req) => match app.subscribe(&req.id).await {
                Ok(stream) => sse(id, stream),
                Err(e) => rpc_response(&id, Err(e)),
            },
            Err(e) => rpc_response(&id, Err(e)),
        },
        _ => {
            let result = app.call(&method, &body).await;
            rpc_response(&id, result)
        }
    }
}

fn sse(id: Value, stream: BoxStream<'static, StreamResponse>) -> Response {
    let events = stream.map(move |item| {
        let body = json!({"jsonrpc": "2.0", "id": id, "result": item});
        Ok::<_, Infallible>(SseEvent::default().data(body.to_string()))
    });
    Sse::new(events).keep_alive(axum::response::sse::KeepAlive::default()).into_response()
}

/// Task id for turn `turn` of run `run_id`.
fn task_id(run_id: &str, turn: usize) -> String {
    format!("{run_id}:{turn}")
}

fn parse_task_id(id: &str) -> Option<(&str, usize)> {
    let (run, turn) = id.rsplit_once(':')?;
    Some((run, turn.parse().ok()?))
}

/// Index of the event that starts each turn.
fn turn_starts(events: &[Event]) -> Vec<usize> {
    events
        .iter()
        .enumerate()
        .filter(|(_, e)| matches!(e.kind, EventKind::RunStarted { .. } | EventKind::UserMessage { .. }))
        .map(|(i, _)| i)
        .collect()
}

fn turn_input(event: &Event) -> Option<&CairnMessage> {
    match &event.kind {
        EventKind::RunStarted { input, .. } => Some(input),
        EventKind::UserMessage { message } => Some(message),
        _ => None,
    }
}

/// Text the agent sees for an incoming A2A message.
fn input_text(message: &Message) -> Result<String, RpcError> {
    let text = message.text();
    if text.trim().is_empty() {
        return Err(RpcError::new(codes::INVALID_PARAMS, "the message has no content"));
    }
    Ok(text)
}

/// "approve" / "deny" from a client answering an approval question.
fn decision(message: &Message) -> Result<bool, RpcError> {
    for part in &message.parts {
        if let Some(approved) = part.data.as_ref().and_then(|d| d["approved"].as_bool()) {
            return Ok(approved);
        }
    }
    let text = message.text().trim().to_lowercase();
    let word = text.split(|c: char| !c.is_alphanumeric()).next().unwrap_or("");
    match word {
        "approve" | "approved" | "yes" | "y" | "ok" => Ok(true),
        "deny" | "denied" | "reject" | "rejected" | "no" | "n" => Ok(false),
        _ => Err(RpcError::new(codes::INVALID_PARAMS, "reply \"approve\" or \"deny\" (or data {\"approved\": bool})")),
    }
}

/// What a `SendMessage` asks for.
enum Plan {
    /// A new conversation (run).
    Start { run_id: String, text: String },
    /// The next turn of a finished conversation.
    Continue { run_id: String, text: String },
    /// An answer to an approval question.
    Decide { run_id: String, approved: bool },
    /// A retried first message: the task already exists.
    Existing,
}

impl App {
    async fn call(&self, method: &str, body: &Value) -> Result<Value, RpcError> {
        match method {
            "SendMessage" => {
                let req: SendMessageRequest = params(body)?;
                to_value(SendMessageResponse::Task(self.send(req).await?))
            }
            "GetTask" => {
                let req: GetTaskRequest = params(body)?;
                let (run, turn) = parse_task_id(&req.id).ok_or_else(|| RpcError::task_not_found(&req.id))?;
                to_value(self.task(run, turn, req.history_length).await?)
            }
            "ListTasks" => to_value(self.list(params(body)?).await?),
            "CancelTask" => {
                let req: TaskIdRequest = params(body)?;
                to_value(self.cancel(&req.id).await?)
            }
            "CreateTaskPushNotificationConfig"
            | "GetTaskPushNotificationConfig"
            | "ListTaskPushNotificationConfigs"
            | "DeleteTaskPushNotificationConfig" => {
                Err(RpcError::new(codes::PUSH_NOTIFICATION_NOT_SUPPORTED, "push notifications are not supported"))
            }
            "GetExtendedAgentCard" => {
                Err(RpcError::new(codes::EXTENDED_AGENT_CARD_NOT_CONFIGURED, "no extended agent card"))
            }
            other => Err(RpcError::new(codes::METHOD_NOT_FOUND, format!("method `{other}` not found"))),
        }
    }

    /// The run's events, if it is one of this agent's runs.
    async fn events(&self, run_id: &str) -> Result<Vec<Event>, RpcError> {
        let events = self.agent.journal().load(run_id).await.map_err(RpcError::internal)?;
        match events.first().map(|e| &e.kind) {
            Some(EventKind::RunStarted { agent, .. }) if agent == self.agent.name() => Ok(events),
            _ => Ok(Vec::new()),
        }
    }

    async fn task(&self, run_id: &str, turn: usize, history_length: Option<i32>) -> Result<Task, RpcError> {
        let events = self.events(run_id).await?;
        self.task_from(&events, run_id, turn, history_length)
            .ok_or_else(|| RpcError::task_not_found(&task_id(run_id, turn)))
    }

    fn task_from(&self, events: &[Event], run_id: &str, turn: usize, history_length: Option<i32>) -> Option<Task> {
        let starts = turn_starts(events);
        let start = *starts.get(turn)?;
        let end = starts.get(turn + 1).copied().unwrap_or(events.len());
        let state = RunState::from_events(&events[..end])?;
        let id = task_id(run_id, turn);
        let at = events[end - 1].at;

        let mut status = self.status(&state, &id, run_id);
        status.timestamp = Some(at);
        let mut artifacts = Vec::new();
        let mut history = Vec::new();
        if let Some(input) = turn_input(&events[start]) {
            let mut m = Message::new(Role::User, format!("{id}-input"), vec![Part::text(input.text())]);
            m.task_id = Some(id.clone());
            m.context_id = Some(run_id.to_owned());
            history.push(m);
        }
        if let Status::Completed { output } = &state.status {
            let mut parts = vec![Part::text(output.clone())];
            if let Some(structured) = &state.structured_output {
                parts.push(Part::data(structured.clone()));
            }
            artifacts.push(Artifact {
                artifact_id: format!("{id}-answer"),
                name: Some("answer".into()),
                description: None,
                parts: parts.clone(),
                metadata: None,
                extensions: Vec::new(),
            });
            let mut m = Message::new(Role::Agent, format!("{id}-answer"), parts);
            m.task_id = Some(id.clone());
            m.context_id = Some(run_id.to_owned());
            history.push(m);
        }
        if let Some(n) = history_length {
            let keep = usize::try_from(n).unwrap_or(0).min(history.len());
            history.drain(..history.len() - keep);
        }
        Some(Task { id, context_id: run_id.to_owned(), status, artifacts, history, metadata: None })
    }

    fn status(&self, state: &RunState, task_id: &str, context_id: &str) -> TaskStatus {
        let message = |text: String| {
            let mut m = Message::new(Role::Agent, format!("{task_id}-status"), vec![Part::text(text)]);
            m.task_id = Some(task_id.to_owned());
            m.context_id = Some(context_id.to_owned());
            Some(m)
        };
        let (state, message) = match &state.status {
            Status::Running => (TaskState::Working, None),
            Status::AwaitingApproval => {
                let calls: Vec<String> =
                    state.awaiting_approval().iter().map(|p| format!("`{}` with {}", p.name, p.input)).collect();
                let question = format!("Approval needed to run {}.", calls.join(", "));
                if self.client_approvals {
                    (TaskState::InputRequired, message(format!("{question} Reply \"approve\" or \"deny\".")))
                } else {
                    (TaskState::Working, message(format!("{question} Waiting for an operator.")))
                }
            }
            Status::Completed { .. } => (TaskState::Completed, None),
            Status::Failed { error } => (TaskState::Failed, message(error.clone())),
            Status::BudgetExceeded { reason } => (TaskState::Failed, message(format!("budget exceeded: {reason}"))),
            Status::Canceled { reason } => (TaskState::Canceled, message(reason.clone())),
        };
        TaskStatus { state, message, timestamp: Some(Utc::now()) }
    }

    /// Work out what a message asks for, and which task it belongs to.
    async fn plan(&self, message: &Message) -> Result<(Plan, String, usize), RpcError> {
        if let Some(tid) = &message.task_id {
            let (run, turn) = parse_task_id(tid).ok_or_else(|| RpcError::task_not_found(tid))?;
            if message.context_id.as_deref().is_some_and(|c| c != run) {
                return Err(RpcError::new(codes::INVALID_PARAMS, "taskId does not belong to contextId"));
            }
            let events = self.events(run).await?;
            let starts = turn_starts(&events);
            if turn >= starts.len() {
                return Err(RpcError::task_not_found(tid));
            }
            let state = RunState::from_events(&events).ok_or_else(|| RpcError::task_not_found(tid))?;
            if turn + 1 < starts.len() || state.status.is_terminal() {
                return Err(RpcError::unsupported(
                    "the task has finished; send a new message with its contextId to continue the conversation",
                ));
            }
            if state.status == Status::AwaitingApproval && self.client_approvals {
                let approved = decision(message)?;
                return Ok((Plan::Decide { run_id: run.to_owned(), approved }, run.to_owned(), turn));
            }
            return Err(RpcError::unsupported("the task is still being worked on"));
        }

        let text = input_text(message)?;
        if let Some(ctx) = &message.context_id {
            let events = self.agent.journal().load(ctx).await.map_err(RpcError::internal)?;
            if events.is_empty() {
                let plan = Plan::Start { run_id: ctx.clone(), text };
                return Ok((plan, ctx.clone(), 0));
            }
            let events = self.events(ctx).await?;
            let state = RunState::from_events(&events).ok_or_else(|| {
                RpcError::new(codes::INVALID_PARAMS, format!("context `{ctx}` belongs to another agent"))
            })?;
            let turns = turn_starts(&events).len();
            if !state.status.is_terminal() {
                return Err(RpcError::unsupported(format!(
                    "task `{}` in this context is still in progress",
                    task_id(ctx, turns - 1)
                )));
            }
            return Ok((Plan::Continue { run_id: ctx.clone(), text }, ctx.clone(), turns));
        }

        // A new conversation, named after the message so a retry finds it.
        let digest = sha256_hex(&format!("{}\0{}", self.agent.name(), message.message_id));
        let run_id = format!("a2a-{}", &digest[..32]);
        if !self.agent.journal().load(&run_id).await.map_err(RpcError::internal)?.is_empty() {
            return Ok((Plan::Existing, run_id, 0));
        }
        Ok((Plan::Start { run_id: run_id.clone(), text }, run_id, 0))
    }

    async fn send(&self, req: SendMessageRequest) -> Result<Task, RpcError> {
        let config = req.configuration.unwrap_or_default();
        if config.task_push_notification_config.is_some() {
            return Err(RpcError::new(codes::PUSH_NOTIFICATION_NOT_SUPPORTED, "push notifications are not supported"));
        }
        let (plan, run_id, turn) = self.plan(&req.message).await?;
        if config.return_immediately {
            let agent = self.agent.clone();
            tokio::spawn(async move {
                if let Err(e) = execute(&agent, plan).await {
                    tracing::warn!(error = %e, "A2A task failed to run");
                }
            });
            // Wait until the turn is in the journal so GetTask can find it.
            for _ in 0..200 {
                if turn_starts(&self.events(&run_id).await?).len() > turn {
                    break;
                }
                tokio::time::sleep(Duration::from_millis(10)).await;
            }
        } else {
            match execute(&self.agent, plan).await {
                Ok(_) | Err(AgentError::RunExists(_)) => {}
                Err(e) => return Err(agent_error(e)),
            }
        }
        self.task(&run_id, turn, config.history_length).await
    }

    async fn stream_message(&self, req: SendMessageRequest) -> Result<BoxStream<'static, StreamResponse>, RpcError> {
        let config = req.configuration.unwrap_or_default();
        if config.task_push_notification_config.is_some() {
            return Err(RpcError::new(codes::PUSH_NOTIFICATION_NOT_SUPPORTED, "push notifications are not supported"));
        }
        let (plan, run_id, turn) = self.plan(&req.message).await?;
        let id = task_id(&run_id, turn);
        let (tx, rx) = mpsc::unbounded_channel::<StreamResponse>();

        if let Plan::Existing = plan {
            let _ = tx.send(StreamResponse::Task(self.task(&run_id, turn, config.history_length).await?));
            return Ok(receiver_stream(rx));
        }
        let mut first = Task {
            id: id.clone(),
            context_id: run_id.clone(),
            status: TaskStatus { state: TaskState::Submitted, message: None, timestamp: Some(Utc::now()) },
            artifacts: Vec::new(),
            history: Vec::new(),
            metadata: None,
        };
        if let Plan::Decide { .. } = plan {
            first.status.state = TaskState::Working;
        }
        let _ = tx.send(StreamResponse::Task(first));

        // The same agent, reporting progress to this stream as it works.
        let progress = {
            let (tx, id, ctx) = (tx.clone(), id.clone(), run_id.clone());
            move |event: &StreamEvent| {
                let text = match event {
                    StreamEvent::TextDelta(t) => t.clone(),
                    StreamEvent::ToolCallStart { name, .. } => format!("\n[calling `{name}`]\n"),
                    _ => return,
                };
                let mut m = Message::new(Role::Agent, uuid::Uuid::now_v7().to_string(), vec![Part::text(text)]);
                m.task_id = Some(id.clone());
                m.context_id = Some(ctx.clone());
                let status = TaskStatus { state: TaskState::Working, message: Some(m), timestamp: Some(Utc::now()) };
                let update =
                    TaskStatusUpdateEvent { task_id: id.clone(), context_id: ctx.clone(), status, metadata: None };
                let _ = tx.send(StreamResponse::StatusUpdate(update));
            }
        };
        let agent = self.agent.to_builder().on_delta(progress).build().map_err(RpcError::internal)?;
        let app = self.clone();
        tokio::spawn(async move {
            let result = execute(&agent, plan).await;
            if let Err(e) = &result {
                if !matches!(e, AgentError::RunExists(_)) {
                    let msg = Message::new(Role::Agent, format!("{id}-error"), vec![Part::text(e.to_string())]);
                    let status =
                        TaskStatus { state: TaskState::Failed, message: Some(msg), timestamp: Some(Utc::now()) };
                    let update = TaskStatusUpdateEvent { task_id: id, context_id: run_id, status, metadata: None };
                    let _ = tx.send(StreamResponse::StatusUpdate(update));
                    return;
                }
            }
            if let Ok(task) = app.task(&run_id, turn, Some(0)).await {
                send_final(&tx, task);
            }
        });
        Ok(receiver_stream(rx))
    }

    async fn subscribe(&self, id: &str) -> Result<BoxStream<'static, StreamResponse>, RpcError> {
        let (run_id, turn) = parse_task_id(id).ok_or_else(|| RpcError::task_not_found(id))?;
        let task = self.task(run_id, turn, None).await?;
        if task.status.state.is_terminal() {
            return Err(RpcError::unsupported("the task has finished"));
        }
        let (tx, rx) = mpsc::unbounded_channel();
        let _ = tx.send(StreamResponse::Task(task.clone()));
        let app = self.clone();
        let run_id = run_id.to_owned();
        tokio::spawn(async move {
            let mut last = (task.status.state, task.status.message.map(|m| m.text()));
            loop {
                tokio::time::sleep(Duration::from_millis(250)).await;
                if tx.is_closed() {
                    return;
                }
                let Ok(task) = app.task(&run_id, turn, Some(0)).await else { return };
                let now = (task.status.state, task.status.message.as_ref().map(Message::text));
                if task.status.state.is_terminal() || task.status.state.is_interrupted() {
                    send_final(&tx, task);
                    return;
                }
                if now != last {
                    let update = TaskStatusUpdateEvent {
                        task_id: task.id,
                        context_id: task.context_id,
                        status: task.status,
                        metadata: None,
                    };
                    let _ = tx.send(StreamResponse::StatusUpdate(update));
                    last = now;
                }
            }
        });
        Ok(receiver_stream(rx))
    }

    async fn cancel(&self, id: &str) -> Result<Task, RpcError> {
        let (run_id, turn) = parse_task_id(id).ok_or_else(|| RpcError::task_not_found(id))?;
        let task = self.task(run_id, turn, None).await?;
        if task.status.state.is_terminal() {
            return Err(RpcError::new(codes::TASK_NOT_CANCELABLE, "the task has already finished"));
        }
        match self.agent.cancel(run_id, "canceled by the A2A client", Some("a2a-client")).await {
            Ok(_) => self.task(run_id, turn, None).await,
            Err(AgentError::Leased { .. }) => {
                Err(RpcError::new(codes::TASK_NOT_CANCELABLE, "the task is being worked on and cannot be interrupted"))
            }
            Err(e) => Err(agent_error(e)),
        }
    }

    async fn list(&self, req: ListTasksRequest) -> Result<ListTasksResponse, RpcError> {
        let page_size = req.page_size.unwrap_or(50).clamp(1, 100);
        let history = Some(req.history_length.unwrap_or(0));
        let mut tasks = Vec::new();
        if let Some(ctx) = &req.context_id {
            let events = self.events(ctx).await?;
            for turn in 0..turn_starts(&events).len() {
                tasks.extend(self.task_from(&events, ctx, turn, history));
            }
        } else {
            let runs = self.agent.journal().list_runs(page_size as usize * 4).await.map_err(RpcError::internal)?;
            for run in runs {
                let events = self.events(&run).await?;
                let turns = turn_starts(&events).len();
                if turns > 0 {
                    tasks.extend(self.task_from(&events, &run, turns - 1, history));
                }
            }
        }
        if let Some(state) = req.status {
            tasks.retain(|t| t.status.state == state);
        }
        if req.include_artifacts != Some(true) {
            tasks.iter_mut().for_each(|t| t.artifacts.clear());
        }
        let total_size = tasks.len() as i32;
        tasks.truncate(page_size as usize);
        Ok(ListTasksResponse { tasks, next_page_token: String::new(), page_size, total_size })
    }
}

/// Run what a message asked for.
async fn execute(agent: &Agent, plan: Plan) -> Result<Option<RunOutcome>, AgentError> {
    match plan {
        Plan::Start { run_id, text } => agent.run_with_id(&run_id, text).await.map(Some),
        Plan::Continue { run_id, text } => agent.continue_run(&run_id, text).await.map(Some),
        Plan::Decide { run_id, approved } => {
            let state = agent.state(&run_id).await?;
            for call in state.awaiting_approval() {
                agent.decide(&run_id, &call.id, approved, Some("a2a-client"), None).await?;
            }
            agent.resume(&run_id).await.map(Some)
        }
        Plan::Existing => Ok(None),
    }
}

/// The answer (if any) and the final status, ending a stream.
fn send_final(tx: &mpsc::UnboundedSender<StreamResponse>, task: Task) {
    for artifact in task.artifacts {
        let _ = tx.send(StreamResponse::ArtifactUpdate(TaskArtifactUpdateEvent {
            task_id: task.id.clone(),
            context_id: task.context_id.clone(),
            artifact,
            append: false,
            last_chunk: true,
            metadata: None,
        }));
    }
    let _ = tx.send(StreamResponse::StatusUpdate(TaskStatusUpdateEvent {
        task_id: task.id,
        context_id: task.context_id,
        status: task.status,
        metadata: None,
    }));
}

fn receiver_stream(rx: mpsc::UnboundedReceiver<StreamResponse>) -> BoxStream<'static, StreamResponse> {
    futures::stream::unfold(rx, |mut rx| async move { rx.recv().await.map(|item| (item, rx)) }).boxed()
}
