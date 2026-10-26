//! # cairn-server
//!
//! Host Cairn agents behind an HTTP API, with a crash-recovery worker and a
//! built-in Dev UI. Agents are Rust code, so the server is a library you
//! embed in your own binary:
//!
//! ```ignore
//! let journal: Arc<dyn Journal> = Arc::new(PostgresJournal::connect(&url).await?);
//! cairn_server::Server::new(journal)
//!     .agent(support_agent)
//!     .agent(billing_agent)
//!     .api_key(std::env::var("CAIRN_API_KEY")?)
//!     .recover_every(Duration::from_secs(30))
//!     .serve("0.0.0.0:8790")
//!     .await?;
//! ```
//!
//! Run several replicas on one PostgreSQL journal: run leases make sure
//! each run is driven by one replica at a time, and the recovery loop picks
//! up runs whose worker died.
//!
//! ## Endpoints
//!
//! | Method | Path | |
//! |---|---|---|
//! | GET | `/ui` | Dev UI |
//! | GET | `/healthz` | liveness |
//! | GET | `/v1/agents` | registered agents |
//! | POST | `/v1/agents/{name}/runs` | start a run: `{"input", "run_id"?, "wait"?}` |
//! | GET | `/v1/runs?limit=` | recent runs |
//! | GET | `/v1/runs/{id}` | run summary and full state |
//! | GET | `/v1/runs/{id}/events` | the run's journal |
//! | GET | `/v1/runs/{id}/state?at=N` | state after event N (time travel) |
//! | POST | `/v1/runs/{id}/approvals/{call_id}` | decide: `{"approved", "by"?, "note"?}` |
//! | POST | `/v1/runs/{id}/resume` | resume a run |
//! | GET | `/v1/approvals` | pending approvals across recent runs |

use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;

use axum::extract::{Path, Query, State};
use axum::http::{HeaderMap, StatusCode, header};
use axum::response::{Html, IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use cairn_runtime::{Agent, AgentError, Event, EventKind, Journal, PendingCall, RunOutcome, RunState, Status};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};

const UI: &str = include_str!("ui.html");

/// Builder and router for hosting agents.
pub struct Server {
    journal: Arc<dyn Journal>,
    agents: HashMap<String, Arc<Agent>>,
    key_hashes: Vec<String>,
    recover_every: Option<Duration>,
}

struct App {
    journal: Arc<dyn Journal>,
    agents: HashMap<String, Arc<Agent>>,
    key_hashes: Vec<String>,
}

fn sha256_hex(s: &str) -> String {
    hex::encode(Sha256::digest(s.as_bytes()))
}

impl Server {
    pub fn new(journal: Arc<dyn Journal>) -> Self {
        Self { journal, agents: HashMap::new(), key_hashes: Vec::new(), recover_every: None }
    }

    /// Host an agent. It is rebuilt to use the server's journal so the API,
    /// the UI and recovery all see its runs.
    pub fn agent(mut self, agent: Agent) -> Self {
        let agent = if Arc::ptr_eq(agent.journal(), &self.journal) {
            agent
        } else {
            agent.to_builder().shared_journal(self.journal.clone()).build().expect("agent was already valid")
        };
        self.agents.insert(agent.name().to_owned(), Arc::new(agent));
        self
    }

    /// Require `Authorization: Bearer <key>` (or `x-api-key`) on the API.
    /// May be called several times; any listed key is accepted.
    pub fn api_key(mut self, key: impl AsRef<str>) -> Self {
        self.key_hashes.push(sha256_hex(key.as_ref()));
        self
    }

    /// Periodically resume runs that are still running but have no live
    /// worker (e.g. after a crash). Leases keep replicas from colliding.
    pub fn recover_every(mut self, interval: Duration) -> Self {
        self.recover_every = Some(interval);
        self
    }

    /// The router. Starts the recovery loop if configured (requires a Tokio runtime).
    pub fn router(self) -> Router {
        let app = Arc::new(App { journal: self.journal, agents: self.agents, key_hashes: self.key_hashes });
        if let Some(every) = self.recover_every {
            tokio::spawn(recovery_loop(app.clone(), every));
        }
        let api = Router::new()
            .route("/v1/agents", get(list_agents))
            .route("/v1/agents/{name}/runs", post(start_run))
            .route("/v1/runs", get(list_runs))
            .route("/v1/runs/{id}", get(get_run))
            .route("/v1/runs/{id}/events", get(get_events))
            .route("/v1/runs/{id}/state", get(get_state))
            .route("/v1/runs/{id}/approvals/{call_id}", post(decide))
            .route("/v1/runs/{id}/resume", post(resume))
            .route("/v1/approvals", get(list_approvals))
            .route_layer(axum::middleware::from_fn_with_state(app.clone(), auth));
        Router::new()
            .route("/healthz", get(|| async { "ok" }))
            .route("/", get(|| async { axum::response::Redirect::temporary("/ui") }))
            .route("/ui", get(|| async { Html(UI) }))
            .merge(api)
            .with_state(app)
    }

    pub async fn serve(self, addr: &str) -> std::io::Result<()> {
        let listener = tokio::net::TcpListener::bind(addr).await?;
        tracing::info!("cairn-server listening on http://{addr} (Dev UI at /ui)");
        axum::serve(listener, self.router()).await
    }
}

async fn auth(
    State(app): State<Arc<App>>,
    headers: HeaderMap,
    request: axum::extract::Request,
    next: axum::middleware::Next,
) -> Response {
    if app.key_hashes.is_empty() {
        return next.run(request).await;
    }
    let presented = headers
        .get(header::AUTHORIZATION)
        .and_then(|v| v.to_str().ok())
        .and_then(|v| v.strip_prefix("Bearer "))
        .or_else(|| headers.get("x-api-key").and_then(|v| v.to_str().ok()));
    match presented {
        Some(key) if app.key_hashes.contains(&sha256_hex(key)) => next.run(request).await,
        _ => error(StatusCode::UNAUTHORIZED, "invalid or missing API key"),
    }
}

fn error(status: StatusCode, message: impl Into<String>) -> Response {
    ApiError(status, message.into()).into_response()
}

/// A small error for helpers; converted to a JSON response at the edge.
struct ApiError(StatusCode, String);

impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        (self.0, Json(json!({ "error": self.1 }))).into_response()
    }
}

fn agent_error(e: AgentError) -> Response {
    let status = match &e {
        AgentError::UnknownRun(_) => StatusCode::NOT_FOUND,
        AgentError::RunExists(_) | AgentError::Leased { .. } | AgentError::NotAwaitingApproval { .. } => {
            StatusCode::CONFLICT
        }
        AgentError::Config(_) => StatusCode::BAD_REQUEST,
        _ => StatusCode::INTERNAL_SERVER_ERROR,
    };
    error(status, e.to_string())
}

/// A compact view of a run for lists and dashboards.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RunSummary {
    pub run_id: String,
    pub agent: String,
    pub kind: String,
    pub status: String,
    pub output: Option<String>,
    pub error: Option<String>,
    pub steps: u32,
    pub tokens: u64,
    pub cost_usd: f64,
    pub started_at: Option<chrono::DateTime<chrono::Utc>>,
    pub updated_at: Option<chrono::DateTime<chrono::Utc>>,
    pub awaiting_approval: Vec<PendingCall>,
}

fn status_name(s: &Status) -> &'static str {
    match s {
        Status::Running => "running",
        Status::AwaitingApproval => "awaiting_approval",
        Status::Completed { .. } => "completed",
        Status::Failed { .. } => "failed",
        Status::BudgetExceeded { .. } => "budget_exceeded",
    }
}

fn summarize(events: &[Event]) -> Option<RunSummary> {
    let state = RunState::from_events(events)?;
    let (output, error) = match &state.status {
        Status::Completed { output } => (Some(output.clone()), None),
        Status::Failed { error } => (None, Some(error.clone())),
        Status::BudgetExceeded { reason } => (None, Some(reason.clone())),
        _ => (None, None),
    };
    let kind = match events.first().map(|e| &e.kind) {
        Some(EventKind::WorkflowStarted { .. }) => "workflow",
        _ => "agent",
    };
    Some(RunSummary {
        run_id: state.run_id.clone(),
        agent: state.agent.clone(),
        kind: kind.into(),
        status: status_name(&state.status).into(),
        output,
        error,
        steps: state.steps,
        tokens: state.usage.total(),
        cost_usd: state.cost_usd,
        started_at: events.first().map(|e| e.at),
        updated_at: events.last().map(|e| e.at),
        awaiting_approval: state.awaiting_approval(),
    })
}

fn outcome_json(o: &RunOutcome) -> Value {
    json!({
        "run_id": o.run_id,
        "status": status_name(&o.status),
        "output": o.output(),
        "structured": o.structured,
        "steps": o.steps,
        "tokens": o.usage.total(),
        "cost_usd": o.cost_usd,
        "awaiting_approval": o.awaiting_approval,
        "error": match &o.status { Status::Failed { error } => Some(error.clone()), Status::BudgetExceeded { reason } => Some(reason.clone()), _ => None },
    })
}

async fn list_agents(State(app): State<Arc<App>>) -> Response {
    let mut names: Vec<&String> = app.agents.keys().collect();
    names.sort();
    Json(json!(names.iter().map(|n| json!({"name": n})).collect::<Vec<_>>())).into_response()
}

#[derive(Deserialize)]
struct StartRun {
    input: String,
    run_id: Option<String>,
    #[serde(default)]
    wait: bool,
}

async fn start_run(State(app): State<Arc<App>>, Path(name): Path<String>, Json(body): Json<StartRun>) -> Response {
    let Some(agent) = app.agents.get(&name).cloned() else {
        return error(StatusCode::NOT_FOUND, format!("no agent named `{name}`"));
    };
    let run_id = body.run_id.unwrap_or_else(|| uuid::Uuid::now_v7().to_string());
    if body.wait {
        return match agent.run_with_id(&run_id, body.input).await {
            Ok(o) => Json(outcome_json(&o)).into_response(),
            Err(e) => agent_error(e),
        };
    }
    if !cairn_runtime::lease::is_plain_run_id(&run_id) {
        return error(StatusCode::BAD_REQUEST, format!("`{run_id}` is not a valid run id"));
    }
    match app.journal.load(&run_id).await {
        Ok(events) if !events.is_empty() => {
            return error(StatusCode::CONFLICT, format!("run `{run_id}` already exists"));
        }
        Ok(_) => {}
        Err(e) => return error(StatusCode::INTERNAL_SERVER_ERROR, e.to_string()),
    }
    let id = run_id.clone();
    tokio::spawn(async move {
        if let Err(e) = agent.run_with_id(&id, body.input).await {
            tracing::warn!(run_id = %id, error = %e, "run did not start");
        }
    });
    (StatusCode::ACCEPTED, Json(json!({ "run_id": run_id }))).into_response()
}

#[derive(Deserialize)]
struct ListQuery {
    limit: Option<usize>,
}

async fn recent(app: &App, limit: usize) -> Result<Vec<RunSummary>, ApiError> {
    let ids =
        app.journal.list_runs(limit).await.map_err(|e| ApiError(StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;
    let mut out = Vec::with_capacity(ids.len());
    for id in ids {
        let events =
            app.journal.load(&id).await.map_err(|e| ApiError(StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;
        if let Some(s) = summarize(&events) {
            out.push(s);
        }
    }
    Ok(out)
}

async fn list_runs(State(app): State<Arc<App>>, Query(q): Query<ListQuery>) -> Response {
    match recent(&app, q.limit.unwrap_or(50).min(500)).await {
        Ok(runs) => Json(runs).into_response(),
        Err(r) => r.into_response(),
    }
}

async fn load(app: &App, id: &str) -> Result<Vec<Event>, ApiError> {
    match app.journal.load(id).await {
        Ok(events) if events.is_empty() => Err(ApiError(StatusCode::NOT_FOUND, format!("no run `{id}`"))),
        Ok(events) => Ok(events),
        Err(e) => Err(ApiError(StatusCode::INTERNAL_SERVER_ERROR, e.to_string())),
    }
}

async fn get_run(State(app): State<Arc<App>>, Path(id): Path<String>) -> Response {
    match load(&app, &id).await {
        Ok(events) => {
            let summary = summarize(&events);
            let state = RunState::from_events(&events);
            Json(json!({ "summary": summary, "state": state, "hosted": summary.as_ref().is_some_and(|s| app.agents.contains_key(&s.agent)) }))
                .into_response()
        }
        Err(r) => r.into_response(),
    }
}

async fn get_events(State(app): State<Arc<App>>, Path(id): Path<String>) -> Response {
    match load(&app, &id).await {
        Ok(events) => Json(events).into_response(),
        Err(r) => r.into_response(),
    }
}

#[derive(Deserialize)]
struct AtQuery {
    at: Option<usize>,
}

async fn get_state(State(app): State<Arc<App>>, Path(id): Path<String>, Query(q): Query<AtQuery>) -> Response {
    match load(&app, &id).await {
        Ok(events) => {
            let end = q.at.map_or(events.len(), |n| (n + 1).min(events.len()));
            Json(RunState::from_events(&events[..end])).into_response()
        }
        Err(r) => r.into_response(),
    }
}

/// The hosted agent that owns a run.
async fn owner(app: &App, id: &str) -> Result<Arc<Agent>, ApiError> {
    let events = load(app, id).await?;
    let name = match events.first().map(|e| &e.kind) {
        Some(EventKind::RunStarted { agent, .. }) => agent.clone(),
        _ => return Err(ApiError(StatusCode::BAD_REQUEST, format!("`{id}` is not an agent run"))),
    };
    app.agents.get(&name).cloned().ok_or_else(|| {
        ApiError(
            StatusCode::CONFLICT,
            format!("agent `{name}` is not hosted by this server, so it cannot act on this run"),
        )
    })
}

#[derive(Deserialize)]
struct Decision {
    approved: bool,
    by: Option<String>,
    note: Option<String>,
}

async fn decide(
    State(app): State<Arc<App>>,
    Path((id, call_id)): Path<(String, String)>,
    Json(d): Json<Decision>,
) -> Response {
    let agent = match owner(&app, &id).await {
        Ok(a) => a,
        Err(r) => return r.into_response(),
    };
    if let Err(e) = agent.decide(&id, &call_id, d.approved, d.by.as_deref(), d.note.as_deref()).await {
        return agent_error(e);
    }
    spawn_resume(agent, id.clone());
    (StatusCode::ACCEPTED, Json(json!({ "run_id": id, "decided": call_id, "approved": d.approved }))).into_response()
}

async fn resume(State(app): State<Arc<App>>, Path(id): Path<String>) -> Response {
    match owner(&app, &id).await {
        Ok(agent) => {
            spawn_resume(agent, id.clone());
            (StatusCode::ACCEPTED, Json(json!({ "run_id": id }))).into_response()
        }
        Err(r) => r.into_response(),
    }
}

fn spawn_resume(agent: Arc<Agent>, id: String) {
    tokio::spawn(async move {
        match agent.resume(&id).await {
            Ok(_) | Err(AgentError::Leased { .. }) => {}
            Err(e) => tracing::warn!(run_id = %id, error = %e, "resume failed"),
        }
    });
}

async fn list_approvals(State(app): State<Arc<App>>) -> Response {
    match recent(&app, 200).await {
        Ok(runs) => {
            let pending: Vec<Value> = runs
                .into_iter()
                .flat_map(|r| {
                    let (run_id, agent) = (r.run_id, r.agent);
                    r.awaiting_approval.into_iter().map(move |c| {
                        json!({ "run_id": run_id, "agent": agent, "call_id": c.id, "tool": c.name, "input": c.input })
                    })
                })
                .collect();
            Json(pending).into_response()
        }
        Err(r) => r.into_response(),
    }
}

/// Resume hosted runs that are still `running` but not being driven.
async fn recovery_loop(app: Arc<App>, every: Duration) {
    let mut tick = tokio::time::interval(every);
    loop {
        tick.tick().await;
        let Ok(ids) = app.journal.list_runs(500).await else { continue };
        for id in ids {
            let Ok(events) = app.journal.load(&id).await else { continue };
            let Some(EventKind::RunStarted { agent, .. }) = events.first().map(|e| &e.kind) else { continue };
            let Some(agent) = app.agents.get(agent).cloned() else { continue };
            let running = RunState::from_events(&events).is_some_and(|s| s.status == Status::Running);
            if running {
                // A live worker holds the lease, so this is a no-op for active runs.
                spawn_resume(agent, id);
            }
        }
    }
}
