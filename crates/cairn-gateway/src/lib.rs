//! # cairn-gateway
//!
//! A drop-in, OpenAI- and Anthropic-compatible endpoint. Point any existing
//! framework at it (LangGraph, CrewAI, OpenAI Agents SDK, Vercel AI SDK,
//! Microsoft Agent Framework, raw SDKs) by changing its base URL, and every
//! model call gains per-key budgets, model allowlists, cross-provider
//! failover and an audit trail. Either wire format can be served by any
//! provider: an OpenAI-SDK client can run on Claude and vice versa.

mod anthropic_api;
pub mod config;
mod openai_api;

use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::Instant;

use axum::Router;
use axum::extract::State;
use axum::http::{HeaderMap, StatusCode, header};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use cairn_models::Pricing;
use cairn_runtime::{Content, Model, ModelError, ModelRequest, ModelResponse, ModelStream, StreamEvent};
use futures::StreamExt;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use tokio::io::AsyncWriteExt;

pub use config::{Config, KeyConfig};

/// Writes one client wire format as server-sent events, incrementally.
pub trait SseRenderer: Send {
    fn start(&mut self) -> String;
    fn delta(&mut self, event: &StreamEvent) -> String;
    fn finish(&mut self, response: &ModelResponse) -> String;
    fn error(&mut self, err: &GatewayError) -> String;
}

pub type ModelFactory = Arc<dyn Fn(&str) -> Result<Arc<dyn Model>, ModelError> + Send + Sync>;

#[derive(Debug, Clone)]
pub struct GatewayError {
    pub status: u16,
    pub kind: &'static str,
    pub message: String,
}

impl GatewayError {
    pub fn bad_request(msg: impl Into<String>) -> Self {
        Self { status: 400, kind: "invalid_request_error", message: msg.into() }
    }
    fn unauthorized() -> Self {
        Self { status: 401, kind: "authentication_error", message: "invalid or missing API key".into() }
    }
    fn forbidden(msg: String) -> Self {
        Self { status: 403, kind: "permission_error", message: msg }
    }
    fn not_found(msg: String) -> Self {
        Self { status: 404, kind: "not_found_error", message: msg }
    }
    fn budget(msg: String) -> Self {
        Self { status: 429, kind: "budget_exceeded", message: msg }
    }
    fn upstream(e: &ModelError) -> Self {
        let status = match e {
            ModelError::Api { status, .. } if *status == 429 => 429,
            ModelError::Api { status, .. } if *status < 500 => *status,
            _ => 502,
        };
        Self { status, kind: "upstream_error", message: e.to_string() }
    }
}

#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize)]
pub struct Spend {
    pub usd: f64,
    pub tokens: u64,
}

/// One line of the audit log.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AuditRecord {
    pub at: chrono::DateTime<chrono::Utc>,
    pub request_id: String,
    pub key: String,
    pub api: String,
    pub model: String,
    pub served_by: Option<String>,
    pub failed_over: Vec<String>,
    pub status: u16,
    pub error: Option<String>,
    pub tokens: u64,
    pub cost_usd: f64,
    pub duration_ms: u64,
    pub tool_calls: Vec<String>,
}

pub struct Gateway {
    config: Config,
    factory: ModelFactory,
    models: Mutex<HashMap<String, Arc<dyn Model>>>,
    spend: Mutex<HashMap<String, Spend>>,
    audit: Option<tokio::sync::Mutex<tokio::fs::File>>,
}

impl Gateway {
    /// Uses `cairn_models::from_spec` (credentials from the environment).
    pub async fn new(config: Config) -> std::io::Result<Arc<Self>> {
        Self::with_factory(config, Arc::new(|spec: &str| cairn_models::from_spec(spec))).await
    }

    pub async fn with_factory(config: Config, factory: ModelFactory) -> std::io::Result<Arc<Self>> {
        let mut spend = HashMap::new();
        let audit = match &config.audit_log {
            Some(path) => {
                if let Some(dir) = path.parent().filter(|d| !d.as_os_str().is_empty()) {
                    tokio::fs::create_dir_all(dir).await?;
                }
                if let Ok(text) = tokio::fs::read_to_string(path).await {
                    for rec in text.lines().filter_map(|l| serde_json::from_str::<AuditRecord>(l).ok()) {
                        let s: &mut Spend = spend.entry(rec.key).or_default();
                        s.usd += rec.cost_usd;
                        s.tokens += rec.tokens;
                    }
                }
                let file = tokio::fs::OpenOptions::new().create(true).append(true).open(path).await?;
                Some(tokio::sync::Mutex::new(file))
            }
            None => None,
        };
        Ok(Arc::new(Self { config, factory, models: Mutex::new(HashMap::new()), spend: Mutex::new(spend), audit }))
    }

    pub fn router(self: &Arc<Self>) -> Router {
        Router::new()
            .route("/healthz", get(|| async { "ok" }))
            .route("/v1/models", get(list_models))
            .route("/v1/chat/completions", post(openai_chat))
            .route("/v1/messages", post(anthropic_messages))
            .route("/v1/cairn/usage", get(usage))
            .with_state(self.clone())
    }

    pub fn spend(&self, key: &str) -> Spend {
        self.spend.lock().unwrap().get(key).copied().unwrap_or_default()
    }

    fn authenticate(&self, headers: &HeaderMap) -> Result<&KeyConfig, GatewayError> {
        let presented = headers
            .get(header::AUTHORIZATION)
            .and_then(|v| v.to_str().ok())
            .and_then(|v| v.strip_prefix("Bearer "))
            .or_else(|| headers.get("x-api-key").and_then(|v| v.to_str().ok()))
            .ok_or_else(GatewayError::unauthorized)?;
        self.config.keys.iter().find(|k| k.matches(presented)).ok_or_else(GatewayError::unauthorized)
    }

    fn resolve(&self, key: &KeyConfig, model: &str) -> Result<Vec<String>, GatewayError> {
        if !key.may_use(model) {
            return Err(GatewayError::forbidden(format!("key `{}` may not use `{model}`", key.name)));
        }
        let specs = if let Some(specs) = self.config.routes.get(model) {
            specs.clone()
        } else if self.config.allow_direct_specs && model.contains(':') {
            vec![model.to_owned()]
        } else {
            return Err(GatewayError::not_found(format!("unknown model `{model}`")));
        };
        if key.budget_usd.is_some() {
            if let Some(spec) = specs.iter().find(|s| !self.config.pricing.contains_key(*s)) {
                return Err(GatewayError::forbidden(format!(
                    "`{spec}` has no pricing, so key `{}`'s USD budget could not be enforced",
                    key.name
                )));
            }
        }
        Ok(specs)
    }

    fn check_budget(&self, key: &KeyConfig) -> Result<(), GatewayError> {
        let s = self.spend(&key.name);
        if let Some(max) = key.budget_usd.filter(|max| s.usd >= *max) {
            return Err(GatewayError::budget(format!("key `{}` spent ${:.4} of ${max:.4}", key.name, s.usd)));
        }
        if let Some(max) = key.budget_tokens.filter(|max| s.tokens >= *max) {
            return Err(GatewayError::budget(format!("key `{}` used {} of {max} tokens", key.name, s.tokens)));
        }
        Ok(())
    }

    fn model(&self, spec: &str) -> Result<Arc<dyn Model>, ModelError> {
        let mut cache = self.models.lock().unwrap();
        if let Some(m) = cache.get(spec) {
            return Ok(m.clone());
        }
        let m = (self.factory)(spec)?;
        cache.insert(spec.to_owned(), m.clone());
        Ok(m)
    }

    fn cost(&self, spec: &str, resp: &ModelResponse) -> f64 {
        self.config.pricing.get(spec).map(|p| Pricing::from(*p).cost(&resp.usage)).unwrap_or(0.0)
    }

    /// Authenticate, authorize and budget-check; returns the key name and
    /// the failover chain for `model`.
    fn authorize(
        &self,
        headers: &HeaderMap,
        model: &str,
        rec: &mut AuditRecord,
    ) -> Result<(String, Vec<String>), GatewayError> {
        let key = self.authenticate(headers)?;
        rec.key = key.name.clone();
        let specs = self.resolve(key, model)?;
        self.check_budget(key)?;
        Ok((key.name.clone(), specs))
    }

    /// Walk the failover chain, calling `call` on each model until one
    /// succeeds or a non-retryable error occurs.
    async fn failover<T, F, Fut>(
        &self,
        specs: &[String],
        rec: &mut AuditRecord,
        call: F,
    ) -> Result<(String, T), GatewayError>
    where
        F: Fn(Arc<dyn Model>) -> Fut,
        Fut: std::future::Future<Output = Result<T, ModelError>>,
    {
        let mut last = None;
        for (i, spec) in specs.iter().enumerate() {
            // A spec that cannot even be constructed (missing credentials, unknown
            // provider) is skipped like an outage so the rest of the chain still serves.
            let (result, unavailable) = match self.model(spec) {
                Ok(m) => (call(m).await, false),
                Err(e) => (Err(e), true),
            };
            match result {
                Ok(v) => return Ok((spec.clone(), v)),
                Err(e) if (e.retryable() || unavailable) && i + 1 < specs.len() => {
                    tracing::warn!(spec, error = %e, "upstream failed, failing over");
                    rec.failed_over.push(format!("{spec}: {e}"));
                    last = Some(e);
                }
                Err(e) => return Err(GatewayError::upstream(&e)),
            }
        }
        Err(GatewayError::upstream(&last.unwrap_or_else(|| ModelError::Other("route has no models".into()))))
    }

    /// Charge the key and fill the audit record for a successful response.
    fn record_success(&self, key: &str, spec: &str, resp: &ModelResponse, rec: &mut AuditRecord) {
        let cost = self.cost(spec, resp);
        let tokens = resp.usage.total();
        {
            let mut spend = self.spend.lock().unwrap();
            let s = spend.entry(key.to_owned()).or_default();
            s.usd += cost;
            s.tokens += tokens;
        }
        rec.served_by = Some(spec.to_owned());
        rec.cost_usd = cost;
        rec.tokens = tokens;
        rec.tool_calls = resp
            .content
            .iter()
            .filter_map(|c| match c {
                Content::ToolUse { name, .. } => Some(name.clone()),
                _ => None,
            })
            .collect();
    }

    async fn complete(
        &self,
        headers: &HeaderMap,
        model: &str,
        req: &ModelRequest,
        rec: &mut AuditRecord,
    ) -> Result<ModelResponse, GatewayError> {
        let (key, specs) = self.authorize(headers, model, rec)?;
        let (spec, resp) = self.failover(&specs, rec, |m| async move { m.complete(req).await }).await?;
        self.record_success(&key, &spec, &resp, rec);
        Ok(resp)
    }

    /// Opens a stream, failing over until the first model accepts. Once
    /// tokens flow to the client the response is committed to that model.
    async fn open_stream(
        &self,
        headers: &HeaderMap,
        model: &str,
        req: &ModelRequest,
        rec: &mut AuditRecord,
    ) -> Result<(String, String, ModelStream), GatewayError> {
        let (key, specs) = self.authorize(headers, model, rec)?;
        let (spec, stream) = self.failover(&specs, rec, |m| async move { m.stream(req).await }).await?;
        Ok((key, spec, stream))
    }

    async fn write_audit(&self, rec: &AuditRecord) {
        if let Some(file) = &self.audit {
            let mut line = serde_json::to_vec(rec).unwrap_or_default();
            line.push(b'\n');
            let mut f = file.lock().await;
            if let Err(e) = f.write_all(&line).await {
                tracing::error!(error = %e, "failed to write audit record");
            }
        }
    }
}

type Parse = fn(&Value) -> Result<(String, ModelRequest, bool), GatewayError>;
type Render = fn(&str, &str, &ModelResponse) -> Value;
type Renderer = fn(&str, &str, bool) -> Box<dyn SseRenderer>;

struct Api {
    name: &'static str,
    id_prefix: &'static str,
    parse: Parse,
    render: Render,
    renderer: Renderer,
    render_error: fn(&GatewayError) -> Value,
}

const OPENAI: Api = Api {
    name: "openai",
    id_prefix: "chatcmpl-",
    parse: openai_api::parse_request,
    render: openai_api::render_response,
    renderer: openai_api::SseOut::boxed,
    render_error: openai_api::render_error,
};

const ANTHROPIC: Api = Api {
    name: "anthropic",
    id_prefix: "msg_",
    parse: anthropic_api::parse_request,
    render: anthropic_api::render_response,
    renderer: anthropic_api::SseOut::boxed,
    render_error: anthropic_api::render_error,
};

async fn handle(gw: Arc<Gateway>, api: &'static Api, headers: HeaderMap, body: Value) -> Response {
    let started = Instant::now();
    let request_id = format!("{}{}", api.id_prefix, uuid::Uuid::now_v7().simple());
    let mut rec = AuditRecord {
        at: chrono::Utc::now(),
        request_id: request_id.clone(),
        key: String::new(),
        api: api.name.into(),
        model: body["model"].as_str().unwrap_or_default().to_owned(),
        served_by: None,
        failed_over: Vec::new(),
        status: 200,
        error: None,
        tokens: 0,
        cost_usd: 0.0,
        duration_ms: 0,
        tool_calls: Vec::new(),
    };

    let (model, req, stream) = match (api.parse)(&body) {
        Ok(parsed) => parsed,
        Err(e) => return error_response(api, &e),
    };
    if stream {
        return stream_response(gw, api, headers, request_id, model, req, rec, started).await;
    }

    let result = gw.complete(&headers, &model, &req, &mut rec).await;
    rec.duration_ms = started.elapsed().as_millis() as u64;
    let response = match result {
        Ok(resp) => axum::Json((api.render)(&request_id, &model, &resp)).into_response(),
        Err(e) => {
            rec.status = e.status;
            rec.error = Some(e.message.clone());
            error_response(api, &e)
        }
    };
    // Unauthenticated noise is not audited against any key.
    if !rec.key.is_empty() {
        gw.write_audit(&rec).await;
    }
    response
}

fn error_response(api: &Api, e: &GatewayError) -> Response {
    let status = StatusCode::from_u16(e.status).unwrap_or(StatusCode::BAD_GATEWAY);
    (status, axum::Json((api.render_error)(e))).into_response()
}

#[allow(clippy::too_many_arguments)]
async fn stream_response(
    gw: Arc<Gateway>,
    api: &'static Api,
    headers: HeaderMap,
    request_id: String,
    model: String,
    req: ModelRequest,
    mut rec: AuditRecord,
    started: Instant,
) -> Response {
    let (key, spec, mut upstream) = match gw.open_stream(&headers, &model, &req, &mut rec).await {
        Ok(opened) => opened,
        Err(e) => {
            rec.status = e.status;
            rec.error = Some(e.message.clone());
            rec.duration_ms = started.elapsed().as_millis() as u64;
            if !rec.key.is_empty() {
                gw.write_audit(&rec).await;
            }
            return error_response(api, &e);
        }
    };

    let mut out = (api.renderer)(&request_id, &model, spec.starts_with("anthropic:"));
    let (tx, rx) = tokio::sync::mpsc::channel::<String>(64);
    tokio::spawn(async move {
        let mut client_alive = tx.send(out.start()).await.is_ok();
        let mut completed = false;
        while client_alive {
            let Some(event) = upstream.next().await else { break };
            let chunk = match event {
                Ok(StreamEvent::Completed(resp)) => {
                    gw.record_success(&key, &spec, &resp, &mut rec);
                    completed = true;
                    out.finish(&resp)
                }
                Ok(delta) => out.delta(&delta),
                Err(e) => {
                    let err = GatewayError::upstream(&e);
                    rec.status = err.status;
                    rec.error = Some(err.message.clone());
                    let _ = tx.send(out.error(&err)).await;
                    break;
                }
            };
            if !chunk.is_empty() {
                client_alive = tx.send(chunk).await.is_ok();
            }
            if completed {
                break;
            }
        }
        if !completed && rec.error.is_none() {
            rec.status = 499;
            rec.error = Some(if client_alive { "upstream stream ended early" } else { "client disconnected" }.into());
            rec.served_by = Some(spec.clone());
        }
        rec.duration_ms = started.elapsed().as_millis() as u64;
        gw.write_audit(&rec).await;
    });

    let body = futures::stream::unfold(rx, |mut rx| async move {
        rx.recv().await.map(|chunk| (Ok::<_, std::convert::Infallible>(chunk), rx))
    });
    (
        [(header::CONTENT_TYPE, "text/event-stream"), (header::CACHE_CONTROL, "no-cache")],
        axum::body::Body::from_stream(body),
    )
        .into_response()
}

async fn openai_chat(
    State(gw): State<Arc<Gateway>>,
    headers: HeaderMap,
    axum::Json(body): axum::Json<Value>,
) -> Response {
    handle(gw, &OPENAI, headers, body).await
}

async fn anthropic_messages(
    State(gw): State<Arc<Gateway>>,
    headers: HeaderMap,
    axum::Json(body): axum::Json<Value>,
) -> Response {
    handle(gw, &ANTHROPIC, headers, body).await
}

async fn list_models(State(gw): State<Arc<Gateway>>, headers: HeaderMap) -> Response {
    match gw.authenticate(&headers) {
        Ok(key) => {
            let data: Vec<Value> = gw
                .config
                .routes
                .keys()
                .filter(|r| key.may_use(r))
                .map(|r| json!({"id": r, "object": "model", "owned_by": "cairn"}))
                .collect();
            axum::Json(json!({"object": "list", "data": data})).into_response()
        }
        Err(e) => (StatusCode::UNAUTHORIZED, axum::Json(openai_api::render_error(&e))).into_response(),
    }
}

async fn usage(State(gw): State<Arc<Gateway>>, headers: HeaderMap) -> Response {
    match gw.authenticate(&headers) {
        Ok(key) => {
            let s = gw.spend(&key.name);
            axum::Json(json!({
                "key": key.name, "spent_usd": s.usd, "budget_usd": key.budget_usd,
                "tokens": s.tokens, "budget_tokens": key.budget_tokens,
            }))
            .into_response()
        }
        Err(e) => (StatusCode::UNAUTHORIZED, axum::Json(openai_api::render_error(&e))).into_response(),
    }
}
