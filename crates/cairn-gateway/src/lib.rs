//! # cairn-gateway
//!
//! A drop-in, OpenAI- and Anthropic-compatible endpoint. Point any existing
//! framework at it (LangGraph, CrewAI, OpenAI Agents SDK, Vercel AI SDK,
//! Microsoft Agent Framework, raw SDKs) by changing its base URL, and every
//! model call gains per-key budgets, model allowlists, cross-provider
//! failover and an audit trail. Either wire format can be served by any
//! provider: an OpenAI-SDK client can run on Claude and vice versa.

pub mod config;
mod anthropic_api;
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
use cairn_runtime::{Content, Model, ModelError, ModelRequest, ModelResponse};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use tokio::io::AsyncWriteExt;

pub use config::{Config, KeyConfig};

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
        if let Some(specs) = self.config.routes.get(model) {
            return Ok(specs.clone());
        }
        if self.config.allow_direct_specs && model.contains(':') {
            return Ok(vec![model.to_owned()]);
        }
        Err(GatewayError::not_found(format!("unknown model `{model}`")))
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

    /// Authenticate, authorize, budget-check, then try each spec in order.
    async fn complete(&self, api: &str, headers: &HeaderMap, model: &str, req: &ModelRequest, rec: &mut AuditRecord) -> Result<ModelResponse, GatewayError> {
        let key = self.authenticate(headers)?;
        rec.key = key.name.clone();
        rec.api = api.to_owned();
        let specs = self.resolve(key, model)?;
        self.check_budget(key)?;

        let mut last = None;
        for (i, spec) in specs.iter().enumerate() {
            // A spec that cannot even be constructed (missing credentials, unknown
            // provider) is skipped like an outage so the rest of the chain still serves.
            let (result, unavailable) = match self.model(spec) {
                Ok(m) => (m.complete(req).await, false),
                Err(e) => (Err(e), true),
            };
            match result {
                Ok(resp) => {
                    let cost = self.cost(spec, &resp);
                    let tokens = resp.usage.total();
                    {
                        let mut spend = self.spend.lock().unwrap();
                        let s = spend.entry(key.name.clone()).or_default();
                        s.usd += cost;
                        s.tokens += tokens;
                    }
                    rec.served_by = Some(spec.clone());
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
                    return Ok(resp);
                }
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
type RenderStream = fn(&str, &str, &ModelResponse) -> String;

struct Api {
    name: &'static str,
    id_prefix: &'static str,
    parse: Parse,
    render: Render,
    render_stream: RenderStream,
    render_error: fn(&GatewayError) -> Value,
}

const OPENAI: Api = Api {
    name: "openai",
    id_prefix: "chatcmpl-",
    parse: openai_api::parse_request,
    render: openai_api::render_response,
    render_stream: openai_api::render_stream,
    render_error: openai_api::render_error,
};

const ANTHROPIC: Api = Api {
    name: "anthropic",
    id_prefix: "msg_",
    parse: anthropic_api::parse_request,
    render: anthropic_api::render_response,
    render_stream: anthropic_api::render_stream,
    render_error: anthropic_api::render_error,
};

async fn handle(gw: Arc<Gateway>, api: &Api, headers: HeaderMap, body: Value) -> Response {
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

    let result = match (api.parse)(&body) {
        Ok((model, req, stream)) => gw.complete(api.name, &headers, &model, &req, &mut rec).await.map(|r| (model, r, stream)),
        Err(e) => Err(e),
    };
    rec.duration_ms = started.elapsed().as_millis() as u64;

    let response = match result {
        Ok((model, resp, false)) => axum::Json((api.render)(&request_id, &model, &resp)).into_response(),
        Ok((model, resp, true)) => (
            [(header::CONTENT_TYPE, "text/event-stream"), (header::CACHE_CONTROL, "no-cache")],
            (api.render_stream)(&request_id, &model, &resp),
        )
            .into_response(),
        Err(e) => {
            rec.status = e.status;
            rec.error = Some(e.message.clone());
            let status = StatusCode::from_u16(e.status).unwrap_or(StatusCode::BAD_GATEWAY);
            (status, axum::Json((api.render_error)(&e))).into_response()
        }
    };
    // Unauthenticated noise is not audited against any key.
    if !rec.key.is_empty() {
        gw.write_audit(&rec).await;
    }
    response
}

async fn openai_chat(State(gw): State<Arc<Gateway>>, headers: HeaderMap, axum::Json(body): axum::Json<Value>) -> Response {
    handle(gw, &OPENAI, headers, body).await
}

async fn anthropic_messages(State(gw): State<Arc<Gateway>>, headers: HeaderMap, axum::Json(body): axum::Json<Value>) -> Response {
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
