use std::future::Future;
use std::marker::PhantomData;
use std::time::Duration;

use async_trait::async_trait;
use schemars::JsonSchema;
use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use sha2::{Digest, Sha256};

/// What a tool does to the world. Policies and retry semantics key off this.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Effect {
    /// No side effects; safe to retry freely.
    Read,
    /// Changes state but can be undone.
    Write,
    /// Cannot be undone (payments, emails, deletes).
    Irreversible,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ToolSpec {
    pub name: String,
    pub description: String,
    pub input_schema: Value,
    pub effect: Effect,
}

/// Passed to every tool invocation.
#[derive(Debug, Clone)]
pub struct ToolCtx {
    pub run_id: String,
    pub call_id: String,
    /// Stable across retries and crash-recovery re-executions of the same
    /// call. Forward it to downstream APIs to get exactly-once effects.
    pub idempotency_key: String,
}

impl ToolCtx {
    pub fn new(run_id: &str, call_id: &str) -> Self {
        let digest = Sha256::digest(format!("{run_id}:{call_id}").as_bytes());
        Self { run_id: run_id.to_owned(), call_id: call_id.to_owned(), idempotency_key: hex::encode(&digest[..16]) }
    }
}

#[derive(Debug, Clone, thiserror::Error)]
#[error("{0}")]
pub struct ToolError(pub String);

impl ToolError {
    pub fn new(msg: impl Into<String>) -> Self {
        Self(msg.into())
    }
}

#[async_trait]
pub trait Tool: Send + Sync {
    fn spec(&self) -> &ToolSpec;
    async fn call(&self, input: Value, ctx: &ToolCtx) -> Result<Value, ToolError>;

    /// Overrides the agent's default tool timeout for this tool.
    fn timeout(&self) -> Option<Duration> {
        None
    }
}

/// Builds a typed tool from an async function. The JSON schema shown to the
/// model is derived from `I`, and model input is validated against it.
///
/// ```ignore
/// #[derive(Deserialize, JsonSchema)]
/// struct GetOrder { order_id: String }
///
/// let get_order = tool("get_order", "Look up an order", Effect::Read,
///     |i: GetOrder, _ctx| async move { Ok(json!({"id": i.order_id})) });
/// ```
pub fn tool<I, O, F, Fut>(name: &str, description: &str, effect: Effect, f: F) -> FnTool<I, O, F>
where
    I: DeserializeOwned + JsonSchema + Send + 'static,
    O: Serialize + Send + 'static,
    F: Fn(I, ToolCtx) -> Fut + Send + Sync + 'static,
    Fut: Future<Output = Result<O, ToolError>> + Send + 'static,
{
    let mut schema = serde_json::to_value(schemars::schema_for!(I)).unwrap_or(Value::Null);
    if let Value::Object(map) = &mut schema {
        map.remove("$schema");
        map.remove("title");
    }
    FnTool {
        spec: ToolSpec { name: name.to_owned(), description: description.to_owned(), input_schema: schema, effect },
        f,
        timeout: None,
        _marker: PhantomData,
    }
}

pub struct FnTool<I, O, F> {
    spec: ToolSpec,
    f: F,
    timeout: Option<Duration>,
    _marker: PhantomData<fn(I) -> O>,
}

impl<I, O, F> FnTool<I, O, F> {
    /// Cancel this tool if it runs longer than `limit`; the model receives
    /// an error result and the run continues.
    pub fn with_timeout(mut self, limit: Duration) -> Self {
        self.timeout = Some(limit);
        self
    }
}

#[async_trait]
impl<I, O, F, Fut> Tool for FnTool<I, O, F>
where
    I: DeserializeOwned + JsonSchema + Send + 'static,
    O: Serialize + Send + 'static,
    F: Fn(I, ToolCtx) -> Fut + Send + Sync + 'static,
    Fut: Future<Output = Result<O, ToolError>> + Send + 'static,
{
    fn spec(&self) -> &ToolSpec {
        &self.spec
    }

    fn timeout(&self) -> Option<Duration> {
        self.timeout
    }

    async fn call(&self, input: Value, ctx: &ToolCtx) -> Result<Value, ToolError> {
        let typed: I = serde_json::from_value(input)
            .map_err(|e| ToolError(format!("invalid input for `{}`: {e}", self.spec.name)))?;
        let out = (self.f)(typed, ctx.clone()).await?;
        serde_json::to_value(out).map_err(|e| ToolError(format!("could not serialize output: {e}")))
    }
}
