//! Hooks around every model and tool call.
//!
//! Middleware sees each step of a run and can inspect, rewrite or refuse
//! it: redact PII before it reaches a provider, block a dangerous tool
//! input, normalise an answer, enforce an organisation-wide rule. Hooks run
//! in the order they were added.
//!
//! What a hook changes is what gets journaled: a rewritten model response
//! is recorded as rewritten, a rewritten tool output as rewritten, so
//! resume and replay see exactly what the agent saw. (Requests are not
//! journaled, so `before_model` rewrites affect only the provider call.)

use async_trait::async_trait;
use serde_json::Value;

use crate::model::{ModelRequest, ModelResponse};
use crate::tool::ToolSpec;

/// Where in a run a hook is called.
#[derive(Debug, Clone)]
pub struct HookCtx {
    pub run_id: String,
    pub agent: String,
    /// The model step (1-based) the hook belongs to.
    pub step: u32,
}

#[async_trait]
pub trait Middleware: Send + Sync {
    /// Before a model call. `Err(reason)` fails the run.
    async fn before_model(&self, _ctx: &HookCtx, _request: &mut ModelRequest) -> Result<(), String> {
        Ok(())
    }

    /// After a model call, before the response is journaled. `Err(reason)`
    /// fails the run.
    async fn after_model(&self, _ctx: &HookCtx, _response: &mut ModelResponse) -> Result<(), String> {
        Ok(())
    }

    /// Before a tool runs (after policy and approval). `Err(reason)` denies
    /// the call and tells the model why.
    async fn before_tool(&self, _ctx: &HookCtx, _tool: &ToolSpec, _input: &mut Value) -> Result<(), String> {
        Ok(())
    }

    /// After a tool runs, before its result is journaled. `Err(reason)`
    /// turns the result into an error result with that reason.
    async fn after_tool(
        &self,
        _ctx: &HookCtx,
        _tool: &ToolSpec,
        _output: &mut Value,
        _is_error: &mut bool,
    ) -> Result<(), String> {
        Ok(())
    }
}
