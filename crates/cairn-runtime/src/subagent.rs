//! Agents as tools: a parent agent delegates a task to a child agent.

use async_trait::async_trait;
use serde_json::{Value, json};

use crate::agent::{Agent, AgentError};
use crate::lease::child_run_id;
use crate::state::Status;
use crate::tool::{Effect, Tool, ToolCtx, ToolError, ToolSpec};

/// Exposes an [`Agent`] as a tool taking `{"task": "..."}`.
///
/// Each call is a durable child run with id `<parent run>.<call id>`: if the
/// parent crashes mid-call, the retried call resumes the child instead of
/// starting over. The child's spend is charged to the parent's budget, and
/// its own USD budget is capped to what the parent has left.
///
/// A child that needs human approval cannot suspend its parent from inside
/// a tool call; the call returns an error naming the child run. Use
/// [`crate::workflow::WorkflowCtx::agent`] when children need approvals.
pub struct AgentTool {
    agent: Agent,
    spec: ToolSpec,
}

impl AgentTool {
    pub fn new(agent: Agent, name: &str, description: &str) -> Self {
        let spec = ToolSpec {
            name: name.to_owned(),
            description: description.to_owned(),
            input_schema: json!({
                "type": "object",
                "properties": {"task": {"type": "string", "description": "What the agent should do, with all needed context."}},
                "required": ["task"]
            }),
            effect: Effect::Write,
        };
        Self { agent, spec }
    }

    /// Override the effect the parent's policy sees (default `Write`).
    pub fn effect(mut self, effect: Effect) -> Self {
        self.spec.effect = effect;
        self
    }
}

#[async_trait]
impl Tool for AgentTool {
    fn spec(&self) -> &ToolSpec {
        &self.spec
    }

    async fn call(&self, input: Value, ctx: &ToolCtx) -> Result<Value, ToolError> {
        let task = input["task"].as_str().ok_or_else(|| ToolError::new("`task` must be a string"))?.to_owned();
        let mut builder = self.agent.to_builder();
        if let Some(remaining) = ctx.remaining_usd {
            if remaining <= 0.0 {
                return Err(ToolError::new("no budget left for a sub-agent"));
            }
            builder = builder.cap_usd(remaining);
        }
        let agent = builder.build().map_err(|e| ToolError::new(e.to_string()))?;
        let child = child_run_id(&ctx.run_id, &format!("agent-{}", ctx.call_id));
        let outcome = match agent.run_with_id(&child, task).await {
            Err(AgentError::RunExists(_)) => agent.resume(&child).await,
            other => other,
        }
        .map_err(|e| ToolError::new(e.to_string()))?;
        ctx.charge(outcome.usage, outcome.cost_usd);
        match &outcome.status {
            Status::Completed { output } => Ok(outcome.structured.clone().unwrap_or_else(|| json!(output))),
            Status::AwaitingApproval => {
                Err(ToolError::new(format!("sub-agent `{}` is waiting for approval in run `{child}`", agent.name())))
            }
            other => Err(ToolError::new(format!("sub-agent `{}` did not finish: {other:?}", agent.name()))),
        }
    }
}
