use std::sync::Arc;
use std::time::Duration;

use async_trait::async_trait;
use cairn_runtime::{Effect, Tool, ToolCtx, ToolError, ToolSpec};
use serde_json::{Value, json};

use crate::client::{A2aClient, A2aError};
use crate::types::*;

/// A remote A2A agent usable as a Cairn [`Tool`].
///
/// The model sends a message and gets back the remote agent's answer plus
/// the `context_id` to continue the conversation, or a `task_id` and
/// `input_required` when the remote agent needs more input.
///
/// Each call's message id is the call's idempotency key, so a call retried
/// after a crash reaches the remote agent as the same message (Cairn agents
/// served with [`A2aServer`](crate::A2aServer) deduplicate on it).
///
/// The default effect is [`Effect::Irreversible`]: Cairn cannot see what the
/// remote agent does, so calls need approval under the default policy until
/// you say otherwise with [`effect`](Self::effect).
pub struct A2aTool {
    client: Arc<A2aClient>,
    spec: ToolSpec,
    timeout: Option<Duration>,
    poll_every: Duration,
}

/// Model-facing tool names must match `^[a-zA-Z0-9_-]{1,64}$`.
fn sanitize(name: &str) -> String {
    let s: String = name
        .chars()
        .map(|c| if c.is_ascii_alphanumeric() || c == '_' || c == '-' { c.to_ascii_lowercase() } else { '_' })
        .take(64)
        .collect();
    if s.is_empty() { "remote_agent".into() } else { s }
}

fn describe(card: &AgentCard) -> String {
    let mut text = format!("Ask the remote agent `{}`: {}", card.name, card.description);
    for skill in &card.skills {
        text.push_str(&format!("\n- {}: {}", skill.name, skill.description));
    }
    text
}

impl A2aTool {
    /// Named and described from the agent's card when the client has one.
    pub fn new(client: Arc<A2aClient>) -> Self {
        let (name, description) = match client.card() {
            Some(card) => (sanitize(&card.name), describe(card)),
            None => ("remote_agent".into(), "Ask a remote agent.".into()),
        };
        let input_schema = json!({
            "type": "object",
            "properties": {
                "message": {"type": "string", "description": "What to ask or tell the agent."},
                "context_id": {"type": "string", "description": "Continue an earlier conversation with this agent."},
                "task_id": {"type": "string", "description": "Answer a task that returned `input_required`."}
            },
            "required": ["message"]
        });
        Self {
            client,
            spec: ToolSpec { name, description, input_schema, effect: Effect::Irreversible },
            timeout: None,
            poll_every: Duration::from_millis(500),
        }
    }

    pub fn name(mut self, name: &str) -> Self {
        self.spec.name = sanitize(name);
        self
    }

    pub fn description(mut self, description: &str) -> Self {
        self.spec.description = description.to_owned();
        self
    }

    pub fn effect(mut self, effect: Effect) -> Self {
        self.spec.effect = effect;
        self
    }

    /// Give up on the remote task after `limit` (overrides the agent's
    /// default tool timeout). Remote agents are often slower than local tools.
    pub fn timeout(mut self, limit: Duration) -> Self {
        self.timeout = Some(limit);
        self
    }

    /// How often to check on a task the remote agent is still working on.
    pub fn poll_every(mut self, interval: Duration) -> Self {
        self.poll_every = interval;
        self
    }
}

fn state_name(state: TaskState) -> &'static str {
    match state {
        TaskState::Unspecified => "unspecified",
        TaskState::Submitted => "submitted",
        TaskState::Working => "working",
        TaskState::Completed => "completed",
        TaskState::Failed => "failed",
        TaskState::Canceled => "canceled",
        TaskState::InputRequired => "input_required",
        TaskState::Rejected => "rejected",
        TaskState::AuthRequired => "auth_required",
    }
}

fn tool_error(e: A2aError) -> ToolError {
    ToolError(format!("remote agent error: {e}"))
}

#[async_trait]
impl Tool for A2aTool {
    fn spec(&self) -> &ToolSpec {
        &self.spec
    }

    fn timeout(&self) -> Option<Duration> {
        self.timeout
    }

    async fn call(&self, input: Value, ctx: &ToolCtx) -> Result<Value, ToolError> {
        let text = input["message"].as_str().ok_or_else(|| ToolError::new("`message` is required"))?;
        let mut message = Message::new(Role::User, ctx.idempotency_key.clone(), vec![Part::text(text)]);
        message.context_id = input["context_id"].as_str().map(str::to_owned);
        message.task_id = input["task_id"].as_str().map(str::to_owned);

        let mut task = match self.client.send_message(&SendMessageRequest::new(message)).await.map_err(tool_error)? {
            SendMessageResponse::Message(reply) => {
                return Ok(json!({"state": "completed", "context_id": reply.context_id, "answer": reply.text()}));
            }
            SendMessageResponse::Task(task) => task,
        };
        while !task.status.state.is_terminal() && !task.status.state.is_interrupted() {
            tokio::time::sleep(self.poll_every).await;
            task = self.client.get_task(&task.id, Some(0)).await.map_err(tool_error)?;
        }

        let state = task.status.state;
        let answer = task.text();
        if matches!(state, TaskState::Failed | TaskState::Rejected | TaskState::Canceled) {
            return Err(ToolError(format!("remote task {}: {answer}", state_name(state))));
        }
        let data: Vec<&Value> = task.artifacts.iter().flat_map(|a| &a.parts).filter_map(|p| p.data.as_ref()).collect();
        let mut out = json!({
            "state": state_name(state),
            "task_id": task.id,
            "context_id": task.context_id,
            "answer": answer,
        });
        if !data.is_empty() {
            out["data"] = json!(data);
        }
        Ok(out)
    }
}
