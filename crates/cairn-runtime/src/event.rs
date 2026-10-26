use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::message::Message;
use crate::model::{ModelResponse, Usage};

/// One immutable fact in a run's history.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Event {
    pub run_id: String,
    /// Position in the run's journal, starting at 0. Gap-free.
    pub seq: u64,
    pub at: DateTime<Utc>,
    pub kind: EventKind,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum EventKind {
    RunStarted {
        agent: String,
        input: Message,
    },
    ModelCalled {
        step: u32,
        response: ModelResponse,
        cost_usd: Option<f64>,
        duration_ms: u64,
    },
    /// Recorded for audit; does not change state. A terminal failure is
    /// followed by `RunFailed`.
    ModelFailed {
        step: u32,
        error: String,
        /// Tokens already paid for before the failure (e.g. rejected answers).
        #[serde(default)]
        usage: Usage,
        #[serde(default)]
        cost_usd: Option<f64>,
    },
    ToolDenied {
        call_id: String,
        tool: String,
        reason: String,
    },
    ApprovalRequested {
        call_id: String,
        tool: String,
        input: Value,
        reason: String,
    },
    ApprovalDecided {
        call_id: String,
        approved: bool,
        by: Option<String>,
        note: Option<String>,
    },
    ToolCompleted {
        call_id: String,
        tool: String,
        output: Value,
        is_error: bool,
        duration_ms: u64,
        /// Model usage the tool reported (e.g. a sub-agent's).
        #[serde(default, skip_serializing_if = "is_zero_usage")]
        usage: Usage,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        cost_usd: Option<f64>,
    },
    BudgetExceeded {
        reason: String,
    },
    RunCompleted {
        output: String,
        /// The validated final answer, when the agent has a finish schema.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        structured: Option<Value>,
    },
    /// Older messages were replaced by a summary (context compaction).
    ContextCompacted {
        /// Messages after the task that the summary replaces.
        replaced: usize,
        summary: String,
        #[serde(default, skip_serializing_if = "is_zero_usage")]
        usage: Usage,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        cost_usd: Option<f64>,
    },
    /// A final answer failed the finish schema; the model is asked again.
    FinishRejected {
        step: u32,
        reason: String,
    },
    RunFailed {
        error: String,
    },
    /// First event of a workflow run.
    WorkflowStarted {
        workflow: String,
        input: Value,
    },
    /// A workflow step finished; replays return this instead of re-running it.
    StepCompleted {
        step: String,
        output: Value,
        is_error: bool,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        child_run: Option<String>,
        #[serde(default, skip_serializing_if = "is_zero_usage")]
        usage: Usage,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        cost_usd: Option<f64>,
    },
    WorkflowCompleted {
        output: Value,
    },
    WorkflowFailed {
        error: String,
    },
}

impl EventKind {
    pub fn name(&self) -> &'static str {
        match self {
            EventKind::RunStarted { .. } => "run_started",
            EventKind::ModelCalled { .. } => "model_called",
            EventKind::ModelFailed { .. } => "model_failed",
            EventKind::ToolDenied { .. } => "tool_denied",
            EventKind::ApprovalRequested { .. } => "approval_requested",
            EventKind::ApprovalDecided { .. } => "approval_decided",
            EventKind::ToolCompleted { .. } => "tool_completed",
            EventKind::BudgetExceeded { .. } => "budget_exceeded",
            EventKind::RunCompleted { .. } => "run_completed",
            EventKind::FinishRejected { .. } => "finish_rejected",
            EventKind::ContextCompacted { .. } => "context_compacted",
            EventKind::RunFailed { .. } => "run_failed",
            EventKind::WorkflowStarted { .. } => "workflow_started",
            EventKind::StepCompleted { .. } => "step_completed",
            EventKind::WorkflowCompleted { .. } => "workflow_completed",
            EventKind::WorkflowFailed { .. } => "workflow_failed",
        }
    }
}

fn is_zero_usage(u: &Usage) -> bool {
    *u == Usage::default()
}
