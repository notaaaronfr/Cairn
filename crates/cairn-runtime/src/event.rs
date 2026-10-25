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
    },
    BudgetExceeded {
        reason: String,
    },
    RunCompleted {
        output: String,
    },
    RunFailed {
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
            EventKind::RunFailed { .. } => "run_failed",
        }
    }
}
