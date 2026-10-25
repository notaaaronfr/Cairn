use std::collections::HashMap;

use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

use crate::event::{Event, EventKind};
use crate::message::{Content, Message, Role};
use crate::model::Usage;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "status", rename_all = "snake_case")]
pub enum Status {
    Running,
    AwaitingApproval,
    Completed { output: String },
    Failed { error: String },
    BudgetExceeded { reason: String },
}

impl Status {
    pub fn is_terminal(&self) -> bool {
        matches!(self, Status::Completed { .. } | Status::Failed { .. } | Status::BudgetExceeded { .. })
    }
}

/// A tool call requested by the model that has not produced a result yet.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PendingCall {
    pub id: String,
    pub name: String,
    pub input: Value,
    pub awaiting_approval: bool,
}

/// The full state of a run, derived purely from its events.
///
/// There is no other source of truth: live execution, resume after a crash
/// and time-travel debugging all go through [`RunState::apply`].
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RunState {
    pub run_id: String,
    pub agent: String,
    pub status: Status,
    pub messages: Vec<Message>,
    pub steps: u32,
    pub usage: Usage,
    pub cost_usd: f64,
    pub pending: Vec<PendingCall>,
    pub approvals: HashMap<String, bool>,
    /// Results gathered for the current batch of tool calls.
    pub tool_results: Vec<Content>,
    pub next_seq: u64,
}

impl RunState {
    /// Rebuild state from a run's events. Returns `None` for an empty slice.
    /// Pass a prefix (`&events[..=n]`) to see the state at any point in time.
    pub fn from_events(events: &[Event]) -> Option<Self> {
        let first = events.first()?;
        let mut state = RunState {
            run_id: first.run_id.clone(),
            agent: String::new(),
            status: Status::Running,
            messages: Vec::new(),
            steps: 0,
            usage: Usage::default(),
            cost_usd: 0.0,
            pending: Vec::new(),
            approvals: HashMap::new(),
            tool_results: Vec::new(),
            next_seq: 0,
        };
        for event in events {
            state.apply(event);
        }
        Some(state)
    }

    pub fn apply(&mut self, event: &Event) {
        debug_assert_eq!(event.seq, self.next_seq, "events must be applied in order");
        self.next_seq = event.seq + 1;

        match &event.kind {
            EventKind::RunStarted { agent, input } => {
                self.agent = agent.clone();
                self.messages = vec![input.clone()];
                self.status = Status::Running;
            }
            EventKind::ModelCalled { response, cost_usd, .. } => {
                self.steps += 1;
                self.usage.add(&response.total_usage());
                self.cost_usd += cost_usd.unwrap_or(0.0);
                self.messages.push(Message::assistant(response.content.clone()));
                self.pending = response
                    .content
                    .iter()
                    .filter_map(|c| match c {
                        Content::ToolUse { id, name, input } => Some(PendingCall {
                            id: id.clone(),
                            name: name.clone(),
                            input: input.clone(),
                            awaiting_approval: false,
                        }),
                        _ => None,
                    })
                    .collect();
            }
            EventKind::ModelFailed { usage, cost_usd, .. } => {
                self.usage.add(usage);
                self.cost_usd += cost_usd.unwrap_or(0.0);
            }
            EventKind::ToolDenied { call_id, reason, .. } => {
                self.resolve(call_id, json!({ "error": format!("tool call denied: {reason}") }), true);
            }
            EventKind::ApprovalRequested { call_id, .. } => {
                if let Some(p) = self.pending.iter_mut().find(|p| &p.id == call_id) {
                    p.awaiting_approval = true;
                }
                self.status = Status::AwaitingApproval;
            }
            EventKind::ApprovalDecided { call_id, approved, .. } => {
                self.approvals.insert(call_id.clone(), *approved);
                if let Some(p) = self.pending.iter_mut().find(|p| &p.id == call_id) {
                    p.awaiting_approval = false;
                }
                if !self.pending.iter().any(|p| p.awaiting_approval) {
                    self.status = Status::Running;
                }
            }
            EventKind::ToolCompleted { call_id, output, is_error, .. } => {
                self.resolve(call_id, output.clone(), *is_error);
            }
            EventKind::BudgetExceeded { reason } => {
                self.status = Status::BudgetExceeded { reason: reason.clone() };
            }
            EventKind::RunCompleted { output } => {
                self.status = Status::Completed { output: output.clone() };
            }
            EventKind::RunFailed { error } => {
                self.status = Status::Failed { error: error.clone() };
            }
        }
    }

    fn resolve(&mut self, call_id: &str, content: Value, is_error: bool) {
        self.pending.retain(|p| p.id != call_id);
        self.tool_results.push(Content::ToolResult { tool_use_id: call_id.to_owned(), content, is_error });
        if self.pending.is_empty() {
            let results = std::mem::take(&mut self.tool_results);
            self.messages.push(Message { role: Role::User, content: results });
        }
    }

    pub fn awaiting_approval(&self) -> Vec<PendingCall> {
        self.pending.iter().filter(|p| p.awaiting_approval).cloned().collect()
    }
}
