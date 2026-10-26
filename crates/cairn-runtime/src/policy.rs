use serde_json::Value;

use crate::tool::{Effect, ToolSpec};

#[derive(Debug, Clone, PartialEq)]
pub enum Decision {
    Allow,
    Deny { reason: String },
    RequireApproval { reason: String },
}

/// Decides, before execution, whether a tool call may run.
pub trait Policy: Send + Sync {
    fn decide(&self, spec: &ToolSpec, input: &Value) -> Decision;
}

impl<F> Policy for F
where
    F: Fn(&ToolSpec, &Value) -> Decision + Send + Sync,
{
    fn decide(&self, spec: &ToolSpec, input: &Value) -> Decision {
        self(spec, input)
    }
}

/// Requires human approval for every call at or above `approve_at`.
/// The default gates irreversible tools.
#[derive(Debug, Clone)]
pub struct EffectPolicy {
    pub approve_at: Option<Effect>,
}

impl Default for EffectPolicy {
    fn default() -> Self {
        Self { approve_at: Some(Effect::Irreversible) }
    }
}

impl Policy for EffectPolicy {
    fn decide(&self, spec: &ToolSpec, _input: &Value) -> Decision {
        match self.approve_at {
            Some(level) if spec.effect >= level => {
                Decision::RequireApproval { reason: format!("`{}` has {:?} effect", spec.name, spec.effect) }
            }
            _ => Decision::Allow,
        }
    }
}

pub struct AllowAll;

impl Policy for AllowAll {
    fn decide(&self, _spec: &ToolSpec, _input: &Value) -> Decision {
        Decision::Allow
    }
}
