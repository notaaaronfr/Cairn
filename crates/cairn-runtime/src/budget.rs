use serde::{Deserialize, Serialize};

/// Hard limits enforced by the runtime before every model call.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct Budget {
    /// Maximum number of model calls.
    pub max_steps: Option<u32>,
    /// Maximum spend in USD (requires the model to report pricing).
    pub max_usd: Option<f64>,
    /// Maximum input + output tokens.
    pub max_tokens: Option<u64>,
}

impl Budget {
    pub fn max_steps(mut self, n: u32) -> Self {
        self.max_steps = Some(n);
        self
    }

    pub fn max_usd(mut self, usd: f64) -> Self {
        self.max_usd = Some(usd);
        self
    }

    pub fn max_tokens(mut self, n: u64) -> Self {
        self.max_tokens = Some(n);
        self
    }

    /// Returns why the budget is exhausted, if it is.
    pub fn exceeded(&self, steps: u32, tokens: u64, usd: f64) -> Option<String> {
        if let Some(max) = self.max_steps {
            if steps >= max {
                return Some(format!("step limit reached ({steps}/{max})"));
            }
        }
        if let Some(max) = self.max_tokens {
            if tokens >= max {
                return Some(format!("token limit reached ({tokens}/{max})"));
            }
        }
        if let Some(max) = self.max_usd {
            if usd >= max {
                return Some(format!("spend limit reached (${usd:.4}/${max:.4})"));
            }
        }
        None
    }
}
