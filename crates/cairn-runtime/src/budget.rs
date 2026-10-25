use serde::{Deserialize, Serialize};

/// Hard limits enforced by the runtime before every model call.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct Budget {
    /// Maximum number of model calls.
    pub max_steps: Option<u32>,
    /// Maximum spend in USD. Every model the agent may call must have
    /// pricing, otherwise [`crate::AgentBuilder::build`] refuses the agent.
    pub max_usd: Option<f64>,
    /// Maximum input + output tokens.
    pub max_tokens: Option<u64>,
    /// Stop instead of calling the model when the remaining budget allows
    /// fewer output tokens than this. Defaults to 1.
    pub min_output_tokens: Option<u32>,
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

    pub fn min_output_tokens(mut self, n: u32) -> Self {
        self.min_output_tokens = Some(n);
        self
    }

    /// Largest output-token limit for the next call that keeps the run
    /// within budget, given an estimate of the call's input tokens.
    ///
    /// `call_cost(input, output)` is the model's worst-case USD for a call
    /// (see [`crate::Model::max_call_cost`]). Returns `Err(reason)` when the
    /// remaining budget cannot pay for `min_output_tokens`.
    pub fn output_cap(
        &self,
        configured: u32,
        used_tokens: u64,
        spent_usd: f64,
        est_input_tokens: u64,
        call_cost: impl Fn(u64, u64) -> Option<f64>,
    ) -> Result<u32, String> {
        let mut cap = u64::from(configured);
        if let Some(max) = self.max_tokens {
            cap = cap.min(max.saturating_sub(used_tokens).saturating_sub(est_input_tokens));
        }
        if let (Some(max), Some(fixed)) = (self.max_usd, call_cost(est_input_tokens, 0)) {
            // Cost is linear in output tokens: fixed + per_token * n.
            let per_token = call_cost(0, 1_000_000).unwrap_or(0.0) / 1_000_000.0;
            let left = max - spent_usd - fixed;
            if per_token > 0.0 {
                cap = cap.min((left / per_token).floor().max(0.0) as u64);
            } else if left < 0.0 {
                cap = 0;
            }
        }
        let min = u64::from(self.min_output_tokens.unwrap_or(1));
        if cap < min {
            return Err(format!(
                "remaining budget allows only {cap} output tokens for the next call (minimum {min}, \
                 estimated input {est_input_tokens} tokens)"
            ));
        }
        Ok(cap as u32)
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
