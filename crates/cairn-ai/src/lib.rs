//! # Cairn
//!
//! Durable, governed, model-agnostic agents.
//!
//! ```ignore
//! use cairn::prelude::*;
//!
//! let agent = Agent::builder("support")
//!     .model(cairn::models::from_spec("anthropic:claude-opus-5-5")?)
//!     .system("You resolve support tickets.")
//!     .tool(get_order)
//!     .tool(refund)                       // Effect::Irreversible -> needs approval
//!     .budget(Budget::default().max_steps(20).max_usd(1.0))
//!     .journal(JsonlJournal::new("runs")?)
//!     .build()?;
//!
//! let outcome = agent.run("Refund order 1042").await?;
//! ```

pub use cairn_runtime::*;

/// Model adapters and `provider:model` construction.
pub mod models {
    pub use cairn_models::*;
}

pub mod prelude {
    pub use cairn_runtime::{
        Agent, AllowAll, Budget, Content, Decision, Effect, EffectPolicy, Event, EventKind, FallbackModel,
        InMemoryJournal, Journal, JsonlJournal, Message, Model, PendingCall, Policy, RunOutcome, RunState, Status,
        Tool, ToolCtx, ToolError, ToolSpec, tool,
    };
}
