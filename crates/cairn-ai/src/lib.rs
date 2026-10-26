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

/// PostgreSQL journal (enable the `postgres` feature).
#[cfg(feature = "postgres")]
pub use cairn_journal_postgres::PostgresJournal;

/// Model Context Protocol client (enable the `mcp` feature).
#[cfg(feature = "mcp")]
pub mod mcp {
    pub use cairn_mcp::*;
}

/// OpenTelemetry export (enable the `otel` feature).
#[cfg(feature = "otel")]
pub mod otel;

pub mod prelude {
    pub use cairn_runtime::{
        Agent, AgentTool, AllowAll, Budget, Content, Decision, Effect, EffectPolicy, Event, EventKind, FallbackModel,
        GroupChat, HookCtx, InMemoryJournal, Journal, JsonlJournal, Message, Middleware, Model, PendingCall, Policy,
        Priced, Pricing, RunOutcome, RunState, Status, StreamEvent, Tool, ToolCtx, ToolError, ToolSpec, Workflow,
        WorkflowCtx, WorkflowError, WorkflowStatus, tool,
    };
}
