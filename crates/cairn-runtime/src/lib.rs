//! # cairn-runtime
//!
//! The core of Cairn: an event-sourced agent runtime.
//!
//! Every model call, tool call, approval and policy decision is appended to a
//! [`Journal`]. The agent's state is *only ever* a fold over those events
//! ([`RunState::apply`]), so the same code path powers live execution, crash
//! recovery, human-in-the-loop suspension and time-travel debugging.

pub mod agent;
pub mod budget;
pub mod eval;
pub mod event;
pub mod journal;
pub mod message;
pub mod model;
pub mod output;
pub mod policy;
pub mod pricing;
pub mod state;
pub mod testing;
pub mod tool;

pub use agent::{Agent, AgentBuilder, AgentError, RunOutcome};
pub use budget::Budget;
pub use event::{Event, EventKind};
pub use journal::{InMemoryJournal, Journal, JournalError, JsonlJournal, LeaseHolder};
pub use message::{Content, Message, Role};
pub use model::{
    FallbackModel, Model, ModelError, ModelRequest, ModelResponse, ModelStream, RejectedAttempt, StopReason,
    StreamEvent, Usage, synthesize_stream,
};
pub use policy::{AllowAll, Decision, EffectPolicy, Policy};
pub use pricing::{Priced, Pricing};
pub use state::{PendingCall, RunState, Status};
pub use tool::{Effect, Tool, ToolCtx, ToolError, ToolSpec, tool};
