//! # cairn-a2a
//!
//! The [Agent2Agent (A2A)](https://a2a-protocol.org) protocol, version 1.0,
//! over its JSON-RPC binding.
//!
//! **Serve** a Cairn agent to any A2A client (other frameworks, other
//! companies' agents):
//!
//! ```ignore
//! let server = A2aServer::new(support_agent, "https://agents.example.com/a2a")
//!     .description("Answers billing and order questions.")
//!     .skill(AgentSkill::new("orders", "Order lookup", "Find and explain orders").tags(["support"]))
//!     .api_key(std::env::var("A2A_KEY")?);
//! server.serve("0.0.0.0:8791").await?;
//! ```
//!
//! Conversations are durable Cairn runs: they survive restarts, show up in
//! `cairn-server`'s Dev UI, and keep their budgets and approval gates.
//!
//! **Call** a remote A2A agent, from code or as a tool for a Cairn agent:
//!
//! ```ignore
//! let remote = Arc::new(A2aClient::discover("https://travel.example.com", [("authorization", "Bearer …")]).await?);
//! let agent = Agent::builder("planner")
//!     .model(model)
//!     .tool(A2aTool::new(remote).effect(Effect::Read).timeout(Duration::from_secs(300)))
//!     .build()?;
//! ```

mod client;
mod server;
mod tool;
pub mod types;

pub use client::{A2aClient, A2aError};
pub use server::A2aServer;
pub use tool::A2aTool;
pub use types::{
    AgentCard, AgentSkill, Artifact, Message, Part, Role, SendMessageRequest, SendMessageResponse, StreamResponse,
    Task, TaskState, TaskStatus,
};
