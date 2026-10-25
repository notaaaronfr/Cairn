//! # cairn-mcp
//!
//! A Model Context Protocol client. Connect to any MCP server over stdio or
//! Streamable HTTP and use its tools in a Cairn agent:
//!
//! ```ignore
//! let whatstask = McpClient::connect_http(
//!     "https://whatstask.example/mcp",
//!     [("authorization", format!("Bearer {}", std::env::var("WHATSTASK_TOKEN")?))],
//! ).await?;
//! let tools = whatstask.tools(ToolOptions::default().effect("submit_work", Effect::Write)).await?;
//! let agent = Agent::builder("worker").model(model).tools(tools).build()?;
//! ```
//!
//! Credentials live in the client's transport and are never shown to the
//! model. Each call forwards Cairn's idempotency key in `_meta` so servers
//! can deduplicate retried calls.

mod client;
mod tool;
mod transport;

pub use client::{McpClient, McpError, RemoteTool};
pub use tool::{McpTool, ToolOptions};
