use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use std::time::Duration;

use async_trait::async_trait;
use cairn_runtime::{Effect, Tool, ToolCtx, ToolError, ToolSpec};
use serde_json::{Value, json};

use crate::client::{McpClient, McpError, RemoteTool};

/// How a server's tools are exposed to an agent.
#[derive(Debug, Clone, Default)]
pub struct ToolOptions {
    prefix: Option<String>,
    effects: HashMap<String, Effect>,
    only: Option<HashSet<String>>,
    timeout: Option<Duration>,
}

impl ToolOptions {
    /// Prefix every tool name (e.g. `"whatstask_"`) to avoid collisions.
    pub fn prefix(mut self, prefix: &str) -> Self {
        self.prefix = Some(prefix.to_owned());
        self
    }

    /// Override the effect derived from the server's annotations.
    pub fn effect(mut self, tool: &str, effect: Effect) -> Self {
        self.effects.insert(tool.to_owned(), effect);
        self
    }

    /// Expose only these tools (by server name).
    pub fn only<I: IntoIterator<Item = S>, S: Into<String>>(mut self, tools: I) -> Self {
        self.only = Some(tools.into_iter().map(Into::into).collect());
        self
    }

    /// Timeout for every tool from this server (overrides the agent default).
    pub fn timeout(mut self, limit: Duration) -> Self {
        self.timeout = Some(limit);
        self
    }
}

/// Effect from MCP tool annotations. Per the spec, a tool that is not
/// read-only is assumed destructive unless it says otherwise, so an
/// unannotated tool is `Irreversible` (and needs approval by default).
fn effect_from(annotations: &Value) -> Effect {
    if annotations["readOnlyHint"].as_bool() == Some(true) {
        Effect::Read
    } else if annotations["destructiveHint"].as_bool() == Some(false) {
        Effect::Write
    } else {
        Effect::Irreversible
    }
}

/// Model-facing tool names must match `^[a-zA-Z0-9_-]{1,64}$`.
fn sanitize(name: &str) -> String {
    let s: String = name.chars().map(|c| if c.is_ascii_alphanumeric() || c == '_' || c == '-' { c } else { '_' }).collect();
    s.chars().take(64).collect()
}

/// An MCP server tool usable as a Cairn [`Tool`].
pub struct McpTool {
    client: Arc<McpClient>,
    remote_name: String,
    spec: ToolSpec,
    timeout: Option<Duration>,
}

impl McpTool {
    pub fn remote_name(&self) -> &str {
        &self.remote_name
    }
}

impl McpClient {
    /// The server's tools as Cairn tools.
    pub async fn tools(self: &Arc<Self>, options: ToolOptions) -> Result<Vec<McpTool>, McpError> {
        let remote = self.list_tools().await?;
        let tools: Vec<McpTool> = remote
            .into_iter()
            .filter(|t| options.only.as_ref().is_none_or(|only| only.contains(&t.name)))
            .map(|t: RemoteTool| {
                let effect = options.effects.get(&t.name).copied().unwrap_or_else(|| effect_from(&t.annotations));
                let name = sanitize(&format!("{}{}", options.prefix.as_deref().unwrap_or(""), t.name));
                McpTool {
                    client: self.clone(),
                    spec: ToolSpec { name, description: t.description, input_schema: t.input_schema, effect },
                    remote_name: t.name,
                    timeout: options.timeout,
                }
            })
            .collect();
        if let Some(only) = &options.only {
            let found: HashSet<&str> = tools.iter().map(|t| t.remote_name.as_str()).collect();
            let missing: Vec<&String> = only.iter().filter(|n| !found.contains(n.as_str())).collect();
            if !missing.is_empty() {
                return Err(McpError::Protocol(format!("server `{}` has no tool(s) {missing:?}", self.server_name())));
            }
        }
        Ok(tools)
    }
}

#[async_trait]
impl Tool for McpTool {
    fn spec(&self) -> &ToolSpec {
        &self.spec
    }

    fn timeout(&self) -> Option<Duration> {
        self.timeout
    }

    async fn call(&self, input: Value, ctx: &ToolCtx) -> Result<Value, ToolError> {
        let meta = json!({"cairn/run_id": ctx.run_id, "cairn/idempotency_key": ctx.idempotency_key});
        let result = self
            .client
            .call_tool(&self.remote_name, input, meta)
            .await
            .map_err(|e| ToolError::new(e.to_string()))?;
        let text = result["content"]
            .as_array()
            .into_iter()
            .flatten()
            .map(|c| match c["type"].as_str() {
                Some("text") => c["text"].as_str().unwrap_or_default().to_owned(),
                Some(other) => format!("[{other} content omitted]"),
                None => String::new(),
            })
            .collect::<Vec<_>>()
            .join("\n");
        if result["isError"].as_bool() == Some(true) {
            return Err(ToolError::new(if text.is_empty() { "tool reported an error".into() } else { text }));
        }
        if !result["structuredContent"].is_null() {
            return Ok(result["structuredContent"].clone());
        }
        Ok(serde_json::from_str(&text).unwrap_or(Value::String(text)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn maps_annotations_to_effects() {
        assert_eq!(effect_from(&json!({"readOnlyHint": true})), Effect::Read);
        assert_eq!(effect_from(&json!({"destructiveHint": false})), Effect::Write);
        assert_eq!(effect_from(&json!({"destructiveHint": true})), Effect::Irreversible);
        assert_eq!(effect_from(&Value::Null), Effect::Irreversible, "spec default is destructive");
    }

    #[test]
    fn sanitizes_names() {
        assert_eq!(sanitize("tasks.list/mine"), "tasks_list_mine");
        assert_eq!(sanitize(&"x".repeat(80)).len(), 64);
    }
}
