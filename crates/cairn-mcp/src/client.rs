use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Duration;

use serde_json::{Value, json};

use crate::transport::{HttpTransport, IoTransport, Transport};

/// Protocol version this client requests. Servers may answer with another
/// version they support; the client then uses that.
pub const PROTOCOL_VERSION: &str = "2025-06-18";
const SETUP_TIMEOUT: Duration = Duration::from_secs(30);

#[derive(Debug, Clone, thiserror::Error)]
pub enum McpError {
    #[error("MCP transport error: {0}")]
    Transport(String),
    #[error("MCP protocol error: {0}")]
    Protocol(String),
    #[error("MCP server error {code}: {message}")]
    Rpc { code: i64, message: String },
    #[error("MCP connection closed")]
    Closed,
    #[error("MCP request timed out: {0}")]
    Timeout(String),
}

/// A tool as advertised by the server.
#[derive(Debug, Clone)]
pub struct RemoteTool {
    pub name: String,
    pub description: String,
    pub input_schema: Value,
    pub annotations: Value,
}

pub struct McpClient {
    transport: Arc<dyn Transport>,
    next_id: AtomicU64,
    server_name: String,
    protocol_version: String,
}

/// Removes an abandoned request from the transport if the caller's future is
/// dropped (e.g. by a tool timeout) and tells the server to stop.
struct InFlight {
    id: u64,
    transport: Arc<dyn Transport>,
    done: bool,
}

impl Drop for InFlight {
    fn drop(&mut self) {
        if self.done {
            return;
        }
        self.transport.abandon(self.id);
        let (transport, id) = (self.transport.clone(), self.id);
        if let Ok(rt) = tokio::runtime::Handle::try_current() {
            rt.spawn(async move {
                let cancel = json!({"jsonrpc": "2.0", "method": "notifications/cancelled",
                                    "params": {"requestId": id, "reason": "client cancelled"}});
                let _ = transport.notify(cancel).await;
            });
        }
    }
}

impl McpClient {
    /// Spawn a local MCP server and talk to it over stdio.
    pub async fn connect_stdio<I, K, V>(
        command: &str,
        args: &[&str],
        env: I,
    ) -> Result<Arc<Self>, McpError>
    where
        I: IntoIterator<Item = (K, V)>,
        K: AsRef<std::ffi::OsStr>,
        V: AsRef<std::ffi::OsStr>,
    {
        let mut child = tokio::process::Command::new(command)
            .args(args)
            .envs(env)
            .stdin(std::process::Stdio::piped())
            .stdout(std::process::Stdio::piped())
            .stderr(std::process::Stdio::piped())
            .kill_on_drop(true)
            .spawn()
            .map_err(|e| McpError::Transport(format!("could not start `{command}`: {e}")))?;
        let stdin = child.stdin.take().expect("piped stdin");
        let stdout = child.stdout.take().expect("piped stdout");
        if let Some(stderr) = child.stderr.take() {
            let name = command.to_owned();
            tokio::spawn(async move {
                use tokio::io::AsyncBufReadExt;
                let mut lines = tokio::io::BufReader::new(stderr).lines();
                while let Ok(Some(line)) = lines.next_line().await {
                    tracing::debug!(server = %name, "{line}");
                }
            });
        }
        Self::initialize(Arc::new(IoTransport::new(stdout, stdin, Some(child)))).await
    }

    /// Talk to a server over any byte stream (pipes, sockets, in-memory).
    pub async fn connect_io(
        reader: impl tokio::io::AsyncRead + Send + Unpin + 'static,
        writer: impl tokio::io::AsyncWrite + Send + Unpin + 'static,
    ) -> Result<Arc<Self>, McpError> {
        Self::initialize(Arc::new(IoTransport::new(reader, writer, None))).await
    }

    /// Connect to a remote server over Streamable HTTP. `headers` (e.g.
    /// `authorization`) are sent on every request and never exposed to models.
    pub async fn connect_http<K, V>(url: &str, headers: impl IntoIterator<Item = (K, V)>) -> Result<Arc<Self>, McpError>
    where
        K: AsRef<str>,
        V: AsRef<str>,
    {
        let mut map = reqwest::header::HeaderMap::new();
        for (k, v) in headers {
            let name = reqwest::header::HeaderName::from_bytes(k.as_ref().as_bytes())
                .map_err(|e| McpError::Transport(format!("bad header name: {e}")))?;
            let value = reqwest::header::HeaderValue::from_str(v.as_ref())
                .map_err(|e| McpError::Transport(format!("bad header value: {e}")))?;
            map.insert(name, value);
        }
        Self::initialize(Arc::new(HttpTransport::new(url.to_owned(), map))).await
    }

    async fn initialize(transport: Arc<dyn Transport>) -> Result<Arc<Self>, McpError> {
        let mut client =
            Self { transport, next_id: AtomicU64::new(1), server_name: String::new(), protocol_version: String::new() };
        let params = json!({
            "protocolVersion": PROTOCOL_VERSION,
            "capabilities": {},
            "clientInfo": {"name": "cairn", "version": env!("CARGO_PKG_VERSION")},
        });
        let result = tokio::time::timeout(SETUP_TIMEOUT, client.request("initialize", params))
            .await
            .map_err(|_| McpError::Timeout("initialize".into()))??;
        client.protocol_version = result["protocolVersion"].as_str().unwrap_or(PROTOCOL_VERSION).to_owned();
        client.server_name = result["serverInfo"]["name"].as_str().unwrap_or("mcp").to_owned();
        client.transport.set_protocol_version(&client.protocol_version);
        client.transport.notify(json!({"jsonrpc": "2.0", "method": "notifications/initialized"})).await?;
        Ok(Arc::new(client))
    }

    pub fn server_name(&self) -> &str {
        &self.server_name
    }

    pub fn protocol_version(&self) -> &str {
        &self.protocol_version
    }

    /// Requests sent but not yet answered.
    pub fn in_flight(&self) -> usize {
        self.transport.in_flight()
    }

    /// Send a JSON-RPC request and return its `result`.
    pub async fn request(&self, method: &str, params: Value) -> Result<Value, McpError> {
        let id = self.next_id.fetch_add(1, Ordering::Relaxed);
        let mut guard = InFlight { id, transport: self.transport.clone(), done: false };
        let message = json!({"jsonrpc": "2.0", "id": id, "method": method, "params": params});
        let response = self.transport.request(id, message).await;
        guard.done = true;
        let response = response?;
        if let Some(err) = response.get("error") {
            return Err(McpError::Rpc {
                code: err["code"].as_i64().unwrap_or(0),
                message: err["message"].as_str().unwrap_or_default().to_owned(),
            });
        }
        Ok(response.get("result").cloned().unwrap_or(Value::Null))
    }

    /// All tools the server offers (follows pagination).
    pub async fn list_tools(&self) -> Result<Vec<RemoteTool>, McpError> {
        let mut tools = Vec::new();
        let mut cursor: Option<String> = None;
        loop {
            let params = match &cursor {
                Some(c) => json!({"cursor": c}),
                None => json!({}),
            };
            let page = tokio::time::timeout(SETUP_TIMEOUT, self.request("tools/list", params))
                .await
                .map_err(|_| McpError::Timeout("tools/list".into()))??;
            for t in page["tools"].as_array().into_iter().flatten() {
                tools.push(RemoteTool {
                    name: t["name"].as_str().unwrap_or_default().to_owned(),
                    description: t["description"].as_str().unwrap_or_default().to_owned(),
                    input_schema: if t["inputSchema"].is_object() {
                        t["inputSchema"].clone()
                    } else {
                        json!({"type": "object", "properties": {}})
                    },
                    annotations: t["annotations"].clone(),
                });
            }
            match page["nextCursor"].as_str() {
                Some(next) if !next.is_empty() => cursor = Some(next.to_owned()),
                _ => return Ok(tools),
            }
        }
    }

    /// Call a tool. `meta` is sent as `_meta` (e.g. idempotency keys).
    pub async fn call_tool(&self, name: &str, arguments: Value, meta: Value) -> Result<Value, McpError> {
        let mut params = json!({"name": name, "arguments": arguments});
        if !meta.is_null() {
            params["_meta"] = meta;
        }
        self.request("tools/call", params).await
    }
}
