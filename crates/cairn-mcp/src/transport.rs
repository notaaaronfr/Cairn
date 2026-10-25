use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use async_trait::async_trait;
use futures::StreamExt;
use serde_json::{Value, json};
use tokio::io::{AsyncBufReadExt, AsyncRead, AsyncWrite, AsyncWriteExt, BufReader};
use tokio::sync::oneshot;

use crate::client::McpError;

/// Delivers JSON-RPC messages to a server.
#[async_trait]
pub(crate) trait Transport: Send + Sync {
    /// Send a request and wait for the response with the same id.
    async fn request(&self, id: u64, message: Value) -> Result<Value, McpError>;
    /// Send a notification (no response).
    async fn notify(&self, message: Value) -> Result<(), McpError>;
    /// Forget an in-flight request (e.g. after the caller gave up).
    fn abandon(&self, _id: u64) {}
    /// Number of requests awaiting a response.
    fn in_flight(&self) -> usize {
        0
    }
    /// Called after a successful `initialize`.
    fn set_protocol_version(&self, _version: &str) {}
}

type Pending = Arc<Mutex<HashMap<u64, oneshot::Sender<Value>>>>;
type Writer = Arc<tokio::sync::Mutex<Box<dyn AsyncWrite + Send + Unpin>>>;

/// Newline-delimited JSON-RPC over a byte stream (stdio, pipes, sockets).
pub(crate) struct IoTransport {
    writer: Writer,
    pending: Pending,
    /// Keeps a spawned server process alive for the transport's lifetime.
    _child: Option<tokio::process::Child>,
}

impl IoTransport {
    pub(crate) fn new(
        reader: impl AsyncRead + Send + Unpin + 'static,
        writer: impl AsyncWrite + Send + Unpin + 'static,
        child: Option<tokio::process::Child>,
    ) -> Self {
        let writer: Writer = Arc::new(tokio::sync::Mutex::new(Box::new(writer)));
        let pending: Pending = Arc::default();
        tokio::spawn(read_loop(reader, writer.clone(), pending.clone()));
        Self { writer, pending, _child: child }
    }
}

async fn write_line(writer: &Writer, message: &Value) -> Result<(), McpError> {
    let mut line = serde_json::to_vec(message).map_err(|e| McpError::Protocol(e.to_string()))?;
    line.push(b'\n');
    let mut w = writer.lock().await;
    w.write_all(&line).await.map_err(|e| McpError::Transport(e.to_string()))?;
    w.flush().await.map_err(|e| McpError::Transport(e.to_string()))
}

async fn read_loop(reader: impl AsyncRead + Send + Unpin, writer: Writer, pending: Pending) {
    let mut lines = BufReader::new(reader).lines();
    loop {
        let line = match lines.next_line().await {
            Ok(Some(line)) => line,
            Ok(None) | Err(_) => break,
        };
        if line.trim().is_empty() {
            continue;
        }
        let Ok(msg) = serde_json::from_str::<Value>(&line) else {
            tracing::debug!(%line, "ignoring non-JSON line from MCP server");
            continue;
        };
        dispatch(msg, &pending, |reply| {
            let writer = writer.clone();
            async move {
                let _ = write_line(&writer, &reply).await;
            }
        })
        .await;
    }
    // Server closed: dropping the senders fails every waiting request.
    pending.lock().unwrap().clear();
}

/// Route one incoming message: responses resolve waiting requests, server
/// requests get a reply via `reply`, notifications are logged.
async fn dispatch<F, Fut>(msg: Value, pending: &Pending, reply: F)
where
    F: FnOnce(Value) -> Fut,
    Fut: std::future::Future<Output = ()>,
{
    let is_response = msg.get("result").is_some() || msg.get("error").is_some();
    match (msg.get("id").and_then(Value::as_u64), msg.get("method").and_then(Value::as_str)) {
        (Some(id), None) if is_response => {
            if let Some(tx) = pending.lock().unwrap().remove(&id) {
                let _ = tx.send(msg);
            }
        }
        (_, Some(method)) if msg.get("id").is_some() => {
            let id = msg["id"].clone();
            let response = if method == "ping" {
                json!({"jsonrpc": "2.0", "id": id, "result": {}})
            } else {
                json!({"jsonrpc": "2.0", "id": id, "error": {"code": -32601, "message": format!("client does not support `{method}`")}})
            };
            reply(response).await;
        }
        (_, Some(method)) => tracing::debug!(method, "MCP notification"),
        _ => tracing::debug!(%msg, "unrecognised MCP message"),
    }
}

#[async_trait]
impl Transport for IoTransport {
    async fn request(&self, id: u64, message: Value) -> Result<Value, McpError> {
        let (tx, rx) = oneshot::channel();
        self.pending.lock().unwrap().insert(id, tx);
        if let Err(e) = write_line(&self.writer, &message).await {
            self.pending.lock().unwrap().remove(&id);
            return Err(e);
        }
        rx.await.map_err(|_| McpError::Closed)
    }

    async fn notify(&self, message: Value) -> Result<(), McpError> {
        write_line(&self.writer, &message).await
    }

    fn abandon(&self, id: u64) {
        self.pending.lock().unwrap().remove(&id);
    }

    fn in_flight(&self) -> usize {
        self.pending.lock().unwrap().len()
    }
}

/// MCP Streamable HTTP transport.
pub(crate) struct HttpTransport {
    url: String,
    headers: reqwest::header::HeaderMap,
    client: reqwest::Client,
    session: Mutex<Option<String>>,
    protocol: Mutex<Option<String>>,
}

impl HttpTransport {
    pub(crate) fn new(url: String, headers: reqwest::header::HeaderMap) -> Self {
        Self { url, headers, client: reqwest::Client::new(), session: Mutex::new(None), protocol: Mutex::new(None) }
    }

    async fn post(&self, message: &Value) -> Result<reqwest::Response, McpError> {
        let mut req = self
            .client
            .post(&self.url)
            .headers(self.headers.clone())
            .header("accept", "application/json, text/event-stream")
            .json(message);
        if let Some(s) = self.session.lock().unwrap().clone() {
            req = req.header("mcp-session-id", s);
        }
        if let Some(v) = self.protocol.lock().unwrap().clone() {
            req = req.header("mcp-protocol-version", v);
        }
        let resp = req.send().await.map_err(|e| McpError::Transport(e.to_string()))?;
        if let Some(s) = resp.headers().get("mcp-session-id").and_then(|v| v.to_str().ok()) {
            *self.session.lock().unwrap() = Some(s.to_owned());
        }
        let status = resp.status();
        if !status.is_success() {
            let body = resp.text().await.unwrap_or_default();
            return Err(McpError::Transport(format!("HTTP {status}: {body}")));
        }
        Ok(resp)
    }

    /// Answer a server-to-client request that arrived on an SSE stream.
    async fn reply(&self, msg: Value, pending: &Pending) {
        dispatch(msg, pending, |reply| async move {
            let _ = self.post(&reply).await;
        })
        .await;
    }
}

#[async_trait]
impl Transport for HttpTransport {
    async fn request(&self, id: u64, message: Value) -> Result<Value, McpError> {
        let resp = self.post(&message).await?;
        let is_sse = resp
            .headers()
            .get("content-type")
            .and_then(|v| v.to_str().ok())
            .is_some_and(|ct| ct.starts_with("text/event-stream"));

        if !is_sse {
            let body: Value = resp.json().await.map_err(|e| McpError::Protocol(e.to_string()))?;
            let found = match body {
                Value::Array(batch) => batch.into_iter().find(|m| m["id"].as_u64() == Some(id)),
                single => Some(single),
            };
            return found.ok_or_else(|| McpError::Protocol("response did not contain the request id".into()));
        }

        // Read events until the response for `id` arrives; answer server
        // requests (e.g. ping) and skip notifications along the way.
        let mut bytes = resp.bytes_stream();
        let mut buf = Vec::<u8>::new();
        let unrelated: Pending = Arc::default();
        loop {
            while let Some(pos) = buf.windows(2).position(|w| w == b"\n\n") {
                let block: Vec<u8> = buf.drain(..pos + 2).collect();
                let data: String = String::from_utf8_lossy(&block)
                    .lines()
                    .filter_map(|l| l.strip_prefix("data:"))
                    .map(|d| d.strip_prefix(' ').unwrap_or(d))
                    .collect::<Vec<_>>()
                    .join("\n");
                let Ok(msg) = serde_json::from_str::<Value>(&data) else { continue };
                let is_response = msg.get("result").is_some() || msg.get("error").is_some();
                if is_response && msg["id"].as_u64() == Some(id) {
                    return Ok(msg);
                }
                self.reply(msg, &unrelated).await;
            }
            match bytes.next().await {
                Some(Ok(chunk)) => buf.extend(chunk.iter().filter(|b| **b != b'\r')),
                Some(Err(e)) => return Err(McpError::Transport(e.to_string())),
                None => return Err(McpError::Closed),
            }
        }
    }

    async fn notify(&self, message: Value) -> Result<(), McpError> {
        self.post(&message).await.map(|_| ())
    }

    fn set_protocol_version(&self, version: &str) {
        *self.protocol.lock().unwrap() = Some(version.to_owned());
    }
}
