use std::sync::atomic::{AtomicU64, Ordering};

use futures::stream::{BoxStream, StreamExt};
use reqwest::header::{ACCEPT, CONTENT_TYPE, HeaderMap, HeaderName, HeaderValue};
use serde::Serialize;
use serde::de::DeserializeOwned;
use serde_json::{Value, json};

use crate::types::*;

#[derive(Debug, thiserror::Error)]
pub enum A2aError {
    #[error("transport error: {0}")]
    Transport(String),
    #[error("HTTP {status}: {body}")]
    Http { status: u16, body: String },
    /// A JSON-RPC error from the agent; `code` is one of [`codes`].
    #[error("A2A error {code}: {message}")]
    Rpc { code: i64, message: String, data: Option<Value> },
    #[error("protocol error: {0}")]
    Protocol(String),
}

impl From<reqwest::Error> for A2aError {
    fn from(e: reqwest::Error) -> Self {
        A2aError::Transport(e.to_string())
    }
}

/// A client for one remote A2A agent over JSON-RPC.
pub struct A2aClient {
    http: reqwest::Client,
    url: String,
    headers: HeaderMap,
    card: Option<AgentCard>,
    next_id: AtomicU64,
}

impl A2aClient {
    /// Fetch the agent's card from `base_url` (or the card URL itself) and
    /// connect to its JSON-RPC interface. `headers` (e.g. `authorization`)
    /// are sent on every request, including the card fetch.
    pub async fn discover<I, K, V>(base_url: &str, headers: I) -> Result<Self, A2aError>
    where
        I: IntoIterator<Item = (K, V)>,
        K: AsRef<str>,
        V: AsRef<str>,
    {
        let headers = header_map(headers)?;
        let card_url = if base_url.ends_with(".json") {
            base_url.to_owned()
        } else {
            format!("{}{AGENT_CARD_PATH}", base_url.trim_end_matches('/'))
        };
        let http = reqwest::Client::new();
        let resp = http.get(&card_url).headers(headers.clone()).send().await?;
        let card: AgentCard = read_json(resp).await?;
        let url = card.jsonrpc_url().map(str::to_owned).ok_or_else(|| {
            A2aError::Protocol(format!("agent `{}` has no JSON-RPC interface for A2A {PROTOCOL_VERSION}", card.name))
        })?;
        Ok(Self { http, url, headers, card: Some(card), next_id: AtomicU64::new(1) })
    }

    /// Connect to a known JSON-RPC endpoint without fetching a card.
    pub fn new(jsonrpc_url: &str) -> Self {
        Self {
            http: reqwest::Client::new(),
            url: jsonrpc_url.to_owned(),
            headers: HeaderMap::new(),
            card: None,
            next_id: AtomicU64::new(1),
        }
    }

    /// Send `name: value` with every request.
    pub fn header(mut self, name: &str, value: &str) -> Result<Self, A2aError> {
        let (n, v) = parse_header(name, value)?;
        self.headers.insert(n, v);
        Ok(self)
    }

    pub fn card(&self) -> Option<&AgentCard> {
        self.card.as_ref()
    }

    pub fn url(&self) -> &str {
        &self.url
    }

    pub async fn send_message(&self, request: &SendMessageRequest) -> Result<SendMessageResponse, A2aError> {
        self.call("SendMessage", request).await
    }

    /// Send one text message and wait for the result.
    pub async fn send_text(&self, text: &str) -> Result<SendMessageResponse, A2aError> {
        self.send_message(&SendMessageRequest::new(Message::user(text))).await
    }

    pub async fn get_task(&self, id: &str, history_length: Option<i32>) -> Result<Task, A2aError> {
        self.call("GetTask", &GetTaskRequest { id: id.to_owned(), history_length, tenant: None }).await
    }

    pub async fn list_tasks(&self, request: &ListTasksRequest) -> Result<ListTasksResponse, A2aError> {
        self.call("ListTasks", request).await
    }

    pub async fn cancel_task(&self, id: &str) -> Result<Task, A2aError> {
        self.call("CancelTask", &TaskIdRequest { id: id.to_owned(), ..Default::default() }).await
    }

    /// Send a message and receive the task's updates as they happen.
    pub async fn send_streaming_message(
        &self,
        request: &SendMessageRequest,
    ) -> Result<BoxStream<'static, Result<StreamResponse, A2aError>>, A2aError> {
        self.stream("SendStreamingMessage", request).await
    }

    /// Receive updates for a task that is still running.
    pub async fn subscribe_to_task(
        &self,
        id: &str,
    ) -> Result<BoxStream<'static, Result<StreamResponse, A2aError>>, A2aError> {
        self.stream("SubscribeToTask", &TaskIdRequest { id: id.to_owned(), ..Default::default() }).await
    }

    fn request(&self, method: &str, params: &impl Serialize, accept: &'static str) -> reqwest::RequestBuilder {
        let id = self.next_id.fetch_add(1, Ordering::Relaxed);
        self.http
            .post(&self.url)
            .headers(self.headers.clone())
            .header("A2A-Version", PROTOCOL_VERSION)
            .header(CONTENT_TYPE, "application/json")
            .header(ACCEPT, accept)
            .json(&json!({"jsonrpc": "2.0", "id": id, "method": method, "params": params}))
    }

    async fn call<T: DeserializeOwned>(&self, method: &str, params: &impl Serialize) -> Result<T, A2aError> {
        let resp = self.request(method, params, "application/json").send().await?;
        let body: Value = read_json(resp).await?;
        rpc_result(body)
    }

    async fn stream(
        &self,
        method: &str,
        params: &impl Serialize,
    ) -> Result<BoxStream<'static, Result<StreamResponse, A2aError>>, A2aError> {
        let resp = self.request(method, params, "text/event-stream").send().await?;
        let status = resp.status();
        if !status.is_success() {
            return Err(A2aError::Http { status: status.as_u16(), body: resp.text().await.unwrap_or_default() });
        }
        let is_sse = resp
            .headers()
            .get(CONTENT_TYPE)
            .and_then(|v| v.to_str().ok())
            .is_some_and(|v| v.starts_with("text/event-stream"));
        if !is_sse {
            // Errors before the stream starts come back as plain JSON-RPC.
            let body: Value = resp.json().await?;
            return Err(rpc_result::<Value>(body)
                .err()
                .unwrap_or_else(|| A2aError::Protocol("expected an event stream".into())));
        }
        Ok(sse_data(resp)
            .map(|data| {
                let body: Value = serde_json::from_str(&data?)
                    .map_err(|e| A2aError::Protocol(format!("invalid stream event: {e}")))?;
                rpc_result(body)
            })
            .boxed())
    }
}

fn parse_header(name: &str, value: &str) -> Result<(HeaderName, HeaderValue), A2aError> {
    let n = HeaderName::from_bytes(name.as_bytes()).map_err(|e| A2aError::Protocol(format!("header {name}: {e}")))?;
    let v = HeaderValue::from_str(value).map_err(|e| A2aError::Protocol(format!("header {name}: {e}")))?;
    Ok((n, v))
}

fn header_map<I, K, V>(headers: I) -> Result<HeaderMap, A2aError>
where
    I: IntoIterator<Item = (K, V)>,
    K: AsRef<str>,
    V: AsRef<str>,
{
    let mut map = HeaderMap::new();
    for (k, v) in headers {
        let (n, v) = parse_header(k.as_ref(), v.as_ref())?;
        map.insert(n, v);
    }
    Ok(map)
}

async fn read_json<T: DeserializeOwned>(resp: reqwest::Response) -> Result<T, A2aError> {
    let status = resp.status();
    let text = resp.text().await?;
    if !status.is_success() {
        // JSON-RPC errors may arrive with a non-200 status; prefer them.
        if let Ok(body) = serde_json::from_str::<Value>(&text) {
            if body.get("error").is_some() {
                return Err(rpc_result::<Value>(body).unwrap_err());
            }
        }
        return Err(A2aError::Http { status: status.as_u16(), body: text });
    }
    serde_json::from_str(&text).map_err(|e| A2aError::Protocol(format!("invalid response: {e}")))
}

fn rpc_result<T: DeserializeOwned>(mut body: Value) -> Result<T, A2aError> {
    if let Some(err) = body.get("error") {
        return Err(A2aError::Rpc {
            code: err["code"].as_i64().unwrap_or(codes::INTERNAL_ERROR),
            message: err["message"].as_str().unwrap_or("unknown error").to_owned(),
            data: err.get("data").cloned(),
        });
    }
    let result = body.get_mut("result").map(Value::take).ok_or_else(|| A2aError::Protocol("no result".into()))?;
    serde_json::from_value(result).map_err(|e| A2aError::Protocol(format!("unexpected result: {e}")))
}

/// The `data` of each server-sent event in a response body.
fn sse_data(resp: reqwest::Response) -> BoxStream<'static, Result<String, A2aError>> {
    fn block_data(block: &[u8]) -> Option<String> {
        let text = String::from_utf8_lossy(block);
        let data: Vec<&str> =
            text.lines().filter_map(|l| l.strip_prefix("data:")).map(|v| v.strip_prefix(' ').unwrap_or(v)).collect();
        (!data.is_empty()).then(|| data.join("\n"))
    }
    let bytes = Box::pin(resp.bytes_stream());
    futures::stream::unfold((bytes, Vec::<u8>::new(), false), |(mut bytes, mut buf, done)| async move {
        if done {
            return None;
        }
        loop {
            if let Some(pos) = buf.windows(2).position(|w| w == b"\n\n") {
                let block: Vec<u8> = buf.drain(..pos + 2).collect();
                if let Some(data) = block_data(&block) {
                    return Some((Ok(data), (bytes, buf, false)));
                }
                continue;
            }
            match bytes.next().await {
                Some(Ok(chunk)) => buf.extend(chunk.iter().filter(|b| **b != b'\r')),
                Some(Err(e)) => return Some((Err(A2aError::Transport(e.to_string())), (bytes, buf, true))),
                None => return block_data(&buf).map(|d| (Ok(d), (bytes, Vec::new(), true))),
            }
        }
    })
    .boxed()
}
