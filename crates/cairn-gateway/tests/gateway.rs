use std::sync::Arc;

use async_trait::async_trait;
use cairn_gateway::{Config, Gateway, ModelFactory};
use cairn_runtime::testing::{ScriptedModel, text_response, tool_call_response};
use cairn_runtime::{Model, ModelError, ModelRequest, ModelResponse};
use serde_json::{Value, json};

struct Down;

#[async_trait]
impl Model for Down {
    fn id(&self) -> String {
        "down:model".into()
    }
    async fn complete(&self, _: &ModelRequest) -> Result<ModelResponse, ModelError> {
        Err(ModelError::Api { status: 503, body: "overloaded".into() })
    }
}

const CONFIG: &str = r#"
allow_direct_specs = true

[[keys]]
name = "team-a"
key = "ck_a"
budget_usd = 0.002

[[keys]]
name = "team-b"
key = "ck_b"
models = ["fast"]

[routes]
smart = ["down:model", "up:model"]
misconfigured = ["missing:model", "up:model"]
fast = ["up:model"]

[pricing."up:model"]
input = 10.0
output = 10.0

# Never serve successfully, but team-a has a USD budget, so they must be priced.
[pricing."down:model"]
input = 10.0
output = 10.0

[pricing."missing:model"]
input = 10.0
output = 10.0
"#;

async fn start(responses: Vec<ModelResponse>, audit: &std::path::Path) -> (String, Arc<Gateway>) {
    let mut config = Config::from_toml(CONFIG).unwrap();
    config.audit_log = Some(audit.to_owned());
    let up: Arc<dyn Model> = Arc::new(ScriptedModel::new(responses));
    let factory: ModelFactory = Arc::new(move |spec: &str| match spec {
        "down:model" => Ok(Arc::new(Down) as Arc<dyn Model>),
        "up:model" => Ok(up.clone()),
        other => Err(ModelError::Other(format!("no such model {other}"))),
    });
    let gw = Gateway::with_factory(config, factory).await.unwrap();
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let router = gw.router();
    tokio::spawn(async move { axum::serve(listener, router).await.unwrap() });
    (format!("http://{addr}"), gw)
}

async fn post(url: &str, key: &str, body: Value) -> (u16, Value) {
    let resp = reqwest::Client::new().post(url).bearer_auth(key).json(&body).send().await.unwrap();
    let status = resp.status().as_u16();
    (status, resp.json().await.unwrap_or(Value::Null))
}

#[tokio::test]
async fn openai_client_gets_failover_tool_calls_and_audit() {
    let dir = tempfile::tempdir().unwrap();
    let audit = dir.path().join("audit.jsonl");
    let (base, _gw) = start(vec![tool_call_response(&[("call_1", "get_order", json!({"id": "42"}))])], &audit).await;

    let (status, body) = post(
        &format!("{base}/v1/chat/completions"),
        "ck_a",
        json!({
            "model": "smart",
            "messages": [{"role": "system", "content": "be useful"}, {"role": "user", "content": "where is order 42?"}],
            "tools": [{"type": "function", "function": {"name": "get_order", "parameters": {"type": "object"}}}]
        }),
    )
    .await;

    assert_eq!(status, 200, "{body}");
    assert_eq!(body["object"], "chat.completion");
    assert_eq!(body["model"], "smart");
    assert_eq!(body["choices"][0]["finish_reason"], "tool_calls");
    let call = &body["choices"][0]["message"]["tool_calls"][0];
    assert_eq!(call["function"]["name"], "get_order");
    assert_eq!(call["function"]["arguments"], r#"{"id":"42"}"#);

    let line = std::fs::read_to_string(&audit).unwrap();
    let rec: Value = serde_json::from_str(line.lines().next().unwrap()).unwrap();
    assert_eq!(rec["key"], "team-a");
    assert_eq!(rec["served_by"], "up:model");
    assert!(rec["failed_over"][0].as_str().unwrap().starts_with("down:model"));
    assert_eq!(rec["tool_calls"][0], "get_order");
}

#[tokio::test]
async fn anthropic_client_round_trip_and_tool_results() {
    let dir = tempfile::tempdir().unwrap();
    let (base, _gw) = start(vec![text_response("Order 42 ships today.")], &dir.path().join("a.jsonl")).await;

    let resp = reqwest::Client::new()
        .post(format!("{base}/v1/messages"))
        .header("x-api-key", "ck_b")
        .json(&json!({
            "model": "fast",
            "max_tokens": 256,
            "system": [{"type": "text", "text": "be useful"}],
            "messages": [
                {"role": "user", "content": "where is order 42?"},
                {"role": "assistant", "content": [{"type": "tool_use", "id": "t1", "name": "get_order", "input": {"id": "42"}}]},
                {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "t1", "content": [{"type": "text", "text": "shipping"}]}]}
            ]
        }))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 200);
    let body: Value = resp.json().await.unwrap();
    assert_eq!(body["type"], "message");
    assert_eq!(body["stop_reason"], "end_turn");
    assert_eq!(body["content"][0]["text"], "Order 42 ships today.");
}

#[tokio::test]
async fn auth_allowlist_and_budget_are_enforced() {
    let dir = tempfile::tempdir().unwrap();
    let audit = dir.path().join("audit.jsonl");
    let (base, gw) = start(vec![text_response("one"), text_response("two")], &audit).await;
    let url = format!("{base}/v1/chat/completions");
    let msg = |model: &str| json!({"model": model, "messages": [{"role": "user", "content": "hi"}]});

    assert_eq!(post(&url, "wrong", msg("fast")).await.0, 401);
    assert_eq!(post(&url, "ck_b", msg("smart")).await.0, 403, "team-b may only use `fast`");
    assert_eq!(post(&url, "ck_a", msg("nope")).await.0, 404);

    // ScriptedModel reports 120 tokens; at $10/Mtok that's $0.0012 per call.
    assert_eq!(post(&url, "ck_a", msg("fast")).await.0, 200);
    assert_eq!(post(&url, "ck_a", msg("fast")).await.0, 200);
    let (status, body) = post(&url, "ck_a", msg("fast")).await;
    assert_eq!(status, 429);
    assert_eq!(body["error"]["type"], "budget_exceeded");
    assert!((gw.spend("team-a").usd - 0.0024).abs() < 1e-9);

    // Spend survives a restart because it is rebuilt from the audit log.
    let (base2, gw2) = start(vec![], &audit).await;
    assert!((gw2.spend("team-a").usd - 0.0024).abs() < 1e-9);
    assert_eq!(post(&format!("{base2}/v1/chat/completions"), "ck_a", msg("fast")).await.0, 429);
}

#[tokio::test]
async fn streaming_clients_receive_sse() {
    let dir = tempfile::tempdir().unwrap();
    let (base, _gw) = start(vec![text_response("hello"), text_response("hi")], &dir.path().join("a.jsonl")).await;
    let client = reqwest::Client::new();

    let text = client
        .post(format!("{base}/v1/chat/completions"))
        .bearer_auth("ck_b")
        .json(&json!({"model": "fast", "stream": true, "messages": [{"role": "user", "content": "x"}]}))
        .send()
        .await
        .unwrap()
        .text()
        .await
        .unwrap();
    assert!(text.contains(r#""content":"hello""#), "{text}");
    assert!(text.trim_end().ends_with("data: [DONE]"));

    let text = client
        .post(format!("{base}/v1/messages"))
        .header("x-api-key", "ck_b")
        .json(&json!({"model": "fast", "stream": true, "max_tokens": 10, "messages": [{"role": "user", "content": "x"}]}))
        .send()
        .await
        .unwrap()
        .text()
        .await
        .unwrap();
    for ev in ["message_start", "content_block_start", "content_block_delta", "message_delta", "message_stop"] {
        assert!(text.contains(&format!("event: {ev}")), "missing {ev}");
    }
}

#[test]
fn hashed_keys_match() {
    let hash = cairn_gateway::config::sha256_hex("secret");
    let k = cairn_gateway::KeyConfig {
        name: "x".into(),
        key: None,
        key_sha256: Some(hash),
        budget_usd: None,
        budget_tokens: None,
        models: None,
    };
    assert!(k.matches("secret"));
    assert!(!k.matches("secreT"));
}

#[tokio::test]
async fn unconfigured_provider_fails_over() {
    let dir = tempfile::tempdir().unwrap();
    let (base, _gw) = start(vec![text_response("served")], &dir.path().join("a.jsonl")).await;
    let (status, body) = post(
        &format!("{base}/v1/chat/completions"),
        "ck_a",
        json!({"model": "misconfigured", "messages": [{"role": "user", "content": "hi"}]}),
    )
    .await;
    assert_eq!(status, 200, "{body}");
    assert_eq!(body["choices"][0]["message"]["content"], "served");
}

/// Fake OpenAI-compatible upstream that streams fragmented SSE slowly.
async fn slow_upstream() -> String {
    use axum::body::Body;
    use axum::routing::post;
    let app = axum::Router::new().route(
        "/v1/chat/completions",
        post(|| async {
            let chunks = vec![
                "data: {\"model\":\"fake-1\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"Hel\"}}]}\n\n",
                // An event split across two TCP writes.
                "data: {\"choices\":[{\"index\":0,\"delta\":{\"con",
                "tent\":\"lo\"}}]}\n\n",
                "data: {\"choices\":[{\"index\":0,\"delta\":{\"tool_calls\":[{\"index\":0,\"id\":\"c1\",\"type\":\"function\",\"function\":{\"name\":\"get_order\",\"arguments\":\"{\\\"id\\\":\"}}]}}]}\n\n",
                "data: {\"choices\":[{\"index\":0,\"delta\":{\"tool_calls\":[{\"index\":0,\"function\":{\"arguments\":\"7}\"}}]}}]}\n\n",
                "data: {\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"tool_calls\"}]}\n\n",
                "data: {\"choices\":[],\"usage\":{\"prompt_tokens\":100,\"completion_tokens\":100}}\n\n",
                "data: [DONE]\n\n",
            ];
            let body = futures::stream::unfold(chunks.into_iter(), |mut it| async move {
                let next = it.next()?;
                tokio::time::sleep(std::time::Duration::from_millis(60)).await;
                Some((Ok::<_, std::convert::Infallible>(next.to_owned()), it))
            });
            ([("content-type", "text/event-stream")], Body::from_stream(body))
        }),
    );
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
    format!("http://{addr}/v1")
}

#[tokio::test]
async fn true_streaming_across_formats_with_billing() {
    use futures::StreamExt;
    let upstream = slow_upstream().await;
    let dir = tempfile::tempdir().unwrap();
    let audit = dir.path().join("audit.jsonl");
    let mut config = Config::from_toml(CONFIG).unwrap();
    config.audit_log = Some(audit.clone());
    config.routes.insert("live".into(), vec!["down:model".into(), "fake:fake-1".into()]);
    config.pricing.insert("fake:fake-1".into(), toml::from_str("input = 1.0\noutput = 1.0").unwrap());
    let factory: ModelFactory = Arc::new(move |spec: &str| match spec {
        "down:model" => Ok(Arc::new(Down) as Arc<dyn Model>),
        "fake:fake-1" => Ok(Arc::new(cairn_models::OpenAiCompatModel::new("fake", "fake-1", upstream.clone(), None))),
        other => Err(ModelError::Other(format!("no such model {other}"))),
    });
    let gw = Gateway::with_factory(config, factory).await.unwrap();
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let router = gw.router();
    tokio::spawn(async move { axum::serve(listener, router).await.unwrap() });

    let started = std::time::Instant::now();
    let resp = reqwest::Client::new()
        .post(format!("http://{addr}/v1/messages"))
        .header("x-api-key", "ck_a")
        .json(&json!({"model": "live", "stream": true, "max_tokens": 100, "messages": [{"role": "user", "content": "hi"}]}))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 200);

    let mut body = resp.bytes_stream();
    let mut text = String::new();
    let mut first_text_at = None;
    while let Some(chunk) = body.next().await {
        text.push_str(&String::from_utf8_lossy(&chunk.unwrap()));
        if first_text_at.is_none() && text.contains("text_delta") {
            first_text_at = Some(started.elapsed());
        }
    }
    let total = started.elapsed();

    // Tokens reached the client well before the upstream finished.
    let first = first_text_at.expect("no text delta");
    assert!(first * 3 < total, "first token at {first:?}, total {total:?} — not streaming");

    let events: Vec<Value> = text
        .lines()
        .filter_map(|l| l.strip_prefix("data: "))
        .map(|d| serde_json::from_str(d).unwrap())
        .collect();
    let types: Vec<&str> = events.iter().map(|e| e["type"].as_str().unwrap()).collect();
    assert_eq!(types.first(), Some(&"message_start"));
    assert_eq!(types.last(), Some(&"message_stop"));

    let text_deltas: Vec<&str> =
        events.iter().filter(|e| e["delta"]["type"] == "text_delta").map(|e| e["delta"]["text"].as_str().unwrap()).collect();
    assert_eq!(text_deltas, ["Hel", "lo"]);

    let tool_start = events.iter().find(|e| e["content_block"]["type"] == "tool_use").expect("tool block");
    assert_eq!(tool_start["index"], 1, "tool_use is the second content block");
    assert_eq!(tool_start["content_block"]["id"], "c1");
    assert_eq!(tool_start["content_block"]["name"], "get_order");
    let args: String = events
        .iter()
        .filter(|e| e["delta"]["type"] == "input_json_delta")
        .map(|e| e["delta"]["partial_json"].as_str().unwrap())
        .collect();
    assert_eq!(serde_json::from_str::<Value>(&args).unwrap(), json!({"id": 7}));

    let stops = events.iter().filter(|e| e["type"] == "content_block_stop").count();
    assert_eq!(stops, 2, "every opened block is closed");
    let end = events.iter().find(|e| e["type"] == "message_delta").unwrap();
    assert_eq!(end["delta"]["stop_reason"], "tool_use");

    // Audit is written after the stream ends; wait briefly for it.
    let mut rec = Value::Null;
    for _ in 0..50 {
        if let Ok(t) = std::fs::read_to_string(&audit) {
            if let Some(line) = t.lines().next() {
                rec = serde_json::from_str(line).unwrap();
                break;
            }
        }
        tokio::time::sleep(std::time::Duration::from_millis(20)).await;
    }
    assert_eq!(rec["served_by"], "fake:fake-1");
    assert!(rec["failed_over"][0].as_str().unwrap().starts_with("down:model"));
    assert_eq!(rec["tokens"], 200);
    assert_eq!(rec["tool_calls"][0], "get_order");
    assert!((gw.spend("team-a").usd - 0.0002).abs() < 1e-12);
}

#[test]
fn usd_budget_with_unpriced_route_is_rejected_at_load() {
    let bad = r#"
[[keys]]
name = "k"
key = "x"
budget_usd = 5.0

[routes]
cheap = ["compat:k2:K2"]
"#;
    let err = Config::from_toml(bad).unwrap_err();
    assert!(err.contains("compat:k2:K2") && err.contains("no [pricing"), "{err}");

    let free = format!("{bad}\n[pricing.\"compat:k2:K2\"]\ninput = 0.0\noutput = 0.0\n");
    assert!(Config::from_toml(&free).is_ok(), "an explicitly free model is allowed");
}

#[tokio::test]
async fn usd_budget_key_cannot_request_unpriced_direct_spec() {
    let dir = tempfile::tempdir().unwrap();
    let (base, _gw) = start(vec![text_response("x")], &dir.path().join("a.jsonl")).await;
    // team-a has budget_usd; "unpriced:model" has no pricing entry.
    let (status, body) = post(
        &format!("{base}/v1/chat/completions"),
        "ck_a",
        json!({"model": "unpriced:model", "messages": [{"role": "user", "content": "hi"}]}),
    )
    .await;
    assert_eq!(status, 403, "{body}");
    assert!(body["error"]["message"].as_str().unwrap().contains("no pricing"));
}
