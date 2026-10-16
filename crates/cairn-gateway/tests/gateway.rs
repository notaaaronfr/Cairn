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
