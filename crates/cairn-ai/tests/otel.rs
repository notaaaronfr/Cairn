//! End-to-end: spans from a real agent run reach an OTLP/HTTP collector.
#![cfg(feature = "otel")]

use std::sync::{Arc, Mutex};

use axum::body::Bytes;
use axum::http::HeaderMap;
use cairn::prelude::*;
use cairn::testing::{ScriptedModel, text_response};

/// (content-type, body) of each export request.
type Request = (String, Vec<u8>);

#[tokio::test(flavor = "multi_thread")]
async fn spans_are_exported_over_otlp_http() {
    let received: Arc<Mutex<Vec<Request>>> = Arc::default();
    let sink = received.clone();
    let app = axum::Router::new().route(
        "/v1/traces",
        axum::routing::post(move |headers: HeaderMap, body: Bytes| {
            let sink = sink.clone();
            async move {
                let ct = headers.get("content-type").and_then(|v| v.to_str().ok()).unwrap_or_default().to_owned();
                sink.lock().unwrap().push((ct, body.to_vec()));
                ""
            }
        }),
    );
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });

    let guard = cairn::otel::init(&format!("http://{addr}/v1/traces"), "cairn-test").unwrap();
    let agent = Agent::builder("support").model(ScriptedModel::new(vec![text_response("hi")])).build().unwrap();
    agent.run("hello").await.unwrap();
    // Dropping the guard flushes the batch; do it off the async thread.
    tokio::task::spawn_blocking(move || drop(guard)).await.unwrap();

    let received = received.lock().unwrap();
    assert!(!received.is_empty(), "collector received nothing");
    let (ct, body) = &received[0];
    assert_eq!(ct, "application/x-protobuf");
    let text = String::from_utf8_lossy(body);
    for expected in ["invoke_agent support", "chat scripted:test", "gen_ai.usage.input_tokens", "cairn-test"] {
        assert!(text.contains(expected), "missing `{expected}` in exported spans");
    }
}
