use axum::Json;
use axum::http::HeaderMap;
use axum::routing::post;
use cairn_models::from_spec;
use cairn_runtime::{Message, ModelRequest};
use serde_json::{Value, json};

#[tokio::test]
async fn compat_spec_reaches_custom_endpoint_with_key() {
    let app = axum::Router::new().route(
        "/v1/chat/completions",
        post(|headers: HeaderMap, Json(body): Json<Value>| async move {
            let auth = headers.get("authorization").and_then(|v| v.to_str().ok()).unwrap_or_default().to_owned();
            Json(json!({
                "model": body["model"],
                "choices": [{"finish_reason": "stop", "message": {"content": format!("auth={auth}")}}],
                "usage": {"prompt_tokens": 3, "completion_tokens": 2}
            }))
        }),
    );
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });

    // SAFETY: this test binary sets these variables once, before any other
    // thread reads the environment, and no other test in it touches them.
    unsafe {
        std::env::set_var("CAIRNTEST_K2_BASE_URL", format!("http://{addr}/v1"));
        std::env::set_var("CAIRNTEST_K2_API_KEY", "sk-test");
    }

    let model = from_spec("compat:cairntest-k2:IFM/K2-Horizon-375B-A23B").unwrap();
    assert_eq!(model.id(), "cairntest-k2:IFM/K2-Horizon-375B-A23B");

    let resp = model
        .complete(&ModelRequest { messages: vec![Message::user("hi")], max_tokens: 10, ..Default::default() })
        .await
        .unwrap();
    assert_eq!(resp.model, "IFM/K2-Horizon-375B-A23B");
    assert_eq!(resp.content[0], cairn_runtime::Content::Text { text: "auth=Bearer sk-test".into() });
}
