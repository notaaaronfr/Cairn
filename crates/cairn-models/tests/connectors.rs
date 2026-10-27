//! HTTP-level tests for Azure OpenAI, native Gemini and Bedrock, against
//! local fake servers that check what the real services would receive.

use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::{Arc, Mutex};

use axum::Json;
use axum::extract::{Path, RawQuery};
use axum::http::{HeaderMap, Uri};
use axum::routing::post;
use cairn_models::{AwsCredentials, BedrockAuth, BedrockModel, GeminiModel, OpenAiCompatModel};
use cairn_runtime::*;
use schemars::JsonSchema;
use serde::Deserialize;
use serde_json::{Value, json};
use sha2::Digest;

async fn serve(app: axum::Router) -> String {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
    format!("http://{addr}")
}

fn header(h: &HeaderMap, k: &str) -> String {
    h.get(k).and_then(|v| v.to_str().ok()).unwrap_or_default().to_owned()
}

/// (deployment, query, api-key header, authorization header)
type AzureCall = (String, String, String, String);
/// (raw path, headers, body)
type BedrockCall = (String, HeaderMap, Vec<u8>);

fn hello() -> ModelRequest {
    ModelRequest { messages: vec![Message::user("hi")], max_tokens: 20, ..Default::default() }
}

#[tokio::test]
async fn azure_openai_key_and_entra_auth() {
    let seen: Arc<Mutex<Vec<AzureCall>>> = Arc::default();
    let s = seen.clone();
    let app = axum::Router::new().route(
        "/openai/deployments/{dep}/chat/completions",
        post(move |Path(dep): Path<String>, RawQuery(q): RawQuery, h: HeaderMap| {
            let s = s.clone();
            async move {
                s.lock().unwrap().push((
                    dep,
                    q.unwrap_or_default(),
                    header(&h, "api-key"),
                    header(&h, "authorization"),
                ));
                Json(json!({"choices": [{"finish_reason": "stop", "message": {"content": "hi from azure"}}],
                            "usage": {"prompt_tokens": 3, "completion_tokens": 2}}))
            }
        }),
    );
    let base = serve(app).await;

    let keyed = OpenAiCompatModel::azure(&base, "gpt-dep", "az-key");
    assert_eq!(keyed.id(), "azure:gpt-dep");
    let r = keyed.complete(&hello()).await.unwrap();
    assert_eq!(r.content[0], Content::Text { text: "hi from azure".into() });

    let calls = Arc::new(AtomicU32::new(0));
    let c = calls.clone();
    let entra = OpenAiCompatModel::azure_entra(&base, "gpt-dep", move || {
        c.fetch_add(1, Ordering::SeqCst);
        Ok("entra-token".into())
    })
    .with_api_version("2025-01-01-preview");
    entra.complete(&hello()).await.unwrap();
    entra.complete(&hello()).await.unwrap();
    assert_eq!(calls.load(Ordering::SeqCst), 2, "token fetched per request");

    let seen = seen.lock().unwrap();
    assert_eq!(seen[0], ("gpt-dep".into(), "api-version=2024-10-21".into(), "az-key".into(), String::new()));
    assert_eq!(seen[1].1, "api-version=2025-01-01-preview");
    assert_eq!(seen[1].3, "Bearer entra-token");
}

#[derive(Deserialize, JsonSchema)]
struct City {
    city: String,
}

#[tokio::test]
async fn gemini_native_function_calling_returns_thought_signatures() {
    let requests: Arc<Mutex<Vec<(String, Value)>>> = Arc::default();
    let r = requests.clone();
    let app = axum::Router::new().route(
        "/v1beta/models/{call}",
        post(move |Path(call): Path<String>, h: HeaderMap, Json(body): Json<Value>| {
            let r = r.clone();
            async move {
                assert_eq!(header(&h, "x-goog-api-key"), "g-key");
                let n = {
                    let mut r = r.lock().unwrap();
                    r.push((call, body));
                    r.len()
                };
                Json(if n == 1 {
                    json!({"candidates": [{"content": {"role": "model", "parts": [
                        {"text": "Looking up weather.", "thought": true},
                        {"functionCall": {"name": "weather", "args": {"city": "Paris"}}, "thoughtSignature": "SIG-1"}
                    ]}, "finishReason": "STOP"}],
                     "usageMetadata": {"promptTokenCount": 20, "candidatesTokenCount": 5, "thoughtsTokenCount": 7}, "modelVersion": "gemini-x-001"})
                } else {
                    json!({"candidates": [{"content": {"role": "model", "parts": [{"text": "Sunny in Paris."}]}, "finishReason": "STOP"}],
                           "usageMetadata": {"promptTokenCount": 40, "candidatesTokenCount": 4}})
                })
            }
        }),
    );
    let base = serve(app).await;
    let model = GeminiModel::new("gemini-x", "g-key").with_base_url(&format!("{base}/v1beta"));
    let agent = Agent::builder("w")
        .model(model)
        .system("be brief")
        .tool(tool("weather", "Weather for a city", Effect::Read, |c: City, _ctx| async move {
            Ok(json!({"city": c.city, "sky": "sunny"}))
        }))
        .build()
        .unwrap();
    let out = agent.run("weather in Paris?").await.unwrap();
    assert_eq!(out.output(), Some("Sunny in Paris."));
    assert_eq!(out.usage.output_tokens, 5 + 7 + 4, "thinking tokens are billed as output");

    let requests = requests.lock().unwrap();
    assert_eq!(requests[0].0, "gemini-x:generateContent");
    let first = &requests[0].1;
    assert_eq!(first["systemInstruction"]["parts"][0]["text"], "be brief");
    assert_eq!(
        first["tools"][0]["functionDeclarations"][0]["parametersJsonSchema"]["properties"]["city"]["type"],
        "string"
    );

    let second = &requests[1].1["contents"];
    let model_turn = &second[1];
    assert_eq!(model_turn["role"], "model");
    let call = model_turn["parts"].as_array().unwrap().iter().find(|p| p.get("functionCall").is_some()).unwrap();
    assert_eq!(call["thoughtSignature"], "SIG-1", "signature returned on the function call");
    assert!(call["functionCall"].get("id").is_none(), "generated ids are not sent to Gemini");
    let response = &second[2]["parts"][0]["functionResponse"];
    assert_eq!(response["name"], "weather");
    assert_eq!(response["response"]["sky"], "sunny");
}

#[tokio::test]
async fn gemini_streams_deltas() {
    let app = axum::Router::new().route(
        "/v1beta/models/{call}",
        post(|Path(call): Path<String>, RawQuery(q): RawQuery| async move {
            assert_eq!((call.as_str(), q.as_deref()), ("gemini-x:streamGenerateContent", Some("alt=sse")));
            let body = [
                json!({"candidates": [{"content": {"parts": [{"text": "Hel"}]}}]}),
                json!({"candidates": [{"content": {"parts": [{"text": "lo"}]}, "finishReason": "STOP"}],
                       "usageMetadata": {"promptTokenCount": 3, "candidatesTokenCount": 2}}),
            ]
            .iter()
            .map(|v| format!("data: {v}\r\n\r\n"))
            .collect::<String>();
            ([("content-type", "text/event-stream")], body)
        }),
    );
    let base = serve(app).await;
    let seen = Arc::new(Mutex::new(String::new()));
    let sink = seen.clone();
    let agent = Agent::builder("w")
        .model(GeminiModel::new("gemini-x", "k").with_base_url(&format!("{base}/v1beta")))
        .on_delta(move |e| {
            if let StreamEvent::TextDelta(t) = e {
                sink.lock().unwrap().push_str(t);
            }
        })
        .build()
        .unwrap();
    let out = agent.run("hi").await.unwrap();
    assert_eq!(out.output(), Some("Hello"));
    assert_eq!(*seen.lock().unwrap(), "Hello");
}

#[tokio::test]
async fn bedrock_sigv4_and_api_key() {
    let seen: Arc<Mutex<Vec<BedrockCall>>> = Arc::default();
    let s = seen.clone();
    let app = axum::Router::new().route(
        "/model/{id}/converse",
        post(move |uri: Uri, h: HeaderMap, body: axum::body::Bytes| {
            let s = s.clone();
            async move {
                s.lock().unwrap().push((uri.path().to_owned(), h, body.to_vec()));
                Json(json!({"output": {"message": {"role": "assistant", "content": [{"text": "hi from bedrock"}]}},
                            "stopReason": "end_turn", "usage": {"inputTokens": 3, "outputTokens": 4}}))
            }
        }),
    );
    let base = serve(app).await;
    let creds = AwsCredentials {
        access_key_id: "AKIDEXAMPLE".into(),
        secret_access_key: "secret".into(),
        session_token: Some("session".into()),
    };
    let signed = BedrockModel::new("anthropic.claude-test-v1:0", "eu-west-1", BedrockAuth::Credentials(creds))
        .with_endpoint(&base);
    assert_eq!(signed.id(), "bedrock:anthropic.claude-test-v1:0");
    let r = signed.complete(&hello()).await.unwrap();
    assert_eq!(r.content[0], Content::Text { text: "hi from bedrock".into() });
    assert_eq!(r.usage.output_tokens, 4);

    let keyed = BedrockModel::new("amazon.nova-pro-v1:0", "us-east-1", BedrockAuth::ApiKey("bedrock-key".into()))
        .with_endpoint(&base);
    keyed.complete(&hello()).await.unwrap();

    let seen = seen.lock().unwrap();
    let (path, h, body) = &seen[0];
    assert_eq!(path, "/model/anthropic.claude-test-v1%3A0/converse", "model id is percent-encoded");
    let auth = header(h, "authorization");
    assert!(auth.starts_with("AWS4-HMAC-SHA256 Credential=AKIDEXAMPLE/"), "{auth}");
    assert!(auth.contains("/eu-west-1/bedrock/aws4_request"), "{auth}");
    assert!(
        auth.contains("SignedHeaders=content-type;host;x-amz-content-sha256;x-amz-date;x-amz-security-token"),
        "{auth}"
    );
    assert_eq!(header(h, "x-amz-content-sha256"), hex::encode(sha2::Sha256::digest(body)));
    assert_eq!(header(h, "x-amz-security-token"), "session");
    let parsed: Value = serde_json::from_slice(body).unwrap();
    assert_eq!(parsed["inferenceConfig"]["maxTokens"], 20);

    assert_eq!(header(&seen[1].1, "authorization"), "Bearer bedrock-key");
}
