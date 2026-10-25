use std::sync::{Arc, Mutex};
use std::time::Duration;

use cairn_mcp::{McpClient, ToolOptions};
use cairn_runtime::testing::{ScriptedModel, text_response, tool_call_response};
use cairn_runtime::*;
use serde_json::{Value, json};
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};

type Log = Arc<Mutex<Vec<Value>>>;

/// Answers one JSON-RPC request like a WhatsTask MCP server would.
fn handle(msg: &Value) -> Option<Value> {
    let id = msg.get("id")?.clone();
    let method = msg["method"].as_str()?;
    let result = match method {
        "initialize" => json!({"protocolVersion": "2025-06-18", "serverInfo": {"name": "whatstask-fake"}, "capabilities": {"tools": {}}}),
        "tools/list" if msg["params"]["cursor"] == "p2" => json!({"tools": [
            {"name": "submit_work", "description": "Submit work", "inputSchema": {"type": "object", "properties": {"task_id": {"type": "string"}}}},
            {"name": "escalate", "description": "Escalate", "annotations": {"destructiveHint": false}},
        ]}),
        "tools/list" => json!({"nextCursor": "p2", "tools": [
            {"name": "list_my_tasks", "description": "List my tasks", "inputSchema": {"type": "object"}, "annotations": {"readOnlyHint": true}},
            {"name": "get_task", "description": "Get a task", "inputSchema": {"type": "object"}, "annotations": {"readOnlyHint": true}},
        ]}),
        "tools/call" => {
            let meta = msg["params"]["_meta"].clone();
            match msg["params"]["name"].as_str().unwrap_or_default() {
                "list_my_tasks" => json!({"content": [{"type": "text", "text": "2 tasks"}],
                                          "structuredContent": {"tasks": [{"id": "T-1"}, {"id": "T-2"}], "meta": meta}}),
                "submit_work" => json!({"content": [{"type": "text", "text": format!("submitted {}", msg["params"]["arguments"]["task_id"])}]}),
                "escalate" => json!({"isError": true, "content": [{"type": "text", "text": "no manager assigned"}]}),
                "hang" => return None,
                other => return Some(json!({"jsonrpc": "2.0", "id": id, "error": {"code": -32602, "message": format!("unknown tool {other}")}})),
            }
        }
        _ => return Some(json!({"jsonrpc": "2.0", "id": id, "error": {"code": -32601, "message": "nope"}})),
    };
    Some(json!({"jsonrpc": "2.0", "id": id, "result": result}))
}

/// In-memory stdio-style server. Sends a `ping` to the client before
/// answering the first tools/call, as real servers may.
async fn fake_server(reader: tokio::io::DuplexStream, mut writer: tokio::io::DuplexStream, log: Log) {
    let mut lines = BufReader::new(reader).lines();
    let mut pinged = false;
    while let Ok(Some(line)) = lines.next_line().await {
        let msg: Value = serde_json::from_str(&line).unwrap();
        log.lock().unwrap().push(msg.clone());
        if msg["method"] == "tools/call" && !pinged {
            pinged = true;
            writer.write_all(b"{\"jsonrpc\":\"2.0\",\"id\":\"srv-1\",\"method\":\"ping\"}\n").await.unwrap();
            writer.write_all(b"{\"jsonrpc\":\"2.0\",\"method\":\"notifications/message\",\"params\":{}}\n").await.unwrap();
        }
        if let Some(reply) = handle(&msg) {
            writer.write_all(format!("{reply}\n").as_bytes()).await.unwrap();
        }
    }
}

async fn connect() -> (Arc<McpClient>, Log) {
    // One pipe per direction, like a child process's stdin and stdout.
    let (c2s_w, c2s_r) = tokio::io::duplex(64 * 1024);
    let (s2c_w, s2c_r) = tokio::io::duplex(64 * 1024);
    let log: Log = Arc::default();
    tokio::spawn(fake_server(c2s_r, s2c_w, log.clone()));
    (McpClient::connect_io(s2c_r, c2s_w).await.unwrap(), log)
}

#[tokio::test]
async fn handshake_pagination_and_effects() {
    let (client, log) = connect().await;
    assert_eq!(client.server_name(), "whatstask-fake");
    assert_eq!(client.protocol_version(), "2025-06-18");
    // Notifications are fire-and-forget; give the server a moment to log it.
    for _ in 0..100 {
        if log.lock().unwrap().iter().any(|m| m["method"] == "notifications/initialized") {
            break;
        }
        tokio::time::sleep(Duration::from_millis(5)).await;
    }
    assert_eq!(log.lock().unwrap()[1]["method"], "notifications/initialized");

    let tools = client.tools(ToolOptions::default().prefix("wt_").effect("submit_work", Effect::Write)).await.unwrap();
    let specs: Vec<(String, Effect)> = tools.iter().map(|t| (t.spec().name.clone(), t.spec().effect)).collect();
    assert_eq!(
        specs,
        [
            ("wt_list_my_tasks".into(), Effect::Read),
            ("wt_get_task".into(), Effect::Read),
            ("wt_submit_work".into(), Effect::Write),       // overridden
            ("wt_escalate".into(), Effect::Write),          // destructiveHint: false
        ]
    );
    let err = client.tools(ToolOptions::default().only(["list_my_tasks", "delete_everything"])).await.err().unwrap();
    assert!(err.to_string().contains("delete_everything"));
}

#[tokio::test]
async fn agent_uses_mcp_tools_end_to_end() {
    let (client, log) = connect().await;
    let tools = client.tools(ToolOptions::default().effect("submit_work", Effect::Write)).await.unwrap();
    let model = Arc::new(ScriptedModel::new(vec![
        tool_call_response(&[("c1", "list_my_tasks", json!({}))]),
        tool_call_response(&[("c2", "submit_work", json!({"task_id": "T-1"})), ("c3", "escalate", json!({}))]),
        text_response("Submitted T-1; escalation failed."),
    ]));
    let agent = Agent::builder("worker").model(model.clone()).tools(tools).build().unwrap();
    let out = agent.run("do my tasks").await.unwrap();
    assert_eq!(out.output(), Some("Submitted T-1; escalation failed."));

    let requests = model.requests();
    let Content::ToolResult { content, is_error: false, .. } = &requests[1].messages[2].content[0] else { panic!() };
    assert_eq!(content["tasks"][0]["id"], "T-1", "structuredContent is passed through");
    assert_eq!(content["meta"]["cairn/idempotency_key"], ToolCtx::new(&out.run_id, "c1").idempotency_key);

    let results = &requests[2].messages[4].content;
    assert!(matches!(&results[0], Content::ToolResult { content, is_error: false, .. } if content == "submitted \"T-1\""));
    assert!(matches!(&results[1], Content::ToolResult { content, is_error: true, .. }
        if content["error"].as_str().unwrap().contains("no manager assigned")));

    // The client answered the server's ping.
    assert!(log.lock().unwrap().iter().any(|m| m["id"] == "srv-1" && m["result"].is_object()));
}

#[tokio::test]
async fn unannotated_tools_need_approval_by_default() {
    let (client, _log) = connect().await;
    let tools = client.tools(ToolOptions::default().only(["submit_work"])).await.unwrap();
    assert_eq!(tools[0].spec().effect, Effect::Irreversible);
    let agent = Agent::builder("w")
        .model(ScriptedModel::new(vec![tool_call_response(&[("c1", "submit_work", json!({"task_id": "T-9"}))])]))
        .tools(tools)
        .build()
        .unwrap();
    assert_eq!(agent.run("x").await.unwrap().status, Status::AwaitingApproval);
}

#[tokio::test]
async fn timed_out_call_is_cancelled_and_cleaned_up() {
    let (client, log) = connect().await;
    let res = tokio::time::timeout(Duration::from_millis(100), client.call_tool("hang", json!({}), Value::Null)).await;
    assert!(res.is_err(), "the call should still be pending");
    tokio::time::sleep(Duration::from_millis(50)).await;
    assert_eq!(client.in_flight(), 0, "abandoned request must be removed");
    assert!(log.lock().unwrap().iter().any(|m| m["method"] == "notifications/cancelled"));
}

#[tokio::test]
async fn stdio_subprocess_server() {
    if std::process::Command::new("python3").arg("--version").output().is_err() {
        eprintln!("skipped: python3 not available");
        return;
    }
    let script = r#"
import sys, json
for line in sys.stdin:
    m = json.loads(line)
    if "id" not in m: continue
    if m["method"] == "initialize":
        r = {"protocolVersion": "2025-06-18", "serverInfo": {"name": "py"}, "capabilities": {}}
    elif m["method"] == "tools/list":
        r = {"tools": [{"name": "echo", "description": "Echo", "inputSchema": {"type": "object"}, "annotations": {"readOnlyHint": True}}]}
    else:
        r = {"content": [{"type": "text", "text": json.dumps(m["params"]["arguments"])}]}
    print(json.dumps({"jsonrpc": "2.0", "id": m["id"], "result": r}), flush=True)
"#;
    let client = McpClient::connect_stdio("python3", &["-c", script], [("PYTHONUNBUFFERED", "1")]).await.unwrap();
    assert_eq!(client.server_name(), "py");
    let tools = client.tools(ToolOptions::default()).await.unwrap();
    let out = tools[0].call(json!({"msg": "hi"}), &ToolCtx::new("r", "c")).await.unwrap();
    assert_eq!(out, json!({"msg": "hi"}));
}

#[tokio::test]
async fn streamable_http_with_session_auth_and_sse() {
    use axum::http::{HeaderMap, StatusCode};
    use axum::response::IntoResponse;

    async fn endpoint(headers: HeaderMap, axum::Json(msg): axum::Json<Value>) -> axum::response::Response {
        if headers.get("authorization").and_then(|v| v.to_str().ok()) != Some("Bearer secret") {
            return StatusCode::UNAUTHORIZED.into_response();
        }
        let method = msg["method"].as_str().unwrap_or_default();
        if method != "initialize" && headers.get("mcp-session-id").and_then(|v| v.to_str().ok()) != Some("sess-42") {
            return (StatusCode::BAD_REQUEST, "missing session").into_response();
        }
        if msg.get("id").is_none() {
            return StatusCode::ACCEPTED.into_response();
        }
        let reply = handle(&msg).unwrap();
        match method {
            "initialize" => ([("mcp-session-id", "sess-42")], axum::Json(reply)).into_response(),
            // Tool calls stream: a progress notification, then the result.
            "tools/call" => {
                let body = format!(
                    "event: message\ndata: {{\"jsonrpc\":\"2.0\",\"method\":\"notifications/progress\",\"params\":{{}}}}\n\nevent: message\ndata: {reply}\n\n"
                );
                ([("content-type", "text/event-stream")], body).into_response()
            }
            _ => axum::Json(reply).into_response(),
        }
    }

    let app = axum::Router::new().route("/mcp", axum::routing::post(endpoint));
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
    let url = format!("http://{addr}/mcp");

    assert!(McpClient::connect_http(&url, [("authorization", "Bearer wrong")]).await.is_err());

    let client = McpClient::connect_http(&url, [("authorization", "Bearer secret")]).await.unwrap();
    let tools = client.tools(ToolOptions::default().only(["list_my_tasks"])).await.unwrap();
    let out = tools[0].call(json!({}), &ToolCtx::new("r", "c")).await.unwrap();
    assert_eq!(out["tasks"][1]["id"], "T-2");
}
