use std::sync::Arc;
use std::sync::atomic::{AtomicU32, Ordering};
use std::time::Duration;

use cairn_runtime::testing::{ScriptedModel, text_response, tool_call_response};
use cairn_runtime::*;
use cairn_server::Server;
use schemars::JsonSchema;
use serde::Deserialize;
use serde_json::{Value, json};

#[derive(Deserialize, JsonSchema)]
struct Id {
    id: u32,
}

fn refund_agent(journal: &Arc<dyn Journal>, refunds: Arc<AtomicU32>, responses: Vec<ModelResponse>) -> Agent {
    let refund = tool("refund", "Refund", Effect::Irreversible, move |i: Id, _ctx| {
        let refunds = refunds.clone();
        async move {
            refunds.fetch_add(1, Ordering::SeqCst);
            Ok(json!({ "refunded": i.id }))
        }
    });
    Agent::builder("support")
        .model(ScriptedModel::new(responses))
        .tool(refund)
        .shared_journal(journal.clone())
        .build()
        .unwrap()
}

async fn serve(server: Server) -> String {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let router = server.router();
    tokio::spawn(async move { axum::serve(listener, router).await.unwrap() });
    format!("http://{addr}")
}

struct Client {
    base: String,
    key: Option<&'static str>,
    http: reqwest::Client,
}

impl Client {
    async fn send(&self, method: reqwest::Method, path: &str, body: Option<Value>) -> (u16, Value) {
        let mut req = self.http.request(method, format!("{}{path}", self.base));
        if let Some(k) = self.key {
            req = req.bearer_auth(k);
        }
        if let Some(b) = body {
            req = req.json(&b);
        }
        let resp = req.send().await.unwrap();
        let status = resp.status().as_u16();
        (status, resp.json().await.unwrap_or(Value::Null))
    }
    async fn get(&self, path: &str) -> (u16, Value) {
        self.send(reqwest::Method::GET, path, None).await
    }
    async fn post(&self, path: &str, body: Value) -> (u16, Value) {
        self.send(reqwest::Method::POST, path, Some(body)).await
    }
    async fn wait_for(&self, run_id: &str, status: &str) -> Value {
        for _ in 0..200 {
            let (_, run) = self.get(&format!("/v1/runs/{run_id}")).await;
            if run["summary"]["status"] == status {
                return run;
            }
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
        panic!("run {run_id} never reached {status}");
    }
}

#[tokio::test]
async fn api_runs_approvals_and_time_travel() {
    let journal: Arc<dyn Journal> = Arc::new(InMemoryJournal::new());
    let refunds = Arc::new(AtomicU32::new(0));
    let agent = refund_agent(
        &journal,
        refunds.clone(),
        vec![tool_call_response(&[("r1", "refund", json!({"id": 7}))]), text_response("Refunded order 7.")],
    );
    let echo = Agent::builder("echo").model(ScriptedModel::new(vec![text_response("pong")])).build().unwrap();
    let base = serve(Server::new(journal.clone()).agent(agent).agent(echo).api_key("secret")).await;
    let anon = Client { base: base.clone(), key: None, http: reqwest::Client::new() };
    let c = Client { base, key: Some("secret"), http: reqwest::Client::new() };

    assert_eq!(anon.get("/v1/runs").await.0, 401);
    assert_eq!(c.get("/v1/agents").await.1, json!([{"name": "echo"}, {"name": "support"}]));

    // Synchronous run on an agent that was built with its own journal:
    // the server rebuilt it onto the shared journal, so the run is visible.
    let (status, out) =
        c.post("/v1/agents/echo/runs", json!({"input": "ping", "wait": true, "run_id": "echo-1"})).await;
    assert_eq!((status, out["output"].clone()), (200, json!("pong")));
    assert_eq!(c.get("/v1/runs/echo-1").await.1["summary"]["status"], "completed");

    // Background run that needs approval.
    let (status, started) = c.post("/v1/agents/support/runs", json!({"input": "refund 7", "run_id": "case-7"})).await;
    assert_eq!((status, started["run_id"].clone()), (202, json!("case-7")));
    c.wait_for("case-7", "awaiting_approval").await;
    let (_, pending) = c.get("/v1/approvals").await;
    assert_eq!(pending[0]["tool"], "refund");
    assert_eq!(pending[0]["input"], json!({"id": 7}));
    assert_eq!(refunds.load(Ordering::SeqCst), 0);

    let (status, _) = c.post("/v1/runs/case-7/approvals/r1", json!({"approved": true, "by": "ops"})).await;
    assert_eq!(status, 202);
    let run = c.wait_for("case-7", "completed").await;
    assert_eq!(run["summary"]["output"], "Refunded order 7.");
    assert_eq!(refunds.load(Ordering::SeqCst), 1);
    assert_eq!(c.post("/v1/runs/case-7/approvals/r1", json!({"approved": true})).await.0, 409, "already decided");

    // Journal and time travel.
    let (_, events) = c.get("/v1/runs/case-7/events").await;
    assert_eq!(events[0]["kind"]["type"], "run_started");
    let (_, early) = c.get("/v1/runs/case-7/state?at=0").await;
    assert_eq!(early["messages"].as_array().unwrap().len(), 1);
    assert_eq!(early["status"]["status"], "running");

    // Most recent first.
    let (_, runs) = c.get("/v1/runs").await;
    assert_eq!(runs[0]["run_id"], "case-7");
    assert!(runs[0]["cost_usd"].as_f64().unwrap() > 0.0);

    // Validation.
    assert_eq!(c.post("/v1/agents/echo/runs", json!({"input": "x", "run_id": "../etc"})).await.0, 400);
    assert_eq!(c.post("/v1/agents/echo/runs", json!({"input": "x", "run_id": "echo-1"})).await.0, 409);
    assert_eq!(c.post("/v1/agents/nobody/runs", json!({"input": "x"})).await.0, 404);
    assert_eq!(c.get("/v1/runs/missing").await.0, 404);
}

#[tokio::test]
async fn recovery_loop_finishes_runs_whose_worker_died() {
    let journal: Arc<dyn Journal> = Arc::new(InMemoryJournal::new());
    let refunds = Arc::new(AtomicU32::new(0));

    // A worker ran until approval, the decision was recorded, then it died
    // before resuming: the run is `running` with nobody driving it.
    let worker = refund_agent(
        &journal,
        refunds.clone(),
        vec![tool_call_response(&[("r1", "refund", json!({"id": 1}))]), text_response("done")],
    );
    let out = worker.run_with_id("orphan", "refund 1").await.unwrap();
    worker.decide(&out.run_id, "r1", true, None, None).await.unwrap();
    drop(worker);

    let agent = refund_agent(&journal, refunds.clone(), vec![text_response("recovered")]);
    let base = serve(Server::new(journal.clone()).agent(agent).recover_every(Duration::from_millis(20))).await;
    let c = Client { base, key: None, http: reqwest::Client::new() };
    let run = c.wait_for("orphan", "completed").await;
    assert_eq!(run["summary"]["output"], "recovered");
    assert_eq!(refunds.load(Ordering::SeqCst), 1);
}

#[tokio::test]
async fn read_only_mode_and_ui() {
    let journal: Arc<dyn Journal> = Arc::new(InMemoryJournal::new());
    let agent = refund_agent(&journal, Arc::default(), vec![tool_call_response(&[("r1", "refund", json!({"id": 1}))])]);
    agent.run_with_id("held", "refund").await.unwrap();

    // No agents hosted: browse only.
    let base = serve(Server::new(journal.clone())).await;
    let c = Client { base: base.clone(), key: None, http: reqwest::Client::new() };
    assert_eq!(c.get("/v1/runs/held").await.1["hosted"], false);
    let (status, body) = c.post("/v1/runs/held/approvals/r1", json!({"approved": true})).await;
    assert_eq!(status, 409);
    assert!(body["error"].as_str().unwrap().contains("not hosted"));

    let html = reqwest::get(format!("{base}/ui")).await.unwrap().text().await.unwrap();
    assert!(html.contains("<title>Cairn Dev UI</title>"));
    assert_eq!(reqwest::get(format!("{base}/healthz")).await.unwrap().text().await.unwrap(), "ok");
}
