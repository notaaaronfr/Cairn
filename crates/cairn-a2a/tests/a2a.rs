use std::sync::Arc;
use std::sync::atomic::{AtomicU32, Ordering};
use std::time::Duration;

use cairn_a2a::types::{ListTasksRequest, SendMessageConfiguration, codes};
use cairn_a2a::*;
use cairn_runtime::testing::{ScriptedModel, text_response, tool_call_response};
use cairn_runtime::{Agent, Effect, Status, tool};
use futures::StreamExt;
use schemars::JsonSchema;
use serde::Deserialize;
use serde_json::{Value, json};

async fn start(server: impl FnOnce(String) -> A2aServer) -> String {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let base = format!("http://{}", listener.local_addr().unwrap());
    let router = server(format!("{base}/a2a")).router();
    tokio::spawn(async move { axum::serve(listener, router).await.unwrap() });
    base
}

fn agent(responses: Vec<cairn_runtime::ModelResponse>) -> (Agent, Arc<ScriptedModel>) {
    let model = Arc::new(ScriptedModel::new(responses));
    (Agent::builder("support").model(model.clone()).build().unwrap(), model)
}

fn task(resp: SendMessageResponse) -> Task {
    match resp {
        SendMessageResponse::Task(t) => t,
        SendMessageResponse::Message(m) => panic!("expected a task, got message {m:?}"),
    }
}

async fn raw_rpc(url: &str, headers: &[(&str, &str)], body: Value) -> (u16, Value) {
    let mut req = reqwest::Client::new().post(url).json(&body);
    for (k, v) in headers {
        req = req.header(*k, *v);
    }
    let resp = req.send().await.unwrap();
    (resp.status().as_u16(), resp.json().await.unwrap_or(Value::Null))
}

#[tokio::test]
async fn serves_a_card_and_runs_multi_turn_conversations() {
    let (agent, model) =
        agent(vec![text_response("Order 42 shipped on Monday."), text_response("It arrives Thursday.")]);
    let base = start(|url| {
        A2aServer::new(agent, &url)
            .description("Answers order questions.")
            .skill(AgentSkill::new("orders", "Orders", "Look up orders").tags(["support"]))
    })
    .await;

    let client = A2aClient::discover(&base, [("x-test", "1")]).await.unwrap();
    let card = client.card().unwrap();
    assert_eq!(card.name, "support");
    assert_eq!(card.skills[0].id, "orders");
    assert_eq!(client.url(), format!("{base}/a2a"));

    // The card is the spec's JSON shape.
    let raw: Value = reqwest::get(format!("{base}/.well-known/agent-card.json")).await.unwrap().json().await.unwrap();
    assert_eq!(raw["supportedInterfaces"][0]["protocolBinding"], "JSONRPC");
    assert_eq!(raw["supportedInterfaces"][0]["protocolVersion"], "1.0");
    assert_eq!(raw["capabilities"]["streaming"], true);

    let first = task(client.send_text("Where is order 42?").await.unwrap());
    assert_eq!(first.status.state, TaskState::Completed);
    assert_eq!(first.text(), "Order 42 shipped on Monday.");
    assert_eq!(first.history.len(), 2);

    // Same context: the agent sees the whole conversation.
    let mut follow_up = Message::user("When will it arrive?");
    follow_up.context_id = Some(first.context_id.clone());
    let second = task(client.send_message(&SendMessageRequest::new(follow_up)).await.unwrap());
    assert_eq!(second.context_id, first.context_id);
    assert_ne!(second.id, first.id);
    assert_eq!(second.text(), "It arrives Thursday.");
    let texts: Vec<String> = model.requests()[1].messages.iter().map(|m| m.text()).collect();
    assert_eq!(texts, ["Where is order 42?", "Order 42 shipped on Monday.", "When will it arrive?"]);

    // Earlier turns stay readable as they were.
    let again = client.get_task(&first.id, None).await.unwrap();
    assert_eq!(again.text(), "Order 42 shipped on Monday.");
    assert_eq!(client.get_task(&first.id, Some(0)).await.unwrap().history.len(), 0);

    let listed = client
        .list_tasks(&ListTasksRequest { context_id: Some(first.context_id.clone()), ..Default::default() })
        .await
        .unwrap();
    assert_eq!(listed.tasks.len(), 2);
    assert!(listed.tasks.iter().all(|t| t.artifacts.is_empty()), "artifacts only on request");
}

#[tokio::test]
async fn a_retried_first_message_returns_the_same_task() {
    let (agent, model) = agent(vec![text_response("done")]);
    let base = start(|url| A2aServer::new(agent, &url)).await;
    let client = A2aClient::discover(&base, Vec::<(String, String)>::new()).await.unwrap();

    let message = Message::user("do it once");
    let a = task(client.send_message(&SendMessageRequest::new(message.clone())).await.unwrap());
    let b = task(client.send_message(&SendMessageRequest::new(message)).await.unwrap());
    assert_eq!(a.id, b.id);
    assert_eq!(b.status.state, TaskState::Completed);
    assert_eq!(model.requests().len(), 1, "the model ran once");
}

#[tokio::test]
async fn streams_progress_then_the_answer() {
    let (agent, _) = agent(vec![text_response("Streaming answer")]);
    let base = start(|url| A2aServer::new(agent, &url)).await;
    let client = A2aClient::discover(&base, Vec::<(String, String)>::new()).await.unwrap();

    let items: Vec<StreamResponse> = client
        .send_streaming_message(&SendMessageRequest::new(Message::user("hi")))
        .await
        .unwrap()
        .map(Result::unwrap)
        .collect()
        .await;
    assert!(matches!(&items[0], StreamResponse::Task(t) if t.status.state == TaskState::Submitted));
    let progress: String = items
        .iter()
        .filter_map(|i| match i {
            StreamResponse::StatusUpdate(u) if u.status.state == TaskState::Working => {
                u.status.message.as_ref().map(Message::text)
            }
            _ => None,
        })
        .collect();
    assert_eq!(progress, "Streaming answer");
    assert!(items.iter().any(
        |i| matches!(i, StreamResponse::ArtifactUpdate(a) if a.last_chunk && a.artifact.parts[0].text.as_deref() == Some("Streaming answer"))
    ));
    assert!(matches!(items.last().unwrap(), StreamResponse::StatusUpdate(u) if u.status.state == TaskState::Completed));
}

#[derive(Deserialize, JsonSchema)]
struct Refund {
    order: u32,
}

fn refund_agent(responses: Vec<cairn_runtime::ModelResponse>, runs: Arc<AtomicU32>) -> Agent {
    let refund = tool("refund", "Refund an order", Effect::Irreversible, move |r: Refund, _ctx| {
        let runs = runs.clone();
        async move {
            runs.fetch_add(1, Ordering::SeqCst);
            Ok(json!({"refunded": r.order}))
        }
    });
    Agent::builder("billing").model(Arc::new(ScriptedModel::new(responses))).tool(refund).build().unwrap()
}

#[tokio::test]
async fn clients_can_answer_approval_questions_when_allowed() {
    let runs = Arc::new(AtomicU32::new(0));
    let agent = refund_agent(
        vec![tool_call_response(&[("c1", "refund", json!({"order": 7}))]), text_response("Refunded order 7.")],
        runs.clone(),
    );
    let base = start(|url| A2aServer::new(agent, &url).client_approvals(true)).await;
    let client = A2aClient::discover(&base, Vec::<(String, String)>::new()).await.unwrap();

    let asked = task(client.send_text("refund order 7").await.unwrap());
    assert_eq!(asked.status.state, TaskState::InputRequired);
    assert!(asked.status.message.as_ref().unwrap().text().contains("`refund`"));

    let mut answer = Message::user("approve");
    answer.task_id = Some(asked.id.clone());
    let done = task(client.send_message(&SendMessageRequest::new(answer)).await.unwrap());
    assert_eq!(done.id, asked.id, "the answer continues the same task");
    assert_eq!(done.status.state, TaskState::Completed);
    assert_eq!(done.text(), "Refunded order 7.");
    assert_eq!(runs.load(Ordering::SeqCst), 1);
}

#[tokio::test]
async fn approvals_wait_for_an_operator_and_tasks_can_be_canceled() {
    let runs = Arc::new(AtomicU32::new(0));
    let agent = refund_agent(vec![tool_call_response(&[("c1", "refund", json!({"order": 7}))])], runs.clone());
    let journal = agent.journal().clone();
    let base = start(|url| A2aServer::new(agent, &url)).await;
    let client = A2aClient::discover(&base, Vec::<(String, String)>::new()).await.unwrap();

    let waiting = task(client.send_text("refund order 7").await.unwrap());
    assert_eq!(waiting.status.state, TaskState::Working);
    assert!(waiting.status.message.unwrap().text().contains("operator"));

    // The client cannot approve it.
    let mut answer = Message::user("approve");
    answer.task_id = Some(waiting.id.clone());
    let err = client.send_message(&SendMessageRequest::new(answer)).await.unwrap_err();
    assert!(matches!(err, A2aError::Rpc { code: codes::UNSUPPORTED_OPERATION, .. }), "{err}");

    let canceled = client.cancel_task(&waiting.id).await.unwrap();
    assert_eq!(canceled.status.state, TaskState::Canceled);
    let err = client.cancel_task(&waiting.id).await.unwrap_err();
    assert!(matches!(err, A2aError::Rpc { code: codes::TASK_NOT_CANCELABLE, .. }), "{err}");
    assert_eq!(runs.load(Ordering::SeqCst), 0);

    // The journal records it like any other run.
    let state = cairn_runtime::RunState::from_events(&journal.load(&canceled.context_id).await.unwrap()).unwrap();
    assert!(matches!(state.status, Status::Canceled { .. }));
}

#[tokio::test]
async fn return_immediately_hands_back_a_task_to_poll() {
    let (agent, _) = agent(vec![text_response("later")]);
    let base = start(|url| A2aServer::new(agent, &url)).await;
    let client = A2aClient::discover(&base, Vec::<(String, String)>::new()).await.unwrap();

    let mut req = SendMessageRequest::new(Message::user("go"));
    req.configuration = Some(SendMessageConfiguration { return_immediately: true, ..Default::default() });
    let mut t = task(client.send_message(&req).await.unwrap());
    for _ in 0..100 {
        if t.status.state.is_terminal() {
            break;
        }
        tokio::time::sleep(Duration::from_millis(20)).await;
        t = client.get_task(&t.id, None).await.unwrap();
    }
    assert_eq!(t.text(), "later");
}

#[tokio::test]
async fn protocol_errors_use_the_spec_codes() {
    let (agent, _) = agent(vec![]);
    let base = start(|url| A2aServer::new(agent, &url).api_key("secret")).await;
    let url = format!("{base}/a2a");
    let auth = [("authorization", "Bearer secret")];

    let card: Value = reqwest::get(format!("{base}/.well-known/agent-card.json")).await.unwrap().json().await.unwrap();
    assert_eq!(card["securitySchemes"]["bearer"]["httpAuthSecurityScheme"]["scheme"], "Bearer");

    let (status, _) =
        raw_rpc(&url, &[], json!({"jsonrpc": "2.0", "id": 1, "method": "GetTask", "params": {"id": "x:0"}})).await;
    assert_eq!(status, 401);

    let cases = [
        (json!({"jsonrpc": "2.0", "id": 1, "method": "GetTask", "params": {"id": "nope:0"}}), codes::TASK_NOT_FOUND),
        (json!({"jsonrpc": "2.0", "id": 2, "method": "Frobnicate"}), codes::METHOD_NOT_FOUND),
        (json!({"jsonrpc": "2.0", "id": 3, "method": "SendMessage", "params": {"nope": 1}}), codes::INVALID_PARAMS),
        (
            json!({"jsonrpc": "2.0", "id": 4, "method": "CreateTaskPushNotificationConfig", "params": {}}),
            codes::PUSH_NOTIFICATION_NOT_SUPPORTED,
        ),
        (
            json!({"jsonrpc": "2.0", "id": 5, "method": "GetExtendedAgentCard"}),
            codes::EXTENDED_AGENT_CARD_NOT_CONFIGURED,
        ),
    ];
    for (body, code) in cases {
        let id = body["id"].clone();
        let (status, resp) = raw_rpc(&url, &auth, body).await;
        assert_eq!(status, 200);
        assert_eq!(resp["id"], id);
        assert_eq!(resp["error"]["code"], code, "{resp}");
    }

    let (_, resp) = raw_rpc(
        &url,
        &[auth[0], ("A2A-Version", "0.3")],
        json!({"jsonrpc": "2.0", "id": 9, "method": "GetTask", "params": {"id": "x:0"}}),
    )
    .await;
    assert_eq!(resp["error"]["code"], codes::VERSION_NOT_SUPPORTED);

    let client = A2aClient::new(&url).header("authorization", "Bearer secret").unwrap();
    let err = client.get_task("missing:0", None).await.unwrap_err();
    assert!(matches!(err, A2aError::Rpc { code: codes::TASK_NOT_FOUND, .. }));
}

#[tokio::test]
async fn tasks_of_other_agents_on_the_same_journal_stay_hidden() {
    let (support, _) = agent(vec![text_response("secret answer")]);
    support.run_with_id("private-run", "internal question").await.unwrap();
    let journal = support.journal().clone();
    let other =
        Agent::builder("public").model(Arc::new(ScriptedModel::new(vec![]))).shared_journal(journal).build().unwrap();
    let base = start(|url| A2aServer::new(other, &url)).await;
    let client = A2aClient::discover(&base, Vec::<(String, String)>::new()).await.unwrap();

    let err = client.get_task("private-run:0", None).await.unwrap_err();
    assert!(matches!(err, A2aError::Rpc { code: codes::TASK_NOT_FOUND, .. }));
    let mut follow_up = Message::user("tell me");
    follow_up.context_id = Some("private-run".into());
    assert!(client.send_message(&SendMessageRequest::new(follow_up)).await.is_err());
    assert!(client.list_tasks(&ListTasksRequest::default()).await.unwrap().tasks.is_empty());
}

#[tokio::test]
async fn a_cairn_agent_calls_a_remote_agent_as_a_tool() {
    let (remote, remote_model) = agent(vec![text_response("Flight BA117 departs 08:40.")]);
    let base = start(|url| A2aServer::new(remote, &url).description("Knows flight times.")).await;
    let client = Arc::new(A2aClient::discover(&base, Vec::<(String, String)>::new()).await.unwrap());

    let planner_model = Arc::new(ScriptedModel::new(vec![
        tool_call_response(&[("c1", "support", json!({"message": "When does BA117 leave?"}))]),
        text_response("Your flight leaves at 08:40."),
    ]));
    let planner = Agent::builder("planner")
        .model(planner_model.clone())
        .tool(A2aTool::new(client).effect(Effect::Read))
        .build()
        .unwrap();

    let out = planner.run("Plan my morning").await.unwrap();
    assert_eq!(out.output(), Some("Your flight leaves at 08:40."));

    // The tool is described from the remote card, and its result reached the model.
    let req = &planner_model.requests()[1];
    assert!(req.tools[0].description.contains("Knows flight times."));
    let result = req.messages[2].content[0].clone();
    let cairn_runtime::Content::ToolResult { content, is_error, .. } = result else { panic!("expected a tool result") };
    assert!(!is_error);
    assert_eq!(content["state"], "completed");
    assert_eq!(content["answer"], "Flight BA117 departs 08:40.");
    assert!(content["context_id"].as_str().unwrap().starts_with("a2a-"));
    assert_eq!(remote_model.requests()[0].messages[0].text(), "When does BA117 leave?");
}
