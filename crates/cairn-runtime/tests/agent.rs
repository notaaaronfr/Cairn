use std::sync::Arc;
use std::sync::atomic::{AtomicU32, Ordering};

use cairn_runtime::testing::{ScriptedModel, text_response, tool_call_response};
use cairn_runtime::*;
use schemars::JsonSchema;
use serde::Deserialize;
use serde_json::{Value, json};

#[derive(Deserialize, JsonSchema)]
struct Id {
    id: u32,
}

fn counting_tool(name: &str, effect: Effect, counter: Arc<AtomicU32>) -> impl Tool + 'static {
    tool(name, "test tool", effect, move |i: Id, _ctx| {
        let counter = counter.clone();
        async move {
            counter.fetch_add(1, Ordering::SeqCst);
            Ok(json!({ "id": i.id, "ok": true }))
        }
    })
}

#[tokio::test]
async fn runs_tool_loop_to_completion() {
    let model = Arc::new(ScriptedModel::new(vec![
        tool_call_response(&[("c1", "lookup", json!({"id": 7}))]),
        text_response("done"),
    ]));
    let calls = Arc::new(AtomicU32::new(0));
    let agent = Agent::builder("t")
        .model(model.clone())
        .tool(counting_tool("lookup", Effect::Read, calls.clone()))
        .build()
        .unwrap();

    let out = agent.run("go").await.unwrap();
    assert_eq!(out.output(), Some("done"));
    assert_eq!(out.steps, 2);
    assert_eq!(calls.load(Ordering::SeqCst), 1);

    // Second request carries the tool result back to the model.
    let second = &model.requests()[1];
    assert_eq!(second.messages.len(), 3);
    assert!(matches!(
        &second.messages[2].content[0],
        Content::ToolResult { tool_use_id, is_error: false, .. } if tool_use_id == "c1"
    ));
    // Tool schema is derived from the input type.
    assert_eq!(second.tools[0].input_schema["properties"]["id"]["type"], "integer");
}

#[tokio::test]
async fn irreversible_tool_suspends_until_approved() {
    let model = Arc::new(ScriptedModel::new(vec![
        tool_call_response(&[("r1", "refund", json!({"id": 1}))]),
        text_response("refunded"),
    ]));
    let calls = Arc::new(AtomicU32::new(0));
    let agent = Agent::builder("t")
        .model(model)
        .tool(counting_tool("refund", Effect::Irreversible, calls.clone()))
        .build()
        .unwrap();

    let out = agent.run("refund").await.unwrap();
    assert_eq!(out.status, Status::AwaitingApproval);
    assert_eq!(out.awaiting_approval[0].id, "r1");
    assert_eq!(calls.load(Ordering::SeqCst), 0, "must not run before approval");

    // Resuming without a decision stays suspended and does not call the model.
    let again = agent.resume(&out.run_id).await.unwrap();
    assert_eq!(again.status, Status::AwaitingApproval);

    agent.decide(&out.run_id, "r1", true, Some("alice"), None).await.unwrap();
    let done = agent.resume(&out.run_id).await.unwrap();
    assert_eq!(done.output(), Some("refunded"));
    assert_eq!(calls.load(Ordering::SeqCst), 1);
}

#[tokio::test]
async fn rejected_call_is_reported_to_model() {
    let model = Arc::new(ScriptedModel::new(vec![
        tool_call_response(&[("r1", "refund", json!({"id": 1}))]),
        text_response("ok, not refunding"),
    ]));
    let calls = Arc::new(AtomicU32::new(0));
    let agent = Agent::builder("t")
        .model(model.clone())
        .tool(counting_tool("refund", Effect::Irreversible, calls.clone()))
        .build()
        .unwrap();

    let out = agent.run("refund").await.unwrap();
    agent.decide(&out.run_id, "r1", false, Some("bob"), Some("over limit")).await.unwrap();
    let done = agent.resume(&out.run_id).await.unwrap();

    assert_eq!(done.output(), Some("ok, not refunding"));
    assert_eq!(calls.load(Ordering::SeqCst), 0);
    let last = model.requests()[1].messages.last().unwrap().clone();
    assert!(matches!(&last.content[0], Content::ToolResult { is_error: true, .. }));
}

#[tokio::test]
async fn custom_policy_can_deny() {
    let model = Arc::new(ScriptedModel::new(vec![
        tool_call_response(&[("c1", "lookup", json!({"id": 666}))]),
        text_response("denied"),
    ]));
    let calls = Arc::new(AtomicU32::new(0));
    let agent = Agent::builder("t")
        .model(model)
        .tool(counting_tool("lookup", Effect::Read, calls.clone()))
        .policy(|_spec: &ToolSpec, input: &Value| {
            if input["id"] == 666 { Decision::Deny { reason: "forbidden id".into() } } else { Decision::Allow }
        })
        .build()
        .unwrap();

    let out = agent.run("go").await.unwrap();
    assert_eq!(out.output(), Some("denied"));
    assert_eq!(calls.load(Ordering::SeqCst), 0);
}

#[tokio::test]
async fn budget_stops_runaway_loops() {
    let looping: Vec<_> = (0..10).map(|i| tool_call_response(&[(&format!("c{i}"), "lookup", json!({"id": i}))])).collect();
    let agent = Agent::builder("t")
        .model(ScriptedModel::new(looping))
        .tool(counting_tool("lookup", Effect::Read, Arc::new(AtomicU32::new(0))))
        .budget(Budget::default().max_steps(3))
        .build()
        .unwrap();

    let out = agent.run("go").await.unwrap();
    assert!(matches!(out.status, Status::BudgetExceeded { .. }));
    assert_eq!(out.steps, 3);
}

#[tokio::test]
async fn model_failure_fails_run() {
    let agent = Agent::builder("t").model(ScriptedModel::new(vec![])).build().unwrap();
    let out = agent.run("go").await.unwrap();
    assert!(matches!(out.status, Status::Failed { .. }));
}

#[tokio::test]
async fn resumes_after_crash_from_disk_without_redoing_work() {
    let dir = tempfile::tempdir().unwrap();
    let calls = Arc::new(AtomicU32::new(0));

    // Process 1: model asks for two tools; the second needs approval, so the
    // run suspends after the first has executed. Then the process "dies".
    let run_id = {
        let agent = Agent::builder("t")
            .model(ScriptedModel::new(vec![tool_call_response(&[
                ("a", "lookup", json!({"id": 1})),
                ("b", "refund", json!({"id": 2})),
            ])]))
            .tool(counting_tool("lookup", Effect::Read, calls.clone()))
            .tool(counting_tool("refund", Effect::Irreversible, calls.clone()))
            .journal(JsonlJournal::new(dir.path()).unwrap())
            .build()
            .unwrap();
        let out = agent.run("go").await.unwrap();
        assert_eq!(out.status, Status::AwaitingApproval);
        out.run_id
    };
    assert_eq!(calls.load(Ordering::SeqCst), 1);

    // Process 2: fresh agent, fresh journal handle, same directory.
    let model = Arc::new(ScriptedModel::new(vec![text_response("all done")]));
    let agent = Agent::builder("t")
        .model(model.clone())
        .tool(counting_tool("lookup", Effect::Read, calls.clone()))
        .tool(counting_tool("refund", Effect::Irreversible, calls.clone()))
        .journal(JsonlJournal::new(dir.path()).unwrap())
        .build()
        .unwrap();
    agent.decide(&run_id, "b", true, None, None).await.unwrap();
    let out = agent.resume(&run_id).await.unwrap();

    assert_eq!(out.output(), Some("all done"));
    assert_eq!(calls.load(Ordering::SeqCst), 2, "lookup must not re-run");
    // Both results were delivered together in one user message.
    let req = &model.requests()[0];
    assert_eq!(req.messages.last().unwrap().content.len(), 2);

    // Time travel: state at every prefix is reconstructible and consistent.
    let events = agent.journal().load(&run_id).await.unwrap();
    for n in 1..=events.len() {
        let s = RunState::from_events(&events[..n]).unwrap();
        assert_eq!(s.next_seq, n as u64);
    }
    assert_eq!(RunState::from_events(&events).unwrap(), agent.state(&run_id).await.unwrap());
}

#[tokio::test]
async fn journal_rejects_conflicting_writers() {
    let j = InMemoryJournal::new();
    let e = |seq| Event {
        run_id: "r".into(),
        seq,
        at: chrono::Utc::now(),
        kind: EventKind::RunFailed { error: "x".into() },
    };
    j.append(&e(0)).await.unwrap();
    assert!(matches!(j.append(&e(0)).await, Err(JournalError::Conflict { .. })));
    assert!(matches!(j.append(&e(5)).await, Err(JournalError::Conflict { .. })));
}

#[test]
fn idempotency_key_is_stable_per_call() {
    assert_eq!(ToolCtx::new("run", "c1").idempotency_key, ToolCtx::new("run", "c1").idempotency_key);
    assert_ne!(ToolCtx::new("run", "c1").idempotency_key, ToolCtx::new("run", "c2").idempotency_key);
}
