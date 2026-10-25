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

#[tokio::test]
async fn streams_deltas_and_journals_only_completed_steps() {
    use std::sync::Mutex;
    let seen = Arc::new(Mutex::new(Vec::new()));
    let sink = seen.clone();
    let agent = Agent::builder("t")
        .model(ScriptedModel::new(vec![
            tool_call_response(&[("c1", "lookup", json!({"id": 1}))]),
            text_response("streamed answer"),
        ]))
        .tool(counting_tool("lookup", Effect::Read, Arc::new(AtomicU32::new(0))))
        .on_delta(move |e| sink.lock().unwrap().push(e.clone()))
        .build()
        .unwrap();

    let out = agent.run("go").await.unwrap();
    assert_eq!(out.output(), Some("streamed answer"));
    let seen = seen.lock().unwrap().clone();
    assert!(seen.iter().any(|e| matches!(e, StreamEvent::ToolCallStart { name, .. } if name == "lookup")));
    assert!(seen.contains(&StreamEvent::TextDelta("streamed answer".into())));
    assert!(!seen.iter().any(|e| matches!(e, StreamEvent::Completed(_))), "Completed is not a delta");

    let events = agent.journal().load(&out.run_id).await.unwrap();
    assert_eq!(events.iter().filter(|e| matches!(e.kind, EventKind::ModelCalled { .. })).count(), 2);
}

/// A model with no pricing configured.
struct Unpriced(ScriptedModel);

#[async_trait::async_trait]
impl Model for Unpriced {
    fn id(&self) -> String {
        "unpriced:model".into()
    }
    async fn complete(&self, r: &ModelRequest) -> Result<ModelResponse, ModelError> {
        self.0.complete(r).await
    }
}

/// Fails with a retryable error so fallback moves on.
struct Down;

#[async_trait::async_trait]
impl Model for Down {
    fn id(&self) -> String {
        "down:model".into()
    }
    async fn complete(&self, _: &ModelRequest) -> Result<ModelResponse, ModelError> {
        Err(ModelError::Api { status: 503, body: "overloaded".into() })
    }
    fn pricing(&self) -> Option<Pricing> {
        Some(Pricing::new(1000.0, 1000.0))
    }
}

#[test]
fn usd_budget_refuses_unpriced_models() {
    let err = Agent::builder("t")
        .model(Unpriced(ScriptedModel::new(vec![])))
        .budget(Budget::default().max_usd(1.0))
        .build()
        .err()
        .expect("must refuse");
    assert!(err.to_string().contains("unpriced:model"), "{err}");

    // Also when the unpriced model hides inside a fallback chain.
    let chain = FallbackModel::new(vec![Arc::new(ScriptedModel::new(vec![])), Arc::new(Unpriced(ScriptedModel::new(vec![])))]);
    let err = Agent::builder("t").model(chain).budget(Budget::default().max_usd(1.0)).build().err().expect("must refuse");
    assert!(err.to_string().contains("unpriced:model"), "{err}");

    // Without a USD budget, unpriced models are fine.
    assert!(Agent::builder("t").model(Unpriced(ScriptedModel::new(vec![]))).build().is_ok());
    // A genuinely free model opts in explicitly.
    let free = Priced::new(Unpriced(ScriptedModel::new(vec![])), Pricing::new(0.0, 0.0));
    assert!(Agent::builder("t").model(free).budget(Budget::default().max_usd(1.0)).build().is_ok());
}

#[tokio::test]
async fn usd_budget_actually_stops_the_run() {
    // ScriptedModel: $1/Mtok, 120 tokens per call => $0.00012 per step.
    let looping: Vec<_> = (0..10).map(|i| tool_call_response(&[(&format!("c{i}"), "lookup", json!({"id": i}))])).collect();
    let agent = Agent::builder("t")
        .model(ScriptedModel::new(looping))
        .tool(counting_tool("lookup", Effect::Read, Arc::new(AtomicU32::new(0))))
        .budget(Budget::default().max_usd(0.0003))
        .build()
        .unwrap();
    let out = agent.run("go").await.unwrap();
    assert!(matches!(out.status, Status::BudgetExceeded { .. }), "{:?}", out.status);
    // Two steps cost $0.00024; a third could not fit in what is left, so the
    // run stops before calling rather than overshooting.
    assert_eq!(out.steps, 2);
    assert!(out.cost_usd <= 0.0003, "{}", out.cost_usd);
}

#[tokio::test]
async fn fallback_prices_the_model_that_served() {
    let mut reply = text_response("hi");
    reply.model = "provider-name-2026-10-01".into(); // provider-reported name differs from id
    let cheap = Arc::new(Priced::new(ScriptedModel::new(vec![reply]), Pricing::new(10.0, 10.0)));
    let chain = FallbackModel::new(vec![Arc::new(Down), cheap]);
    let agent = Agent::builder("t").model(chain).budget(Budget::default().max_usd(1.0)).build().unwrap();
    let out = agent.run("go").await.unwrap();
    // 120 tokens at $10/Mtok on the serving model, not $1000/Mtok (the max).
    assert!((out.cost_usd - 0.0012).abs() < 1e-12, "{}", out.cost_usd);
}

#[tokio::test]
async fn hung_tool_times_out_and_run_continues() {
    use std::time::{Duration, Instant};
    let hang = tool("fetch_page", "Fetch a page", Effect::Read, |_i: Id, _ctx| async move {
        tokio::time::sleep(Duration::from_secs(30)).await;
        Ok(json!("never"))
    })
    .with_timeout(Duration::from_millis(50));
    let slow_default = tool("slow", "Slow", Effect::Read, |_i: Id, _ctx| async move {
        tokio::time::sleep(Duration::from_secs(30)).await;
        Ok(json!("never"))
    });

    let model = Arc::new(ScriptedModel::new(vec![
        tool_call_response(&[("c1", "fetch_page", json!({"id": 1})), ("c2", "slow", json!({"id": 2}))]),
        text_response("page was unreachable"),
    ]));
    let agent = Agent::builder("t")
        .model(model.clone())
        .tool(hang)
        .tool(slow_default)
        .tool_timeout(Duration::from_millis(80)) // agent default, used by `slow`
        .build()
        .unwrap();

    let started = Instant::now();
    let out = agent.run("go").await.unwrap();
    assert!(started.elapsed() < Duration::from_secs(2), "took {:?}", started.elapsed());
    assert_eq!(out.output(), Some("page was unreachable"));

    let requests = model.requests();
    let results = &requests[1].messages.last().unwrap().content;
    for (r, name) in results.iter().zip(["fetch_page", "slow"]) {
        let Content::ToolResult { content, is_error, .. } = r else { panic!("not a tool result") };
        assert!(*is_error);
        assert!(content["error"].as_str().unwrap().contains(&format!("`{name}` timed out")), "{content}");
    }
}

mod usable_fallback {
    use super::*;

    /// A ScriptedModel with its own id and price, standing in for K2 or Haiku.
    struct Named {
        id: &'static str,
        inner: ScriptedModel,
        price: f64,
    }

    #[async_trait::async_trait]
    impl Model for Named {
        fn id(&self) -> String {
            self.id.into()
        }
        async fn complete(&self, r: &ModelRequest) -> Result<ModelResponse, ModelError> {
            self.inner.complete(r).await
        }
        fn pricing(&self) -> Option<Pricing> {
            Some(Pricing::new(self.price, self.price))
        }
    }

    fn k2(responses: Vec<ModelResponse>) -> Arc<Named> {
        Arc::new(Named { id: "k2:K2", inner: ScriptedModel::new(responses), price: 0.0 })
    }
    fn haiku(responses: Vec<ModelResponse>) -> Arc<Named> {
        Arc::new(Named { id: "anthropic:haiku", inner: ScriptedModel::new(responses), price: 10.0 })
    }

    fn truncated(text: &str) -> ModelResponse {
        ModelResponse { stop_reason: StopReason::MaxTokens, ..text_response(text) }
    }

    async fn run(chain: FallbackModel) -> (Agent, RunOutcome) {
        let agent = Agent::builder("t").model(chain).budget(Budget::default().max_usd(1.0)).build().unwrap();
        let out = agent.run("go").await.unwrap();
        (agent, out)
    }

    async fn first_model_call(agent: &Agent, run_id: &str) -> ModelResponse {
        let events = agent.journal().load(run_id).await.unwrap();
        events
            .into_iter()
            .find_map(|e| match e.kind {
                EventKind::ModelCalled { response, .. } => Some(response),
                _ => None,
            })
            .unwrap()
    }

    #[tokio::test]
    async fn truncated_reply_falls_back_and_is_still_paid_for() {
        let (k, h) = (k2(vec![truncated("The handoff is: {\"summ")]), haiku(vec![text_response("complete answer")]));
        let (agent, out) = run(FallbackModel::new(vec![k, h.clone()])).await;
        assert_eq!(out.output(), Some("complete answer"));

        let resp = first_model_call(&agent, &out.run_id).await;
        assert_eq!(resp.served_by.as_deref(), Some("anthropic:haiku"));
        assert_eq!(resp.rejected.len(), 1);
        assert_eq!(resp.rejected[0].model, "k2:K2");
        assert!(resp.rejected[0].reason.contains("cut off"));
        // Both attempts' tokens are counted; only Haiku's cost money.
        assert_eq!(out.usage.total(), 240);
        assert!((out.cost_usd - 120.0 * 10.0 / 1e6).abs() < 1e-12, "{}", out.cost_usd);
    }

    #[tokio::test]
    async fn empty_reply_falls_back() {
        let (k, h) = (k2(vec![text_response("   ")]), haiku(vec![text_response("real answer")]));
        let (_, out) = run(FallbackModel::new(vec![k, h])).await;
        assert_eq!(out.output(), Some("real answer"));
    }

    #[tokio::test]
    async fn bad_or_incomplete_json_falls_back_good_json_is_kept() {
        let fields = ["summary", "work", "risks", "confidence"];
        let partial = text_response(r#"{"summary": "done", "work": "x"}"#);
        let good = text_response(r#"```json
{"summary": "s", "work": "w", "risks": ["r"], "confidence": 0.8}
```"#);

        let (agent, out) =
            run(FallbackModel::new(vec![k2(vec![partial]), haiku(vec![good.clone()])]).require_json_fields(&fields)).await;
        let resp = first_model_call(&agent, &out.run_id).await;
        assert!(resp.rejected[0].reason.contains("missing required field(s): risks, confidence"), "{:?}", resp.rejected);
        assert_eq!(resp.served_by.as_deref(), Some("anthropic:haiku"));

        // When K2 gets it right, Haiku is never called.
        let h = haiku(vec![]);
        let (_, out) = run(FallbackModel::new(vec![k2(vec![good]), h.clone()]).require_json_fields(&fields)).await;
        assert!(out.output().is_some());
        assert!(h.inner.requests().is_empty());
    }

    #[tokio::test]
    async fn tool_call_turns_are_not_held_to_the_json_rule() {
        let k = k2(vec![tool_call_response(&[("c1", "lookup", json!({"id": 1}))]), text_response(r#"{"summary":"s"}"#)]);
        let chain = FallbackModel::new(vec![k, haiku(vec![])]).require_json_fields(&["summary"]);
        let agent = Agent::builder("t")
            .model(chain)
            .tool(counting_tool("lookup", Effect::Read, Arc::new(AtomicU32::new(0))))
            .build()
            .unwrap();
        let out = agent.run("go").await.unwrap();
        assert_eq!(out.output(), Some(r#"{"summary":"s"}"#));
    }

    #[tokio::test]
    async fn all_unusable_fails_the_run_but_keeps_the_spend() {
        let (k, h) = (k2(vec![text_response("")]), haiku(vec![truncated("abc")]));
        let (agent, out) = run(FallbackModel::new(vec![k, h])).await;
        let Status::Failed { error } = &out.status else { panic!("{:?}", out.status) };
        assert!(error.contains("k2:K2: reply is empty") && error.contains("anthropic:haiku: reply was cut off"), "{error}");
        assert_eq!(out.usage.total(), 240);
        assert!(out.cost_usd > 0.0);
        let state = agent.state(&out.run_id).await.unwrap();
        assert_eq!(state.cost_usd, out.cost_usd);
    }

    #[tokio::test]
    async fn streaming_never_leaks_a_rejected_answer() {
        use std::sync::Mutex;
        let seen = Arc::new(Mutex::new(String::new()));
        let sink = seen.clone();
        let chain = FallbackModel::new(vec![k2(vec![truncated("BAD PARTIAL")]), haiku(vec![text_response("good")])]);
        let agent = Agent::builder("t")
            .model(chain)
            .on_delta(move |e| {
                if let StreamEvent::TextDelta(t) = e {
                    sink.lock().unwrap().push_str(t);
                }
            })
            .build()
            .unwrap();
        let out = agent.run("go").await.unwrap();
        assert_eq!(out.output(), Some("good"));
        assert_eq!(*seen.lock().unwrap(), "good");
    }
}

mod budget_cap {
    use super::*;

    fn looping(n: usize) -> Vec<ModelResponse> {
        (0..n).map(|i| tool_call_response(&[(&format!("c{i}"), "lookup", json!({"id": i}))])).collect()
    }

    fn agent(model: Arc<ScriptedModel>, budget: Budget) -> Agent {
        Agent::builder("t")
            .model(model)
            .tool(counting_tool("lookup", Effect::Read, Arc::new(AtomicU32::new(0))))
            .max_tokens(4096)
            .budget(budget)
            .build()
            .unwrap()
    }

    #[tokio::test]
    async fn token_budget_shrinks_each_steps_answer_limit() {
        let model = Arc::new(ScriptedModel::new(looping(10)));
        let out = agent(model.clone(), Budget::default().max_tokens(1500)).run("go").await.unwrap();
        assert!(matches!(out.status, Status::BudgetExceeded { .. }), "{:?}", out.status);

        let caps: Vec<u32> = model.requests().iter().map(|r| r.max_tokens).collect();
        assert!(caps[0] < 1500, "first cap {} must fit the 1500-token budget", caps[0]);
        assert!(caps.windows(2).all(|w| w[1] < w[0]), "caps must shrink as the budget is used: {caps:?}");
        // Each call's used tokens + its cap never exceeded what was left.
        assert!(out.usage.total() <= 1500, "{}", out.usage.total());
    }

    #[tokio::test]
    async fn usd_budget_shrinks_answer_limit() {
        // $1/Mtok: $0.0006 buys 600 tokens in total.
        let model = Arc::new(ScriptedModel::new(looping(10)));
        let out = agent(model.clone(), Budget::default().max_usd(0.0006)).run("go").await.unwrap();
        assert!(matches!(out.status, Status::BudgetExceeded { .. }));
        let first = model.requests()[0].max_tokens;
        assert!(first < 600, "{first}");
        assert!(out.cost_usd <= 0.0006 + 1e-12, "{}", out.cost_usd);
    }

    #[tokio::test]
    async fn stops_before_calling_when_budget_cannot_buy_minimum() {
        let model = Arc::new(ScriptedModel::new(looping(1)));
        let out = agent(model.clone(), Budget::default().max_tokens(5_000).min_output_tokens(10_000)).run("go").await.unwrap();
        let Status::BudgetExceeded { reason } = &out.status else { panic!("{:?}", out.status) };
        assert!(reason.contains("minimum 10000"), "{reason}");
        assert!(model.requests().is_empty(), "no model call may be made");
    }

    #[tokio::test]
    async fn fallback_cap_accounts_for_every_paid_attempt() {
        let single = Arc::new(ScriptedModel::new(vec![text_response("a")]));
        agent(single.clone(), Budget::default().max_usd(0.001)).run("go").await.unwrap();

        let a = Arc::new(ScriptedModel::new(vec![text_response("a")]));
        let chain = FallbackModel::new(vec![a.clone(), Arc::new(ScriptedModel::new(vec![]))]);
        Agent::builder("t").model(chain).max_tokens(4096).budget(Budget::default().max_usd(0.001)).build().unwrap().run("go").await.unwrap();

        let (one, two) = (single.requests()[0].max_tokens, a.requests()[0].max_tokens);
        assert!(two < one, "a two-model chain must leave room for both attempts: {two} vs {one}");
    }
}

mod finish_schema {
    use super::*;

    #[derive(Debug, Deserialize, JsonSchema, PartialEq)]
    struct Handoff {
        summary: String,
        work: String,
        risks: Vec<String>,
        /// 0.0 – 1.0
        #[schemars(range(min = 0.0, max = 1.0))]
        confidence: f64,
    }

    const GOOD: &str = r#"{"summary":"Fixed login","work":"Patched session expiry","risks":["sessions reset once"],"confidence":0.9}"#;

    fn agent(model: Arc<ScriptedModel>) -> Agent {
        Agent::builder("worker")
            .model(model)
            .system("You are a WhatsTask worker.")
            .tool(counting_tool("lookup", Effect::Read, Arc::new(AtomicU32::new(0))))
            .finish_as::<Handoff>()
            .build()
            .unwrap()
    }

    #[tokio::test]
    async fn valid_handoff_completes_with_structured_output() {
        let model = Arc::new(ScriptedModel::new(vec![
            tool_call_response(&[("c1", "lookup", json!({"id": 1}))]),
            text_response(&format!("Done!\n```json\n{GOOD}\n```")),
        ]));
        let out = agent(model.clone()).run("fix login").await.unwrap();
        let handoff: Handoff = out.output_as().unwrap().unwrap();
        assert_eq!(handoff.risks, ["sessions reset once"]);

        let system = model.requests()[0].system.clone().unwrap();
        assert!(system.starts_with("You are a WhatsTask worker."));
        assert!(system.contains("\"risks\"") && system.contains("only a JSON object"), "{system}");
    }

    #[tokio::test]
    async fn missing_risks_gets_one_corrective_retry() {
        let model = Arc::new(ScriptedModel::new(vec![
            text_response(r#"{"summary":"Fixed login","work":"Patched","confidence":0.9}"#),
            text_response(GOOD),
        ]));
        let agent = agent(model.clone());
        let out = agent.run("fix login").await.unwrap();
        assert!(out.structured.is_some());

        let correction = model.requests()[1].messages.last().unwrap().text();
        assert!(correction.contains("rejected") && correction.contains("risks"), "{correction}");
        let events = agent.journal().load(&out.run_id).await.unwrap();
        assert_eq!(events.iter().filter(|e| matches!(e.kind, EventKind::FinishRejected { .. })).count(), 1);
    }

    #[tokio::test]
    async fn fails_after_retry_and_validates_more_than_presence() {
        // Second answer has every field but confidence is out of range.
        let model = Arc::new(ScriptedModel::new(vec![
            text_response("I fixed it, trust me."),
            text_response(r#"{"summary":"s","work":"w","risks":[],"confidence":7}"#),
            text_response(GOOD),
        ]));
        let out = agent(model.clone()).run("fix").await.unwrap();
        let Status::Failed { error } = &out.status else { panic!("{:?}", out.status) };
        assert!(error.contains("after 1 retry") && error.contains("/confidence"), "{error}");
        assert_eq!(model.requests().len(), 2, "exactly one retry");
        assert!(out.structured.is_none());
    }

    #[tokio::test]
    async fn zero_retries_fails_immediately() {
        let model = Arc::new(ScriptedModel::new(vec![text_response("not json")]));
        let agent = Agent::builder("w").model(model).finish_as::<Handoff>().finish_retries(0).build().unwrap();
        let out = agent.run("x").await.unwrap();
        assert!(matches!(&out.status, Status::Failed { error } if error.contains("not valid JSON")));
    }

    #[test]
    fn invalid_schema_is_a_config_error() {
        let err = Agent::builder("w")
            .model(ScriptedModel::new(vec![]))
            .finish_schema(json!({"type": "not-a-type"}))
            .build()
            .err()
            .unwrap();
        assert!(err.to_string().contains("invalid finish schema"), "{err}");
    }
}

mod evals {
    use super::*;
    use cairn_runtime::eval::{Completed, EvalCase, Evaluator, FieldsMatch, Judge, LiveTools};

    const HANDOFF: &str = r#"{"summary":"Fixed","work":"Patched T-1","risks":["none"],"confidence":0.9}"#;

    struct Counters {
        reads: Arc<AtomicU32>,
        writes: Arc<AtomicU32>,
    }

    fn tools(c: &Counters) -> (impl Tool + 'static, impl Tool + 'static) {
        let reads = c.reads.clone();
        let get_task = tool("get_task", "Get a task", Effect::Read, move |i: Id, _ctx| {
            let reads = reads.clone();
            async move {
                reads.fetch_add(1, Ordering::SeqCst);
                Ok(json!({"id": i.id, "title": format!("live task {}", i.id)}))
            }
        });
        let writes = c.writes.clone();
        let submit = tool("submit_work", "Submit", Effect::Irreversible, move |i: Id, _ctx| {
            let writes = writes.clone();
            async move {
                writes.fetch_add(1, Ordering::SeqCst);
                Ok(json!({"submitted": i.id}))
            }
        });
        (get_task, submit)
    }

    /// Production agent ("Haiku") whose run gets recorded.
    async fn production(c: &Counters, journal: Arc<dyn Journal>) -> (Agent, String) {
        let (get_task, submit) = tools(c);
        let agent = Agent::builder("worker")
            .model(ScriptedModel::new(vec![
                tool_call_response(&[("c1", "get_task", json!({"id": 1}))]),
                tool_call_response(&[("c2", "submit_work", json!({"id": 1}))]),
                text_response(HANDOFF),
            ]))
            .tool(get_task)
            .tool(submit)
            .policy(AllowAll)
            .shared_journal(journal)
            .finish_schema(json!({"type": "object", "required": ["summary", "work", "risks", "confidence"]}))
            .build()
            .unwrap();
        let out = agent.run("Work on task 1").await.unwrap();
        assert!(out.structured.is_some());
        (agent, out.run_id)
    }

    #[tokio::test]
    async fn replay_on_candidate_never_repeats_side_effects() {
        let c = Counters { reads: Arc::default(), writes: Arc::default() };
        let journal: Arc<dyn Journal> = Arc::new(InMemoryJournal::new());
        let (prod, run_id) = production(&c, journal.clone()).await;
        assert_eq!((c.reads.load(Ordering::SeqCst), c.writes.load(Ordering::SeqCst)), (1, 1));
        let events_before = journal.load(&run_id).await.unwrap().len();

        // Candidate ("K2"): same reads/writes as recorded, plus an extra
        // unrecorded read and an unrecorded write.
        let k2 = Arc::new(ScriptedModel::new(vec![
            tool_call_response(&[("k1", "get_task", json!({"id": 1})), ("k2", "get_task", json!({"id": 2}))]),
            tool_call_response(&[("k3", "submit_work", json!({"id": 1})), ("k4", "submit_work", json!({"id": 2}))]),
            text_response(HANDOFF),
        ]));
        let judge = ScriptedModel::new(vec![text_response(r#"{"score": 0.8, "reason": "same outcome"}"#)]);
        let candidate = prod.to_builder().model(k2.clone()).build().unwrap();
        let cases = EvalCase::load(journal.as_ref(), std::slice::from_ref(&run_id)).await.unwrap();
        let report = Evaluator::new(candidate)
            .scorer(Completed)
            .scorer(FieldsMatch::new(&["summary", "risks"]))
            .scorer(Judge::new(judge, "Same task outcome and honest risks."))
            .run(&cases)
            .await;

        // No real write happened; the unrecorded read ran live.
        assert_eq!(c.writes.load(Ordering::SeqCst), 1, "replay must never execute a write");
        assert_eq!(c.reads.load(Ordering::SeqCst), 2, "only the unrecorded read ran live");

        let requests = k2.requests();
        let reads = &requests[1].messages.last().unwrap().content;
        assert!(matches!(&reads[0], Content::ToolResult { content, .. } if content["title"] == "live task 1"), "recorded result served");
        assert!(matches!(&reads[1], Content::ToolResult { content, .. } if content["title"] == "live task 2"));
        let writes = &requests[2].messages.last().unwrap().content;
        assert!(matches!(&writes[0], Content::ToolResult { content, is_error: false, .. } if content["submitted"] == 1));
        assert!(matches!(&writes[1], Content::ToolResult { content, is_error: true, .. }
            if content["error"].as_str().unwrap().contains("not executed")));

        // Scores and report.
        assert_eq!(report.mean("completed"), 1.0);
        assert_eq!(report.mean("fields_match(summary,risks)"), 1.0);
        assert_eq!(report.mean("judge"), 0.8);
        assert_eq!(report.pass_rate(0.8), 1.0);
        assert_eq!(report.pass_rate(0.9), 0.0);
        assert!(report.baseline_cost() > 0.0 && report.candidate_cost() > 0.0);
        let md = report.to_markdown();
        assert!(md.contains("| judge |") && md.contains("**Mean:**"), "{md}");

        // Production history untouched.
        assert_eq!(journal.load(&run_id).await.unwrap().len(), events_before);
    }

    #[tokio::test]
    async fn never_mode_blocks_unrecorded_reads_too() {
        let c = Counters { reads: Arc::default(), writes: Arc::default() };
        let journal: Arc<dyn Journal> = Arc::new(InMemoryJournal::new());
        let (prod, run_id) = production(&c, journal.clone()).await;
        let k2 = Arc::new(ScriptedModel::new(vec![tool_call_response(&[("k1", "get_task", json!({"id": 7}))]), text_response(HANDOFF)]));
        let cases = EvalCase::load(journal.as_ref(), &[run_id]).await.unwrap();
        Evaluator::new(prod.to_builder().model(k2.clone()).build().unwrap()).live_tools(LiveTools::Never).run(&cases).await;
        assert_eq!(c.reads.load(Ordering::SeqCst), 1, "no live read during replay");
    }

    #[tokio::test]
    async fn failing_candidate_scores_zero() {
        let c = Counters { reads: Arc::default(), writes: Arc::default() };
        let journal: Arc<dyn Journal> = Arc::new(InMemoryJournal::new());
        let (prod, run_id) = production(&c, journal.clone()).await;
        let k2 = ScriptedModel::new(vec![text_response("I did it"), text_response("still no json")]);
        let cases = EvalCase::load(journal.as_ref(), &[run_id]).await.unwrap();
        let report = Evaluator::new(prod.to_builder().model(k2).build().unwrap()).scorer(Completed).run(&cases).await;
        assert_eq!(report.mean("completed"), 0.0);
        assert_eq!(report.pass_rate(0.5), 0.0);
    }
}

mod leases {
    use super::*;
    use std::time::Duration;

    fn worker(name: &str, journal: Arc<dyn Journal>, refunds: Arc<AtomicU32>, responses: Vec<ModelResponse>) -> Agent {
        let refund = tool("refund", "Refund", Effect::Irreversible, move |_i: Id, _ctx| {
            let refunds = refunds.clone();
            async move {
                tokio::time::sleep(Duration::from_millis(150)).await; // long enough to overlap
                refunds.fetch_add(1, Ordering::SeqCst);
                Ok(json!("ok"))
            }
        });
        Agent::builder("support").model(ScriptedModel::new(responses)).tool(refund).shared_journal(journal).worker_id(name).build().unwrap()
    }

    #[tokio::test]
    async fn only_one_worker_drives_a_run() {
        let journal: Arc<dyn Journal> = Arc::new(InMemoryJournal::new());
        let refunds = Arc::new(AtomicU32::new(0));
        let a = worker("a", journal.clone(), refunds.clone(), vec![
            tool_call_response(&[("r1", "refund", json!({"id": 1}))]),
            text_response("done"),
        ]);
        let b = worker("b", journal.clone(), refunds.clone(), vec![text_response("done")]);

        let out = a.run("refund").await.unwrap();
        a.decide(&out.run_id, "r1", true, None, None).await.unwrap();
        let (ra, rb) = tokio::join!(a.resume(&out.run_id), b.resume(&out.run_id));

        assert_eq!(refunds.load(Ordering::SeqCst), 1, "the refund ran exactly once");
        let (won, lost) = if ra.is_ok() { (ra, rb) } else { (rb, ra) };
        assert_eq!(won.unwrap().output(), Some("done"));
        assert!(matches!(lost, Err(AgentError::Leased { .. })), "{lost:?}");

        // The lease is released when the run is done.
        assert!(journal.acquire_lease(&out.run_id, "someone-else", Duration::from_secs(1)).await.unwrap().is_ok());
    }

    #[tokio::test]
    async fn crashed_workers_lease_expires_and_another_takes_over() {
        let journal: Arc<dyn Journal> = Arc::new(InMemoryJournal::new());
        let refunds = Arc::new(AtomicU32::new(0));
        let a = worker("a", journal.clone(), refunds.clone(), vec![tool_call_response(&[("r1", "refund", json!({"id": 1}))])]);
        let out = a.run("refund").await.unwrap();
        a.decide(&out.run_id, "r1", true, None, None).await.unwrap();

        // Worker "crashed" took the lease and died without releasing it.
        journal.acquire_lease(&out.run_id, "crashed", Duration::from_millis(60)).await.unwrap().unwrap();
        let b = worker("b", journal.clone(), refunds.clone(), vec![text_response("recovered")]);
        let err = b.resume(&out.run_id).await.unwrap_err();
        assert!(matches!(&err, AgentError::Leased { owner, .. } if owner == "crashed"), "{err}");

        tokio::time::sleep(Duration::from_millis(80)).await;
        assert_eq!(b.resume(&out.run_id).await.unwrap().output(), Some("recovered"));
        assert_eq!(refunds.load(Ordering::SeqCst), 1);
    }

    #[tokio::test]
    async fn long_runs_keep_their_lease_by_renewing() {
        let journal: Arc<dyn Journal> = Arc::new(InMemoryJournal::new());
        let refunds = Arc::new(AtomicU32::new(0));
        // 30ms lease, 150ms tool: without renewal the lease would expire mid-call.
        let a = worker("a", journal.clone(), refunds.clone(), vec![
            tool_call_response(&[("r1", "refund", json!({"id": 1}))]),
            text_response("done"),
        ])
        .to_builder()
        .lease_ttl(Duration::from_millis(30))
        .build()
        .unwrap();
        let b = worker("b", journal.clone(), refunds.clone(), vec![text_response("stolen")]);

        let out = a.run("refund").await.unwrap();
        a.decide(&out.run_id, "r1", true, None, None).await.unwrap();
        let run_id = out.run_id.clone();
        let driving = tokio::spawn(async move { a.resume(&run_id).await });

        tokio::time::sleep(Duration::from_millis(100)).await; // > 3 lease lengths in
        let err = b.resume(&out.run_id).await.unwrap_err();
        assert!(matches!(&err, AgentError::Leased { owner, .. } if owner == "a"), "{err}");

        assert_eq!(driving.await.unwrap().unwrap().output(), Some("done"));
        assert_eq!(refunds.load(Ordering::SeqCst), 1);
    }
}
