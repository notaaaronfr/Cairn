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
