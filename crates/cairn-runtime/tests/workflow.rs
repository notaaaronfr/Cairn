use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};

use cairn_runtime::testing::{ScriptedModel, text_response, tool_call_response};
use cairn_runtime::*;
use schemars::JsonSchema;
use serde::Deserialize;
use serde_json::json;

fn journal() -> Arc<dyn Journal> {
    Arc::new(InMemoryJournal::new())
}

fn agent(name: &str, j: &Arc<dyn Journal>, responses: Vec<ModelResponse>) -> (Agent, Arc<ScriptedModel>) {
    let model = Arc::new(ScriptedModel::new(responses));
    let a = Agent::builder(name).model(model.clone()).shared_journal(j.clone()).build().unwrap();
    (a, model)
}

#[tokio::test]
async fn steps_are_replayed_not_rerun_after_a_crash() {
    let j = journal();
    let calls = Arc::new(AtomicU32::new(0));
    let crash = Arc::new(AtomicBool::new(true));
    let wf = {
        let (calls, crash) = (calls.clone(), crash.clone());
        Workflow::new("pipeline", j.clone(), move |ctx: WorkflowCtx, input: String| {
            let (calls, crash) = (calls.clone(), crash.clone());
            async move {
                let c = calls.clone();
                let a: String = ctx
                    .step("fetch", move || async move {
                        c.fetch_add(1, Ordering::SeqCst);
                        Ok::<_, String>(format!("fetched {input}"))
                    })
                    .await?;
                if crash.load(Ordering::SeqCst) {
                    // Simulates the process dying between steps.
                    return Err(WorkflowError::Journal("simulated crash".into()));
                }
                let c = calls.clone();
                let b: String = ctx
                    .step("fetch", move || async move {
                        c.fetch_add(1, Ordering::SeqCst);
                        Ok::<_, String>("second".to_owned())
                    })
                    .await?;
                Ok(format!("{a} + {b}"))
            }
        })
    };

    assert!(wf.run_with_id("wf-1", "x".into()).await.is_err());
    assert_eq!(calls.load(Ordering::SeqCst), 1);

    crash.store(false, Ordering::SeqCst);
    let out = wf.resume("wf-1").await.unwrap();
    assert_eq!(out.output().unwrap(), "fetched x + second");
    assert_eq!(calls.load(Ordering::SeqCst), 2, "first step was replayed, not re-run");

    // Repeated names get stable ids; a finished run resumes to its result.
    let events = j.load("wf-1").await.unwrap();
    let steps: Vec<_> = events
        .iter()
        .filter_map(|e| match &e.kind {
            EventKind::StepCompleted { step, .. } => Some(step.clone()),
            _ => None,
        })
        .collect();
    assert_eq!(steps, ["fetch", "fetch#2"]);
    assert_eq!(wf.resume("wf-1").await.unwrap().output().unwrap(), "fetched x + second");
    assert!(matches!(wf.run_with_id("wf-1", "x".into()).await, Err(WorkflowError::RunExists(_))));
}

#[tokio::test]
async fn sequential_and_concurrent_agents_with_cost_rollup() {
    let j = journal();
    let (research, _) = agent("research", &j, vec![text_response("facts")]);
    let (write, write_model) = agent("write", &j, vec![text_response("draft")]);
    let (legal, _) = agent("legal", &j, vec![text_response("ok-legal")]);
    let (tone, _) = agent("tone", &j, vec![text_response("ok-tone")]);

    let wf = Workflow::new("doc", j.clone(), move |ctx: WorkflowCtx, input: String| {
        let (research, write, legal, tone) = (
            research.to_builder().build().unwrap(),
            write.to_builder().build().unwrap(),
            legal.to_builder().build().unwrap(),
            tone.to_builder().build().unwrap(),
        );
        async move {
            let draft = ctx.sequential("draft", &[&research, &write], &input).await?;
            let reviews = ctx.concurrent("review", &[&legal, &tone], &draft.text()).await?;
            Ok(reviews.iter().map(AgentResult::text).collect::<Vec<_>>())
        }
    });
    let out = wf.run_with_id("doc-1", "topic".into()).await.unwrap();
    assert_eq!(out.output().unwrap(), &["ok-legal", "ok-tone"]);
    assert_eq!(write_model.requests()[0].messages[0].text(), "facts", "output flows to the next agent");
    // 4 child runs at 120 tokens / $0.00012 each.
    assert_eq!(out.usage.total(), 480);
    assert!((out.cost_usd - 0.00048).abs() < 1e-12);
    assert!(!j.load("doc-1.draft-0-research").await.unwrap().is_empty(), "child runs have derived ids");
}

#[derive(Deserialize, JsonSchema)]
struct Id {
    id: u32,
}

#[tokio::test]
async fn child_approval_suspends_and_resumes_the_workflow() {
    let j = journal();
    let refunds = Arc::new(AtomicU32::new(0));
    let r = refunds.clone();
    let refund = tool("refund", "Refund", Effect::Irreversible, move |i: Id, _ctx| {
        let r = r.clone();
        async move {
            assert_eq!(i.id, 7);
            r.fetch_add(1, Ordering::SeqCst);
            Ok(json!("refunded"))
        }
    });
    let support = Agent::builder("support")
        .model(ScriptedModel::new(vec![
            tool_call_response(&[("r1", "refund", json!({"id": 7}))]),
            text_response("done"),
        ]))
        .tool(refund)
        .shared_journal(j.clone())
        .build()
        .unwrap();
    let support2 = support.to_builder().build().unwrap();
    let wf = Workflow::new("case", j.clone(), move |ctx: WorkflowCtx, input: String| {
        let support = support2.to_builder().build().unwrap();
        async move { ctx.agent("resolve", &support, input).await?.into_text() }
    });

    let out = wf.run_with_id("case-1", "refund order 7".into()).await.unwrap();
    let WorkflowStatus::Suspended(waiting) = &out.status else { panic!("{:?}", out.status) };
    assert_eq!(waiting, &["case-1.resolve".to_owned()]);
    assert_eq!(refunds.load(Ordering::SeqCst), 0);

    support.decide("case-1.resolve", "r1", true, Some("ops"), None).await.unwrap();
    let out = wf.resume("case-1").await.unwrap();
    assert_eq!(out.output().unwrap(), "done");
    assert_eq!(refunds.load(Ordering::SeqCst), 1);
}

#[tokio::test]
async fn workflow_budget_caps_children() {
    let j = journal();
    // Children refuse to call a model unless the budget buys 50 output tokens.
    let child = |name: &str, r| {
        let (a, m) = agent(name, &j, r);
        (a.to_builder().budget(Budget::default().min_output_tokens(50)).build().unwrap(), m)
    };
    let (first, _) = child("first", vec![text_response("a")]);
    let (second, second_model) = child("second", vec![text_response("b")]);
    // One step costs $0.00012; the workflow may spend $0.00015 in total, so
    // the second child's capped budget cannot buy 50 output tokens.
    let wf = Workflow::new("budgeted", j.clone(), move |ctx: WorkflowCtx, input: String| {
        let (first, second) = (first.to_builder().build().unwrap(), second.to_builder().build().unwrap());
        async move { ctx.sequential("s", &[&first, &second], &input).await.map(|r| r.text()) }
    })
    .max_usd(0.00015);
    let out = wf.run("go".into()).await.unwrap();
    assert!(matches!(out.status, WorkflowStatus::Failed(ref e) if e.contains("BudgetExceeded")), "{:?}", out.status);
    assert!(second_model.requests().is_empty(), "the second child could not afford a call");
    assert!(out.cost_usd <= 0.00015);
}

#[tokio::test]
async fn handoff_passes_control_with_a_note() {
    let j = journal();
    let (triage, _) = agent(
        "triage",
        &j,
        vec![
            tool_call_response(&[("h1", "handoff_to_billing", json!({"note": "customer wants a refund for #42"}))]),
            text_response("Passing you to billing."),
        ],
    );
    let (billing, billing_model) = agent("billing", &j, vec![text_response("Refund for #42 issued.")]);
    let wf = Workflow::new("support", j.clone(), move |ctx: WorkflowCtx, input: String| {
        let (triage, billing) = (triage.to_builder().build().unwrap(), billing.to_builder().build().unwrap());
        async move { Ok(ctx.handoff("h", &[&triage, &billing], "triage", &input, 3).await?.text()) }
    });
    let out = wf.run("I was double charged".into()).await.unwrap();
    assert_eq!(out.output().unwrap(), "Refund for #42 issued.");
    let seen = billing_model.requests()[0].messages[0].text();
    assert!(seen.contains("I was double charged") && seen.contains("refund for #42"), "{seen}");
    // Billing was offered a way back to triage.
    assert!(billing_model.requests()[0].tools.iter().any(|t| t.name == "handoff_to_triage"));
}

#[tokio::test]
async fn group_chats_round_robin_and_managed() {
    let j = journal();
    let (writer, _) = agent("writer", &j, vec![text_response("draft v1"), text_response("draft v2")]);
    let (critic, _) = agent("critic", &j, vec![text_response("needs work"), text_response("TERMINATE: good")]);
    let wf = Workflow::new("rr", j.clone(), move |ctx: WorkflowCtx, task: String| {
        let (w, c) = (writer.to_builder().build().unwrap(), critic.to_builder().build().unwrap());
        async move {
            let r = ctx.group_chat("chat", GroupChat::round_robin(&[&w, &c], 10), &task).await?;
            Ok((r.rounds, r.answer, r.transcript.len()))
        }
    });
    let out = wf.run("write a haiku".into()).await.unwrap();
    assert_eq!(out.output().unwrap(), &(4, Some("TERMINATE: good".to_owned()), 4));

    let (writer, _) = agent("writer", &j, vec![text_response("final haiku")]);
    let (critic, _) = agent("critic", &j, vec![]);
    let (manager, _) = agent(
        "manager",
        &j,
        vec![
            text_response(r#"{"done": false, "next": "writer"}"#),
            text_response(r#"{"done": true, "answer": "final haiku"}"#),
        ],
    );
    let wf = Workflow::new("managed", j.clone(), move |ctx: WorkflowCtx, task: String| {
        let (w, c, m) = (
            writer.to_builder().build().unwrap(),
            critic.to_builder().build().unwrap(),
            manager.to_builder().build().unwrap(),
        );
        async move {
            let r = ctx.group_chat("chat", GroupChat::managed(&[&w, &c], &m, 5), &task).await?;
            Ok(r.answer)
        }
    });
    let out = wf.run("write a haiku".into()).await.unwrap();
    assert_eq!(out.output().unwrap().as_deref(), Some("final haiku"));
}

#[tokio::test]
async fn agent_as_tool_charges_parent_and_resumes_by_id() {
    let j = journal();
    let (child, child_model) = agent("researcher", &j, vec![text_response("42 is the answer")]);
    let parent = Agent::builder("lead")
        .model(ScriptedModel::new(vec![
            tool_call_response(&[("c1", "research", json!({"task": "find the answer"}))]),
            text_response("The answer is 42."),
        ]))
        .tool(AgentTool::new(child, "research", "Research a question"))
        .shared_journal(j.clone())
        .budget(Budget::default().max_usd(1.0))
        .build()
        .unwrap();
    let out = parent.run("what is the answer?").await.unwrap();
    assert_eq!(out.output(), Some("The answer is 42."));
    assert_eq!(child_model.requests()[0].messages[0].text(), "find the answer");
    // Parent: 2 calls; child: 1 call — all charged to the parent.
    assert_eq!(out.usage.total(), 360);
    assert!((out.cost_usd - 0.00036).abs() < 1e-12, "{}", out.cost_usd);

    // A retried call (e.g. after a parent crash) resumes the same child run.
    let tool = AgentTool::new(agent("researcher", &j, vec![]).0, "research", "r");
    let ctx = ToolCtx::new(&out.run_id, "c1");
    assert_eq!(tool.call(json!({"task": "find the answer"}), &ctx).await.unwrap(), json!("42 is the answer"));
}
