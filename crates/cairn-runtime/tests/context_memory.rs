use std::sync::Arc;

use cairn_runtime::testing::{ScriptedModel, text_response, tool_call_response};
use cairn_runtime::*;
use schemars::JsonSchema;
use serde::Deserialize;
use serde_json::json;

#[derive(Deserialize, JsonSchema)]
struct Id {
    id: u32,
}

/// A lookup tool returning a large, distinctive payload.
fn lookup() -> impl Tool + 'static {
    tool("lookup", "Look up a record", Effect::Read, |i: Id, _ctx| async move {
        Ok(json!({ "id": i.id, "blob": format!("record-{}-", i.id).repeat(60) }))
    })
}

fn looping(n: u32) -> Vec<ModelResponse> {
    let mut r: Vec<_> =
        (1..=n).map(|i| tool_call_response(&[(&format!("c{i}"), "lookup", json!({"id": i}))])).collect();
    r.push(text_response("done"));
    r
}

#[tokio::test]
async fn compaction_summarizes_older_turns_and_is_journaled() {
    let main = Arc::new(ScriptedModel::new(looping(4)));
    let summarizer = Arc::new(ScriptedModel::new(vec![text_response("Looked up records 1-3; all exist."); 3]));
    let agent = Agent::builder("worker")
        .model(main.clone())
        .tool(lookup())
        .context(ContextPolicy::default().compact_at(500).keep_recent(2).summarize_with(summarizer.clone()))
        .budget(Budget::default().max_usd(1.0))
        .build()
        .unwrap();
    let out = agent.run("Check records 1 to 4").await.unwrap();
    assert_eq!(out.output(), Some("done"));

    // The summarizer saw a plain transcript of the older turns.
    let s = summarizer.requests();
    assert!(!s.is_empty(), "compaction happened");
    let transcript = s[0].messages[0].text();
    assert!(transcript.contains("[called lookup with {\"id\":1}]"), "{transcript}");

    // After compaction the main model sees task + summary, then recent turns.
    let requests = main.requests();
    let at = requests
        .iter()
        .position(|r| r.messages[0].text().contains("[Summary of earlier work"))
        .expect("compacted request");
    let after = &requests[at];
    assert!(after.messages[0].text().starts_with("Check records 1 to 4"));
    assert!(after.messages[0].text().contains("Looked up records 1-3"));
    assert_eq!(after.messages[1].role, Role::Assistant, "a kept turn starts with the assistant");
    // Without compaction this request would have had two more messages
    // (one more call and its result) than the one before it.
    assert!(after.messages.len() < requests[at - 1].messages.len() + 2);

    // Journaled, counted, and replayable.
    let events = agent.journal().load(&out.run_id).await.unwrap();
    let compactions = events
        .iter()
        .filter(|e| matches!(&e.kind, EventKind::ContextCompacted { summary, .. } if summary.contains("records 1-3")))
        .count();
    assert_eq!(compactions, s.len(), "every summary is journaled");
    assert_eq!(out.steps, 5);
    assert_eq!(out.usage.total(), (5 + compactions as u64) * 120, "main calls plus summaries are all counted");
    assert_eq!(RunState::from_events(&events).unwrap(), agent.state(&out.run_id).await.unwrap());
}

#[tokio::test]
async fn trimming_shortens_old_results_in_requests_only() {
    let main = Arc::new(ScriptedModel::new(looping(3)));
    let agent = Agent::builder("worker")
        .model(main.clone())
        .tool(lookup())
        .context(ContextPolicy::default().max_tool_result_chars(40).keep_full_results(1))
        .build()
        .unwrap();
    let out = agent.run("go").await.unwrap();

    let last = main.requests().pop().unwrap();
    let results: Vec<&serde_json::Value> = last
        .messages
        .iter()
        .flat_map(|m| m.content.iter())
        .filter_map(|c| match c {
            Content::ToolResult { content, .. } => Some(content),
            _ => None,
        })
        .collect();
    assert_eq!(results.len(), 3);
    assert!(results[0].as_str().unwrap().contains("more characters omitted"));
    assert!(results[1].as_str().unwrap().contains("more characters omitted"));
    assert!(results[2].is_object(), "the latest batch is kept whole");

    let events = agent.journal().load(&out.run_id).await.unwrap();
    let journaled = events.iter().find_map(|e| match &e.kind {
        EventKind::ToolCompleted { output, .. } => Some(output.clone()),
        _ => None,
    });
    assert!(journaled.unwrap()["blob"].as_str().unwrap().len() > 400, "the journal keeps full results");
}

#[test]
fn usd_budget_refuses_an_unpriced_summarizer() {
    struct Unpriced;
    #[async_trait::async_trait]
    impl Model for Unpriced {
        fn id(&self) -> String {
            "unpriced:summarizer".into()
        }
        async fn complete(&self, _: &ModelRequest) -> Result<ModelResponse, ModelError> {
            unreachable!()
        }
    }
    let err = Agent::builder("w")
        .model(ScriptedModel::new(vec![]))
        .context(ContextPolicy::default().compact_at(10).summarize_with(Unpriced))
        .budget(Budget::default().max_usd(1.0))
        .build()
        .err()
        .unwrap();
    assert!(err.to_string().contains("unpriced:summarizer"), "{err}");
}

#[tokio::test]
async fn memory_tools_remember_and_recall_per_scope() {
    let store: Arc<dyn MemoryStore> = Arc::new(InMemoryMemory::new());
    let model = Arc::new(ScriptedModel::new(vec![
        tool_call_response(&[(
            "m1",
            "remember",
            json!({"note": "Prefers express shipping to Berlin", "tags": ["shipping"]}),
        )]),
        text_response("Noted."),
    ]));
    let agent =
        Agent::builder("assistant").model(model).tools(memory_tools(store.clone(), "customer-42")).build().unwrap();
    agent.run("Please always ship express to my Berlin address").await.unwrap();
    store.remember("customer-42", "Invoices go to accounts@example.com", &[]).await.unwrap();
    store.remember("customer-7", "Prefers standard shipping", &[]).await.unwrap();

    let hits = store.recall("customer-42", "which shipping does this customer prefer?", 5).await.unwrap();
    assert_eq!(hits.len(), 1, "only matching notes in this scope");
    assert!(hits[0].text.contains("express"));

    // The recall tool, through an agent.
    let model = Arc::new(ScriptedModel::new(vec![
        tool_call_response(&[("r1", "recall", json!({"query": "shipping preference"}))]),
        text_response("You prefer express."),
    ]));
    let agent = Agent::builder("assistant")
        .model(model.clone())
        .tools(memory_tools(store.clone(), "customer-42"))
        .build()
        .unwrap();
    agent.run("How do I like my orders shipped?").await.unwrap();
    let result = &model.requests()[1].messages[2].content[0];
    assert!(
        matches!(result, Content::ToolResult { content, .. } if content[0]["note"].as_str().unwrap().contains("express"))
    );

    assert!(store.forget("customer-42", &hits[0].id).await.unwrap());
    assert!(store.recall("customer-42", "shipping", 5).await.unwrap().is_empty());
}

#[tokio::test]
async fn recall_middleware_adds_relevant_notes_to_the_system_prompt() {
    let store: Arc<dyn MemoryStore> = Arc::new(InMemoryMemory::new());
    store.remember("u1", "Allergic to peanuts", &[]).await.unwrap();
    store.remember("u1", "Lives in Lisbon", &[]).await.unwrap();
    let model = Arc::new(ScriptedModel::new(vec![text_response("Here is a peanut-free recipe.")]));
    let agent = Agent::builder("chef")
        .model(model.clone())
        .system("You are a chef.")
        .middleware(RecallMiddleware::new(store, "u1").limit(3))
        .build()
        .unwrap();
    agent.run("Suggest a snack without peanuts").await.unwrap();
    let system = model.requests()[0].system.clone().unwrap();
    assert!(system.starts_with("You are a chef."));
    assert!(system.contains("Allergic to peanuts"));
    assert!(!system.contains("Lisbon"), "irrelevant notes are not added");
}
