//! Requires PostgreSQL. Set CAIRN_TEST_DATABASE_URL, e.g.
//! `postgres://cairn@localhost:54329/cairn_test`. Skipped when unset.

use std::sync::Arc;
use std::sync::atomic::{AtomicU32, Ordering};

use cairn_journal_postgres::PostgresJournal;
use cairn_runtime::testing::{ScriptedModel, text_response, tool_call_response};
use cairn_runtime::*;
use schemars::JsonSchema;
use serde::Deserialize;
use serde_json::json;

async fn journal() -> Option<PostgresJournal> {
    let url = std::env::var("CAIRN_TEST_DATABASE_URL").ok()?;
    Some(PostgresJournal::connect(&url).await.expect("connect"))
}

macro_rules! require_db {
    () => {
        match journal().await {
            Some(j) => j,
            None => {
                eprintln!("skipped: CAIRN_TEST_DATABASE_URL not set");
                return;
            }
        }
    };
}

fn event(run_id: &str, seq: u64) -> Event {
    Event {
        run_id: run_id.into(),
        seq,
        at: chrono::Utc::now(),
        kind: EventKind::RunFailed { error: format!("e{seq}") },
    }
}

fn new_run_id() -> String {
    uuid::Uuid::now_v7().to_string()
}

#[tokio::test]
async fn append_load_and_reject_duplicates_and_gaps() {
    let j = require_db!();
    let run = new_run_id();
    j.append(&event(&run, 0)).await.unwrap();
    j.append(&event(&run, 1)).await.unwrap();
    assert!(matches!(j.append(&event(&run, 1)).await, Err(JournalError::Conflict { expected: 2, got: 1, .. })));
    assert!(matches!(j.append(&event(&run, 5)).await, Err(JournalError::Conflict { expected: 2, got: 5, .. })));
    let loaded = j.load(&run).await.unwrap();
    assert_eq!(loaded.len(), 2);
    assert_eq!(loaded[1], event(&run, 1).clone_with_at(loaded[1].at));
    assert!(j.load("no-such-run").await.unwrap().is_empty());
}

trait CloneWithAt {
    fn clone_with_at(self, at: chrono::DateTime<chrono::Utc>) -> Self;
}
impl CloneWithAt for Event {
    fn clone_with_at(mut self, at: chrono::DateTime<chrono::Utc>) -> Self {
        self.at = at;
        self
    }
}

#[tokio::test]
async fn exactly_one_concurrent_writer_wins_each_position() {
    let j = require_db!();
    let run = new_run_id();
    j.append(&event(&run, 0)).await.unwrap();

    let mut tasks = Vec::new();
    for _ in 0..24 {
        let (j, run) = (j.clone(), run.clone());
        tasks.push(tokio::spawn(async move { j.append(&event(&run, 1)).await }));
    }
    let mut wins = 0;
    for t in tasks {
        match t.await.unwrap() {
            Ok(()) => wins += 1,
            Err(JournalError::Conflict { .. }) => {}
            Err(e) => panic!("unexpected error: {e}"),
        }
    }
    assert_eq!(wins, 1);
    assert_eq!(j.load(&run).await.unwrap().len(), 2);
}

#[derive(Deserialize, JsonSchema)]
struct Id {
    id: u32,
}

fn agent(j: PostgresJournal, responses: Vec<ModelResponse>, refunds: Arc<AtomicU32>) -> Agent {
    let refund = tool("refund", "Refund", Effect::Irreversible, move |i: Id, _ctx| {
        let refunds = refunds.clone();
        async move {
            refunds.fetch_add(1, Ordering::SeqCst);
            Ok(json!({ "refunded": i.id }))
        }
    });
    Agent::builder("support").model(ScriptedModel::new(responses)).tool(refund).journal(j).build().unwrap()
}

#[tokio::test]
async fn approval_survives_process_boundaries() {
    let j1 = require_db!();
    let refunds = Arc::new(AtomicU32::new(0));

    // Process A: run until it needs approval, then go away.
    let a = agent(j1.clone(), vec![tool_call_response(&[("r1", "refund", json!({"id": 9}))])], refunds.clone());
    let out = a.run("refund order 9").await.unwrap();
    assert_eq!(out.status, Status::AwaitingApproval);
    assert!(j1.runs_awaiting_approval().await.unwrap().contains(&out.run_id));
    drop(a);

    // Process B: brand-new connection pool and agent.
    let j2 = journal().await.unwrap();
    let b = agent(j2.clone(), vec![text_response("refunded")], refunds.clone());
    b.decide(&out.run_id, "r1", true, Some("ops"), None).await.unwrap();
    let done = b.resume(&out.run_id).await.unwrap();

    assert_eq!(done.output(), Some("refunded"));
    assert_eq!(refunds.load(Ordering::SeqCst), 1);
    assert!(!j2.runs_awaiting_approval().await.unwrap().contains(&out.run_id));
    let events = j2.load(&out.run_id).await.unwrap();
    assert_eq!(RunState::from_events(&events).unwrap().status, Status::Completed { output: "refunded".into() });
}

#[tokio::test]
async fn racing_resumers_cannot_corrupt_a_run() {
    let j = require_db!();
    let refunds = Arc::new(AtomicU32::new(0));
    let a = agent(j.clone(), vec![tool_call_response(&[("r1", "refund", json!({"id": 1}))])], refunds.clone());
    let out = a.run("refund").await.unwrap();
    a.decide(&out.run_id, "r1", true, None, None).await.unwrap();

    // Two workers pick up the same approved run at once.
    let w1 = agent(j.clone(), vec![text_response("done")], refunds.clone());
    let w2 = agent(j.clone(), vec![text_response("done")], refunds.clone());
    let (r1, r2) = tokio::join!(w1.resume(&out.run_id), w2.resume(&out.run_id));

    // The lease admits one driver; the other is told who holds it (or, if it
    // arrives after the run finished, sees the finished run).
    let ok = [&r1, &r2].iter().filter(|r| r.is_ok()).count();
    let leased = [&r1, &r2].iter().filter(|r| matches!(r, Err(AgentError::Leased { .. }))).count();
    assert_eq!(ok + leased, 2, "{r1:?} {r2:?}");
    assert!(ok >= 1);

    // Whatever the interleaving, the journal is a single consistent history.
    let events = j.load(&out.run_id).await.unwrap();
    for (i, e) in events.iter().enumerate() {
        assert_eq!(e.seq, i as u64);
    }
    let state = RunState::from_events(&events).unwrap();
    assert_eq!(state.status, Status::Completed { output: "done".into() });
    assert_eq!(events.iter().filter(|e| matches!(e.kind, EventKind::ToolCompleted { .. })).count(), 1);
    assert_eq!(refunds.load(Ordering::SeqCst), 1, "leases make the refund run exactly once");
}

#[tokio::test]
async fn expired_lease_is_taken_over_using_the_database_clock() {
    let j = require_db!();
    let run = new_run_id();
    let ttl = std::time::Duration::from_millis(200);
    assert!(j.acquire_lease(&run, "crashed", ttl).await.unwrap().is_ok());
    let held = j.acquire_lease(&run, "b", ttl).await.unwrap().unwrap_err();
    assert_eq!(held.owner, "crashed");
    assert!(j.renew_lease(&run, "crashed", ttl).await.unwrap());
    assert!(!j.renew_lease(&run, "b", ttl).await.unwrap(), "only the holder can renew");

    tokio::time::sleep(std::time::Duration::from_millis(300)).await;
    assert!(j.acquire_lease(&run, "b", ttl).await.unwrap().is_ok(), "expired lease is taken over");
    assert!(!j.renew_lease(&run, "crashed", ttl).await.unwrap(), "the old holder has lost it");
    j.release_lease(&run, "b").await.unwrap();
    assert!(j.acquire_lease(&run, "c", ttl).await.unwrap().is_ok());
}

#[tokio::test]
async fn memory_full_text_recall_and_run_listing() {
    use cairn_journal_postgres::PostgresMemory;
    use cairn_runtime::memory::MemoryStore;
    let j = require_db!();
    let mem = PostgresMemory::from_pool(j.pool().clone()).await.unwrap();
    let scope = format!("customer-{}", new_run_id());
    mem.remember(&scope, "Prefers express shipping to the Berlin office", &["shipping".into()]).await.unwrap();
    let invoice = mem.remember(&scope, "Invoices go to accounts@example.com", &[]).await.unwrap();
    mem.remember("someone-else", "Prefers standard shipping", &[]).await.unwrap();

    let hits = mem.recall(&scope, "Which shipping option does the customer prefer?", 5).await.unwrap();
    assert_eq!(hits.len(), 1);
    assert!(hits[0].text.contains("express"));
    assert_eq!(hits[0].tags, ["shipping"]);
    assert!(mem.recall(&scope, "?!", 5).await.unwrap().is_empty());
    assert!(mem.forget(&scope, &invoice).await.unwrap());
    assert!(mem.recall(&scope, "invoices", 5).await.unwrap().is_empty());

    // list_runs: most recently active first.
    let (a, b) = (new_run_id(), new_run_id());
    j.append(&event(&a, 0)).await.unwrap();
    j.append(&event(&b, 0)).await.unwrap();
    let recent = j.list_runs(50).await.unwrap();
    let pos = |id: &str| recent.iter().position(|r| r == id).unwrap();
    assert!(pos(&b) < pos(&a));
}
