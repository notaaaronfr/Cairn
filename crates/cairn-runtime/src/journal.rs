use std::collections::HashMap;
use std::path::PathBuf;

use async_trait::async_trait;
use tokio::io::AsyncWriteExt;
use tokio::sync::Mutex;

use crate::event::Event;

#[derive(Debug, thiserror::Error)]
pub enum JournalError {
    #[error("sequence conflict for run {run_id}: expected {expected}, got {got}")]
    Conflict { run_id: String, expected: u64, got: u64 },
    #[error("io error: {0}")]
    Io(#[from] std::io::Error),
    #[error("corrupt journal entry: {0}")]
    Corrupt(String),
    #[error("journal backend error: {0}")]
    Backend(String),
}

/// Append-only, per-run event storage.
///
/// Implementations must reject an append whose `seq` is not exactly the next
/// position; this is what makes concurrent resumers safe.
#[async_trait]
pub trait Journal: Send + Sync {
    async fn append(&self, event: &Event) -> Result<(), JournalError>;
    async fn load(&self, run_id: &str) -> Result<Vec<Event>, JournalError>;
}

#[derive(Default)]
pub struct InMemoryJournal {
    runs: Mutex<HashMap<String, Vec<Event>>>,
}

impl InMemoryJournal {
    pub fn new() -> Self {
        Self::default()
    }
}

#[async_trait]
impl Journal for InMemoryJournal {
    async fn append(&self, event: &Event) -> Result<(), JournalError> {
        let mut runs = self.runs.lock().await;
        let events = runs.entry(event.run_id.clone()).or_default();
        let expected = events.len() as u64;
        if event.seq != expected {
            return Err(JournalError::Conflict { run_id: event.run_id.clone(), expected, got: event.seq });
        }
        events.push(event.clone());
        Ok(())
    }

    async fn load(&self, run_id: &str) -> Result<Vec<Event>, JournalError> {
        Ok(self.runs.lock().await.get(run_id).cloned().unwrap_or_default())
    }
}

/// One JSON-lines file per run, fsynced on every append.
///
/// Sequence checks are enforced within this process; use a database-backed
/// journal when several processes may resume the same run.
pub struct JsonlJournal {
    dir: PathBuf,
    next_seq: Mutex<HashMap<String, u64>>,
}

impl JsonlJournal {
    pub fn new(dir: impl Into<PathBuf>) -> Result<Self, JournalError> {
        let dir = dir.into();
        std::fs::create_dir_all(&dir)?;
        Ok(Self { dir, next_seq: Mutex::new(HashMap::new()) })
    }

    pub fn path_for(&self, run_id: &str) -> PathBuf {
        self.dir.join(format!("{run_id}.jsonl"))
    }

    /// Parse a journal file directly (used by tooling).
    pub fn read_file(path: &std::path::Path) -> Result<Vec<Event>, JournalError> {
        let text = std::fs::read_to_string(path)?;
        parse_lines(&text)
    }
}

fn parse_lines(text: &str) -> Result<Vec<Event>, JournalError> {
    text.lines()
        .filter(|l| !l.trim().is_empty())
        .map(|l| serde_json::from_str(l).map_err(|e| JournalError::Corrupt(e.to_string())))
        .collect()
}

#[async_trait]
impl Journal for JsonlJournal {
    async fn append(&self, event: &Event) -> Result<(), JournalError> {
        let mut seqs = self.next_seq.lock().await;
        let expected = match seqs.get(&event.run_id) {
            Some(n) => *n,
            None => self.load(&event.run_id).await?.len() as u64,
        };
        if event.seq != expected {
            return Err(JournalError::Conflict { run_id: event.run_id.clone(), expected, got: event.seq });
        }

        let mut line = serde_json::to_vec(event).map_err(|e| JournalError::Corrupt(e.to_string()))?;
        line.push(b'\n');
        let mut file = tokio::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(self.path_for(&event.run_id))
            .await?;
        file.write_all(&line).await?;
        file.sync_data().await?;

        seqs.insert(event.run_id.clone(), expected + 1);
        Ok(())
    }

    async fn load(&self, run_id: &str) -> Result<Vec<Event>, JournalError> {
        match tokio::fs::read_to_string(self.path_for(run_id)).await {
            Ok(text) => parse_lines(&text),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(Vec::new()),
            Err(e) => Err(e.into()),
        }
    }
}
