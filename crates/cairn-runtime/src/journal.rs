use std::collections::HashMap;
use std::path::PathBuf;
use std::time::{Duration, Instant};

use chrono::{DateTime, Utc};

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

/// Who currently holds a run's lease.
#[derive(Debug, Clone, PartialEq)]
pub struct LeaseHolder {
    pub owner: String,
    pub expires_at: DateTime<Utc>,
}

/// Append-only, per-run event storage.
///
/// Journals may also implement run leases so that only one worker drives a
/// run at a time. The defaults grant every lease (no coordination), which is
/// correct for single-worker setups; sequence checks still prevent corrupt
/// histories either way.
///
/// Implementations must reject an append whose `seq` is not exactly the next
/// position; this is what makes concurrent resumers safe.
#[async_trait]
pub trait Journal: Send + Sync {
    async fn append(&self, event: &Event) -> Result<(), JournalError>;
    async fn load(&self, run_id: &str) -> Result<Vec<Event>, JournalError>;

    /// Run ids, most recently active first, at most `limit`. Journals that
    /// cannot enumerate runs return an empty list.
    async fn list_runs(&self, _limit: usize) -> Result<Vec<String>, JournalError> {
        Ok(Vec::new())
    }

    /// Take (or re-take) the lease on `run_id` for `ttl`. Returns the current
    /// holder if another live owner has it.
    async fn acquire_lease(
        &self,
        _run_id: &str,
        _owner: &str,
        _ttl: Duration,
    ) -> Result<Result<(), LeaseHolder>, JournalError> {
        Ok(Ok(()))
    }

    /// Extend a lease this owner holds. `false` means it was lost.
    async fn renew_lease(&self, _run_id: &str, _owner: &str, _ttl: Duration) -> Result<bool, JournalError> {
        Ok(true)
    }

    async fn release_lease(&self, _run_id: &str, _owner: &str) -> Result<(), JournalError> {
        Ok(())
    }
}

/// In-process lease table shared by the built-in journals.
#[derive(Default)]
struct Leases(std::sync::Mutex<HashMap<String, (String, Instant)>>);

impl Leases {
    fn acquire(&self, run_id: &str, owner: &str, ttl: Duration) -> Result<(), LeaseHolder> {
        let mut map = self.0.lock().unwrap();
        let now = Instant::now();
        if let Some((holder, until)) = map.get(run_id) {
            if holder != owner && *until > now {
                let left = chrono::Duration::from_std(*until - now).unwrap_or_default();
                return Err(LeaseHolder { owner: holder.clone(), expires_at: Utc::now() + left });
            }
        }
        map.insert(run_id.to_owned(), (owner.to_owned(), now + ttl));
        Ok(())
    }

    fn renew(&self, run_id: &str, owner: &str, ttl: Duration) -> bool {
        let mut map = self.0.lock().unwrap();
        match map.get_mut(run_id) {
            Some((holder, until)) if holder == owner => {
                *until = Instant::now() + ttl;
                true
            }
            _ => false,
        }
    }

    fn release(&self, run_id: &str, owner: &str) {
        let mut map = self.0.lock().unwrap();
        if map.get(run_id).is_some_and(|(holder, _)| holder == owner) {
            map.remove(run_id);
        }
    }
}

#[derive(Default)]
pub struct InMemoryJournal {
    runs: Mutex<HashMap<String, Vec<Event>>>,
    leases: Leases,
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

    async fn list_runs(&self, limit: usize) -> Result<Vec<String>, JournalError> {
        let runs = self.runs.lock().await;
        let mut ids: Vec<(DateTime<Utc>, &String)> =
            runs.iter().filter_map(|(id, events)| events.last().map(|e| (e.at, id))).collect();
        ids.sort_by(|a, b| b.0.cmp(&a.0).then_with(|| a.1.cmp(b.1)));
        Ok(ids.into_iter().take(limit).map(|(_, id)| id.clone()).collect())
    }

    async fn acquire_lease(
        &self,
        run_id: &str,
        owner: &str,
        ttl: Duration,
    ) -> Result<Result<(), LeaseHolder>, JournalError> {
        Ok(self.leases.acquire(run_id, owner, ttl))
    }

    async fn renew_lease(&self, run_id: &str, owner: &str, ttl: Duration) -> Result<bool, JournalError> {
        Ok(self.leases.renew(run_id, owner, ttl))
    }

    async fn release_lease(&self, run_id: &str, owner: &str) -> Result<(), JournalError> {
        self.leases.release(run_id, owner);
        Ok(())
    }
}

/// One JSON-lines file per run, fsynced on every append.
///
/// Sequence checks are enforced within this process; use a database-backed
/// journal when several processes may resume the same run.
pub struct JsonlJournal {
    dir: PathBuf,
    next_seq: Mutex<HashMap<String, u64>>,
    /// Leases are in-process only, like the sequence checks.
    leases: Leases,
}

impl JsonlJournal {
    pub fn new(dir: impl Into<PathBuf>) -> Result<Self, JournalError> {
        let dir = dir.into();
        std::fs::create_dir_all(&dir)?;
        Ok(Self { dir, next_seq: Mutex::new(HashMap::new()), leases: Leases::default() })
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
        let mut file =
            tokio::fs::OpenOptions::new().create(true).append(true).open(self.path_for(&event.run_id)).await?;
        file.write_all(&line).await?;
        file.sync_data().await?;

        seqs.insert(event.run_id.clone(), expected + 1);
        Ok(())
    }

    async fn list_runs(&self, limit: usize) -> Result<Vec<String>, JournalError> {
        let mut entries = tokio::fs::read_dir(&self.dir).await?;
        let mut runs = Vec::new();
        while let Some(entry) = entries.next_entry().await? {
            let path = entry.path();
            if path.extension().and_then(|e| e.to_str()) != Some("jsonl") {
                continue;
            }
            let modified = entry.metadata().await?.modified().unwrap_or(std::time::SystemTime::UNIX_EPOCH);
            if let Some(stem) = path.file_stem().and_then(|s| s.to_str()) {
                runs.push((modified, stem.to_owned()));
            }
        }
        runs.sort_by(|a, b| b.0.cmp(&a.0).then_with(|| a.1.cmp(&b.1)));
        Ok(runs.into_iter().take(limit).map(|(_, id)| id).collect())
    }

    async fn load(&self, run_id: &str) -> Result<Vec<Event>, JournalError> {
        match tokio::fs::read_to_string(self.path_for(run_id)).await {
            Ok(text) => parse_lines(&text),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(Vec::new()),
            Err(e) => Err(e.into()),
        }
    }

    async fn acquire_lease(
        &self,
        run_id: &str,
        owner: &str,
        ttl: Duration,
    ) -> Result<Result<(), LeaseHolder>, JournalError> {
        Ok(self.leases.acquire(run_id, owner, ttl))
    }

    async fn renew_lease(&self, run_id: &str, owner: &str, ttl: Duration) -> Result<bool, JournalError> {
        Ok(self.leases.renew(run_id, owner, ttl))
    }

    async fn release_lease(&self, run_id: &str, owner: &str) -> Result<(), JournalError> {
        self.leases.release(run_id, owner);
        Ok(())
    }
}
