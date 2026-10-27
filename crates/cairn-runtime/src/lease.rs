//! Run leases shared by agents and workflows.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;

use crate::journal::{Journal, JournalError, LeaseHolder};

/// Holds a run's lease and renews it in the background until dropped.
pub(crate) struct Lease {
    lost: Arc<AtomicBool>,
    renewer: tokio::task::JoinHandle<()>,
    journal: Arc<dyn Journal>,
    run_id: String,
    owner: String,
}

impl Lease {
    /// Take the lease on `run_id`, or report who holds it.
    pub(crate) async fn acquire(
        journal: &Arc<dyn Journal>,
        run_id: &str,
        owner: &str,
        ttl: Duration,
    ) -> Result<Result<Lease, LeaseHolder>, JournalError> {
        if let Err(holder) = journal.acquire_lease(run_id, owner, ttl).await? {
            return Ok(Err(holder));
        }
        let lost = Arc::new(AtomicBool::new(false));
        let (j, o, flag, id) = (journal.clone(), owner.to_owned(), lost.clone(), run_id.to_owned());
        let renewer = tokio::spawn(async move {
            let mut tick = tokio::time::interval((ttl / 3).max(Duration::from_millis(10)));
            tick.tick().await;
            loop {
                tick.tick().await;
                if !matches!(j.renew_lease(&id, &o, ttl).await, Ok(true)) {
                    tracing::warn!(run_id = %id, owner = %o, "lost run lease");
                    flag.store(true, Ordering::SeqCst);
                    return;
                }
            }
        });
        Ok(Ok(Lease { lost, renewer, journal: journal.clone(), run_id: run_id.to_owned(), owner: owner.to_owned() }))
    }

    pub(crate) fn lost_flag(&self) -> Arc<AtomicBool> {
        self.lost.clone()
    }

    pub(crate) fn lost(&self) -> bool {
        self.lost.load(Ordering::SeqCst)
    }

    /// Stop renewing and give the lease back.
    pub(crate) async fn release(self) {
        self.renewer.abort();
        if let Err(e) = self.journal.release_lease(&self.run_id, &self.owner).await {
            tracing::warn!(run_id = %self.run_id, error = %e, "could not release run lease; it will expire");
        }
    }
}

impl Drop for Lease {
    fn drop(&mut self) {
        self.renewer.abort();
    }
}

/// Run ids name files in the JSONL journal, so they must be plain names.
pub fn is_plain_run_id(id: &str) -> bool {
    !id.is_empty()
        && id.len() <= 128
        && !id.starts_with('.')
        && id.chars().all(|c| c.is_ascii_alphanumeric() || matches!(c, '-' | '_' | '.'))
}

/// A child run id derived from a parent run and a step, always plain and
/// stable: the same parent and step always give the same id.
pub fn child_run_id(parent: &str, step: &str) -> String {
    let clean: String =
        step.chars().map(|c| if c.is_ascii_alphanumeric() || matches!(c, '-' | '_') { c } else { '-' }).collect();
    let id = format!("{parent}.{clean}");
    if id.len() <= 128 {
        return id;
    }
    use sha2::{Digest, Sha256};
    let digest = hex::encode(&Sha256::digest(id.as_bytes())[..12]);
    let keep = 128 - digest.len() - 1;
    format!("{}.{digest}", &id[..keep.min(id.len())])
}
