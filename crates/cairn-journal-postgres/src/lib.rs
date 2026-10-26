//! PostgreSQL-backed [`Journal`] for Cairn.
//!
//! Appends are atomic compare-and-set on the run's next sequence number, so
//! any number of processes and machines can drive, resume and approve the
//! same runs: exactly one writer wins each position, the rest get
//! [`JournalError::Conflict`] and can reload and retry.

use std::time::Duration;

use async_trait::async_trait;
use cairn_runtime::memory::{MemoryError, MemoryItem, MemoryStore};
use cairn_runtime::{Event, Journal, JournalError, LeaseHolder};
use sqlx::postgres::{PgPool, PgPoolOptions};

const SCHEMA: &str = r#"
CREATE TABLE IF NOT EXISTS cairn_events (
    run_id  TEXT        NOT NULL,
    seq     BIGINT      NOT NULL,
    at      TIMESTAMPTZ NOT NULL,
    kind    TEXT        NOT NULL,
    event   JSONB       NOT NULL,
    PRIMARY KEY (run_id, seq)
);
CREATE INDEX IF NOT EXISTS cairn_events_kind_at ON cairn_events (kind, at);
CREATE TABLE IF NOT EXISTS cairn_memory (
    id         TEXT        PRIMARY KEY,
    scope      TEXT        NOT NULL,
    text       TEXT        NOT NULL,
    tags       JSONB       NOT NULL DEFAULT '[]',
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    search     TSVECTOR    GENERATED ALWAYS AS (to_tsvector('simple', text)) STORED
);
CREATE INDEX IF NOT EXISTS cairn_memory_scope ON cairn_memory (scope, created_at DESC);
CREATE INDEX IF NOT EXISTS cairn_memory_search ON cairn_memory USING GIN (search);
CREATE TABLE IF NOT EXISTS cairn_leases (
    run_id     TEXT        PRIMARY KEY,
    owner      TEXT        NOT NULL,
    expires_at TIMESTAMPTZ NOT NULL
);
"#;

/// Arbitrary constant identifying Cairn's schema-migration advisory lock.
const SCHEMA_LOCK: i64 = 0x6361_6972_6e00_0001;

#[derive(Clone)]
pub struct PostgresJournal {
    pool: PgPool,
}

fn backend(e: sqlx::Error) -> JournalError {
    JournalError::Backend(e.to_string())
}

impl PostgresJournal {
    /// Connect and create the schema if needed.
    pub async fn connect(url: &str) -> Result<Self, JournalError> {
        let pool = PgPoolOptions::new().max_connections(16).connect(url).await.map_err(backend)?;
        Self::from_pool(pool).await
    }

    /// Use an existing pool and create the schema if needed.
    pub async fn from_pool(pool: PgPool) -> Result<Self, JournalError> {
        // `CREATE ... IF NOT EXISTS` races when several replicas start at
        // once; serialize schema setup behind a transaction-scoped lock.
        let mut tx = pool.begin().await.map_err(backend)?;
        sqlx::query("SELECT pg_advisory_xact_lock($1)").bind(SCHEMA_LOCK).execute(&mut *tx).await.map_err(backend)?;
        sqlx::raw_sql(SCHEMA).execute(&mut *tx).await.map_err(backend)?;
        tx.commit().await.map_err(backend)?;
        Ok(Self { pool })
    }

    pub fn pool(&self) -> &PgPool {
        &self.pool
    }

    /// Run ids that are suspended waiting for a human decision.
    pub async fn runs_awaiting_approval(&self) -> Result<Vec<String>, JournalError> {
        // A run is awaiting approval if it has an approval request with no
        // matching decision and has not reached a terminal event.
        sqlx::query_scalar(
            r#"
            SELECT DISTINCT r.run_id FROM cairn_events r
            WHERE r.kind = 'approval_requested'
              AND NOT EXISTS (
                SELECT 1 FROM cairn_events d
                WHERE d.run_id = r.run_id AND d.kind = 'approval_decided'
                  AND d.event->'kind'->>'call_id' = r.event->'kind'->>'call_id')
              AND NOT EXISTS (
                SELECT 1 FROM cairn_events t
                WHERE t.run_id = r.run_id
                  AND t.kind IN ('run_completed', 'run_failed', 'budget_exceeded'))
            ORDER BY r.run_id
            "#,
        )
        .fetch_all(&self.pool)
        .await
        .map_err(backend)
    }
}

#[async_trait]
impl Journal for PostgresJournal {
    async fn append(&self, event: &Event) -> Result<(), JournalError> {
        let json = serde_json::to_value(event).map_err(|e| JournalError::Corrupt(e.to_string()))?;
        let seq = i64::try_from(event.seq).map_err(|_| JournalError::Corrupt("seq out of range".into()))?;

        // Insert only if `seq` is exactly the next position. A concurrent
        // writer for the same position loses on the primary key.
        let result = sqlx::query(
            r#"
            INSERT INTO cairn_events (run_id, seq, at, kind, event)
            SELECT $1, $2, $3, $4, $5
            WHERE (SELECT COALESCE(MAX(seq) + 1, 0) FROM cairn_events WHERE run_id = $1) = $2
            "#,
        )
        .bind(&event.run_id)
        .bind(seq)
        .bind(event.at)
        .bind(event.kind.name())
        .bind(json)
        .execute(&self.pool)
        .await;

        let inserted = match result {
            Ok(r) => r.rows_affected() == 1,
            Err(sqlx::Error::Database(e)) if e.is_unique_violation() => false,
            Err(e) => return Err(backend(e)),
        };
        if inserted {
            return Ok(());
        }
        let expected: i64 = sqlx::query_scalar("SELECT COALESCE(MAX(seq) + 1, 0) FROM cairn_events WHERE run_id = $1")
            .bind(&event.run_id)
            .fetch_one(&self.pool)
            .await
            .map_err(backend)?;
        Err(JournalError::Conflict { run_id: event.run_id.clone(), expected: expected as u64, got: event.seq })
    }

    async fn list_runs(&self, limit: usize) -> Result<Vec<String>, JournalError> {
        sqlx::query_scalar("SELECT run_id FROM cairn_events GROUP BY run_id ORDER BY MAX(at) DESC, run_id LIMIT $1")
            .bind(i64::try_from(limit).unwrap_or(i64::MAX))
            .fetch_all(&self.pool)
            .await
            .map_err(backend)
    }

    /// Leases use the database clock, so worker clock skew does not matter.
    async fn acquire_lease(
        &self,
        run_id: &str,
        owner: &str,
        ttl: Duration,
    ) -> Result<Result<(), LeaseHolder>, JournalError> {
        let taken: Option<String> = sqlx::query_scalar(
            r#"
            INSERT INTO cairn_leases (run_id, owner, expires_at)
            VALUES ($1, $2, now() + make_interval(secs => $3))
            ON CONFLICT (run_id) DO UPDATE SET owner = EXCLUDED.owner, expires_at = EXCLUDED.expires_at
            WHERE cairn_leases.expires_at < now() OR cairn_leases.owner = EXCLUDED.owner
            RETURNING owner
            "#,
        )
        .bind(run_id)
        .bind(owner)
        .bind(ttl.as_secs_f64())
        .fetch_optional(&self.pool)
        .await
        .map_err(backend)?;
        if taken.is_some() {
            return Ok(Ok(()));
        }
        let holder: Option<(String, chrono::DateTime<chrono::Utc>)> =
            sqlx::query_as("SELECT owner, expires_at FROM cairn_leases WHERE run_id = $1")
                .bind(run_id)
                .fetch_optional(&self.pool)
                .await
                .map_err(backend)?;
        match holder {
            Some((owner, expires_at)) => Ok(Err(LeaseHolder { owner, expires_at })),
            // Released between the two statements: try again.
            None => Box::pin(self.acquire_lease(run_id, owner, ttl)).await,
        }
    }

    async fn renew_lease(&self, run_id: &str, owner: &str, ttl: Duration) -> Result<bool, JournalError> {
        let updated = sqlx::query(
            "UPDATE cairn_leases SET expires_at = now() + make_interval(secs => $3) WHERE run_id = $1 AND owner = $2",
        )
        .bind(run_id)
        .bind(owner)
        .bind(ttl.as_secs_f64())
        .execute(&self.pool)
        .await
        .map_err(backend)?;
        Ok(updated.rows_affected() == 1)
    }

    async fn release_lease(&self, run_id: &str, owner: &str) -> Result<(), JournalError> {
        sqlx::query("DELETE FROM cairn_leases WHERE run_id = $1 AND owner = $2")
            .bind(run_id)
            .bind(owner)
            .execute(&self.pool)
            .await
            .map_err(backend)?;
        Ok(())
    }

    async fn load(&self, run_id: &str) -> Result<Vec<Event>, JournalError> {
        let rows: Vec<serde_json::Value> =
            sqlx::query_scalar("SELECT event FROM cairn_events WHERE run_id = $1 ORDER BY seq")
                .bind(run_id)
                .fetch_all(&self.pool)
                .await
                .map_err(backend)?;
        rows.into_iter().map(|v| serde_json::from_value(v).map_err(|e| JournalError::Corrupt(e.to_string()))).collect()
    }
}

/// PostgreSQL-backed long-term memory with full-text search. Shares the
/// journal's database and schema setup.
#[derive(Clone)]
pub struct PostgresMemory {
    pool: PgPool,
}

impl PostgresMemory {
    pub async fn connect(url: &str) -> Result<Self, JournalError> {
        Ok(Self { pool: PostgresJournal::connect(url).await?.pool })
    }

    /// Reuse a journal's pool (creates the schema if needed).
    pub async fn from_pool(pool: PgPool) -> Result<Self, JournalError> {
        Ok(Self { pool: PostgresJournal::from_pool(pool).await?.pool })
    }
}

fn mem_err(e: sqlx::Error) -> MemoryError {
    MemoryError(e.to_string())
}

#[async_trait]
impl MemoryStore for PostgresMemory {
    async fn remember(&self, scope: &str, text: &str, tags: &[String]) -> Result<String, MemoryError> {
        let id = uuid::Uuid::now_v7().to_string();
        sqlx::query("INSERT INTO cairn_memory (id, scope, text, tags) VALUES ($1, $2, $3, $4)")
            .bind(&id)
            .bind(scope)
            .bind(text)
            .bind(serde_json::json!(tags))
            .execute(&self.pool)
            .await
            .map_err(mem_err)?;
        Ok(id)
    }

    /// Ranks notes matching any query word (OR semantics), most relevant first.
    async fn recall(&self, scope: &str, query: &str, limit: usize) -> Result<Vec<MemoryItem>, MemoryError> {
        let words: Vec<String> =
            query.split(|c: char| !c.is_alphanumeric()).filter(|w| w.len() > 1).map(str::to_lowercase).collect();
        if words.is_empty() {
            return Ok(Vec::new());
        }
        let tsquery = words.join(" | ");
        let rows: Vec<(String, String, serde_json::Value, chrono::DateTime<chrono::Utc>)> = sqlx::query_as(
            r#"
            SELECT id, text, tags, created_at FROM cairn_memory
            WHERE scope = $1 AND search @@ to_tsquery('simple', $2)
            ORDER BY ts_rank(search, to_tsquery('simple', $2)) DESC, created_at DESC
            LIMIT $3
            "#,
        )
        .bind(scope)
        .bind(&tsquery)
        .bind(i64::try_from(limit).unwrap_or(i64::MAX))
        .fetch_all(&self.pool)
        .await
        .map_err(mem_err)?;
        Ok(rows
            .into_iter()
            .map(|(id, text, tags, created_at)| MemoryItem {
                id,
                text,
                tags: serde_json::from_value(tags).unwrap_or_default(),
                created_at,
            })
            .collect())
    }

    async fn forget(&self, scope: &str, id: &str) -> Result<bool, MemoryError> {
        let r = sqlx::query("DELETE FROM cairn_memory WHERE scope = $1 AND id = $2")
            .bind(scope)
            .bind(id)
            .execute(&self.pool)
            .await
            .map_err(mem_err)?;
        Ok(r.rows_affected() == 1)
    }
}
