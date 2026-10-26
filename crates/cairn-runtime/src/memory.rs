//! Long-term memory across runs.
//!
//! A [`MemoryStore`] keeps short notes per *scope* (a user, a customer, a
//! project). Agents use it two ways:
//!
//! * as tools — [`memory_tools`] gives the model `remember` and `recall`;
//! * automatically — [`RecallMiddleware`] looks up notes relevant to the
//!   task before each model call and adds them to the system prompt.
//!
//! [`InMemoryMemory`] ranks notes by keyword relevance. Backends with
//! semantic (embedding) search can implement the same trait; the PostgreSQL
//! store in `cairn-journal-postgres` uses full-text search.

use std::collections::HashMap;
use std::sync::Arc;

use async_trait::async_trait;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use tokio::sync::RwLock;

use crate::middleware::{HookCtx, Middleware};
use crate::model::ModelRequest;
use crate::tool::{Effect, Tool, ToolCtx, ToolError, ToolSpec};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MemoryItem {
    pub id: String,
    pub text: String,
    #[serde(default)]
    pub tags: Vec<String>,
    pub created_at: DateTime<Utc>,
}

#[derive(Debug, thiserror::Error)]
#[error("memory store error: {0}")]
pub struct MemoryError(pub String);

#[async_trait]
pub trait MemoryStore: Send + Sync {
    /// Save a note; returns its id.
    async fn remember(&self, scope: &str, text: &str, tags: &[String]) -> Result<String, MemoryError>;
    /// Notes most relevant to `query`, best first.
    async fn recall(&self, scope: &str, query: &str, limit: usize) -> Result<Vec<MemoryItem>, MemoryError>;
    async fn forget(&self, scope: &str, id: &str) -> Result<bool, MemoryError>;
}

pub(crate) fn words(text: &str) -> Vec<String> {
    text.split(|c: char| !c.is_alphanumeric()).filter(|w| w.len() > 1).map(str::to_lowercase).collect()
}

/// Rank items by keyword relevance (TF-IDF over the scope), ties by recency.
pub fn rank(items: &[MemoryItem], query: &str, limit: usize) -> Vec<MemoryItem> {
    let terms = words(query);
    let docs: Vec<Vec<String>> = items.iter().map(|i| words(&format!("{} {}", i.text, i.tags.join(" ")))).collect();
    let n = items.len() as f64;
    let mut scored: Vec<(f64, &MemoryItem)> = items
        .iter()
        .zip(&docs)
        .map(|(item, doc)| {
            let score = terms
                .iter()
                .map(|t| {
                    let tf = doc.iter().filter(|w| *w == t).count() as f64;
                    if tf == 0.0 {
                        return 0.0;
                    }
                    let df = docs.iter().filter(|d| d.contains(t)).count() as f64;
                    (1.0 + tf.ln()) * (1.0 + n / df).ln()
                })
                .sum::<f64>();
            (score, item)
        })
        .filter(|(score, _)| *score > 0.0)
        .collect();
    scored.sort_by(|a, b| b.0.total_cmp(&a.0).then_with(|| b.1.created_at.cmp(&a.1.created_at)));
    scored.into_iter().take(limit).map(|(_, i)| i.clone()).collect()
}

/// Process-local memory store (tests, single-process apps).
#[derive(Default)]
pub struct InMemoryMemory {
    scopes: RwLock<HashMap<String, Vec<MemoryItem>>>,
}

impl InMemoryMemory {
    pub fn new() -> Self {
        Self::default()
    }
}

#[async_trait]
impl MemoryStore for InMemoryMemory {
    async fn remember(&self, scope: &str, text: &str, tags: &[String]) -> Result<String, MemoryError> {
        let id = uuid::Uuid::now_v7().to_string();
        let item = MemoryItem { id: id.clone(), text: text.to_owned(), tags: tags.to_vec(), created_at: Utc::now() };
        self.scopes.write().await.entry(scope.to_owned()).or_default().push(item);
        Ok(id)
    }

    async fn recall(&self, scope: &str, query: &str, limit: usize) -> Result<Vec<MemoryItem>, MemoryError> {
        let scopes = self.scopes.read().await;
        Ok(scopes.get(scope).map(|items| rank(items, query, limit)).unwrap_or_default())
    }

    async fn forget(&self, scope: &str, id: &str) -> Result<bool, MemoryError> {
        let mut scopes = self.scopes.write().await;
        let Some(items) = scopes.get_mut(scope) else { return Ok(false) };
        let before = items.len();
        items.retain(|i| i.id != id);
        Ok(items.len() != before)
    }
}

enum Kind {
    Remember,
    Recall,
}

/// `remember` / `recall` tools bound to one store and scope.
pub struct MemoryTool {
    store: Arc<dyn MemoryStore>,
    scope: String,
    kind: Kind,
    spec: ToolSpec,
}

/// The `remember` and `recall` tools for `scope`.
pub fn memory_tools(store: Arc<dyn MemoryStore>, scope: &str) -> Vec<MemoryTool> {
    vec![
        MemoryTool {
            store: store.clone(),
            scope: scope.to_owned(),
            kind: Kind::Remember,
            spec: ToolSpec {
                name: "remember".into(),
                description: "Save a short, self-contained note worth knowing in future conversations \
                              (preferences, facts, decisions). Do not save secrets."
                    .into(),
                input_schema: json!({
                    "type": "object",
                    "properties": {
                        "note": {"type": "string"},
                        "tags": {"type": "array", "items": {"type": "string"}}
                    },
                    "required": ["note"]
                }),
                effect: Effect::Write,
            },
        },
        MemoryTool {
            store,
            scope: scope.to_owned(),
            kind: Kind::Recall,
            spec: ToolSpec {
                name: "recall".into(),
                description: "Search saved notes from earlier conversations.".into(),
                input_schema: json!({
                    "type": "object",
                    "properties": {"query": {"type": "string"}, "limit": {"type": "integer", "minimum": 1, "maximum": 20}},
                    "required": ["query"]
                }),
                effect: Effect::Read,
            },
        },
    ]
}

#[async_trait]
impl Tool for MemoryTool {
    fn spec(&self) -> &ToolSpec {
        &self.spec
    }

    async fn call(&self, input: Value, _ctx: &ToolCtx) -> Result<Value, ToolError> {
        let err = |e: MemoryError| ToolError::new(e.to_string());
        match self.kind {
            Kind::Remember => {
                let note = input["note"].as_str().filter(|n| !n.trim().is_empty());
                let note = note.ok_or_else(|| ToolError::new("`note` must be a non-empty string"))?;
                let tags: Vec<String> = input["tags"]
                    .as_array()
                    .into_iter()
                    .flatten()
                    .filter_map(|t| t.as_str().map(str::to_owned))
                    .collect();
                let id = self.store.remember(&self.scope, note, &tags).await.map_err(err)?;
                Ok(json!({ "saved": id }))
            }
            Kind::Recall => {
                let query = input["query"].as_str().ok_or_else(|| ToolError::new("`query` must be a string"))?;
                let limit = input["limit"].as_u64().unwrap_or(5).clamp(1, 20) as usize;
                let items = self.store.recall(&self.scope, query, limit).await.map_err(err)?;
                Ok(json!(
                    items
                        .iter()
                        .map(|i| json!({"id": i.id, "note": i.text, "saved": i.created_at}))
                        .collect::<Vec<_>>()
                ))
            }
        }
    }
}

/// Adds notes relevant to the run's task to the system prompt before each
/// model call. The lookup uses the run's first user message; what is added
/// is visible in the request (and in traces).
pub struct RecallMiddleware {
    store: Arc<dyn MemoryStore>,
    scope: String,
    limit: usize,
}

impl RecallMiddleware {
    pub fn new(store: Arc<dyn MemoryStore>, scope: &str) -> Self {
        Self { store, scope: scope.to_owned(), limit: 5 }
    }

    pub fn limit(mut self, n: usize) -> Self {
        self.limit = n;
        self
    }
}

#[async_trait]
impl Middleware for RecallMiddleware {
    async fn before_model(&self, _ctx: &HookCtx, request: &mut ModelRequest) -> Result<(), String> {
        let Some(task) = request.messages.first().map(|m| m.text()) else { return Ok(()) };
        let items = self.store.recall(&self.scope, &task, self.limit).await.map_err(|e| e.to_string())?;
        if items.is_empty() {
            return Ok(());
        }
        let notes: String = items.iter().map(|i| format!("- {}\n", i.text)).collect();
        let block = format!("Notes saved from earlier conversations (may be relevant):\n{notes}");
        request.system = Some(match request.system.take() {
            Some(s) => format!("{s}\n\n{block}"),
            None => block,
        });
        Ok(())
    }
}
