//! Context management for long runs.
//!
//! Two tools, usable together:
//!
//! * **Tool-result trimming** shortens older tool results in each request
//!   (the journal keeps them whole). Deterministic and free.
//! * **Compaction** summarises older turns with a model once a request's
//!   estimated size passes a threshold. The summary is journaled
//!   (`ContextCompacted`), so resume and replay see exactly the same
//!   conversation, and its cost counts toward the run's budget.
//!
//! Compaction keeps the original task and the most recent turns verbatim,
//! and never separates a tool call from its result.

use std::sync::Arc;

use serde_json::{Value, json};

use crate::message::{Content, Message, Role};
use crate::model::{Model, ModelError, ModelRequest, Usage};

const SUMMARY_PROMPT: &str = "You are compacting the history of an AI agent's run so it can continue \
with less context. Summarize the conversation below for the agent: keep the facts it learned, \
decisions made, tool results that still matter (with exact values such as ids and amounts), \
open questions, and what remains to be done. Be concise; do not invent anything.";

/// How an agent keeps long runs within its context window.
#[derive(Clone, Default)]
pub struct ContextPolicy {
    compact_at: Option<u64>,
    keep_recent: Option<usize>,
    summarizer: Option<Arc<dyn Model>>,
    max_tool_result_chars: Option<usize>,
    keep_full_results: Option<usize>,
}

impl std::fmt::Debug for ContextPolicy {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ContextPolicy")
            .field("compact_at", &self.compact_at)
            .field("keep_recent", &self.keep_recent)
            .field("summarizer", &self.summarizer.as_ref().map(|m| m.id()))
            .field("max_tool_result_chars", &self.max_tool_result_chars)
            .finish()
    }
}

impl ContextPolicy {
    /// Compact once a request's estimated input exceeds `tokens`.
    pub fn compact_at(mut self, tokens: u64) -> Self {
        self.compact_at = Some(tokens);
        self
    }

    /// Messages kept verbatim at the end when compacting (default 6).
    pub fn keep_recent(mut self, messages: usize) -> Self {
        self.keep_recent = Some(messages);
        self
    }

    /// Model that writes summaries (default: the agent's own model). A
    /// cheaper model is usually fine.
    pub fn summarize_with(mut self, model: impl Model + 'static) -> Self {
        self.summarizer = Some(Arc::new(model));
        self
    }

    /// Shorten tool results to `chars` in requests, except the most recent
    /// [`keep_full_results`](Self::keep_full_results) batches.
    pub fn max_tool_result_chars(mut self, chars: usize) -> Self {
        self.max_tool_result_chars = Some(chars);
        self
    }

    /// Recent tool-result batches never shortened (default 2).
    pub fn keep_full_results(mut self, batches: usize) -> Self {
        self.keep_full_results = Some(batches);
        self
    }

    pub(crate) fn summarizer(&self) -> Option<&Arc<dyn Model>> {
        self.summarizer.as_ref()
    }

    pub(crate) fn should_compact(&self, estimated_input: u64) -> bool {
        self.compact_at.is_some_and(|limit| estimated_input > limit)
    }

    /// How many messages after the task to replace with a summary, or
    /// `None` if there is nothing worth compacting. The first kept message
    /// is always an assistant turn, so tool calls stay with their results
    /// and roles keep alternating.
    pub(crate) fn compaction_cut(&self, messages: &[Message]) -> Option<usize> {
        let n = messages.len();
        let mut keep_start = n.saturating_sub(self.keep_recent.unwrap_or(6)).max(1);
        while keep_start < n && messages[keep_start].role != Role::Assistant {
            keep_start += 1;
        }
        (keep_start < n && keep_start > 2).then_some(keep_start - 1)
    }

    /// Shorten older tool results in a request (the journal is untouched).
    pub(crate) fn trim(&self, request: &mut ModelRequest) {
        let Some(limit) = self.max_tool_result_chars else { return };
        let keep = self.keep_full_results.unwrap_or(2);
        let batches: Vec<usize> = request
            .messages
            .iter()
            .enumerate()
            .filter(|(_, m)| m.content.iter().any(|c| matches!(c, Content::ToolResult { .. })))
            .map(|(i, _)| i)
            .collect();
        let old = batches.len().saturating_sub(keep);
        for &i in &batches[..old] {
            for c in &mut request.messages[i].content {
                if let Content::ToolResult { content, .. } = c {
                    let text = match &*content {
                        Value::String(s) => s.clone(),
                        other => other.to_string(),
                    };
                    if text.chars().count() > limit {
                        let kept: String = text.chars().take(limit).collect();
                        let dropped = text.chars().count() - limit;
                        *content = json!(format!("{kept}… [{dropped} more characters omitted]"));
                    }
                }
            }
        }
    }
}

/// Render messages as plain text for the summarizer.
fn transcript(messages: &[Message]) -> String {
    let mut out = String::new();
    for m in messages {
        let who = if m.role == Role::User { "user" } else { "assistant" };
        for c in &m.content {
            let line = match c {
                Content::Text { text } => text.clone(),
                Content::ToolUse { name, input, .. } => format!("[called {name} with {input}]"),
                Content::ToolResult { content, is_error, .. } => {
                    format!("[tool {}: {}]", if *is_error { "error" } else { "result" }, content)
                }
                Content::Thinking { .. } => continue,
            };
            out.push_str(&format!("{who}: {line}\n"));
        }
    }
    out
}

/// Summarize `messages`, returning the summary and what it cost.
pub(crate) async fn summarize(
    model: &dyn Model,
    messages: &[Message],
    max_tokens: u32,
) -> Result<(String, Usage, Option<f64>), ModelError> {
    let request = ModelRequest {
        system: Some(SUMMARY_PROMPT.into()),
        messages: vec![Message::user(transcript(messages))],
        max_tokens,
        ..Default::default()
    };
    let resp = model.complete(&request).await?;
    let cost = model.cost_usd(&resp);
    let text = crate::message::content_text(&resp.content).trim().to_owned();
    if text.is_empty() {
        return Err(ModelError::Other("summarizer returned an empty summary".into()));
    }
    Ok((text, resp.total_usage(), cost))
}
