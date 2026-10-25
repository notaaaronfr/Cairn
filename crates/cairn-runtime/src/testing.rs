//! Deterministic models for tests and offline examples.

use std::collections::VecDeque;
use std::sync::Mutex;

use async_trait::async_trait;
use serde_json::Value;

use crate::message::Content;
use crate::model::{Model, ModelError, ModelRequest, ModelResponse, StopReason, Usage};
use crate::pricing::Pricing;

/// Returns pre-scripted responses in order and records every request.
pub struct ScriptedModel {
    responses: Mutex<VecDeque<ModelResponse>>,
    requests: Mutex<Vec<ModelRequest>>,
}

impl ScriptedModel {
    pub fn new(responses: Vec<ModelResponse>) -> Self {
        Self { responses: Mutex::new(responses.into()), requests: Mutex::new(Vec::new()) }
    }

    pub fn requests(&self) -> Vec<ModelRequest> {
        self.requests.lock().unwrap().clone()
    }
}

#[async_trait]
impl Model for ScriptedModel {
    fn id(&self) -> String {
        "scripted:test".into()
    }

    async fn complete(&self, request: &ModelRequest) -> Result<ModelResponse, ModelError> {
        self.requests.lock().unwrap().push(request.clone());
        self.responses
            .lock()
            .unwrap()
            .pop_front()
            .ok_or_else(|| ModelError::Other("scripted model has no responses left".into()))
    }

    /// $1 per million tokens of every kind, so tests can exercise USD budgets.
    fn pricing(&self) -> Option<Pricing> {
        Some(Pricing { input: 1.0, output: 1.0, cache_read: 1.0, cache_write: 1.0 })
    }
}

pub fn text_response(text: &str) -> ModelResponse {
    ModelResponse {
        model: "test".into(),
        content: vec![Content::Text { text: text.into() }],
        stop_reason: StopReason::EndTurn,
        usage: Usage { input_tokens: 100, output_tokens: 20, ..Default::default() },
        ..Default::default()
    }
}

pub fn tool_call_response(calls: &[(&str, &str, Value)]) -> ModelResponse {
    ModelResponse {
        model: "test".into(),
        content: calls
            .iter()
            .map(|(id, name, input)| Content::ToolUse { id: (*id).into(), name: (*name).into(), input: input.clone() })
            .collect(),
        stop_reason: StopReason::ToolUse,
        usage: Usage { input_tokens: 100, output_tokens: 20, ..Default::default() },
        ..Default::default()
    }
}
