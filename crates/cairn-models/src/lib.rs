//! Model adapters for Cairn.
//!
//! * [`AnthropicModel`] speaks the Messages API natively and preserves
//!   extended thinking and prompt caching.
//! * [`OpenAiCompatModel`] speaks Chat Completions, which covers OpenAI,
//!   Google Gemini (OpenAI-compatible endpoint), Ollama, vLLM, OpenRouter,
//!   Groq, Together and most self-hosted servers.

mod anthropic;
mod openai;
mod sse;

pub use anthropic::AnthropicModel;
pub use openai::OpenAiCompatModel;
pub use cairn_runtime::{Priced, Pricing};

use std::sync::Arc;

use cairn_runtime::{Model, ModelError};

/// Build a model from a `provider:model` spec, reading credentials from the
/// environment.
///
/// | spec prefix  | env var             | endpoint                                  |
/// |--------------|---------------------|-------------------------------------------|
/// | `anthropic:` | `ANTHROPIC_API_KEY` | api.anthropic.com                         |
/// | `openai:`    | `OPENAI_API_KEY`    | api.openai.com                            |
/// | `gemini:`    | `GEMINI_API_KEY`    | Gemini OpenAI-compatible endpoint         |
/// | `ollama:`    | —                   | `OLLAMA_BASE_URL` or localhost:11434      |
/// | `openrouter:`| `OPENROUTER_API_KEY`| openrouter.ai                             |
pub fn from_spec(spec: &str) -> Result<Arc<dyn Model>, ModelError> {
    let (provider, model) =
        spec.split_once(':').ok_or_else(|| ModelError::Other(format!("expected `provider:model`, got `{spec}`")))?;
    let key = |var: &str| std::env::var(var).map_err(|_| ModelError::Other(format!("{var} is not set")));
    let model: Arc<dyn Model> = match provider {
        "anthropic" => Arc::new(AnthropicModel::new(model, key("ANTHROPIC_API_KEY")?)),
        "openai" => Arc::new(OpenAiCompatModel::openai(model, key("OPENAI_API_KEY")?)),
        "gemini" => Arc::new(OpenAiCompatModel::gemini(model, key("GEMINI_API_KEY")?)),
        "openrouter" => Arc::new(OpenAiCompatModel::openrouter(model, key("OPENROUTER_API_KEY")?)),
        "ollama" => {
            let base = std::env::var("OLLAMA_BASE_URL").unwrap_or_else(|_| "http://localhost:11434/v1".into());
            Arc::new(OpenAiCompatModel::new("ollama", model, base, None))
        }
        other => return Err(ModelError::Other(format!("unknown provider `{other}`"))),
    };
    Ok(model)
}

pub(crate) async fn send_json(
    request: reqwest::RequestBuilder,
    body: &serde_json::Value,
) -> Result<serde_json::Value, ModelError> {
    let resp = request.json(body).send().await.map_err(|e| ModelError::Transport(e.to_string()))?;
    let status = resp.status();
    let text = resp.text().await.map_err(|e| ModelError::Transport(e.to_string()))?;
    if !status.is_success() {
        return Err(ModelError::Api { status: status.as_u16(), body: text });
    }
    serde_json::from_str(&text).map_err(|e| ModelError::Decode(e.to_string()))
}

pub(crate) fn merge_options(body: &mut serde_json::Value, options: Option<&serde_json::Value>) {
    if let (Some(target), Some(serde_json::Value::Object(extra))) = (body.as_object_mut(), options) {
        for (k, v) in extra {
            target.insert(k.clone(), v.clone());
        }
    }
}

pub(crate) fn result_string(v: &serde_json::Value) -> String {
    match v {
        serde_json::Value::String(s) => s.clone(),
        other => other.to_string(),
    }
}
