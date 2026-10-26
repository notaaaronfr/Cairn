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
pub use cairn_runtime::{Priced, Pricing};
pub use openai::OpenAiCompatModel;

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
/// | `compat:<name>:` | `<NAME>_API_KEY` (optional) | `<NAME>_BASE_URL` (required)  |
///
/// `compat:` reaches any OpenAI-compatible server: `compat:k2:IFM/K2-Horizon-375B-A23B`
/// reads `K2_BASE_URL` and `K2_API_KEY`, and its [`Model::id`] is
/// `k2:IFM/K2-Horizon-375B-A23B`. `<NAME>` is the name upper-cased with
/// non-alphanumeric characters replaced by `_`.
pub fn from_spec(spec: &str) -> Result<Arc<dyn Model>, ModelError> {
    let (provider, model) =
        spec.split_once(':').ok_or_else(|| ModelError::Other(format!("expected `provider:model`, got `{spec}`")))?;
    let key = |var: &str| std::env::var(var).map_err(|_| ModelError::Other(format!("{var} is not set")));
    let model: Arc<dyn Model> = match provider {
        "anthropic" => Arc::new(AnthropicModel::new(model, key("ANTHROPIC_API_KEY")?)),
        "openai" => Arc::new(OpenAiCompatModel::openai(model, key("OPENAI_API_KEY")?)),
        "gemini" => Arc::new(OpenAiCompatModel::gemini(model, key("GEMINI_API_KEY")?)),
        "openrouter" => Arc::new(OpenAiCompatModel::openrouter(model, key("OPENROUTER_API_KEY")?)),
        "compat" => {
            let (name, model) = model
                .split_once(':')
                .filter(|(n, m)| !n.is_empty() && !m.is_empty())
                .ok_or_else(|| ModelError::Other(format!("expected `compat:<name>:<model>`, got `{spec}`")))?;
            let prefix = env_prefix(name);
            let base = key(&format!("{prefix}_BASE_URL"))?;
            let api_key = std::env::var(format!("{prefix}_API_KEY")).ok().filter(|k| !k.is_empty());
            Arc::new(OpenAiCompatModel::new(name, model, base, api_key))
        }
        "ollama" => {
            let base = std::env::var("OLLAMA_BASE_URL").unwrap_or_else(|_| "http://localhost:11434/v1".into());
            Arc::new(OpenAiCompatModel::new("ollama", model, base, None))
        }
        other => return Err(ModelError::Other(format!("unknown provider `{other}`"))),
    };
    Ok(model)
}

fn env_prefix(name: &str) -> String {
    name.chars().map(|c| if c.is_ascii_alphanumeric() { c.to_ascii_uppercase() } else { '_' }).collect()
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn env_prefix_normalizes_names() {
        assert_eq!(env_prefix("k2"), "K2");
        assert_eq!(env_prefix("my-vllm.prod"), "MY_VLLM_PROD");
    }

    #[test]
    fn compat_spec_errors_are_actionable() {
        let err = from_spec("compat:k2").err().unwrap().to_string();
        assert!(err.contains("compat:<name>:<model>"), "{err}");
        let err = from_spec("compat:cairn_unset_zz:m").err().unwrap().to_string();
        assert!(err.contains("CAIRN_UNSET_ZZ_BASE_URL is not set"), "{err}");
    }
}
