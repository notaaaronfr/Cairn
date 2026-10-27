//! Serve a Cairn agent over A2A with no API keys needed: the "model" echoes.
//!
//! ```sh
//! cargo run -p cairn-ai-a2a --example echo_server -- 127.0.0.1:8791
//! curl http://127.0.0.1:8791/.well-known/agent-card.json
//! ```
//!
//! Swap `Echo` for a real model (`cairn_models::from_spec("anthropic:…")`)
//! to serve a real agent.

use async_trait::async_trait;
use cairn_a2a::{A2aServer, AgentSkill};
use cairn_runtime::{Agent, Content, Model, ModelError, ModelRequest, ModelResponse, StopReason, Usage};

struct Echo;

#[async_trait]
impl Model for Echo {
    fn id(&self) -> String {
        "echo".into()
    }

    async fn complete(&self, request: &ModelRequest) -> Result<ModelResponse, ModelError> {
        let last = request.messages.last().map(|m| m.text()).unwrap_or_default();
        Ok(ModelResponse {
            model: "echo".into(),
            content: vec![Content::Text { text: format!("echo: {last}") }],
            stop_reason: StopReason::EndTurn,
            usage: Usage { input_tokens: 1, output_tokens: 1, ..Default::default() },
            ..Default::default()
        })
    }
}

#[tokio::main]
async fn main() -> std::io::Result<()> {
    let addr = std::env::args().nth(1).unwrap_or_else(|| "127.0.0.1:8791".into());
    let agent = Agent::builder("echo").model(Echo).build().expect("valid agent");
    A2aServer::new(agent, &format!("http://{addr}/a2a"))
        .description("Repeats what you say.")
        .skill(AgentSkill::new("echo", "Echo", "Repeats the input").tags(["demo"]))
        .serve(&addr)
        .await
}
