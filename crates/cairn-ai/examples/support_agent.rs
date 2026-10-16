//! A support agent that can look up orders freely but needs human approval
//! to issue refunds.
//!
//! Offline (scripted model):   cargo run --example support_agent
//! Live:  CAIRN_MODEL=anthropic:claude-opus-5-5 cargo run --example support_agent
//!        (also works with openai:..., gemini:..., ollama:..., openrouter:...)
//!
//! Then inspect the run:      cargo run -p cairn-ai-cli -- show runs/<run_id>.jsonl

use std::sync::Arc;

use cairn::prelude::*;
use cairn::testing::{ScriptedModel, text_response, tool_call_response};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use serde_json::json;

#[derive(Deserialize, JsonSchema)]
struct GetOrder {
    /// Order number, e.g. "1042".
    order_id: String,
}

#[derive(Serialize)]
struct Order {
    order_id: String,
    item: String,
    total_usd: f64,
    status: String,
}

#[derive(Deserialize, JsonSchema)]
struct Refund {
    order_id: String,
    amount_usd: f64,
    reason: String,
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let get_order = tool("get_order", "Look up an order by id.", Effect::Read, |i: GetOrder, _ctx| async move {
        Ok(Order { order_id: i.order_id, item: "Noise-cancelling headphones".into(), total_usd: 249.0, status: "delivered, damaged".into() })
    });

    let refund = tool(
        "refund",
        "Refund an order. Irreversible.",
        Effect::Irreversible,
        |i: Refund, ctx: ToolCtx| async move {
            // Pass ctx.idempotency_key to the payment provider for exactly-once refunds.
            println!("  💸 refunding ${:.2} on order {} ({}) [idempotency key {}]", i.amount_usd, i.order_id, i.reason, ctx.idempotency_key);
            Ok(json!({ "refund_id": "rf_123", "status": "succeeded" }))
        },
    );

    let model: Arc<dyn Model> = match std::env::var("CAIRN_MODEL") {
        Ok(spec) => cairn::models::from_spec(&spec)?,
        Err(_) => {
            println!("(CAIRN_MODEL not set — using a scripted model)\n");
            Arc::new(ScriptedModel::new(vec![
                tool_call_response(&[("call_1", "get_order", json!({ "order_id": "1042" }))]),
                tool_call_response(&[(
                    "call_2",
                    "refund",
                    json!({ "order_id": "1042", "amount_usd": 249.0, "reason": "arrived damaged" }),
                )]),
                text_response("Your order 1042 arrived damaged, so I've refunded the full $249.00."),
            ]))
        }
    };

    let agent = Agent::builder("support")
        .model(model)
        .system("You are a support agent. Look up the order before acting. Refund damaged items in full.")
        .tool(get_order)
        .tool(refund)
        .budget(Budget::default().max_steps(10).max_usd(0.50))
        .journal(JsonlJournal::new("runs")?)
        .build()?;

    let mut outcome = agent.run("Order 1042 arrived damaged. I'd like a refund.").await?;

    while outcome.status == Status::AwaitingApproval {
        for call in &outcome.awaiting_approval {
            println!("⏸  approval needed: {} {}", call.name, call.input);
            // A real app would notify a human and resume later — even from another process.
            agent.decide(&outcome.run_id, &call.id, true, Some("ops@example.com"), None).await?;
            println!("✅ approved by ops@example.com");
        }
        outcome = agent.resume(&outcome.run_id).await?;
    }

    println!("\nstatus: {:?}", outcome.status);
    println!("steps: {}, tokens: {}, cost: ${:.6}", outcome.steps, outcome.usage.total(), outcome.cost_usd);
    println!("journal: runs/{}.jsonl", outcome.run_id);
    Ok(())
}
