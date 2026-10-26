use std::sync::{Arc, Mutex};

use async_trait::async_trait;
use cairn_runtime::testing::{ScriptedModel, text_response, tool_call_response};
use cairn_runtime::*;
use schemars::JsonSchema;
use serde::Deserialize;
use serde_json::{Value, json};

#[derive(Deserialize, JsonSchema)]
struct Refund {
    amount: f64,
}

fn refund_tool(seen: Arc<Mutex<Vec<f64>>>) -> impl Tool + 'static {
    tool("refund", "Refund", Effect::Write, move |r: Refund, _ctx| {
        let seen = seen.clone();
        async move {
            seen.lock().unwrap().push(r.amount);
            Ok(json!({"refunded": r.amount, "card": "4111-1111-1111-1111"}))
        }
    })
}

/// Redacts card numbers going to the model, caps refunds, masks tool output.
struct Compliance;

#[async_trait]
impl Middleware for Compliance {
    async fn before_model(&self, _: &HookCtx, req: &mut ModelRequest) -> Result<(), String> {
        for m in &mut req.messages {
            for c in &mut m.content {
                if let Content::Text { text } = c {
                    *text = text.replace("4111-1111-1111-1111", "[CARD]");
                }
            }
        }
        Ok(())
    }

    async fn before_tool(&self, _: &HookCtx, tool: &ToolSpec, input: &mut Value) -> Result<(), String> {
        if tool.name == "refund" {
            let amount = input["amount"].as_f64().unwrap_or(0.0);
            if amount > 1000.0 {
                return Err(format!("refunds over $1000 are not allowed (asked for ${amount})"));
            }
            if amount > 100.0 {
                input["amount"] = json!(100.0); // clamp
            }
        }
        Ok(())
    }

    async fn after_tool(&self, _: &HookCtx, _: &ToolSpec, output: &mut Value, _: &mut bool) -> Result<(), String> {
        if output.get("card").is_some() {
            output["card"] = json!("****1111");
        }
        Ok(())
    }

    async fn after_model(&self, _: &HookCtx, resp: &mut ModelResponse) -> Result<(), String> {
        for c in &mut resp.content {
            if let Content::Text { text } = c {
                *text = text.trim().to_owned();
            }
        }
        Ok(())
    }
}

#[tokio::test]
async fn hooks_redact_clamp_block_and_mask() {
    let seen = Arc::new(Mutex::new(Vec::new()));
    let model = Arc::new(ScriptedModel::new(vec![
        tool_call_response(&[("c1", "refund", json!({"amount": 250.0})), ("c2", "refund", json!({"amount": 5000.0}))]),
        text_response("   Refunded $100; the larger refund was blocked.   "),
    ]));
    let agent = Agent::builder("billing")
        .model(model.clone())
        .tool(refund_tool(seen.clone()))
        .middleware(Compliance)
        .build()
        .unwrap();

    let out = agent.run("My card 4111-1111-1111-1111 was charged twice").await.unwrap();
    assert_eq!(out.output(), Some("Refunded $100; the larger refund was blocked."), "after_model trimmed");

    let requests = model.requests();
    assert_eq!(requests[0].messages[0].text(), "My card [CARD] was charged twice", "before_model redacted");
    assert_eq!(*seen.lock().unwrap(), [100.0], "250 clamped to 100; 5000 never ran");

    let results = &requests[1].messages.last().unwrap().content;
    assert!(
        matches!(&results[0], Content::ToolResult { content, is_error: false, .. } if content["card"] == "****1111")
    );
    assert!(matches!(&results[1], Content::ToolResult { content, is_error: true, .. }
        if content["error"].as_str().unwrap().contains("over $1000")));

    // The journal holds what the agent saw: masked output, trimmed answer.
    let events = agent.journal().load(&out.run_id).await.unwrap();
    assert!(
        events
            .iter()
            .any(|e| matches!(&e.kind, EventKind::ToolCompleted { output, .. } if output["card"] == "****1111"))
    );
    assert!(
        events.iter().any(
            |e| matches!(&e.kind, EventKind::ToolDenied { reason, .. } if reason.contains("blocked by middleware"))
        )
    );
}

struct Order(&'static str, Arc<Mutex<Vec<&'static str>>>);

#[async_trait]
impl Middleware for Order {
    async fn before_model(&self, _: &HookCtx, _: &mut ModelRequest) -> Result<(), String> {
        self.1.lock().unwrap().push(self.0);
        if self.0 == "block" { Err("policy says no".into()) } else { Ok(()) }
    }
}

#[tokio::test]
async fn hooks_run_in_order_and_can_fail_the_run() {
    let log = Arc::new(Mutex::new(Vec::new()));
    let model = Arc::new(ScriptedModel::new(vec![text_response("never")]));
    let agent = Agent::builder("a")
        .model(model.clone())
        .middleware(Order("first", log.clone()))
        .middleware(Order("block", log.clone()))
        .middleware(Order("never-reached", log.clone()))
        .build()
        .unwrap();
    let out = agent.run("hi").await.unwrap();
    assert!(matches!(&out.status, Status::Failed { error } if error.contains("policy says no")));
    assert_eq!(*log.lock().unwrap(), ["first", "block"]);
    assert!(model.requests().is_empty(), "the model was never called");
}
