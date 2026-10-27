//! Separate test binary: tracing caches whether a span site is enabled, so
//! span assertions must not share a process with tests that run without a
//! subscriber.

use std::collections::HashMap;
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::{Arc, Mutex};

use cairn_runtime::testing::{ScriptedModel, text_response, tool_call_response};
use cairn_runtime::*;
use schemars::JsonSchema;
use serde::Deserialize;
use serde_json::json;

#[derive(Deserialize, JsonSchema)]
#[allow(dead_code)]
struct Refund {
    amount: f64,
}

/// Collects spans and their fields.
type Recorded = Vec<(String, HashMap<String, String>)>;

#[derive(Clone, Default)]
struct Spans(Arc<Mutex<Recorded>>);

struct Visitor<'a>(&'a mut HashMap<String, String>);

impl tracing::field::Visit for Visitor<'_> {
    fn record_debug(&mut self, field: &tracing::field::Field, value: &dyn std::fmt::Debug) {
        self.0.insert(field.name().to_owned(), format!("{value:?}").trim_matches('"').to_owned());
    }
}

impl<S: tracing::Subscriber + for<'a> tracing_subscriber::registry::LookupSpan<'a>> tracing_subscriber::Layer<S>
    for Spans
{
    fn on_new_span(
        &self,
        attrs: &tracing::span::Attributes<'_>,
        id: &tracing::span::Id,
        _: tracing_subscriber::layer::Context<'_, S>,
    ) {
        let mut fields = HashMap::new();
        attrs.record(&mut Visitor(&mut fields));
        fields.insert("__id".into(), id.into_u64().to_string());
        self.0.lock().unwrap().push((attrs.metadata().name().to_owned(), fields));
    }

    fn on_record(
        &self,
        id: &tracing::span::Id,
        values: &tracing::span::Record<'_>,
        _: tracing_subscriber::layer::Context<'_, S>,
    ) {
        let mut spans = self.0.lock().unwrap();
        if let Some((_, fields)) = spans.iter_mut().rev().find(|(_, f)| f["__id"] == id.into_u64().to_string()) {
            values.record(&mut Visitor(fields));
        }
    }
}

#[tokio::test]
async fn emits_opentelemetry_genai_spans() {
    use tracing_subscriber::layer::SubscriberExt;
    let spans = Spans::default();
    let _guard = tracing::subscriber::set_default(tracing_subscriber::registry().with(spans.clone()));
    tracing::callsite::rebuild_interest_cache();

    let calls = Arc::new(AtomicU32::new(0));
    let c = calls.clone();
    let agent = Agent::builder("support")
        .model(ScriptedModel::new(vec![
            tool_call_response(&[("c1", "refund", json!({"amount": 5.0}))]),
            text_response("ok"),
        ]))
        .tool(tool("refund", "Refund", Effect::Write, move |_: Refund, _ctx| {
            let c = c.clone();
            async move {
                c.fetch_add(1, Ordering::SeqCst);
                Ok(json!("done"))
            }
        }))
        .build()
        .unwrap();
    agent.run("refund me").await.unwrap();

    let spans = spans.0.lock().unwrap();
    let find = |name: &str| spans.iter().filter(|(n, _)| n == name).map(|(_, f)| f.clone()).collect::<Vec<_>>();
    let agent_span = &find("invoke_agent")[0];
    assert_eq!(agent_span["gen_ai.operation.name"], "invoke_agent");
    assert_eq!(agent_span["gen_ai.agent.name"], "support");

    let chats = find("chat");
    assert_eq!(chats.len(), 2);
    assert_eq!(chats[0]["gen_ai.provider.name"], "scripted");
    assert_eq!(chats[0]["gen_ai.usage.input_tokens"], "100");
    assert_eq!(chats[0]["gen_ai.usage.output_tokens"], "20");
    assert_eq!(chats[0]["gen_ai.response.finish_reasons"], "tooluse");

    let tool_span = &find("execute_tool")[0];
    assert_eq!(tool_span["gen_ai.tool.name"], "refund");
    assert_eq!(tool_span["gen_ai.tool.call.id"], "c1");
    assert_eq!(calls.load(Ordering::SeqCst), 1);
}
