//! Export Cairn's spans to any OpenTelemetry backend over OTLP/HTTP.
//!
//! Cairn emits `invoke_agent`, `chat` and `execute_tool` spans with
//! OpenTelemetry GenAI attributes (`gen_ai.request.model`,
//! `gen_ai.usage.input_tokens`, `gen_ai.tool.name`, …). This helper wires
//! them to an OTLP collector (Datadog, Grafana, Honeycomb, Jaeger, Azure
//! Monitor, …):
//!
//! ```ignore
//! let _otel = cairn::otel::init("http://localhost:4318/v1/traces", "support-worker")?;
//! // ... run agents; spans are batched and exported. Dropping the guard flushes them.
//! ```
//!
//! Already using `tracing-subscriber`? Add [`layer`](crate::otel::layer) to your own registry instead.

use opentelemetry::trace::TracerProvider as _;
use opentelemetry_otlp::{SpanExporter, WithExportConfig};
use opentelemetry_sdk::Resource;
use opentelemetry_sdk::trace::SdkTracerProvider;
use tracing_subscriber::layer::SubscriberExt;
use tracing_subscriber::util::SubscriberInitExt;

/// Flushes and shuts down the exporter when dropped.
pub struct OtelGuard {
    provider: SdkTracerProvider,
}

impl Drop for OtelGuard {
    fn drop(&mut self) {
        if let Err(e) = self.provider.shutdown() {
            eprintln!("cairn: OpenTelemetry shutdown failed: {e}");
        }
    }
}

/// Build a provider exporting to `endpoint` (e.g. `http://localhost:4318/v1/traces`).
pub fn provider(
    endpoint: &str,
    service_name: &str,
) -> Result<SdkTracerProvider, Box<dyn std::error::Error + Send + Sync>> {
    let exporter = SpanExporter::builder().with_http().with_endpoint(endpoint).build()?;
    Ok(SdkTracerProvider::builder()
        .with_batch_exporter(exporter)
        .with_resource(Resource::builder().with_service_name(service_name.to_owned()).build())
        .build())
}

/// A `tracing` layer sending spans to `provider`, for use in your own subscriber.
pub fn layer<S>(provider: &SdkTracerProvider) -> impl tracing_subscriber::Layer<S>
where
    S: tracing::Subscriber + for<'a> tracing_subscriber::registry::LookupSpan<'a>,
{
    tracing_opentelemetry::layer().with_tracer(provider.tracer("cairn"))
}

/// Install a global subscriber that exports spans over OTLP/HTTP and logs
/// to stderr (filtered by `RUST_LOG`, default `info`).
pub fn init(endpoint: &str, service_name: &str) -> Result<OtelGuard, Box<dyn std::error::Error + Send + Sync>> {
    let provider = provider(endpoint, service_name)?;
    let filter = tracing_subscriber::EnvFilter::try_from_default_env().unwrap_or_else(|_| "info".into());
    tracing_subscriber::registry()
        .with(filter)
        .with(tracing_subscriber::fmt::layer().with_writer(std::io::stderr))
        .with(layer(&provider))
        .try_init()?;
    Ok(OtelGuard { provider })
}
