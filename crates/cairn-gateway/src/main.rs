use std::path::PathBuf;

use cairn_gateway::{Config, Gateway};
use clap::Parser;

#[derive(Parser)]
#[command(name = "cairn-gateway", about = "Drop-in OpenAI/Anthropic-compatible gateway with budgets, failover and audit")]
struct Args {
    #[arg(long, short, default_value = "gateway.toml")]
    config: PathBuf,
    /// Print the SHA-256 of a key for use as `key_sha256`, then exit.
    #[arg(long)]
    hash_key: Option<String>,
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::try_from_default_env().unwrap_or_else(|_| "info".into()))
        .init();
    let args = Args::parse();
    if let Some(k) = args.hash_key {
        println!("{}", cairn_gateway::config::sha256_hex(&k));
        return Ok(());
    }
    let text = std::fs::read_to_string(&args.config).map_err(|e| format!("{}: {e}", args.config.display()))?;
    let config = Config::from_toml(&text)?;
    let listen = config.listen.clone();
    let gateway = Gateway::new(config).await?;
    let listener = tokio::net::TcpListener::bind(&listen).await?;
    tracing::info!("cairn-gateway listening on http://{listen}");
    axum::serve(listener, gateway.router()).await?;
    Ok(())
}
