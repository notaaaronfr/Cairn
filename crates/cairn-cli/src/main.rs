use std::path::{Path, PathBuf};
use std::process::ExitCode;

use cairn_runtime::{Content, Event, EventKind, JsonlJournal, RunState};
use clap::{Parser, Subcommand};

#[derive(Parser)]
#[command(name = "cairn", about = "Inspect and time-travel through Cairn run journals")]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Print a run's timeline.
    Show { journal: PathBuf },
    /// Print the reconstructed run state after event `--at` (default: latest).
    State {
        journal: PathBuf,
        #[arg(long)]
        at: Option<u64>,
    },
    /// Browse a directory of JSONL journals in the Dev UI (read-only).
    Ui {
        /// Directory containing <run_id>.jsonl files.
        dir: PathBuf,
        #[arg(long, default_value = "127.0.0.1:8790")]
        addr: String,
    },
}

fn main() -> ExitCode {
    let cli = Cli::parse();
    let result = match cli.command {
        Command::Show { journal } => show(&journal),
        Command::State { journal, at } => state(&journal, at),
        Command::Ui { dir, addr } => ui(dir, &addr),
    };
    match result {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("error: {e}");
            ExitCode::FAILURE
        }
    }
}

fn ui(dir: PathBuf, addr: &str) -> Result<(), String> {
    let journal = JsonlJournal::new(&dir).map_err(|e| format!("{}: {e}", dir.display()))?;
    let runtime = tokio::runtime::Runtime::new().map_err(|e| e.to_string())?;
    println!("Cairn Dev UI for {} at http://{addr}/ui (read-only; Ctrl-C to stop)", dir.display());
    runtime.block_on(cairn_server::Server::new(std::sync::Arc::new(journal)).serve(addr)).map_err(|e| e.to_string())
}

fn load(path: &Path) -> Result<Vec<Event>, String> {
    JsonlJournal::read_file(path).map_err(|e| format!("{}: {e}", path.display()))
}

fn show(path: &Path) -> Result<(), String> {
    let events = load(path)?;
    let Some(first) = events.first() else { return Err("journal is empty".into()) };
    println!("run {}", first.run_id);
    for e in &events {
        println!("{:>4}  {}  {:<18} {}", e.seq, e.at.format("%H:%M:%S%.3f"), e.kind.name(), summary(&e.kind));
    }
    let state = RunState::from_events(&events).expect("non-empty");
    println!(
        "\nstatus: {}   steps: {}   tokens: {}   cost: ${:.6}",
        serde_json::to_value(&state.status).map(|v| v["status"].to_string()).unwrap_or_default(),
        state.steps,
        state.usage.total(),
        state.cost_usd
    );
    Ok(())
}

fn state(path: &Path, at: Option<u64>) -> Result<(), String> {
    let events = load(path)?;
    let end = match at {
        Some(n) if (n as usize) < events.len() => n as usize + 1,
        Some(n) => return Err(format!("event {n} does not exist (journal has {})", events.len())),
        None => events.len(),
    };
    let state = RunState::from_events(&events[..end]).ok_or("journal is empty")?;
    println!("{}", serde_json::to_string_pretty(&state).map_err(|e| e.to_string())?);
    Ok(())
}

fn truncate(s: &str, n: usize) -> String {
    if s.chars().count() <= n { s.to_owned() } else { format!("{}…", s.chars().take(n).collect::<String>()) }
}

fn summary(kind: &EventKind) -> String {
    match kind {
        EventKind::RunStarted { agent, input } => format!("agent={agent} input={:?}", truncate(&input.text(), 60)),
        EventKind::ModelCalled { step, response, cost_usd, duration_ms } => {
            let parts: Vec<String> = response
                .content
                .iter()
                .map(|c| match c {
                    Content::Text { text } => format!("text({:?})", truncate(text, 50)),
                    Content::ToolUse { name, .. } => format!("call({name})"),
                    Content::Thinking { .. } => "thinking".into(),
                    Content::ToolResult { .. } => "result".into(),
                })
                .collect();
            format!(
                "step={step} model={} {}ms tokens={} cost={} → {}",
                response.model,
                duration_ms,
                response.usage.total(),
                cost_usd.map(|c| format!("${c:.6}")).unwrap_or("-".into()),
                parts.join(", ")
            )
        }
        EventKind::ModelFailed { step, error, .. } => format!("step={step} {}", truncate(error, 80)),
        EventKind::ToolDenied { tool, reason, .. } => format!("{tool}: {reason}"),
        EventKind::ApprovalRequested { tool, input, reason, .. } => {
            format!("{tool} {} ({reason})", truncate(&input.to_string(), 60))
        }
        EventKind::ApprovalDecided { approved, by, .. } => {
            format!("{} by {}", if *approved { "approved" } else { "rejected" }, by.as_deref().unwrap_or("?"))
        }
        EventKind::ToolCompleted { tool, output, is_error, duration_ms, .. } => format!(
            "{tool}{} {duration_ms}ms → {}",
            if *is_error { " ERROR" } else { "" },
            truncate(&output.to_string(), 60)
        ),
        EventKind::BudgetExceeded { reason } => reason.clone(),
        EventKind::RunCompleted { output, .. } => format!("{:?}", truncate(output, 70)),
        EventKind::ContextCompacted { replaced, summary, .. } => {
            format!("replaced {replaced} messages → {}", truncate(summary, 70))
        }
        EventKind::FinishRejected { step, reason } => format!("step={step} {}", truncate(reason, 80)),
        EventKind::RunFailed { error } => truncate(error, 80),
        EventKind::RunCanceled { reason, by } => format!("{reason} (by {})", by.as_deref().unwrap_or("?")),
        EventKind::UserMessage { message } => format!("input={:?}", truncate(&message.text(), 60)),
        EventKind::WorkflowStarted { workflow, input } => {
            format!("workflow={workflow} input={}", truncate(&input.to_string(), 60))
        }
        EventKind::StepCompleted { step, output, is_error, child_run, cost_usd, .. } => format!(
            "{step}{}{} cost={} → {}",
            if *is_error { " ERROR" } else { "" },
            child_run.as_deref().map(|c| format!(" child={c}")).unwrap_or_default(),
            cost_usd.map(|c| format!("${c:.6}")).unwrap_or("-".into()),
            truncate(&output.to_string(), 60)
        ),
        EventKind::WorkflowCompleted { output } => truncate(&output.to_string(), 80),
        EventKind::WorkflowFailed { error } => truncate(error, 80),
    }
}
