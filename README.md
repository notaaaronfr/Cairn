# Cairn

[![CI](https://github.com/whatstask/Cairn/actions/workflows/ci.yml/badge.svg)](https://github.com/whatstask/Cairn/actions/workflows/ci.yml)
[![crates.io](https://img.shields.io/crates/v/cairn-ai.svg?cacheSeconds=3600)](https://crates.io/crates/cairn-ai)
[![docs.rs](https://img.shields.io/docsrs/cairn-ai)](https://docs.rs/cairn-ai)
[![License: Apache-2.0](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)
[![MSRV 1.85](https://img.shields.io/badge/rust-1.85%2B-orange.svg)](https://www.rust-lang.org)

Durable, governed, model-agnostic AI agents, written in Rust.

> **Status:** v0.1 — early release. The core is covered by unit, integration and PostgreSQL tests, but
> public APIs may still change before 1.0. See [CHANGELOG.md](CHANGELOG.md).

Every model call, tool call, approval and policy decision is appended to an
event journal. Agent state is **only** a fold over that journal, so one code
path gives you live execution, crash recovery, human-in-the-loop and
time-travel debugging.

## Install

```toml
[dependencies]
cairn = { package = "cairn-ai", version = "0.1" }
# optional features: "postgres" (PostgresJournal), "mcp" (MCP client)
# latest main instead: cairn = { package = "cairn-ai", git = "https://github.com/whatstask/Cairn" }
tokio = { version = "1", features = ["full"] }
```

Requires Rust 1.85+. The gateway and CLI are binaries in this repo
(`cargo install cairn-gateway cairn-ai-cli`). API docs: [docs.rs/cairn-ai](https://docs.rs/cairn-ai).

## Use it from any framework today: the gateway

Change one line in your existing app (LangGraph, CrewAI, OpenAI Agents SDK, Vercel AI SDK,
Microsoft Agent Framework, raw OpenAI/Anthropic SDKs) and every model call gets per-key budgets,
model allowlists, cross-provider failover and an audit trail:

```bash
cp gateway.example.toml gateway.toml          # set keys, routes, pricing
cargo run -p cairn-gateway -- --config gateway.toml
```

```python
# OpenAI SDK, served by Claude (or anything in the route's failover chain)
client = OpenAI(base_url="http://127.0.0.1:8787/v1", api_key="ck_dev_support")
client.chat.completions.create(model="smart", messages=[...])

# Anthropic SDK, served by GPT, Gemini or a local model just as easily
client = Anthropic(base_url="http://127.0.0.1:8787", api_key="ck_dev_support")
```

Endpoints: `POST /v1/chat/completions`, `POST /v1/messages`, `GET /v1/models`,
`GET /v1/cairn/usage`, `GET /healthz`. Streaming is true token streaming in both wire formats,
whichever provider serves the request; failover applies until the first token is sent.

## Native framework quickstart

```bash
cargo test --workspace                       # run the test suite
cargo run --example support_agent            # offline demo (scripted model)
CAIRN_MODEL=anthropic:claude-opus-5-5 cargo run --example support_agent

cargo run -p cairn-ai-cli -- show  runs/<run_id>.jsonl          # timeline
cargo run -p cairn-ai-cli -- state runs/<run_id>.jsonl --at 3   # state at event 3
```

Model specs: `anthropic:…`, `openai:…`, `gemini:…`, `ollama:…`, `openrouter:…`
(keys from `ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, `GEMINI_API_KEY`, `OPENROUTER_API_KEY`),
and `compat:<name>:<model>` for any OpenAI-compatible server — e.g.
`compat:k2:IFM/K2-Horizon-375B-A23B` reads `K2_BASE_URL` and `K2_API_KEY`.

### Streaming and Postgres

```rust
let agent = Agent::builder("support")
    .model(cairn::models::from_spec("anthropic:claude-opus-5-5")?)
    .on_delta(|d| if let StreamEvent::TextDelta(t) = d { print!("{t}") })   // live tokens
    .journal(cairn::PostgresJournal::connect(&database_url).await?)        // features = ["postgres"]
    .build()?;
```

With several workers, each run is driven by one worker at a time: `resume` takes a lease (renewed
in the background, 60 s by default via `.lease_ttl(..)`), other workers get `AgentError::Leased`, and a
crashed worker's runs can be taken over once its lease expires. Postgres leases use the database clock.

A worker serving an outside queue can name its runs after the job: `agent.run_with_id("task-42-1", input)`
refuses an id already in the journal, so after a crash it can `resume` by the same id without having stored
a generated one first.

Run the Postgres tests with `CAIRN_TEST_DATABASE_URL=postgres://user@host/db cargo test -p cairn-journal-postgres`
(they are skipped when the variable is unset).

### Reliability controls

```rust
let k2 = Priced::new(cairn::models::from_spec("compat:k2:IFM/K2-Horizon-375B-A23B")?, Pricing::new(0.0, 0.0));
let haiku = Priced::new(cairn::models::from_spec("anthropic:claude-haiku-5-5")?, Pricing::new(/* $/Mtok */ 1.0, 5.0));

let agent = Agent::builder("worker")
    // K2 first; Haiku if K2 fails *or* its answer is cut off, empty, or missing fields.
    .model(FallbackModel::new(vec![Arc::new(k2), Arc::new(haiku)])
        .require_json_fields(&["summary", "work", "risks", "confidence"])
        .keep_last_answer())                      // if none is usable, let finish_as's retry fix the last
    .finish_as::<Handoff>()                       // can only finish with a valid handoff (1 retry)
    .budget(Budget::default().max_usd(0.25))     // refused at build() if any model is unpriced;
                                                  // each step's max_tokens shrinks to fit what's left
    .tool(fetch_page.with_timeout(Duration::from_secs(20)))   // default tool timeout: 120s
    .build()?;
```

(Example prices are placeholders — use your providers' current rates.)

### Workflows and multi-agent orchestration

```rust
let triage = Workflow::new("triage", journal.clone(), |ctx: WorkflowCtx, ticket: String| async move {
    let facts = ctx.agent("research", &researcher, &ticket).await?.into_text()?;          // durable child run
    let reviews = ctx.concurrent("review", &[&legal, &tone], &facts).await?;               // in parallel
    let reply = ctx.handoff("reply", &[&writer, &billing], "writer", &facts, 3).await?;     // agents hand off
    Ok(reply.text())
})
.max_usd(1.00);                                       // children's budgets are capped to what is left

let outcome = triage.run_with_id("ticket-42", ticket).await?;   // after a crash or approval: triage.resume("ticket-42")
```

Every `ctx.step(..)` / `ctx.agent(..)` result is journaled: `resume` skips finished steps, resumes a
child agent that was mid-run, and a child waiting for approval suspends the whole workflow
(`WorkflowStatus::Suspended`) until it is decided. Patterns: `sequential`, `concurrent`, `handoff`,
`group_chat` (round-robin or manager-selected). Any agent can also be a tool of another with
`AgentTool::new(agent, "research", "…")`; the child's spend is charged to the parent's budget.

### MCP tools

```rust
// features = ["mcp"]
use cairn::mcp::{McpClient, ToolOptions};

let whatstask = McpClient::connect_http(&url, [("authorization", format!("Bearer {token}"))]).await?;
let tools = whatstask
    .tools(ToolOptions::default().effect("submit_work", Effect::Write))   // list_my_tasks, get_task, …
    .await?;
let agent = Agent::builder("worker").model(model).tools(tools).build()?;
```

Stdio servers: `McpClient::connect_stdio("npx", &["-y", "some-mcp-server"], env)`. Tool effects come
from MCP annotations (`readOnlyHint` → Read, `destructiveHint: false` → Write, otherwise Irreversible,
which requires approval by default). Credentials stay in the client; every call carries Cairn's
idempotency key in `_meta`.

### Evals by replay

```rust
use cairn::eval::{Completed, EvalCase, Evaluator, FieldsMatch, Judge};

let cases = EvalCase::load(journal.as_ref(), &recent_run_ids).await?;          // recorded production runs
let k2_worker = production_agent.to_builder().model(k2).build()?;               // candidate configuration
let report = Evaluator::new(k2_worker)
    .scorer(Completed)
    .scorer(FieldsMatch::new(&["risks"]))
    .scorer(Judge::new(haiku, "Same outcome; risks stated honestly."))
    .run(&cases)
    .await;
println!("{}", report.to_markdown());                                           // scores + cost, baseline vs candidate
assert!(report.pass_rate(0.8) >= 0.9);                                          // gate CI
```

Replays never repeat side effects: matching tool calls get the recorded result, unmatched read-only
calls may run live (`LiveTools::ReadOnly`, default) and writes never run. Replays use their own
in-memory journal.

## Principles

1. **The journal is the source of truth.** Resume, replay, audit and evals all derive from it.
2. **No hidden prompts.** Everything sent to a model is inspectable and overridable.
3. **Model-agnostic without the lowest common denominator.** One neutral message format;
   provider features (signed thinking, prompt caching) survive via tagged `provider_data`
   and per-provider `provider_options`.
4. **Tools declare their effects.** `Read` / `Write` / `Irreversible` drive policy, and every
   call gets a stable idempotency key for exactly-once side effects across retries and crashes.
5. **Hard budgets.** Steps, tokens and USD are enforced before every model call, each call's
   answer limit is capped to what is left, and a USD budget is refused unless every model is priced.

## Layout

| Crate | Purpose |
|---|---|
| `cairn-runtime` | Events, journal, state fold, agent loop, tools, policy, budgets |
| `cairn-models` | Anthropic (native) and OpenAI-compatible adapters (OpenAI, Gemini, Ollama, vLLM, OpenRouter…) |
| `cairn-ai` | Facade crate users depend on (`use cairn::prelude::*`) |
| `cairn-gateway` | Drop-in OpenAI/Anthropic-compatible gateway: budgets, allowlists, failover, audit |
| `cairn-journal-postgres` | PostgreSQL journal: many processes/machines can drive, approve and resume the same runs |
| `cairn-ai-mcp` | MCP client (stdio + Streamable HTTP): any MCP server's tools as Cairn tools |
| `cairn-ai-cli` | `cairn show` / `cairn state` journal inspector |

## Roadmap

Done in 0.1:

- Durable event-sourced loop, approvals, policies, budgets, JSONL and PostgreSQL journals, CLI
- Gateway: OpenAI + Anthropic wire formats, any-to-any translation, budgets, failover, audit
- True token streaming across providers, the agent (`on_delta`) and the gateway
- Run leases, `run_with_id`, budget hardening, usability-checked fallback, `finish_schema`
- MCP client (stdio + Streamable HTTP), evals by replay, per-tool timeouts, `compat:` model specs
- Durable workflows: sequential, concurrent, handoff and group-chat orchestration; agents as tools

Next:

- [ ] Gateway spend shared across replicas (Postgres-backed)
- [ ] Parallel execution of read-only tools; tool retries
- [ ] Context engine: compaction policies, token budgets per section
- [ ] Per-agent credential brokering for MCP (OAuth)
- [ ] OpenTelemetry GenAI spans

## Contributing

Issues and pull requests are welcome — see [CONTRIBUTING.md](CONTRIBUTING.md). Please report security
issues privately as described in [SECURITY.md](SECURITY.md).

Created by **Aaron David Charles** · [charlesaarondavid.com](https://charlesaarondavid.com) · `@charlesaarondavid`


## License

Copyright 2026 Aaron David Charles.
Licensed under the [Apache License, Version 2.0](LICENSE). See [NOTICE](NOTICE).
