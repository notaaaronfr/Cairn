# Changelog

All notable changes to Cairn are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project uses
[Semantic Versioning](https://semver.org/). Before 1.0, minor versions may contain breaking changes.

## [Unreleased] — 0.2.0

### Changed
- `gemini:` model specs now use the native Gemini API instead of its OpenAI-compatible endpoint,
  so thought signatures survive multi-step tool use. `OpenAiCompatModel::gemini` remains available.

### Added
- Durable workflows (`Workflow`, `WorkflowCtx`): journaled `step` and `agent` calls, resume after
  crashes and approvals, workflow-wide USD budgets that cap each child agent.
- Multi-agent orchestration: `sequential`, `concurrent`, `handoff` and `group_chat` (round-robin
  or manager-selected).
- `AgentTool`: use an agent as another agent's tool; the child's spend is charged to the parent.
- `Middleware`: `before_model`, `after_model`, `before_tool` and `after_tool` hooks to inspect,
  rewrite or refuse each step; rewrites are what get journaled.
- OpenTelemetry: `invoke_agent`, `chat` and `execute_tool` spans with GenAI semantic-convention
  attributes; `cairn::otel::init` exports them over OTLP/HTTP (feature `otel`).
- `cairn-server`: host agents behind an HTTP API (start, inspect, time-travel, approve, resume), with
  API-key auth, a crash-recovery loop that resumes orphaned runs, and a built-in Dev UI at `/ui`.
- `cairn ui <dir>`: browse a directory of JSONL journals in the Dev UI.
- Model connectors: Azure OpenAI (`azure:` spec; API keys or Microsoft Entra ID tokens), Amazon
  Bedrock (`bedrock:` spec; Converse and ConverseStream with SigV4 or Bedrock API keys, prompt
  caching), and a native Gemini adapter (thought signatures, streaming) that also serves Vertex AI.
- `Journal::list_runs` (in-memory, JSONL and PostgreSQL journals).
- `ToolCtx::charge` and `ToolCtx::remaining_usd` for tools that spend money; tool spend is journaled
  on `ToolCompleted` and counts toward budgets.

## [0.1.0] — 2026-10-26

First public release.

### Runtime (`cairn-runtime`, used through `cairn-ai`)
- Event-sourced agent loop: every model call, tool call, approval and policy decision is journaled, and
  run state is a fold over the journal — one code path for live runs, crash recovery and time travel.
- Typed tools with declared effects (`Read` / `Write` / `Irreversible`) and stable idempotency keys.
- Human approval for gated tools; durable suspension and `resume`, across processes.
- Policies (effect-based or custom), per-tool timeouts (120 s default).
- Budgets on steps, tokens and USD. USD budgets are refused unless every model is priced, and each
  step's answer limit shrinks to fit what is left.
- `FallbackModel`: moves on for failed requests and for unusable answers (truncated, empty, missing
  JSON fields, custom checks); rejected answers still count toward budgets. `keep_last_answer()`
  lets a finish-schema retry repair an almost-right answer.
- `finish_schema` / `finish_as::<T>()`: a run can only end with a validated structured answer.
- Token streaming via `on_delta`.
- Journals: in-memory, JSON lines, and PostgreSQL (`cairn-journal-postgres`).
- Run leases so one worker drives a run at a time; `run_with_id` to name runs after external jobs.
- Evals by replay: re-run recorded runs on a candidate model without repeating side effects, score
  them (`Completed`, `FieldsMatch`, LLM `Judge`) and compare cost.

### Models (`cairn-models`)
- Native Anthropic Messages adapter (signed thinking, prompt caching, streaming).
- OpenAI-compatible adapter for OpenAI, Gemini, OpenRouter, Ollama, vLLM and any
  `compat:<name>:<model>` endpoint; replays `reasoning_content` to reasoning models such as K2.

### Gateway (`cairn-gateway`)
- Drop-in OpenAI- and Anthropic-compatible endpoint: any client format on any provider, true token
  streaming, per-key USD/token budgets, model allowlists, failover chains and an audit log.

### MCP (`cairn-ai-mcp`)
- Client for stdio and Streamable HTTP servers; tool annotations map to effects (unannotated tools
  need approval by default); idempotency keys travel in `_meta`.

### Tooling (`cairn-ai-cli`)
- `cairn show` (timeline) and `cairn state --at N` (state at any event) for journals.

[Unreleased]: https://github.com/whatstask/Cairn/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/whatstask/Cairn/releases/tag/v0.1.0
