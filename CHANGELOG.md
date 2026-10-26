# Changelog

All notable changes to Cairn are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project uses
[Semantic Versioning](https://semver.org/). Before 1.0, minor versions may contain breaking changes.

## [0.1.0] — Unreleased

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

[0.1.0]: https://github.com/whatstask/Cairn/releases/tag/v0.1.0
