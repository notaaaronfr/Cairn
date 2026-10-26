# Contributing to Cairn

Thanks for helping. Bug reports, fixes, docs and new adapters are all welcome.

## Development setup

```bash
git clone https://github.com/whatstask/Cairn && cd Cairn
cargo test --workspace --all-features        # Rust 1.85+
cargo run --example support_agent            # offline demo, no API keys needed
```

PostgreSQL tests are skipped unless `CAIRN_TEST_DATABASE_URL` is set. To run them locally:

```bash
docker run -d --name cairn-pg -e POSTGRES_USER=cairn -e POSTGRES_PASSWORD=cairn \
  -e POSTGRES_DB=cairn_test -p 5432:5432 postgres:16
CAIRN_TEST_DATABASE_URL=postgres://cairn:cairn@localhost:5432/cairn_test cargo test --workspace --all-features
```

## Before opening a pull request

CI runs all of these; running them first saves a round trip:

```bash
cargo fmt --all
cargo clippy --workspace --all-targets --all-features -- -D warnings
cargo test --workspace --all-features
RUSTDOCFLAGS="-D warnings" cargo doc --workspace --no-deps --all-features
```

## Guidelines

- **Tests without network.** Use `cairn_runtime::testing::ScriptedModel` or a local fake server; tests
  must not call real model providers.
- **The journal is a public format.** Runs recorded by one version must load in the next: add new
  event fields with `#[serde(default)]`, never rename or remove them.
- **Side effects are explicit.** New tool-facing features must respect `Effect` and idempotency keys.
- **No hidden prompts.** Anything Cairn adds to a model request must be documented and visible in
  the request.
- **Changelog.** Add user-visible changes to the `Unreleased` section of `CHANGELOG.md`.

## License

By contributing, you agree that your contributions are licensed under the Apache License, Version
2.0, the same license as the project.
