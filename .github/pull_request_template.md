## What and why

<!-- What does this change, and what problem does it solve? -->

## How it was tested

<!-- New or updated tests; anything run manually. -->

## Checklist

- [ ] `cargo fmt --all` and `cargo clippy --workspace --all-targets --all-features` are clean
- [ ] `cargo test --workspace --all-features` passes (set `CAIRN_TEST_DATABASE_URL` to include Postgres tests)
- [ ] Public API changes are documented and listed in `CHANGELOG.md`
- [ ] Journal format changes stay backward compatible (new fields use `#[serde(default)]`)
