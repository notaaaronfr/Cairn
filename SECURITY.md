# Security policy

## Reporting a vulnerability

Please **do not** open a public issue. Report privately through GitHub:
[Report a vulnerability](https://github.com/whatstask/Cairn/security/advisories/new).

Include what is affected, how to reproduce it, and the impact you expect. We will acknowledge the
report, keep you updated while we investigate, and credit you in the advisory unless you prefer
otherwise.

## Supported versions

| Version | Supported |
|---|---|
| 0.1.x | Yes |

## Security model, in brief

- **Credentials never reach models.** Provider keys, gateway keys and MCP auth headers live in
  adapters and transports, not in prompts or tool results.
- **Side effects are gated.** Tools declare effects; irreversible tools need approval under the
  default policy, and MCP tools without annotations are treated as irreversible.
- **Exactly-once effects need idempotency.** Every tool call carries a stable idempotency key (also
  sent to MCP servers in `_meta`); forward it to downstream APIs.
- **Gateway keys** can be stored as SHA-256 hashes (`key_sha256`); the audit log records usage, not
  request bodies.
- **Replays are side-effect free.** Evals serve recorded tool results and never execute writes.
