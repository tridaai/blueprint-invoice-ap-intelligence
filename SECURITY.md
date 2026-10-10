# Security

## Prototype — not hardened for exposure

This blueprint is a **reference prototype**. Optional API-key auth exists
(`API_KEY`), but it is a shared-secret gate for local demos, not an
identity system.

- Run it locally or inside a network you control.
- **Do not expose it to the public internet** as-is. A production
  deployment needs authenticated access, per-client data isolation, and
  the rest of the hardening listed in `docs/architecture.md` (Section 13).

## Data

- All bundled data is synthetic. Never put real supplier, invoice,
  purchase-order, or bank data into this prototype, and never include
  real data in issues, pull requests, logs, or screenshots.
- Supplier text is treated as untrusted input throughout: it is
  screened for prompt injection and bank-change fraud patterns at
  ingest, flagged sentences are removed before any prompt is built, and
  the guardrail layer — not the model — decides what may reach a
  supplier letter or AP packet. Bank details appearing in documents are
  never transcribed into drafts.
- The code reads optional LLM API keys from environment variables. The
  application loads a `.env` file from the repo root at startup (real
  environment variables take precedence); `.env.example` lists every
  variable. `.env` is git-ignored — never commit a filled-in one.

## Reporting a concern

If you find a security concern in this repository, please report it
privately through Trida AI's contact route at
https://trida.ai/intake rather than opening a public issue.
