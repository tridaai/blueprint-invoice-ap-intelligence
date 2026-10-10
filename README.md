<p align="center">
  <img src="assets/tridaai-logo.png" alt="Trida AI" width="360">
</p>

# Trida AI

## Blueprint — Invoice & Accounts-Payable Document Intelligence Agent

**Built on NVIDIA NIM.** A working prototype of the agent an accounts-payable team actually needs at the inbox: a supplier invoice arrives (scanned document + email), and the agent does the intake work end to end — OCR, document verification, field extraction, a **three-way match computed in code** (invoice vs purchase order vs goods receipt), duplicate detection, triage, policy-grounded diagnosis, scored next steps, and a drafted AP packet + supplier query letter — then **stops at the finance approver's gate**.

The lead deployment profile is NVIDIA NIM end to end: **Nemotron OCR** for the document text layer, a **Nemotron LLM** for reasoning and drafting, **NV-Embed** embeddings for policy retrieval. NIM endpoints are OpenAI-compatible, so the same code also runs on Anthropic, OpenAI, or a local Ollama — the engine is portable, NIM is the headline.

The promises are precise, and enforced in code, not in prompts:

- **The agent never pays an invoice, and never says one has been (or will be) paid.** Payment language in a draft fails validation and blocks approval.
- **The agent never acts on bank-detail changes.** A "remit to our new account" request inside a document is treated as a fraud pattern: flagged, blocked, and never echoed into a draft.
- **The agent never approves its own work.** A named finance approver decides, with a reason, at a checkpointed gate.

Everything is synthetic data. There is no real supplier, invoice, or bank detail anywhere in this repository.

---

## Try it in 3 commands

```bash
git clone https://github.com/tridaai/blueprint-invoice-ap-intelligence.git
cd blueprint-invoice-ap-intelligence
docker compose up --build        # the real stack: api + PostgreSQL/pgvector + MinIO
```

Then open http://localhost:8000 — pick a sample invoice, watch the match table, the triage cross-check, and the approver gate. The stack boots with no keys (offline fallback); add an `nvapi-` key to run it on NVIDIA NIM (below).

No Docker? The offline demo runs anywhere Python 3.10–3.12 does:

```bash
uv sync --extra dev
uv run ap-agent-demo             # one invoice, traced end to end
```

---

## The problem

AP teams drown in document work that is simple to describe and miserable to do: read the scan, check it against the PO and the delivery note, spot the price that crept up 8%, catch the invoice that was already paid last month, notice the email asking you to "update the bank account" — which is how payment-diversion fraud actually arrives. The judgement is real; the hours are clerical. This blueprint automates the clerical hours and keeps the judgement with a human.

## What this agent does

1. **OCR** — each document gets a text layer and a mean confidence via the `OcrEngine` port (Nemotron OCR NIM, or the offline text-layer reader).
2. **Vision verify** — a `DocVision` port answers: is this an invoice at all, and is it legible? (Nemotron VL NIM, or a deterministic metadata equivalent.)
3. **Screen** — supplier text is screened sentence-by-sentence for prompt-injection patterns and, on a separate channel, for bank-change fraud patterns. Flagged sentences are withheld from prompts and drafts; fraud excerpts are digit-masked.
4. **Extract** — header + line items via the LLM with per-field confidence, cross-checked deterministically against the provided fields.
5. **Three-way match (in code)** — line quantities, unit prices, tax, totals, and currency against the PO and goods receipt, with tolerance rules in code (price: 2% or 1.00, whichever is greater; totals: 1% or 1.00). The model never does this arithmetic.
6. **Duplicate detection** — same supplier + invoice number + amount against everything the store has seen (AP-DUP-01).
7. **Triage** — rules first ({clean_match, price_variance, quantity_variance, missing_po, missing_receipt, duplicate_suspected, unreadable, not_an_invoice}), then an LLM cross-check with an explicit resolution policy: agreement stands, strong rules beat a disagreeing model, a weak rule yields to a confident model.
8. **Retrieve** — the synthetic AP policy corpus (12 policies) via keyword, semantic (pgvector), or hybrid retrieval with reciprocal-rank fusion.
9. **Diagnose** — a bounded tool loop (search_policies, match_facts, supplier_history, po_lookup) grounded in the computed match, plus supplier history and prior approver feedback from the store.
10. **Options** — five next steps proposed by the model (or templates), always scored and ranked by code; a fraud block zeroes the approve option's fit.
11. **Draft** — a supplier query letter + an internal AP packet, fenced against untrusted supplier text.
12. **Verify + review** — a self-check (amounts must come from the match, citations must be retrieved, no payment language) and an independent reviewer that did not draft.
13. **Guardrails (code)** — `no_payment_language`, `no_prohibited_promises`, `no_pii_in_draft`, citations present, invoice id present. Failures trigger a bounded repair loop; a still-failing draft **cannot be approved** (the API returns 422).
14. **Gate** — the run checkpoints and waits. A finance approver approves or rejects with name + reason. An autonomy recommendation (auto-triage eligible only for clean matches under the approval threshold with every check green) is printed, never acted on.

## Quickstart

### A. NVIDIA NIM — hosted (the lead profile)

Get an `nvapi-` key from build.nvidia.com, then:

```bash
cp .env.example .env   # the file is pre-filled for the NIM profile — paste your key
uv sync --extra dev --extra llm --extra postgres --extra s3
uv run ap-agent-demo   # the trace header names the backend that produced it
```

`.env.example` documents the full NIM profile: Nemotron LLM (`MODEL_BACKEND=openai` + the integrate endpoint), Nemotron OCR (`OCR_ENGINE=nim`), Nemotron VL (`VISION_ENGINE=nim`), NV-Embed (`RETRIEVER=hybrid` + `OPENAI_EMBEDDING_MODEL=nvidia/nv-embed-v1`). For self-hosted NIM containers, point the `*_BASE_URL` variables at your containers instead — no code changes.

> Live NIM runs are the one path not yet exercised end-to-end by the maintainers (no key in the dev sandbox); the NIM seams are covered by fake-based tests. The OCR endpoint's chat-completions shape is a documented adapter assumption — see `docs/architecture.md`.

### B. The full stack locally (compose)

```bash
docker compose up --build
```

This runs **the real stack locally** — the same components a production deployment uses: the stateless API service, PostgreSQL 16 with pgvector (approvals store + LangGraph checkpointer + policy embeddings in one database), and MinIO (S3-compatible document storage). State lives in the services, never in the app container or loose files. Add `-f docker-compose.local.yml --profile local-llm` for a local Ollama.

### C. Offline smoke test (no keys, no Docker)

```bash
uv sync --extra dev
make demo        # traced run over a sample invoice
make test        # 96 tests
make evals       # golden set: 28/28 outcome + match status
```

The offline path uses a deterministic template backend and the documents' provided text layer. It is labelled everywhere it appears: it proves the plumbing, not the model quality. The LLM-judge eval pack (`make llm-evals`) refuses to run without a real provider — exit code 2 — rather than grade its own templates.

## The 12 sample invoices

| ID | Scenario | Designed outcome |
|---|---|---|
| INV-1001 | Clean three-way match | clean_match |
| INV-1002 | Unit price +8% over PO | price_variance (high) |
| INV-1003 | Short shipment (80 of 100 received) | quantity_variance |
| INV-1004 | Same supplier/number/amount as INV-1001 | duplicate_suspected |
| INV-1005 | No purchase order | missing_po |
| INV-1006 | PO present, no goods receipt | missing_receipt |
| INV-1007 | Clean match, EUR | clean_match |
| INV-1008 | Tax does not reconcile | price_variance |
| INV-1009 | Poor scan, OCR confidence 0.32 | unreadable |
| INV-1010 | **Adversarial:** "updated bank account" in notes | fraud flag + block (triage stays clean_match) |
| INV-1011 | **Adversarial:** injection in supplier email | flagged + stripped; clean_match |
| INV-1012 | **Trap:** note relays a payment-date demand | clean_match, but the draft fails `no_payment_language` → unapprovable offline |

Duplicate detection is store-backed: INV-1004 only duplicates because INV-1001 was processed first — run the batch (`ap-agent --all`) or the API in order to see it.

## Configuration

Everything is environment variables (`.env` supported; real env wins). The full reference is `.env.example`. The load-bearing ones:

| Variable | Meaning |
|---|---|
| `MODEL_BACKEND` | `openai` (incl. NIM) · `anthropic` · `ollama` · unset/`mock` = offline fallback |
| `OPENAI_BASE_URL` / `OPENAI_MODEL` / `OPENAI_API_KEY` | OpenAI-compatible endpoint (NIM: `https://integrate.api.nvidia.com/v1`, `nvapi-` key, a Nemotron model) |
| `OCR_ENGINE` / `OCR_MODEL` / `OCR_BASE_URL` | `nim` enables the Nemotron OCR port |
| `VISION_ENGINE` / `VISION_MODEL` / `VISION_BASE_URL` | `nim` enables the Nemotron VL port |
| `RETRIEVER` | `keyword` · `semantic` · `hybrid` (semantic/hybrid use pgvector) |
| `DATABASE_URL` | PostgreSQL — approvals, checkpoints, vectors. Unset = in-memory test doubles |
| `S3_BUCKET` / `S3_ENDPOINT_URL` / `S3_ACCESS_KEY` / `S3_SECRET_KEY` | S3-compatible document storage (MinIO in compose) |
| `RUN_TOKEN_BUDGET` | Per-run token budget; later provider steps degrade to templates when exceeded |
| `ACTION_WEBHOOK_URL` | Opt-in: POSTed once on approval — the only external action available |
| `API_KEY` | When set, data endpoints require `X-API-Key` |

## Architecture

Short version: a LangGraph state machine over narrow ports (`ModelBackend`, `OcrEngine`, `DocVision`, `Retriever`, `Store`, `ObjectStore`, `Checkpointer`, `EventSink`) — each capability swaps via env-driven wiring without touching the graph. All numbers that matter (match, tolerances, option scores, autonomy) are computed in code; the model writes prose and proposes, never arithmetic and never decisions.

Read [`docs/architecture.md`](docs/architecture.md) for the production topology, the Postgres/pgvector rationale, the migration story, and the NIM adapter assumptions.

## Eval results

Golden set: 28 synthetic cases (the 12 samples + 16 variants), run through one service instance in queue order. Current: **outcome 28/28, match status 28/28** — a regression gate (≥95% required), not a real-world accuracy claim. The opt-in LLM pack (`evals/run_llm_evals.py`) additionally judges draft grounding/cleanliness with a live provider.

## Repository structure

```
src/ap_agent/     the agent: graph, matching, triage, screening, guardrails,
                  backends, retriever, store, checkpoints, storage, api, cli
src/ap_agent/migrations/   packaged SQL migrations (mirrored at repo root)
migrations/       numbered SQL schema versions (0001 core, 0002 vectors, 0003 checkpoints)
evals/            golden.jsonl + deterministic runner + opt-in LLM-judge pack
tests/            96 hermetic tests + Postgres integration tests (gated on DATABASE_URL)
data/sample/      mirrors of the bundled samples + policy corpus
docs/             architecture.md
```

## QA checklist (for reviewers)

1. `uv sync --extra dev --extra llm --extra postgres --extra s3` — clean install.
2. `make test` — expect 96 passed, 3 skipped (the Postgres integration tests skip without `DATABASE_URL`).
3. `make evals` — expect 28/28 and PASS.
4. `make demo` — trace a clean invoice; `uv run ap-agent-demo --index 9` traces the bank-fraud sample (blocked, details masked).
5. `docker compose up --build` — console at :8000, MinIO console at :9001; `/readiness` reports the Postgres store. (First build is the verification run — the dev sandbox has no Docker daemon.)
6. With `DATABASE_URL=postgresql://ap:ap@localhost:5432/ap` (compose Postgres published via the local override), re-run the test suite — the 3 integration tests now run: Postgres store round-trip, checkpointer pause/resume across instances, pgvector retrieval.
7. Optional: set the NIM profile in `.env` and `make llm-evals` for the live-provider pack.

## Limitations

- Synthetic data only; the policy corpus is illustrative, not legal or accounting advice.
- Live NVIDIA NIM verification is a follow-up (no key available during the build); the NIM OCR/vision/LLM seams are fake-tested, and the OCR chat-shape is an adapter assumption documented in code and architecture docs.
- The Docker image and compose stack are build-unverified in the dev sandbox (no daemon) — step 5 of the QA checklist is the first real build.
- This is a reference prototype, not a production AP system: no ERP integration, no real payment rails, no SSO. The production-hardening notes in `docs/architecture.md` describe what changes.

## License

Apache 2.0 — see [LICENSE](LICENSE). Contributions welcome; see [CONTRIBUTING.md](CONTRIBUTING.md) and [SECURITY.md](SECURITY.md).
