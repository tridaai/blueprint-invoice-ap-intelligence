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
3. **Screen** — supplier text is screened sentence-by-sentence for prompt-injection patterns and, on a separate channel, for bank-change fraud patterns. Flagged sentences are withheld from prompts and drafts; fraud excerpts are digit-masked. The bank identifiers a document states are also checked against the **supplier master record** (the `SupplierMaster` port — the bank details AP verified out-of-band): a mismatch is itself a fraud signal; a match is evidence that downgrades a pattern hit pointing at the account already on file.
4. **Extract** — header + line items via the LLM with per-field confidence, cross-checked deterministically against the provided fields.
5. **Three-way match (in code)** — line quantities, unit prices, tax, totals, and currency against the PO and goods receipt, with tolerance rules in code (price: 2% or 1.00, whichever is greater; totals: 1% or 1.00). The model never does this arithmetic.
6. **Duplicate detection** — same supplier + invoice number + amount against everything the store has seen (AP-DUP-01).
7. **Triage** — rules first ({clean_match, price_variance, quantity_variance, missing_po, missing_receipt, duplicate_suspected, unreadable, not_an_invoice}), then an LLM cross-check with an explicit resolution policy: agreement stands, strong rules beat a disagreeing model, a weak rule yields to a confident model.
8. **Retrieve** — the synthetic AP policy corpus (12 policies) via keyword, semantic (pgvector), or hybrid retrieval with reciprocal-rank fusion.
9. **Diagnose** — a bounded tool loop (search_policies, match_facts, supplier_history, po_lookup) grounded in the computed match, plus supplier history and prior approver feedback from the store.
10. **Options** — five next steps proposed by the model (or templates), always scored and ranked by code; a fraud block zeroes the approve option's fit. Early-payment terms are priced in code first ("2/10 net 30": gross saving vs the cost of paying early at `COST_OF_CAPITAL_ANNUAL_PCT`), and a worthwhile net saving raises the approve option's score and is quantified in its wording and in the packet.
11. **Draft** — a supplier query letter + an internal AP packet, fenced against untrusted supplier text.
12. **Verify + review** — a self-check (amounts must come from the match, citations must be retrieved, no payment language) and an independent reviewer that did not draft.
13. **Guardrails (code)** — `no_payment_language`, `no_prohibited_promises`, `no_pii_in_draft`, citations present **and within the retrieved set**, **`figures_must_trace`** (every amount in the supplier letter traces to the computed fact block — letters that restate arithmetic in prose drift), invoice id present. Failures trigger a bounded repair loop; a still-failing draft **cannot be approved** (the API returns 422).
14. **Gate** — the run checkpoints and waits. A finance approver approves or rejects with name + reason. An autonomy recommendation (auto-triage eligible only for clean matches under the approval threshold with every check green) is printed, never acted on.

Beyond the single invoice, the follow-on surfaces work over the processed queue:

15. **Credit-note matching** — a credit note that arrives later is tied to the invoice it references, and its amount is compared against the overbilling *recomputed from that invoice's stored match facts* (price/quantity/short-receipt overbilling, grossed at the invoice's tax rate): settles / partial / exceeds / no variance. Report only (`POST /credit-notes/match`).
16. **Statement reconciliation** — a supplier statement is lined up against the processed invoices (matched / amount differs / never processed / missing from the statement) with a summary a human confirms (`POST /statements/reconcile`). Report only — confirming changes the report, never an invoice.
17. **Ops surfaces** — `GET /metrics` (runs, outcomes, decisions, latency, model spend, aggregated from the records) and `GET /audit/export` (JSON/CSV audit trail: conclusion, human decision + reason, dispatch). The approval webhook can be HMAC-signed (`ACTION_WEBHOOK_SECRET`) so the receiving system can verify it.
18. **Payment-run proposals** — the approved invoices are grouped by currency + due window (overdue swept into the first window) into a proposed pay list with totals and still-capturable discount amounts summed in code (`POST /payment-runs/propose`). A finance lead confirms the run (`POST /payment-runs/{run_id}/confirm`); members of a confirmed run are never re-proposed. The agent proposes — it never pays.
19. **AP aging queue** — the approver's worklist: invoices still at the gate, aged by due date into the standard buckets (overdue first), with the triage mix, per-currency totals, and each item's discount deadline (`GET /queue`). A pure read model; the console renders it as a panel.
20. **Duplicate-payment recovery packs** — when duplicate detection fires on an invoice whose original was *already approved*, holding the copy is not enough: the pack computes the evidence from both records, drafts the supplier recovery letter (credit note or refund) and the internal memo through the standard verify + guardrail path, and stores it for a human to send (`POST /invoices/{id}/recovery-pack`). The agent recovers nothing itself.
21. **Multi-invoice bundles** — one arrival carrying several invoices is segmented in code at invoice-number anchors (never by the model); each segment is parsed (lines, totals, dates, PO reference), mapped to the bundle's PO/goods-receipt records, and run as its own invoice with its own gate (`POST /bundles/analyze`). The stored summary reports per-invoice outcomes and totals.
22. **Payment-run hand-off file** — a confirmed run renders as a bank-ready CSV (documented layout) into object storage (`POST /payment-runs/{id}/handoff`); the run record keeps the receipt (object key + sha256). The file is the adapter to the customer's banking/ERP system — generating it changes no payment state, and a human carries it the last mile.
23. **Recovery-case lifecycle** — a recovery pack opens a case that operators walk (`opened → supplier_contacted → credit_received → closed`), one audited transition at a time (`POST /invoices/{id}/recovery-case/transition`); the console renders the case on duplicate invoices. The agent records the case — it contacts nobody and books no credit.
24. **ERP posting example** — `examples/erp_posting.py` shapes an approved result into a generic double-entry journal (purchases / input tax / AP control) with a stable idempotency key, refusing unapproved results and unreconciled figures. No vendor SDK — the transport into a real ERP is the customer's step.
25. **Governed supplier-master changes** — a bank-detail change becomes a request object (migration 0009): proposed by an operator or from a document (the details an invoice states, overlaid on the current record), decided by a *second* operator — proposer ≠ approver — with the out-of-band verification reference recorded, and applied only from its effective date (`/suppliers/master-changes`). Until a change is approved and in force, the master check keeps returning a mismatch for the new details.
26. **Four-eyes on large payment runs** — a run whose net total is above `PAYMENT_RUN_FOUR_EYES_THRESHOLD` (default 25000) is stamped at proposal and its hand-off export refuses — with the reason spelled out — until a second confirmer, different from the first, has also confirmed it (`POST /payment-runs/{run_id}/confirm-second`). A total exactly at the threshold is not above it.
27. **Recovery-case aging** — open duplicate-payment cases age in the queue and `/metrics`: total age, the age of the status each case is stuck in, age buckets, and an SLA flag when one status outlasts `RECOVERY_CASE_SLA_DAYS` (default 14). The SLA runs on status age — a case that moved yesterday is moving, not stuck.
28. **ERP posting receipts** — the example adapter's return path: the ERP's outcome (`posted` + its document number, or `failed` + its reason, under the posting's idempotency key) is recorded back onto the invoice (`POST /invoices/{id}/posting-receipt`) and into the audit export. A posted receipt replays by key but can never be replaced by a different posting; the agent itself still posts nothing.

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
make test        # 222 tests
make evals       # golden set: 30/30 outcome + match status, credit-note 1/1, recovery 1/1
```

The offline path uses a deterministic template backend and the documents' provided text layer. It is labelled everywhere it appears: it proves the plumbing, not the model quality. The LLM-judge eval pack (`make llm-evals`) refuses to run without a real provider — exit code 2 — rather than grade its own templates.

## The 14 sample invoices

| ID | Scenario | Designed outcome |
|---|---|---|
| INV-1001 | Clean three-way match, "2/10 net 30" terms | clean_match (+ a priced early-payment saving) |
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
| INV-1013 | Supplier portal resends NB-4471 *after* INV-1001 was approved | duplicate_suspected → the recovery-pack case |
| INV-1014 | Clean match; note states the bank details already on file | clean_match — the supplier-master check turns the details into evidence (`checked_match`), not a block |

Duplicate detection is store-backed: INV-1004 only duplicates because INV-1001 was processed first — run the batch (`ap-agent --all`) or the API in order to see it. A bundled multi-invoice arrival ships too (`sample_bundles.json`: BND-4001, two Northbeam invoices in one scan against PO-9015).

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
| `ACTION_WEBHOOK_SECRET` | When set, the webhook body is HMAC-SHA256 signed (`X-Trida-Signature`) |
| `COST_OF_CAPITAL_ANNUAL_PCT` | Cost of capital (default 6.0) used to price early-payment discounts |
| `PAYMENT_RUN_FOUR_EYES_THRESHOLD` | A payment run above this net total needs a second, different confirmer before hand-off (default 25000) |
| `RECOVERY_CASE_SLA_DAYS` | Days a recovery case may sit in one status before the queue flags it stuck (default 14) |
| `API_KEY` | When set, data endpoints require `X-API-Key` |

## Architecture

Short version: a LangGraph state machine over narrow ports (`ModelBackend`, `OcrEngine`, `DocVision`, `Retriever`, `Store`, `ObjectStore`, `SupplierMaster`, `Checkpointer`, `EventSink`) — each capability swaps via env-driven wiring without touching the graph. All numbers that matter (match, tolerances, option scores, autonomy) are computed in code; the model writes prose and proposes, never arithmetic and never decisions.

Read [`docs/architecture.md`](docs/architecture.md) for the production topology, the Postgres/pgvector rationale, the migration story, and the NIM adapter assumptions.

## Eval results

Golden set: 30 synthetic cases (the 14 samples + 16 variants), run through one service instance in queue order. Current: **outcome 30/30, match status 30/30**, plus the bundled credit-note case **1/1** (the note settles its invoice's computed variance exactly) and the duplicate-recovery case **1/1** (pack drafted for the approved original, guardrails passed) — a regression gate (≥95% required), not a real-world accuracy claim. The opt-in LLM pack (`evals/run_llm_evals.py`) additionally judges draft grounding/cleanliness with a live provider.

## Repository structure

```
src/ap_agent/     the agent: graph, matching, triage, screening, guardrails,
                  backends, retriever, store, checkpoints, storage, api, cli
src/ap_agent/migrations/   packaged SQL migrations (mirrored at repo root)
migrations/       numbered SQL schema versions (0001 core, 0002 vectors, 0003 checkpoints, 0004 reconciliations, 0005 payment runs, 0006 recovery packs, 0007 bundles, 0008 supplier master)
evals/            golden.jsonl + deterministic runner + opt-in LLM-judge pack
examples/         worked seam examples (erp_posting.py: approved result → ERP journal)
tests/            222 hermetic tests + Postgres integration tests (gated on DATABASE_URL)
data/sample/      mirrors of the bundled samples + policy corpus
docs/             architecture.md
```

## QA checklist (for reviewers)

1. `uv sync --extra dev --extra llm --extra postgres --extra s3` — clean install.
2. `make test` — expect 222 passed, 3 skipped (the Postgres integration tests skip without `DATABASE_URL`).
3. `make evals` — expect 30/30 + credit-note 1/1 + recovery pack 1/1 and PASS.
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
