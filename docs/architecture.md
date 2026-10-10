# Architecture — Invoice & Accounts-Payable Document Intelligence Agent

This document explains how the blueprint is built and why: the graph, the split between code and model, the production data topology, the NVIDIA NIM deployment profiles, the guardrails, the checkpointed gate, and the path from this prototype to a production AP deployment.

> **Provenance.** The structural patterns in this blueprint — composable capability components behind narrow interfaces, NIM-first deployment profiles, an in-repo evaluation harness, guardrails enforced as code, run telemetry as a first-class output — follow the conventions of NVIDIA's published agent blueprints, which we treat as the reference playbook for how production agent systems are packaged. The implementation, documentation, policy corpus, and sample data in this repository are original work, written for this blueprint in the idiom of the Trida AI blueprint series. No code, prose, or assets are copied from NVIDIA repositories or any other source.

---

## 1. Problem framing

Accounts payable intake is high-volume, document-bound, and fraud-exposed. Before a finance approver can decide anything, someone must: read a scan of uncertain quality, verify it is an invoice at all, re-key its fields, compare every line against the purchase order and the goods receipt, check the tax arithmetic, look for a duplicate of an invoice already in the system, and decide which exception — if any — this is. Two failure modes matter more than all the others:

- **Paying twice** (duplicates) and **paying wrong** (price/quantity drift) — controlled by exact arithmetic, not judgement.
- **Payment-diversion fraud** — the "our bank account has changed" message, which arrives through exactly the channel this agent reads. Any automation that transcribes supplier text into payment instructions is an attack surface; this design treats that pattern as a first-class, separately-handled threat.

The design rule follows: **everything checkable is checked in code; the model is used for what it is good at (reading, summarising, drafting), and never for arithmetic, policy, or decisions.**

## 2. System shape

A LangGraph state machine runs one invoice through a fixed pipeline. Every external capability sits behind a port (a `typing.Protocol` in `ports.py`), and every port has an environment-selected implementation assembled in `wiring.py`. The graph never imports a vendor SDK.

```
                ┌──────────────────────────── FastAPI / CLI ───────────────────────────┐
                │  analyze · analyze/stream (SSE) · get · approve · reject · console     │
                └────────────────────────────────┬───────────────────────────────────────┘
                                                 │ InvoiceService
   ┌─────────────────────────────────────────────▼──────────────────────────────────────┐
   │ LangGraph pipeline                                                                  │
   │                                                                                     │
   │  ocr → vision → screen → extract → match → classify ─┬─ retrieve (policies)         │
   │                                                      ├─ discrepancies (code)        │
   │                                                      └─ history (store)             │
   │  → diagnose (bounded tool loop) → options (scored in code) → draft                  │
   │  → verify → review → validate (guardrails) → [bounded repair] → gate                │
   │  → approval_gate (interrupt) ⇢ resume on approver decision                          │
   └──────┬──────────┬──────────┬──────────┬──────────┬──────────┬───────────┬────────────┘
          │          │          │          │          │          │           │
     OcrEngine   DocVision  ModelBackend Retriever   Store   ObjectStore Checkpointer
       (NIM /      (NIM VL /  (NIM LLM /  (keyword /  (Postgres/ (S3 /      (Postgres /
        offline)    metadata)  Anthropic / semantic /  memory)   memory)     memory)
                               Ollama /    hybrid+
                               mock)       RRF)
```

Node order is deliberate. Screening runs before extraction and drafting so untrusted text is fenced or stripped before any model sees it. Matching runs before triage because triage consumes computed facts, not model impressions. The three fan-out nodes (retrieve, discrepancies, history) are independent reads over the same state. Verification, review, and validation are three separate mechanisms on purpose: a checklist (deterministic), a second model pass (judgement), and hard rules (code) fail differently, and a draft must survive all three.

## 3. Production data topology

There are no local files in the runtime story. One database is the system of record; one object store holds blobs; the application containers are stateless and interchangeable.

```
                        ┌────────────────────────┐
   approvers ──────────▶│  api (stateless, ×N)   │──────────┐
                        └──────┬──────────┬──────┘          │ approval
                               │          │                 ▼ webhook (opt-in)
              SQL (psycopg 3)  │          │  S3 API    AP system of record
                               ▼          ▼
                 ┌──────────────────┐  ┌──────────────────┐
                 │ PostgreSQL 16    │  │ Object storage   │
                 │ + pgvector       │  │ (S3 / MinIO)     │
                 │ ─ approvals      │  │ document blobs,  │
                 │ ─ checkpoints    │  │ keyed by         │
                 │ ─ policy vectors │  │ documents'       │
                 └──────────────────┘  │ object_key       │
                                       └──────────────────┘
                 ┌──────────────────┐
                 │ NVIDIA NIM       │  LLM · OCR · VL · embeddings
                 │ hosted or self-  │  (OpenAI-compatible endpoints)
                 │ hosted           │
                 └──────────────────┘
```

**Why PostgreSQL for everything.** The approvals store, the LangGraph checkpointer, and the policy vectors have the same operational profile — modest volume, strong consistency needs, one backup story. Splitting them across a database, a file store, and a separate vector service buys nothing at AP scale and triples the failure surface. pgvector keeps embeddings in the same transactions and the same backups as the records they describe; the corpus here is a dozen policies, and pgvector remains comfortable orders of magnitude beyond that. We chose it over a standalone vector database for one-database simplicity, and the `Retriever` port means that choice is reversible.

**Migrations, not ORM magic.** Schema lives in `migrations/` as numbered SQL files (`0001_core`, `0002_policy_embeddings`, `0003_checkpoints`), mirrored into the package (`src/ap_agent/migrations/`) so installed deployments run the same files the repo reviews — a test asserts the copies are identical. A tiny runner records applied versions in `schema_migrations` and applies pending files in order, one transaction each, when a Postgres-backed store or checkpointer is constructed. Production systems version their schema; this keeps that habit cheap.

**Object storage port.** Documents reference `object_key`s, not local paths. The OCR stage resolves a key through the `ObjectStore` port (boto3 against any S3-compatible endpoint; MinIO in compose) when a document carries no inline text. Unset `S3_BUCKET` disables the port and documents must carry their text layer — the hermetic default for tests and demos.

**Runtime posture.** Multi-stage Dockerfile (locked uv build, slim runtime, non-root user, healthcheck), `/health` for liveness and `/readiness` for store connectivity (503 when the database is unreachable, so an orchestrator stops routing), structured JSON logs with a per-request ID (`X-Request-ID` honoured inbound, returned outbound), configuration exclusively through environment variables, and graceful shutdown through the FastAPI lifespan. Batch analysis goes through the service layer, so concurrent workers share Postgres state exactly as in production — the duplicate sample behaves identically in a batch and in the API.

## 4. The NVIDIA NIM profiles

NIM services speak the OpenAI chat-completions dialect, so "built on NIM" is a configuration profile of the same provider seam rather than a fork of the engine:

| Capability | Port | NIM profile | Offline equivalent |
|---|---|---|---|
| Reasoning / extraction / drafting | `ModelBackend` (`openai`) | `OPENAI_BASE_URL=https://integrate.api.nvidia.com/v1`, `nvapi-` key, Nemotron model (e.g. `nvidia/llama-3.3-nemotron-super-49b-v1`) | deterministic templates (`mock`) |
| OCR | `OcrEngine` (`nim`) | `OCR_ENGINE=nim`, `OCR_MODEL=nvidia/nemotron-ocr-v1` | the document's provided text layer + confidence metadata |
| Document vision | `DocVision` (`nim`) | `VISION_ENGINE=nim`, a Nemotron VL model | deterministic verdict from provided metadata |
| Embeddings | `Retriever` (`semantic`/`hybrid`) | `OPENAI_EMBEDDING_MODEL=nvidia/nv-embed-v1` on the same endpoint | keyword retrieval |

Self-hosted NIM is the same profile with `*_BASE_URL` pointed at the containers (the local compose override documents the per-service URLs). Anthropic, plain OpenAI, and Ollama remain first-class `ModelBackend` choices — the engine is portable by construction.

**Adapter assumption (stated plainly).** The OCR NIM integration posts the document text to the endpoint's chat-completions shape and asks for a JSON reply of `{text, confidence}`. That request shape is an adapter seam: it isolates the one place that changes if a deployment's OCR service expects a dedicated document payload, and it is called out in the `ocr.py` docstring. Live NIM verification — hosted key against all four services — is a follow-up item; every NIM seam is covered by fake-based tests in the meantime.

## 5. The match, and why it is code

`matching.py` computes the three-way match exactly: per-line price variance (percent and absolute), quantity billed vs ordered vs received, line status in {matched, price_variance, quantity_variance, short_received, missing_on_po, missing_on_invoice, unmatched}, tax as stated vs implied-by-arithmetic, totals reconciliation, and currency consistency. Tolerances are constants with policy citations (AP-MATCH-01: price within 2% or 1.00, whichever is greater; totals within 1% or 1.00). Match status rolls up to {full_match, partial_match, no_match, no_po, no_receipt}; a line that corresponds but disagrees still counts as a partial match, reserving no_match for invoices whose lines do not correspond at all.

The match output is the single source of truth downstream: triage rules read it, the verify checklist requires every amount in a draft to come from it, and option scoring takes its match component from it. A model that "feels" the invoice matches cannot override arithmetic that says otherwise — the cross-check policy (next section) formalises the one narrow exception.

## 6. Triage and the cross-check policy

Rules classify first, from computed facts, with a fixed priority: not-an-invoice (vision) → unreadable → duplicate → missing PO → missing receipt → price variance → quantity variance → totals/tax/currency fallback → clean. Price outranks quantity when both occur (price is the money question). Fraud flags deliberately do **not** change the outcome — a clean-matching invoice with a bank-change request is a clean match under a fraud hold, and conflating the two would hide both facts.

When a provider backend is present, it independently classifies from the same facts, and `crosscheck.py` resolves the pair: agreement stands; if the rules are confident (≥ 0.85) they are authoritative over a disagreeing model; if the rules are weak and the model is confident (≥ 0.85) the model's outcome is adopted and the fact is recorded in the triage signals; with no usable model reply the run is rules-only. Every run's result carries the full cross-check record — both verdicts and the resolution — because "the AI said so" is not an audit trail.

## 7. Diagnosis with a bounded tool loop

Diagnosis is the one genuinely agentic step. A provider backend may call up to `DIAGNOSIS_MAX_TOOL_CALLS` tools (default 4, hard cap 6) — `search_policies`, `match_facts`, `supplier_history`, `po_lookup` — each returning computed facts, never new instructions. Every path degrades honestly: no tool support → single-shot LLM diagnosis; no usable reply → a template summary composed from the same facts, labelled `source: template`. Supplier history and prior approver feedback (recorded reject/approve reasons for the same supplier) are injected as evidence lines, which is how the feedback loop teaches later runs without any training.

## 8. Guardrails as code

`guardrails.py` validates the final draft pair; approval is refused (HTTP 422) for any run whose validation failed.

- `no_payment_language` — payment-made / payment-will-be-made assertions (with a negation window so the required disclaimer — "no payment is made or scheduled by this letter" — passes), plus any bank-content pattern: account numbers, sort codes, IBANs, "new bank account", remittance-to-account instructions.
- `no_prohibited_promises` — outcome guarantees and time commitments.
- `no_pii_in_draft` — SSN and Luhn-valid card patterns (invoice numbers and amounts must not trip it, and tests pin that).
- `policy_citations_present`, `references_invoice_id`, `states_next_step`.

The bank-change defence is layered, deliberately: screening flags the pattern and blocks the approve option + autonomy; sanitisation strips the sentences from prompts; flags carry digit-masked excerpts and draft-safe labels; the diagnosis carries the label, never the sentence; and the guardrail rejects any draft in which bank content survived anyway. INV-1010 exercises every layer. INV-1012 is the complementary trap: a supplier note relaying a payment-date demand is *not* an injection (the screen correctly leaves reported speech alone) — it is the drafter's job not to repeat it, and the guardrail's job to catch it when the drafter does.

A failing draft enters a bounded repair loop (redraft with the failures named, up to `GUARDRAIL_REPAIR_MAX_ATTEMPTS`); the original failure stays on the record either way.

## 9. The gate, checkpoints, and autonomy

The graph ends in an interrupt: the run pauses at `approval_gate` with its full state in the checkpointer (Postgres saver over psycopg 3, mirroring the reference saver semantics — the pinned LangGraph ships base and in-memory savers only, so the saver is implemented against the existing port; an integration test pauses on one saver instance and resumes on another to prove restart survival). The service records the decision in the store (the system of record — graph threads are workflow state, not the ledger), then resumes the thread to completion. Approving requires the run's guardrail validation to have passed.

The autonomy recommendation is computed, printed, and never acted on: eligible only for `clean_match`, low severity, `full_match`, at or under the approval threshold, no fraud flags, validation green, no repair, an agreeing cross-check, and an unblocked reviewer. Its purpose is to make the automation roadmap discussable with data instead of vibes.

The approval webhook (`ACTION_WEBHOOK_URL`) is the single external action in the system, opt-in, fired after a human decision — it announces that a packet was approved; it does not pay anything. When `ACTION_WEBHOOK_SECRET` is set, the body is HMAC-SHA256 signed (`X-Trida-Signature: sha256=<hex>`) so the receiving system can verify origin and integrity before acting on it.

## 10. Observability and cost control

Every run returns telemetry: backend, model, call count, token usage, estimated cost (a small price table; models outside it — including Nemotron — report `None`, shown as n/a, never invented), latency, and per-node timings via the event sink (the API's SSE stream is the same events). `RUN_TOKEN_BUDGET` wraps the backend in a guard that, once the budget is spent, hides the optional provider capabilities so later steps degrade to their deterministic paths instead of failing mid-run — the run completes, labelled honestly.

Structured JSON logging with request IDs threads the API surface; the CLI's `--stream` flag prints the same event stream for a local run. Two read surfaces aggregate the store itself: `GET /metrics` (runs, outcomes, decisions, mean latency, token/cost totals summed only over runs that reported them — null, not zero, when none did) and `GET /audit/export` (JSON/CSV: one row per invoice with the triage conclusion, the human decision and its reason, and the dispatch outcome). Aggregating at read time keeps the store the single source of truth — there is no counters table to drift.

## 11. Evaluation

Two packs, both in-repo. The deterministic pack (`evals/run_evals.py`) runs 28 golden cases — the 12 samples plus 16 constructed variants — through one service instance in queue order (order is load-bearing for the duplicate case) and gates on triage-outcome and match-status accuracy at 95%; it then matches the bundled sample credit notes against the populated store, so a note can only settle a variance its invoice actually computed. The opt-in LLM pack (`evals/run_llm_evals.py`) runs the same cases against a live provider and adds an LLM judge for draft grounding and cleanliness; it exits 2 without a provider rather than grade the offline templates. Neither pack claims real-world accuracy: the golden set is synthetic and exists as a regression gate.

## 12. Testing posture

The pytest suite (128 tests) is hermetic: in-memory doubles for the store, checkpointer, and object store; scripted fake backends for the provider seams; fake NIM engines for the OCR/vision ports. Test doubles are test design, not a runtime mode — the documented runtime stores are Postgres and S3. Postgres behaviour is covered by integration tests gated on `DATABASE_URL` (store round-trip, checkpointer pause/resume across instances, pgvector retrieval with a stubbed embeddings client) that skip cleanly when no database is configured.

## 13. Follow-on AP work: discounts, credit notes, statements

Three surfaces operate over the processed queue rather than a single run, all with the same discipline — code computes, humans decide, nothing external happens:

- **Early-payment discounts** (`discounts.py`) are priced inside the run: terms are parsed from the intake (`payment_terms` field or a document field), the gross saving is compared against the cost of paying early at the configured cost of capital, and a worthwhile net saving enters the approve option's score (capped) and is quantified in the packet. The deadline is stated as a condition — the payment-language guardrail rejected an earlier "paid by" phrasing, correctly, and the packet wording was changed rather than the guardrail.
- **Credit-note matching** (`creditnotes.py`) answers "does this credit settle a variance we computed?" from stored match facts: overbilling is recomposed per line (price, quantity, short receipt, missing-on-PO) and grossed at the invoice's effective tax rate, so the comparison is tax-inclusive like the credit itself. It is a query: no invoice state changes.
- **Statement reconciliation** (`reconciliation.py`) lines a supplier statement up against the store's history projection. Its summary is composed deterministically (`summary_source: template`) by design: on this surface a plain summary beats a fluent one that might invent a figure. Reports persist (migration 0004) for human confirmation; confirmation touches the report only.

## 14. From prototype to production

What changes for a real AP deployment, in rough order: SSO + role-based access on the gate (approver ≠ requester enforcement); an ERP/AP-system integration behind the webhook (the packet becomes a posting, still human-triggered); bank-detail master data with out-of-band change workflows (the agent's block becomes one signal among several); document ingestion from the mailroom/scanner pipeline writing blobs to object storage; per-entity policy corpora replacing the synthetic 12; NIM self-hosting or hosted-key management in the platform's secret store; and the LLM-judge pack running in CI against recorded provider budgets. What does not change: the ports, the match arithmetic, the guardrails, or the rule that the agent prepares and a human decides.
