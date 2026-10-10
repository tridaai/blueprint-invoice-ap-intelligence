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
16. **Statement reconciliation** — a supplier statement is lined up against the processed invoices (matched / amount differs / never processed / missing from the statement) with a summary a human confirms (`POST /statements/reconcile`). Lines are duplicate-screened too (surface 33). Report only — confirming changes the report, never an invoice.
17. **Ops surfaces** — `GET /metrics` (runs, outcomes, decisions, latency, model spend, aggregated from the records) and `GET /audit/export` (JSON/CSV audit trail: conclusion, human decision + reason, dispatch). The approval webhook can be HMAC-signed (`ACTION_WEBHOOK_SECRET`) so the receiving system can verify it.
18. **Payment-run proposals** — the approved invoices are grouped by currency + due window (overdue swept into the first window) into a proposed pay list with totals and still-capturable discount amounts summed in code (`POST /payment-runs/propose`). A finance lead confirms the run (`POST /payment-runs/{run_id}/confirm`); members of a confirmed run are never re-proposed. The agent proposes — it never pays.
19. **AP aging queue** — the approver's worklist: invoices still at the gate, aged by due date into the standard buckets (overdue first), with the triage mix, per-currency totals, and each item's discount deadline (`GET /queue`). A pure read model; the console renders it as a panel.
20. **Duplicate-payment recovery packs** — when duplicate detection fires on an invoice whose original was *already approved*, holding the copy is not enough: the pack computes the evidence from both records, drafts the supplier recovery letter (credit note or refund) and the internal memo through the standard verify + guardrail path, and stores it for a human to send (`POST /invoices/{id}/recovery-pack`). The agent recovers nothing itself.
21. **Multi-invoice bundles** — one arrival carrying several invoices is segmented in code at invoice-number anchors (never by the model); each segment is parsed (lines, totals, dates, PO reference), mapped to the bundle's PO/goods-receipt records, and run as its own invoice with its own gate (`POST /bundles/analyze`). The stored summary reports per-invoice outcomes and totals.
22. **Payment-run hand-off file** — a confirmed run renders as a bank-ready file into object storage (`POST /payment-runs/{id}/handoff`): a CSV (documented layout v1) or an ISO 20022 pain.001 credit-transfer XML (`?format=pain001`, mapping in `docs/handoff-formats.md`); the run record keeps the receipt (object key + sha256 + format). The file is the adapter to the customer's banking/ERP system — generating it changes no payment state, and a human carries it the last mile.
23. **Recovery-case lifecycle** — a recovery pack opens a case that operators walk (`opened → supplier_contacted → credit_received → closed`), one audited transition at a time (`POST /invoices/{id}/recovery-case/transition`); the console renders the case on duplicate invoices. The agent records the case — it contacts nobody and books no credit.
24. **ERP posting example** — `examples/erp_posting.py` shapes an approved result into a generic double-entry journal (purchases / input tax / AP control) with a stable idempotency key, refusing unapproved results and unreconciled figures. No vendor SDK — the transport into a real ERP is the customer's step.
25. **Governed supplier-master changes** — a bank-detail change becomes a request object (migration 0009): proposed by an operator or from a document (the details an invoice states, overlaid on the current record), decided by a *second* operator — proposer ≠ approver — with the out-of-band verification reference recorded, and applied only from its effective date (`/suppliers/master-changes`). Until a change is approved and in force, the master check keeps returning a mismatch for the new details.
26. **Four-eyes on large payment runs** — a run whose net total is above its four-eyes threshold (default 25000; per-currency via `PAYMENT_RUN_FOUR_EYES_THRESHOLDS` — see 37) is stamped at proposal and its hand-off export refuses — with the reason spelled out — until a second confirmer, different from the first, has also confirmed it (`POST /payment-runs/{run_id}/confirm-second`). A total exactly at the threshold is not above it.
27. **Recovery-case aging** — open duplicate-payment cases age in the queue and `/metrics`: total age, the age of the status each case is stuck in, age buckets, and an SLA flag when one status outlasts `RECOVERY_CASE_SLA_DAYS` (default 14). The SLA runs on status age — a case that moved yesterday is moving, not stuck.
28. **ERP posting receipts** — the example adapter's return path: the ERP's outcome (`posted` + its document number, or `failed` + its reason, under the posting's idempotency key) is recorded back onto the invoice (`POST /invoices/{id}/posting-receipt`) and into the audit export. A posted receipt replays by key but can never be replaced by a different posting; the agent itself still posts nothing.
29. **Discount drift check at export** — a run confirmed while a discount was capturable can be handed off after the window closed. Export re-prices every discounted member from its stored invoice as of the export date: a drifted member is exported at its full gross amount (the amount actually owed once the window closes — the stale net would short-pay the supplier), the drift is itemised on the run record (`handoff_drift`, with the lost amount and the closed deadline), and the clean members export exactly as confirmed.
30. **Master-coverage metrics** — `/metrics` reports the fraud screen's blind spot: the share of processed invoices (and of spend) whose supplier has a master record, the recorded bank-check verdict mix (match / mismatch / no master record / no details stated / not checked), and the governed changes in flight to close the gap. A pending change counts as in flight, not as coverage.
31. **Master-change expiry** — a pending change nobody decides within `MASTER_CHANGE_EXPIRY_DAYS` (default 30) expires: the sweep runs with the promotion sweep and at analysis time, a late decision attempt expires the request on the spot, and the expiry is recorded in the request's audit history. Expired changes are decidable no longer — re-proposing restarts the governed path with fresh eyes.
32. **Approved-not-posted reconciliation** — the receipt loop's open items: approved invoices with no posted receipt past `POSTING_RECEIPT_GRACE_DAYS` (default 7), oldest first, at `GET /reconciliation/approved-not-posted` and as a count in `/metrics`. A failed receipt keeps its item open with the ERP's reason; a posted receipt closes it. Approvals now stamp `decided_at`, the age anchor (legacy records fall back to submission, labelled).
33. **Duplicate screening across arrival shapes** — duplicate identity is judged on a canonical key (supplier name punctuation/case normalised, invoice number reduced to alphanumerics), so a resend is caught whichever shape it arrives in: a bundle segment that re-sends a portal single is flagged with the single as its twin (and the recovery path opens for it), a single that re-sends a bundle segment likewise, and a repeated segment inside one bundle too. Statement lines are screened on the same key: a line whose number AP holds twice, or that the statement itself lists twice, is flagged with the twin reference in the reconciliation report.
34. **Hand-off drift + lag metrics** — `/metrics` aggregates the export story over the stored runs: runs exported with drift, members drifted, discount value lost per currency — beside confirm→export and proposal→export lag stats, because the lag is the process leak that causes the drift.
35. **Open-posting escalation** — an open posting past `POSTING_ESCALATION_DAYS` (default 21, never below the grace window) stops being a report line and joins the approver queue (`GET /queue`) as an action item with the ERP failure reason attached; `/metrics` counts the escalated items.
36. **Recovery-case escalation webhook** — the queue flags a case whose status outlasted the SLA; the sweep (`POST /recovery-cases/escalate`) also *tells* someone: a signed `recovery_escalation` webhook (the same opt-in `ACTION_WEBHOOK_URL` seam and HMAC scheme as approvals) carrying the case's aging snapshot. Each dispatch is recorded in the pack's own escalation ledger, deduped per status crossing — re-sweeping never re-fires, a case that moves and stalls again escalates for the new crossing, and with no webhook configured nothing is recorded.
37. **Per-currency four-eyes thresholds** — `PAYMENT_RUN_FOUR_EYES_THRESHOLDS` (`EUR=50000,JPY=3000000`) gives currencies their own gates, with the global threshold as fallback in both directions; the figure that applied is stamped on the run at proposal (`four_eyes_threshold`), named in its note and the console, and quoted in the second-confirmation and export refusals.
38. **Creditor-account completion at export** — the hand-off's creditor account is completed from the governed supplier master where (and only where) it holds approved bank details for the member's supplier: the IBAN as the pain.001 `IBAN` element, else the account number with the sort code as creditor agent; every other member keeps the `SUPPLIER-MASTER-AT-IMPORT` marker, and the debtor side always keeps its. The export receipt (`handoff_creditor_entries`) records completed vs marked per member, masked labels only; the CSV carries the account plus its source (layout v2).
39. **Drift forecasting at confirmation** — confirmation projects the drift leak forward: each discounted member's deadline is set against the run's expected export date (proposal date + the measured mean proposal→export lag, the same lag `/metrics` publishes), and members projected to lapse are flagged on the run (`drift_forecast`, with `drift_forecast_basis` stating the lag it rests on — or that no exported runs exist yet). Labelled a forecast everywhere; the export-time drift check remains the authority.
40. **Forecast-accuracy tracking** — the forecast is now graded: `/metrics` sets every exported run's confirmation-time flags against its measured export drift (flagged & drifted, flagged & saved, the unflagged misses, and the quiet majority), with precision, recall, and the discount value on each side of the ledger, overall and per currency. Runs whose confirmation honestly made no forecast are unscored — their drift is reported separately, never counted against a forecast nobody made.
41. **Creditor-completion coverage** — `/metrics` also prices the marker risk over *exports* (master coverage counts records): per currency and per hand-off format, the share of exported members — and of exported value — whose creditor account was completed from the governed master versus the share that still carried the `SUPPLIER-MASTER-AT-IMPORT` marker, read off each run's completion receipt. Receipt-less legacy exports are counted as such, so the denominator never quietly shrinks.
42. **Four-eyes aging in the queue** — a four-eyes run confirmed once is locked out of export until its second confirmer arrives, and now that wait ages: past `PAYMENT_RUN_FOUR_EYES_AGING_DAYS` (default 2) the run joins the approver queue (`GET /queue`) as an action item with the first confirmer named — the person to nudge, and the one person who cannot be the second.
43. **Escalation digest** — the day's escalations composed into one operator summary (`GET /digest`, console panel): recovery cases past their SLA, escalated open postings, invoices stuck at the gate, four-eyes stalls, plus the day's run news — discounts forecast to lapse on runs confirmed today, drift realised on runs exported today. Composed deterministically from the records, in the reconciliation summary's discipline; the agent sends nothing.
44. **Stored digest artifacts** — the computed digest describes the records as they are now, so a day's escalations cannot be re-derived once the records move. An operator can store the day's first-computed digest (`POST /digest/{date}/store`, migration 0010): read back verbatim at `GET /digest/{date}`, stamped with who stored it and when, provenance `stored` — first-computed wins, a stored day is never overwritten, and `GET /digest` stays the live recomputation, labelled `computed`. `GET /digest/history` reads the stored series as headline counts with day-over-day changes against the previous stored day; days nobody stored are absent, not interpolated. The stored copy is the one a human forwards or files.
45. **Marker-risk onboarding worklist** — completion coverage prices the marker; `GET /suppliers/marker-worklist` (console panel) turns the same export receipts into the work: suppliers ranked by marked export value (export-time net, ranked within currency — values are never compared across currencies), each with its marked value, invoice count and ids, export count, and last marked export date. A supplier the governed master now covers with approved bank details — the completion's own has-details rule, checked against the live master — resolves off the list, counted with its formerly-marked value so the queue's shrinkage is visible. The console's per-supplier button prefills the governed master-change form: onboarding stays a human's governed act, one click away.
46. **Worklist flow state** — the worklist joins the governed change pipeline, so a listed row knows where its onboarding stands. A supplier whose latest master change is still live — pending inside its decision window, or approved and awaiting its effective date — reads **onboarding in flight**: change id, proposer, and age on the row, the decision owed by a second operator named as owed, and the console stops offering a duplicate proposal for it. A supplier whose latest change was rejected wears the rejection — decider, date, and the recorded note — so re-proposing is a deliberate act against a visible refusal, and an expired pending change reads as open work again. Header counts track in-flight and rejected suppliers beside the resolved count.
47. **Posting-escalation ladder** — the open-posting escalation gains a second rung. Past twice the escalation threshold, a posting that is *still* open fires the signed `posting_escalation_urgent` event — same opt-in seam, same HMAC scheme, the rung named in the payload — so an unanswered first shout compounds loudly instead of expiring. Dedupe is per episode *and* rung on the invoice record's ledger, and a sweep climbs at most one rung per posting: a posting first seen past the second threshold still hears level 1 first. A posted receipt remains the only reset, and `/metrics` splits recorded pushes by rung.
48. **Digest auto-store sweep** — the stored-digest history no longer depends on someone remembering to press Store. `POST /digest/store-due` pins every unstored day inside a bounded lookback (`DIGEST_STORE_LOOKBACK_DAYS`, default 7) through the same first-computed-wins path a human uses, and reports stored vs already-stored per day. Sweep-stored days are attributed to `system` in their own provenance — a backfilled day never masquerades as a morning ritual — and days a human stored keep their copy and their name, untouched.
49. **Priced four-eyes wait** — the second confirmation used to close the four-eyes wait silently; now it prices it. When the second confirmation lands, the run record is stamped with the member discounts that were capturable at the first confirmation and lapsed before the second (`four_eyes_wait_lapses`, totalled in `four_eyes_wait_cost`), charged under the same capturable-on-the-deadline-day rule as everything else: a discount already gone before the wait began is never charged to it, and a same-day second confirmation costs zero. `/metrics` aggregates the stamps (`payment_runs.four_eyes_wait`) beside the forecast accuracy — the control's cost as a number, next to the leaks it prevents.
46. **Forecast-aware four-eyes waits** — the four-eyes aging threshold is per run now. The configured `PAYMENT_RUN_FOUR_EYES_AGING_DAYS` is the patience for a run whose discounts are safe; a run whose confirmation-time drift forecast flags members lapsing before the expected export cannot export until its second confirmation lands, so the wait itself is the leak — its threshold tightens to 0 days and it joins the queue (`GET /queue`) on the first day of the wait, carrying the lapse count and the projected value at risk, sorted ahead of merely-old waits. The forecast is read as stamped at confirmation, never recomputed in the queue. The digest says it too: lapses on unseconded runs are flagged `blocked_on_four_eyes` — the projection is blocked on the wait, not the export.
47. **Open-posting escalation webhook** — open postings past `POSTING_ESCALATION_DAYS` get the recovery cases' push treatment: the sweep (`POST /postings/escalate`) fires the signed `posting_escalation` webhook over the same opt-in `ACTION_WEBHOOK_URL` seam and HMAC scheme, carrying the posting's snapshot (invoice, supplier, amount, approver, age, and the ERP's failure reason when a receipt failed). Each dispatch is ledgered on the invoice record itself (`posting_escalations`, beside the receipt fields), deduped per open episode — an episode runs from the approval, or from the latest failed receipt, since a failure is new information worth one escalation of its own; a posted receipt closes the item and is the reset. With no webhook configured nothing fires and nothing is recorded; `/metrics` counts the recorded pushes beside the items past the line.
50. **Hand-off routing profiles** — the export stops being one-size-fits-all. A governed profile (`GET/POST /handoff/profiles`, migration 0011) records the payee side's routing facts per supplier or per currency — preferred format (CSV or pain.001), lead days for the pain.001 requested execution date, and whether the payee's bank rejects at-import creditor markers — set with a named actor and an append-only audit trail on the profile record. Export resolution is deterministic: a format named on the export call wins, then the run's currency profile, then a supplier profile the run's profiled members agree on in unison, else the CSV default; the governing profile's lead days push `ReqExctnDt` to the later of the window end and export + lead. The run stamps the governing profile and the execution date written. And a bank that rejects markers is flagged at *proposal* time (`handoff_profile_flags`) for any member the master holds no bank details for — the import refusal is prevented at the desk, not discovered at the bank. `/metrics` counts the profiles in force.
51. **Wait-cost budgets** — the priced four-eyes wait gains an owner. `FOUR_EYES_WAIT_BUDGETS` sets a per-currency monthly ceiling on the discount value the second-confirmation waits may destroy; `/metrics` reports the current month per currency (cost charged to the month the second confirmation landed, the ceiling in force, the excess), and the digest names every breached currency each morning until the month turns. A currency with no budget configured is unbudgeted — reported with a null budget, never breached against an invented zero.
52. **Worklist expiry early warning** — in-flight onboarding rows now age against the decision window they are waiting out: days of window left on the row, and inside the warning band (`MASTER_CHANGE_EXPIRY_WARNING_DAYS`, default 7) the pending change is flagged expiring soon and counted in the worklist header. The digest carries the same warning — change id, proposer, age, days left, and the marked value still riding on the onboarding — as decisions owed while a decision can still rescue them, instead of day 31 silently returning the supplier to open work. Approved changes never expire and carry no warning.
53. **Digest-store coverage** — the audit file, audited. `/metrics` now reports the stored digests' completeness (`digest_store`): stored days by attribution (a human's store, the sweep's `system` backfill, unattributed copies), the stored range's bounds, and every gap inside it — split at the sweep's reach, so gaps a sweep run can still heal are named apart from gaps that predate the lookback and can never backfill. A missing day no longer reads like a quiet day.
54. **Drafting hardened against the flaws a live run exposed** — on a live Nemotron run the AP packet draft carried none of the policies the diagnosis cited (the independent reviewer blocked it — the spine working) and the supplier letter's next step drifted into prose the guardrail could not find, with a repair loop that only knew the draft "failed". The packet is now composed with the diagnosis's retrieved policies appended in code when the model drops them — retrieved-only, so the citations guardrail still polices everything — and the letter prompt names the required `Next step:` line as a line, not a description. A failed check now reaches the repair beside its fix ("add a line beginning 'Next step:' …"), per check, in both the repair instruction and the model's correction block. Fake-tested with a stub backend reproducing both live flaws; no guardrail and no reviewer standard moved.
55. **The duplicate screen, read back as a twin graph** — the audit export gains the twin columns (`duplicate_of`, `arrival_shape`, `twin_arrival_shape`, `twin_cluster`, `twin_cluster_size`) and `GET /audit/twins` returns the clusters: which records were screened as twins of which, on what key evidence (the duplicate-of edge, or a shared canonical key at a matching total), and how each twin arrived — single upload, or re-cut as its own invoice out of a remittance bundle, so a supplier resending one invoice three different ways reads as one cluster, not three coincidences. Statement lines stay reconciliation's business; the graph is over invoice records.
56. **pain.001 batch booking** — a governing profile can now call for batch booking: the export groups the run's lines into one `PmtInf` per requested execution date (`BtchBookg` true, per-group control sums) for banks that book per value date. The lead-days interaction is per group — a line's group date is the later of its due date and the profile's lead floor; the floor wins when later, the due date wins when later — and the completion and drift gates hold inside each group by construction. Each completion entry is stamped with its member's execution date, and creditor-completion coverage splits by PmtInf grouping. The default single-`PmtInf` file is byte-identical (a second golden file pins the batch shape); the CSV stays flat.
57. **Profile effectiveness, graded** — `/metrics` grades every hand-off profile over the exports it governed: routed exports and format mix, the marker rate on its members against the ungoverned population's, whether suppliers its marker-rejection stance flagged at proposal time stopped exporting marked, and whether the lead promise was kept by the dates actually written. A profile governing nothing — or governing no safer than nothing — is named as configuration theatre in its own notes.
58. **The duplicate screen, moved into the pipeline** — a live judged slice scored the INV-1004 resend as a clean match: the AP-DUP-01 screen existed only because the service computed the verdict over the store and passed it in, and the LLM eval pack called `run_invoice` directly, with no store — the screen never ran. The pipeline now computes the verdict itself from whatever priors it is handed (a caller-supplied verdict is discarded when priors ride along), and the LLM pack runs its cases through one shared service in file order, the production path. A deterministic control that depends on the caller remembering to invoke it is not a control.
59. **Outcome claims trace like figures** — the same live judge caught a letter claiming "received and matched" over a partial quantity-variance match. The subject's claim is now composed from the computed outcome in the template drafter and the provider fallback alike (an unknown outcome defaults to "under review", never "matched"), and a new guardrail — `outcome_claim_must_trace` — fails a draft whose claim exceeds the computed outcome: outcome claims require a `clean_match` outcome, the match claim requires a `full_match` status, so a duplicate may state its full match but never a clean outcome. The repair directive teaches the fix by name; the catch is reproduced with a stub and repaired end-to-end.
60. **Twin clusters escalate into the queue** — the twin graph was audit-only; the approver works the queue. A cluster now joins the worklist as one case, its graph attached (members + screened edges), when any trigger fires: size (`TWIN_CLUSTER_MIN_SIZE`, default 3), combined value in a single currency (`TWIN_CLUSTER_VALUE_THRESHOLD`, default 10000), or an approved twin spanning a waiting one — the double-payment shape itself, which seats first in work order. Under every trigger a cluster stays audit-only.
61. **A judge budget a reasoning model can think in** — the LLM pack's judge lost 4 of 6 live verdicts to "reply was not a JSON object": a reasoning judge spends its completion budget thinking before the verdict exists, and 300 tokens ran out mid-thought. The judge budget is now its own — `JUDGE_MAX_TOKENS` (default 2000) or `--judge-max-tokens` — and is recorded in `llm_results.json`, so a harness failure can never again masquerade as a drafting score.

## Quickstart

### A. NVIDIA NIM — hosted (the lead profile)

Get an `nvapi-` key from build.nvidia.com, then:

```bash
cp .env.example .env   # the file is pre-filled for the NIM profile — paste your key
uv sync --extra dev --extra llm --extra postgres --extra s3
uv run ap-agent-demo   # the trace header names the backend that produced it
```

`.env.example` documents the full NIM profile: Nemotron LLM (`MODEL_BACKEND=openai` + the integrate endpoint), Nemotron OCR (`OCR_ENGINE=nim`), Nemotron VL (`VISION_ENGINE=nim`), NV-Embed (`RETRIEVER=hybrid` + `OPENAI_EMBEDDING_MODEL=nvidia/nemotron-3-embed-1b`). For self-hosted NIM containers, point the `*_BASE_URL` variables at your containers instead — no code changes.

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
make test        # 462 tests
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
| `ACTION_WEBHOOK_URL` | Opt-in: POSTed once on approval, and by the recovery-escalation sweep on an SLA breach — the only external action available |
| `ACTION_WEBHOOK_SECRET` | When set, the webhook body is HMAC-SHA256 signed (`X-Trida-Signature`) |
| `COST_OF_CAPITAL_ANNUAL_PCT` | Cost of capital (default 6.0) used to price early-payment discounts |
| `PAYMENT_RUN_FOUR_EYES_THRESHOLD` | A payment run above this net total needs a second, different confirmer before hand-off (default 25000) |
| `PAYMENT_RUN_FOUR_EYES_THRESHOLDS` | Per-currency four-eyes thresholds (`EUR=50000,JPY=3000000`); the global threshold is the fallback for unlisted currencies |
| `RECOVERY_CASE_SLA_DAYS` | Days a recovery case may sit in one status before the queue flags it stuck (default 14) |
| `MASTER_CHANGE_EXPIRY_DAYS` | Days a pending supplier-master change may sit undecided before it expires and must be re-proposed (default 30) |
| `POSTING_RECEIPT_GRACE_DAYS` | Days an approved invoice may go without a posted receipt before the approved-not-posted report flags it (default 7) |
| `POSTING_ESCALATION_DAYS` | Days an open posting may age before it escalates into the approver queue as an action item (default 21, never below the grace window) |
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
tests/            462 hermetic tests + Postgres integration tests (gated on DATABASE_URL)
data/sample/      mirrors of the bundled samples + policy corpus
docs/             architecture.md
```

## QA checklist (for reviewers)

1. `uv sync --extra dev --extra llm --extra postgres --extra s3` — clean install.
2. `make test` — expect 462 passed, 3 skipped (the Postgres integration tests skip without `DATABASE_URL`).
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
