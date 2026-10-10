"""Service layer: run + approver-decision bookkeeping.

Approving a draft records the approver's decision and marks the
packet final. By default it performs NO external action (no email,
no supplier portal call, nothing bound). The one opt-in exception is
output routing: when ``ACTION_WEBHOOK_URL`` is configured, the
approved packet is POSTed to that endpoint — the customer's system
of choice — and the outcome is recorded on the result (see
``dispatch_approval_webhook``). When ``ACTION_WEBHOOK_SECRET`` is
also set, the body is HMAC-SHA256 signed (``X-Trida-Signature``
header) so the receiver can verify origin and integrity.

Records live in an approval store (``store.py``): PostgreSQL when
``DATABASE_URL`` is set (production — analyses and decisions survive
restarts and are shared across processes), the in-memory store as a
test double otherwise.

The approver gate is checkpointed (``checkpoints.py``, on unless
``CHECKPOINTS=off``): an analysis pauses its graph at the gate with
the state persisted under the invoice id, and approve/reject
resume that thread with the decision. The split is deliberate — the
store is the record of decisions (and the only thing the decision
flow reads); the checkpointer holds graph state. A missing/finished
thread never blocks a decision.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import threading
import urllib.request
from datetime import datetime, timezone
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

from .bundles import segment_bundle
from .checkpoints import get_checkpointer
from .config import env_float, env_str, load_dotenv
from .creditnotes import evaluate_credit_note
from .matching import _parse_date
from .paymentruns import propose_payment_runs as _propose_runs
from .paymentruns import run_member_ids
from .queue import build_queue
from .recovery import build_recovery_pack
from .reconciliation import reconcile_statement as _reconcile
from .graph import resume_approval, run_invoice
from .model_backends import ModelBackend, get_backend
from .ocr import get_ocr_engine
from .ports import Checkpointer, DocVision, EventSink, OcrEngine
from .retriever import Retriever, get_retriever
from .schemas import (
    AgentResult,
    BundleInvoiceOutcome,
    BundleResult,
    CreditNoteInput,
    CreditNoteMatch,
    InvoiceBundleInput,
    InvoiceInput,
    PaymentRun,
    QueueView,
    RecoveryPack,
    StatementReconciliation,
    SupplierStatement,
)
from .store import ApprovalRecord, default_store, find_duplicate, supplier_summary
from .vision import get_vision_engine

__all__ = [
    "AUDIT_COLUMNS",
    "ApprovalRecord",
    "BatchItem",
    "InvoiceService",
    "Store",
]

# The audit export's stable column order (see InvoiceService.audit_rows).
AUDIT_COLUMNS = [
    "invoice_id",
    "supplier_name",
    "invoice_number",
    "total_amount",
    "currency",
    "outcome",
    "severity",
    "match_status",
    "status",
    "decision",
    "decided_by",
    "decision_reason",
    "dispatch_status",
    "external_action_taken",
    "backend",
    "model",
    "estimated_cost_usd",
]


# Store protocol re-export for call sites that typed against the service.
from .ports import Store  # noqa: E402,F401


@dataclass
class BatchItem:
    """One outcome of a batch analysis: a result, or the error that
    stopped that one invoice — never the whole batch."""

    invoice_id: str
    result: AgentResult | None = None
    error: str | None = None


def dispatch_approval_webhook(record: ApprovalRecord) -> str | None:
    """POST the approved packet to ``ACTION_WEBHOOK_URL`` (off by default).

    This is the output-routing seam — the thin adapter between an
    approval and the customer's system of choice (their policy admin
    system, a work queue, an automation endpoint). Returns ``None``
    when no URL is configured (the default: no external action at
    all), ``"sent"`` on a 2xx response, and ``"failed"`` on any error
    or non-2xx — a failed dispatch never undoes the approval; it is
    recorded on the result for follow-up. Short timeout on purpose:
    an approval must not hang on a downstream system.
    """
    load_dotenv()
    url = env_str("ACTION_WEBHOOK_URL")
    if not url:
        return None
    result = record.result
    payload = {
        "invoice_id": result.invoice_id,
        "status": result.status,
        "decided_by": result.decided_by,
        "triage": result.triage.model_dump(mode="json"),
        "drafts": {
            "subject": result.drafts.subject if result.drafts else "",
            "supplier_letter": result.drafts.supplier_letter if result.drafts else "",
            "ap_packet": result.drafts.ap_packet if result.drafts else "",
        },
    }
    body = json.dumps(payload).encode("utf-8")
    headers = {"Content-Type": "application/json"}
    secret = env_str("ACTION_WEBHOOK_SECRET")
    if secret:
        # HMAC-SHA256 over the exact body bytes, so the receiving
        # system can verify the packet came from this agent and was
        # not altered in transit: hex(HMAC(secret, body)).
        signature = hmac.new(
            secret.encode("utf-8"), body, hashlib.sha256
        ).hexdigest()
        headers["X-Trida-Signature"] = f"sha256={signature}"
    request = urllib.request.Request(
        url,
        data=body,
        headers=headers,
        method="POST",
    )
    timeout = env_float("ACTION_WEBHOOK_TIMEOUT_SECONDS", 5.0)
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return "sent" if 200 <= response.status < 300 else "failed"
    except Exception:  # dispatch failure is recorded, never raised
        return "failed"


@dataclass
class InvoiceService:
    backend: ModelBackend | None = None
    retriever: Retriever | None = None
    store: Store | None = None
    # The gate checkpointer: a saver instance, None (= resolve from
    # the environment, CHECKPOINTS, default on), or False (= disabled,
    # whatever the environment says — the CLI batch path uses this
    # because it persists nothing).
    checkpointer: Checkpointer | bool | None = None
    # The document-intelligence ports (OCR + vision) and the object
    # store documents resolve through. None = resolve from the
    # environment on first use.
    ocr_engine: OcrEngine | None = None
    vision_engine: DocVision | None = None
    object_store: object | None = None
    _resolved_store: Store | None = field(default=None, repr=False)
    _resolved_checkpointer: Checkpointer | None = field(default=None, repr=False)
    _checkpointer_resolved: bool = field(default=False, repr=False)

    def _get_store(self) -> Store:
        if self._resolved_store is None:
            self._resolved_store = self.store or default_store()
        return self._resolved_store

    def _get_checkpointer(self) -> Checkpointer | None:
        if not self._checkpointer_resolved:
            self._checkpointer_resolved = True
            if self.checkpointer is False:
                self._resolved_checkpointer = None
            elif self.checkpointer is not None:
                self._resolved_checkpointer = self.checkpointer  # type: ignore[assignment]
            else:
                self._resolved_checkpointer = get_checkpointer()
        return self._resolved_checkpointer

    def _resume_thread(self, invoice_id: str, decision: dict) -> None:
        """Complete the checkpointed graph thread with a decision.

        Best-effort by design: the store record the caller just saved
        is the decision of record. A thread that is missing or already
        finished (the analysis ran with checkpoints off, modes were
        mixed) has nothing to resume, and a resume failure never
        unmakes a recorded human decision.
        """
        checkpointer = self._get_checkpointer()
        if checkpointer is None:
            return
        try:
            resume_approval(invoice_id, decision, checkpointer)
        except Exception:
            pass

    def analyze(
        self, invoice: InvoiceInput | dict, event_sink: EventSink | None = None
    ) -> AgentResult:
        # Backend and retriever come from the environment (MODEL_BACKEND /
        # RETRIEVER, with the repo-root .env loaded) unless injected.
        # ``event_sink`` (optional) receives the run's structured events
        # (see events.py) — the API's streaming endpoint passes one.
        backend = self.backend or get_backend()
        if self.retriever is None:
            self.retriever = get_retriever()
        model = (
            invoice
            if isinstance(invoice, InvoiceInput)
            else InvoiceInput.model_validate(invoice)
        )
        return self._analyze_model(model, backend, self.retriever, event_sink=event_sink)

    def _analyze_model(
        self,
        model: InvoiceInput,
        backend: ModelBackend,
        retriever: Retriever,
        event_sink: EventSink | None = None,
    ) -> AgentResult:
        # Memory: what the store already knows about this supplier and
        # this supplier becomes diagnosis evidence for the new analysis.
        # The raw entries ride along too — the diagnosis tool loop
        # (provider mode) reads supplier / supplier history through them.
        priors = self._get_store().prior_invoices(
            exclude_invoice_id=model.invoice_id
        )
        history = self._history_summary(model, priors)
        # Feedback loop: what approvers said about earlier cases
        # for this supplier (with their reasons) joins the same
        # evidence — the next diagnosis shows the pattern of human
        # oversight.
        feedback = self._feedback_for(model)
        if feedback:
            history = {**(history or {}), "feedback": feedback}
        # Duplicate detection (AP-DUP-01): same supplier + invoice
        # number + total as a prior analysed invoice. Computed here,
        # where the store lives; the match facts carry it from here.
        duplicate_of = find_duplicate(priors, model)
        result = run_invoice(
            model,
            backend=backend,
            retriever=retriever,
            history=history,
            priors=priors,
            event_sink=event_sink,
            checkpointer=self._get_checkpointer(),
            thread_id=model.invoice_id,
            ocr_engine=self.ocr_engine or get_ocr_engine(),
            vision_engine=self.vision_engine or get_vision_engine(),
            object_store=self.object_store,
            duplicate_of=duplicate_of,
        )
        self._get_store().save(
            ApprovalRecord(result=result, invoice=model.model_dump(mode="json"))
        )
        return result

    def analyze_batch(
        self,
        invoices: list[InvoiceInput | dict],
        concurrency: int = 1,
    ) -> list[BatchItem]:
        """Analyse many invoices, in input order, optionally concurrently.

        Each item runs the same pipeline as :meth:`analyze` and gets
        its own result object. Injected backend/retriever are shared
        (test doubles are stateless); otherwise every item resolves
        fresh ones from the environment, so a provider backend's usage
        counters and a retriever's per-run stats never race across
        threads. The store is shared deliberately — batch items see
        each other as memory, like a real intake queue — and its
        writes are serialised by the store itself (PostgreSQL: one
        connection per call; in-memory: a lock). An invoice that
        fails (bad payload, provider down) lands in its item's
        ``error``; it never fails the batch.
        """

        def run_one(raw: InvoiceInput | dict) -> BatchItem:
            invoice_id = (
                raw.get("invoice_id", "unknown")
                if isinstance(raw, dict)
                else getattr(raw, "invoice_id", "unknown")
            )
            try:
                model = (
                    raw
                    if isinstance(raw, InvoiceInput)
                    else InvoiceInput.model_validate(raw)
                )
                backend = self.backend or get_backend()
                retriever = self.retriever or get_retriever()
                result = self._analyze_model(model, backend, retriever)
                return BatchItem(invoice_id=model.invoice_id, result=result)
            except Exception as exc:  # one bad invoice never fails the batch
                return BatchItem(
                    invoice_id=str(invoice_id),
                    error=f"{type(exc).__name__}: {exc}",
                )

        workers = max(1, int(concurrency))
        if workers == 1 or len(invoices) <= 1:
            return [run_one(item) for item in invoices]
        semaphore = threading.Semaphore(workers)

        def guarded(item: InvoiceInput | dict) -> BatchItem:
            with semaphore:
                return run_one(item)

        with ThreadPoolExecutor(max_workers=workers) as pool:
            return list(pool.map(guarded, invoices))

    def _feedback_for(self, invoice: InvoiceInput) -> list[dict]:
        """Recent approver feedback for this supplier.

        Bounded to the last 3 matching decisions (the store returns
        newest first), reasons truncated — feedback is evidence, not
        a transcript.
        """
        entries = self._get_store().decision_feedback(
            exclude_invoice_id=invoice.invoice_id
        )
        if not entries:
            return []
        matched: list[dict] = []
        for entry in entries:
            if not (invoice.supplier_name and entry["supplier_name"] == invoice.supplier_name):
                continue
            matched.append(
                {
                    "decision": entry["decision"],
                    "reason": entry["reason"][:160],
                    "match": "supplier",
                }
            )
            if len(matched) == 3:
                break
        return matched

    def _history_summary(
        self, invoice: InvoiceInput, priors: list[dict] | None = None
    ) -> dict | None:
        """Summarise prior analysed invoices for this supplier.

        Returns None when there is nothing to remember.
        """
        if priors is None:
            priors = self._get_store().prior_invoices(
                exclude_invoice_id=invoice.invoice_id
            )
        if not priors:
            return None
        summary = supplier_summary(priors, invoice.supplier_name)
        if not summary["supplier_count"]:
            return None
        hits = [
            e
            for e in priors
            if invoice.supplier_name and e["supplier_name"] == invoice.supplier_name
        ]
        return {
            "supplier": invoice.supplier_name,
            "supplier_count": summary["supplier_count"],
            "supplier_recent_outcomes": [e["outcome"] for e in hits[:3]],
            "supplier_outcome_counts": summary["supplier_outcome_counts"],
        }

    def readiness(self) -> dict:
        """Readiness probe for /readiness: the store answers, or the
        service is not ready. The in-memory double always answers —
        it is process-local by design and the probe says so."""
        store = self._get_store()
        ping = getattr(store, "ping", None)
        db_ok = bool(ping()) if ping is not None else True
        return {
            "ready": db_ok,
            "store": getattr(store, "name", "unknown"),
            "database": "ok" if db_ok else "unreachable",
        }

    def close(self) -> None:
        """Release store resources at shutdown (per-call connections
        make this a no-op for the shipped stores; the hook exists so
        the API lifespan has one place to call)."""
        store = self._resolved_store
        close = getattr(store, "close", None) if store is not None else None
        if close is not None:
            close()

    def approve(
        self, invoice_id: str, approver: str, reason: str = ""
    ) -> AgentResult:
        store = self._get_store()
        record = store.get(invoice_id)
        if record is None:
            raise KeyError(f"Unknown invoice_id: {invoice_id} (analyze it first)")
        if record.result.status != "awaiting_approver":
            raise ValueError(
                f"Invoice {invoice_id} is not awaiting an approver "
                f"(status: {record.result.status})"
            )
        if record.result.validation is not None and not record.result.validation.passed:
            failed = "; ".join(
                f"{c.name}: {c.detail}"
                for c in record.result.validation.checks
                if not c.passed
            )
            raise ValueError(
                "Draft failed guardrail validation and cannot be approved: " + failed
            )
        record.approved = True
        record.approver = approver
        record.approve_reason = reason
        record.result.status = "approved"
        record.result.decision = "approved"
        record.result.decided_by = approver
        record.result.decision_reason = reason or None
        # Output routing: by default there is NO external action. When
        # the operator configures ACTION_WEBHOOK_URL, the approved
        # packet is POSTed to that endpoint (the customer's system of
        # choice) and the outcome is recorded — a failed dispatch
        # leaves the approval standing and says so on the result.
        record.result.external_action_taken = False
        dispatch = dispatch_approval_webhook(record)
        if dispatch is not None:
            record.dispatch_status = dispatch
            record.result.dispatch_status = dispatch
            if dispatch == "sent":
                record.result.external_action_taken = True
        store.save(record)
        self._resume_thread(
            invoice_id,
            {"decision": "approved", "actor": approver, "reason": reason},
        )
        return record.result

    def reject(self, invoice_id: str, reviewer: str, reason: str = "") -> AgentResult:
        store = self._get_store()
        record = store.get(invoice_id)
        if record is None:
            raise KeyError(f"Unknown invoice_id: {invoice_id} (analyze it first)")
        if record.result.status != "awaiting_approver":
            raise ValueError(
                f"Invoice {invoice_id} is not awaiting an approver "
                f"(status: {record.result.status})"
            )
        record.rejected_by = reviewer
        record.reject_reason = reason
        record.result.status = "rejected"
        record.result.decision = "rejected"
        record.result.decided_by = reviewer
        record.result.decision_reason = reason or None
        record.result.external_action_taken = False
        store.save(record)
        self._resume_thread(
            invoice_id,
            {"decision": "rejected", "actor": reviewer, "reason": reason},
        )
        return record.result

    def get(self, invoice_id: str) -> AgentResult | None:
        record = self._get_store().get(invoice_id)
        return record.result if record else None

    def metrics_snapshot(self) -> dict:
        """Operational metrics over the stored runs: volumes,
        outcomes, decisions, latency and model spend.

        Aggregated from the records at read time — the store is the
        system of record, so the numbers can never drift from a
        separate counters table. Token/cost totals sum only the runs
        that reported them (provider mode); when none did, they are
        null rather than a dressed-up zero.
        """
        records = self._get_store().list_records()
        by_status: dict[str, int] = {}
        by_outcome: dict[str, int] = {}
        latencies: list[float] = []
        input_tokens = 0
        output_tokens = 0
        tokens_seen = False
        costs: list[float] = []
        for record in records:
            result = record.result
            by_status[result.status] = by_status.get(result.status, 0) + 1
            outcome = result.triage.outcome
            by_outcome[outcome] = by_outcome.get(outcome, 0) + 1
            telemetry = result.telemetry
            if telemetry is None:
                continue
            latencies.append(telemetry.latency_seconds)
            if telemetry.input_tokens is not None:
                tokens_seen = True
                input_tokens += telemetry.input_tokens
                output_tokens += telemetry.output_tokens or 0
            if telemetry.estimated_cost_usd is not None:
                costs.append(telemetry.estimated_cost_usd)
        return {
            "runs_total": len(records),
            "by_status": by_status,
            "by_outcome": by_outcome,
            "decisions": {
                "approved": by_status.get("approved", 0),
                "rejected": by_status.get("rejected", 0),
                "awaiting_approver": by_status.get("awaiting_approver", 0),
            },
            "mean_latency_seconds": (
                round(sum(latencies) / len(latencies), 3) if latencies else None
            ),
            "total_input_tokens": input_tokens if tokens_seen else None,
            "total_output_tokens": output_tokens if tokens_seen else None,
            "total_estimated_cost_usd": (
                round(sum(costs), 4) if costs else None
            ),
        }

    def audit_rows(self) -> list[dict]:
        """The audit trail: one flat row per processed invoice —
        what the pipeline concluded, what the human decided, and what
        left the building (dispatch). Column order is stable; the
        API renders it as JSON or CSV."""
        rows: list[dict] = []
        for record in self._get_store().list_records():
            result = record.result
            invoice = record.invoice or {}
            telemetry = result.telemetry
            rows.append(
                {
                    "invoice_id": result.invoice_id,
                    "supplier_name": result.supplier_name,
                    "invoice_number": invoice.get("invoice_number", ""),
                    "total_amount": invoice.get("total_amount"),
                    "currency": invoice.get("currency", ""),
                    "outcome": result.triage.outcome,
                    "severity": result.triage.severity,
                    "match_status": result.facts.match_status,
                    "status": result.status,
                    "decision": result.decision,
                    "decided_by": result.decided_by,
                    "decision_reason": result.decision_reason,
                    "dispatch_status": result.dispatch_status,
                    "external_action_taken": result.external_action_taken,
                    "backend": telemetry.backend if telemetry else "",
                    "model": telemetry.model if telemetry else "",
                    "estimated_cost_usd": (
                        telemetry.estimated_cost_usd if telemetry else None
                    ),
                }
            )
        return rows

    def match_credit_note(
        self, note: CreditNoteInput | dict
    ) -> CreditNoteMatch:
        """Match a supplier credit note against the processed invoices.

        A pure query over the store: the note is tied to the invoice
        it references (supplier + supplier invoice number), that
        invoice's overbilled amount is recomputed from its stored
        match facts, and the report says whether the credit settles
        the variance. Nothing is written — an approver acts on the
        report.
        """
        model = (
            note
            if isinstance(note, CreditNoteInput)
            else CreditNoteInput.model_validate(note)
        )
        return evaluate_credit_note(model, self._get_store().list_records())

    def reconcile_statement(
        self, statement: SupplierStatement | dict
    ) -> StatementReconciliation:
        """Reconcile a supplier statement against the processed
        invoices and store the report for confirmation.

        The comparison itself is computed in ``reconciliation.py``
        over the store's history projection. The report is stored so
        a human can confirm it later; reconciliation writes nothing
        to any invoice record.
        """
        model = (
            statement
            if isinstance(statement, SupplierStatement)
            else SupplierStatement.model_validate(statement)
        )
        history = self._get_store().prior_invoices(exclude_invoice_id="")
        report = _reconcile(model, history)
        self._get_store().save_reconciliation(report)
        return report

    def get_reconciliation(
        self, statement_id: str
    ) -> StatementReconciliation | None:
        return self._get_store().get_reconciliation(statement_id)

    def propose_payment_runs(
        self, window_days: int = 7, as_of: str = ""
    ) -> list[PaymentRun]:
        """Propose payment runs from the approved invoices.

        Grouping and totals are computed in ``paymentruns.py`` over
        the store's records. Invoices already inside a confirmed run
        are excluded — a confirmed pay list is not re-proposed. Each
        fresh proposal is stored as a draft object; re-proposing
        refreshes an unconfirmed run with the same window, and never
        overwrites a confirmed one.
        """
        store = self._get_store()
        anchor = _parse_date(as_of) if as_of else None
        confirmed_members: set[str] = set()
        confirmed_ids: set[str] = set()
        for existing in store.list_payment_runs():
            if existing.status == "confirmed":
                confirmed_ids.add(existing.run_id)
                confirmed_members |= run_member_ids(existing)
        runs = _propose_runs(
            store.list_records(),
            window_days=window_days,
            as_of=anchor,
            exclude_invoice_ids=confirmed_members,
        )
        stamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
        saved: list[PaymentRun] = []
        for run in runs:
            if run.run_id in confirmed_ids:
                continue  # a confirmed run is the record; never refreshed
            run.proposed_at = stamp
            store.save_payment_run(run)
            saved.append(run)
        return saved

    def list_payment_runs(self) -> list[PaymentRun]:
        return self._get_store().list_payment_runs()

    def analyze_bundle(
        self, bundle: InvoiceBundleInput | dict
    ) -> BundleResult:
        """Analyse one multi-invoice arrival: segment it in code
        (``bundles.py``), then run every segment through the full
        pipeline as its own invoice — its own record, its own gate.
        A segment that fails lands in its outcome's ``error``; it
        never fails the bundle. The summary is stored so the arrival
        can be retrieved as one batch later."""
        model = (
            bundle
            if isinstance(bundle, InvoiceBundleInput)
            else InvoiceBundleInput.model_validate(bundle)
        )
        invoices, note = segment_bundle(model)
        outcomes: list[BundleInvoiceOutcome] = []
        for invoice in invoices:
            try:
                backend = self.backend or get_backend()
                retriever = self.retriever or get_retriever()
                result = self._analyze_model(invoice, backend, retriever)
                outcomes.append(
                    BundleInvoiceOutcome(
                        invoice_id=invoice.invoice_id,
                        invoice_number=result.invoice_number,
                        status=result.status,
                        outcome=result.triage.outcome,
                        severity=result.triage.severity,
                        total_amount=invoice.total_amount,
                        currency=invoice.currency,
                    )
                )
            except Exception as exc:  # one bad segment never fails the bundle
                outcomes.append(
                    BundleInvoiceOutcome(
                        invoice_id=invoice.invoice_id,
                        invoice_number=invoice.invoice_number,
                        status="error",
                        total_amount=invoice.total_amount,
                        currency=invoice.currency,
                        error=f"{type(exc).__name__}: {exc}",
                    )
                )
        outcome_counts: dict[str, int] = {}
        totals: dict[str, float] = {}
        for outcome in outcomes:
            key = outcome.outcome or "error"
            outcome_counts[key] = outcome_counts.get(key, 0) + 1
            if outcome.status != "error" and outcome.total_amount is not None:
                totals[outcome.currency] = round(
                    totals.get(outcome.currency, 0.0) + float(outcome.total_amount),
                    2,
                )
        summary = BundleResult(
            bundle_id=model.bundle_id,
            supplier_name=model.supplier_name,
            segment_count=len(invoices),
            processed_count=sum(1 for o in outcomes if o.status != "error"),
            outcomes=outcomes,
            outcome_counts=outcome_counts,
            totals_by_currency=totals,
            note=note,
        )
        self._get_store().save_bundle(summary)
        return summary

    def get_bundle(self, bundle_id: str) -> BundleResult | None:
        return self._get_store().get_bundle(bundle_id)

    def draft_duplicate_recovery(self, invoice_id: str) -> RecoveryPack:
        """Draft the duplicate-payment recovery pack for one invoice.

        The case exists when duplicate detection fired on the
        invoice and the twin it duplicates was already approved —
        one payment in flight, a second copy to recover against.
        The evidence and amounts are computed from the two stored
        records; the letter and memo go through the standard verify +
        guardrail path (``recovery.py``) and the pack is stored for
        a human. Idempotent: an existing pack is returned, not
        redrafted. Drafting changes no invoice's state.
        """
        store = self._get_store()
        existing = store.get_recovery_pack(invoice_id)
        if existing is not None:
            return existing
        record = store.get(invoice_id)
        if record is None:
            raise KeyError(f"Unknown invoice_id: {invoice_id} (analyze it first)")
        result = record.result
        twin_id = result.facts.duplicate_of
        if result.triage.outcome != "duplicate_suspected" or not twin_id:
            raise ValueError(
                f"Invoice {invoice_id} is not a suspected duplicate "
                f"(outcome: {result.triage.outcome}) — no recovery case"
            )
        twin = store.get(twin_id)
        if twin is None:
            raise ValueError(
                f"Invoice {invoice_id} duplicates {twin_id}, which is "
                "not in the store — the recovery case cannot be evidenced"
            )
        if twin.result.status != "approved":
            raise ValueError(
                f"Invoice {invoice_id} duplicates {twin_id}, but the "
                f"original is {twin.result.status}, not approved — "
                "recovery starts once the original has been approved; "
                "until then the duplicate control at the gate is enough"
            )
        backend = self.backend or get_backend()
        pack = build_recovery_pack(record, twin, backend)
        store.save_recovery_pack(pack)
        return pack

    def get_recovery_pack(self, invoice_id: str) -> RecoveryPack | None:
        return self._get_store().get_recovery_pack(invoice_id)

    def queue_view(self, as_of: str = "") -> QueueView:
        """The AP aging queue: invoices still at the gate, aged by
        due date, overdue first, with the triage mix. A pure read
        over the store — it decides and writes nothing."""
        anchor = _parse_date(as_of) if as_of else None
        return build_queue(self._get_store().list_records(), as_of=anchor)

    def get_payment_run(self, run_id: str) -> PaymentRun | None:
        return self._get_store().get_payment_run(run_id)

    def confirm_payment_run(self, run_id: str, actor: str) -> PaymentRun:
        """Record a finance lead's confirmation of a proposed run.

        Confirmation touches the run record only: no invoice changes
        state and no payment is executed — the confirmed run is the
        pay list the customer's AP system of record acts on.
        """
        store = self._get_store()
        run = store.get_payment_run(run_id)
        if run is None:
            raise KeyError(f"Unknown run_id: {run_id} (propose it first)")
        if run.status == "confirmed":
            raise ValueError(
                f"Payment run {run_id} was already confirmed by "
                f"{run.confirmed_by}"
            )
        run.status = "confirmed"
        run.confirmed_by = actor
        run.confirmed_at = datetime.now(timezone.utc).isoformat(
            timespec="seconds"
        )
        store.save_payment_run(run)
        return run

    def confirm_reconciliation(
        self, statement_id: str, actor: str
    ) -> StatementReconciliation:
        """Record a human's confirmation of a reconciliation report.

        Confirmation touches the report record only — the invoices it
        describes keep whatever state their own gate decisions gave
        them."""
        store = self._get_store()
        report = store.get_reconciliation(statement_id)
        if report is None:
            raise KeyError(
                f"Unknown statement_id: {statement_id} (reconcile it first)"
            )
        if report.confirmed_by:
            raise ValueError(
                f"Statement {statement_id} was already confirmed by "
                f"{report.confirmed_by}"
            )
        report.confirmed_by = actor
        report.confirmed_at = datetime.now(timezone.utc).isoformat(
            timespec="seconds"
        )
        store.save_reconciliation(report)
        return report
