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
from datetime import datetime, timedelta, timezone
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

from .bundles import segment_bundle
from .checkpoints import get_checkpointer
from .config import env_float, env_int, env_str, load_dotenv
from .creditnotes import evaluate_credit_note
from .matching import _parse_date
from .paymentruns import discount_drift, handoff_metrics, wait_cost_budget_status
from .paymentruns import price_four_eyes_wait as _price_four_eyes_wait
from .paymentruns import drift_forecast as _drift_forecast
from .paymentruns import propose_payment_runs as _propose_runs
from .paymentruns import run_member_ids
from .queue import build_queue, recovery_case_aging
from .recovery import build_recovery_pack
from .reconciliation import reconcile_statement as _reconcile
from .graph import resume_approval, run_invoice
from .master_changes import change_age_days, is_change_expired
from .master_changes import next_request_id, proposed_from_invoice
from .model_backends import ModelBackend, get_backend
from .ocr import get_ocr_engine
from .ports import Checkpointer, DocVision, EventSink, OcrEngine
from .retriever import Retriever, get_retriever
from .schemas import (
    AgentResult,
    ApprovedNotPostedReport,
    BundleInvoiceOutcome,
    BundleResult,
    CreditNoteInput,
    CreditNoteMatch,
    DigestHistory,
    DigestStoreSweep,
    DigestStoreSweepEntry,
    EscalationDigest,
    HandoffProfile,
    HandoffProfileEvent,
    MarkerWorklist,
    InvoiceBundleInput,
    InvoiceInput,
    MasterChangeEvent,
    MasterChangeRequest,
    OpenPostingItem,
    PaymentRun,
    PostingEscalation,
    PostingReceipt,
    QueueView,
    RecoveryEscalation,
    RecoveryPack,
    StatementReconciliation,
    SupplierMasterRecord,
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
    "posting_status",
    "posting_reference",
]


# Store protocol re-export for call sites that typed against the service.
from .ports import Store  # noqa: E402,F401


# Default four-eyes threshold for payment runs: a run whose net
# total is above this needs a second, different confirmer before
# its hand-off file can be generated (PAYMENT_RUN_FOUR_EYES_THRESHOLD
# overrides it). The threshold is per-currency in effect:
# PAYMENT_RUN_FOUR_EYES_THRESHOLDS ("EUR=50000,JPY=3000000") maps
# currencies to their own figures, and the global value is the
# fallback for currencies the map does not name.
PAYMENT_RUN_FOUR_EYES_DEFAULT = 25_000.0


def payment_run_four_eyes_threshold() -> float:
    """The configured global four-eyes threshold for payment runs
    — the fallback for currencies with no per-currency figure."""
    load_dotenv()
    return env_float(
        "PAYMENT_RUN_FOUR_EYES_THRESHOLD", PAYMENT_RUN_FOUR_EYES_DEFAULT
    )


def payment_run_four_eyes_thresholds() -> dict[str, float]:
    """The per-currency four-eyes thresholds, from
    ``PAYMENT_RUN_FOUR_EYES_THRESHOLDS`` — a comma list of
    ``CODE=amount`` pairs. Malformed pairs are skipped: a typo in
    one currency's entry must not silently change another's gate,
    and a currency lost to a typo falls back to the global
    threshold rather than to no gate at all."""
    load_dotenv()
    raw = env_str("PAYMENT_RUN_FOUR_EYES_THRESHOLDS", "") or ""
    thresholds: dict[str, float] = {}
    for part in raw.split(","):
        code, sep, value = part.partition("=")
        if not sep or not code.strip():
            continue
        try:
            thresholds[code.strip().upper()] = float(value.strip())
        except ValueError:
            continue
    return thresholds


def payment_run_four_eyes_threshold_for(currency: str) -> float:
    """The four-eyes threshold that applies to one currency: its
    per-currency figure when configured, else the global one."""
    return payment_run_four_eyes_thresholds().get(
        (currency or "").upper(), payment_run_four_eyes_threshold()
    )


def four_eyes_wait_budgets() -> dict[str, float]:
    """The per-currency monthly budgets for the priced four-eyes
    wait, from ``FOUR_EYES_WAIT_BUDGETS`` — a comma list of
    ``CODE=amount`` pairs, parsed like the four-eyes thresholds.
    A currency with no entry is *unbudgeted*: its wait cost is
    still priced and reported, it simply has no ceiling to
    breach — an absent budget is never read as a zero budget,
    and a malformed pair is skipped rather than fatal."""
    load_dotenv()
    raw = env_str("FOUR_EYES_WAIT_BUDGETS", "") or ""
    budgets: dict[str, float] = {}
    for part in raw.split(","):
        code, sep, value = part.partition("=")
        if not sep or not code.strip():
            continue
        try:
            budgets[code.strip().upper()] = float(value.strip())
        except ValueError:
            continue
    return budgets


def four_eyes_wait_budget_for(currency: str) -> float | None:
    """The monthly wait-cost budget for one currency, or None
    when the currency carries no budget."""
    return four_eyes_wait_budgets().get((currency or "").upper())


# Default aging threshold for a four-eyes run's second
# confirmation: a confirmed-once run waiting longer than this for
# its second confirmer joins the approver queue as an action item
# (PAYMENT_RUN_FOUR_EYES_AGING_DAYS overrides it). Short on
# purpose — the wait is pure stall: the run cannot export, and
# its members' discount windows keep closing while it sits.
PAYMENT_RUN_FOUR_EYES_AGING_DEFAULT_DAYS = 2


def four_eyes_aging_days() -> int:
    """The configured four-eyes aging threshold, in days since
    the first confirmation."""
    load_dotenv()
    return max(
        0,
        env_int(
            "PAYMENT_RUN_FOUR_EYES_AGING_DAYS",
            PAYMENT_RUN_FOUR_EYES_AGING_DEFAULT_DAYS,
        ),
    )


# Default recovery-case SLA: a case may sit in one status this many
# days before the queue flags it as stuck (RECOVERY_CASE_SLA_DAYS
# overrides it).
RECOVERY_CASE_SLA_DEFAULT_DAYS = 14


def recovery_case_sla_days() -> int:
    """The configured recovery-case SLA, in days per status."""
    load_dotenv()
    return max(1, env_int("RECOVERY_CASE_SLA_DAYS", RECOVERY_CASE_SLA_DEFAULT_DAYS))


# Default decision window for a pending master change: a request
# nobody has decided within this many days expires (MASTER_CHANGE_
# EXPIRY_DAYS overrides it) — a live bank-detail change must not
# sit undecided, waiting for a careless approval.
MASTER_CHANGE_EXPIRY_DEFAULT_DAYS = 30


def master_change_expiry_days() -> int:
    """The configured expiry window for pending master changes."""
    load_dotenv()
    return max(
        1, env_int("MASTER_CHANGE_EXPIRY_DAYS", MASTER_CHANGE_EXPIRY_DEFAULT_DAYS)
    )


# Default grace window for the posting-receipt loop: an approved
# invoice with no posted receipt older than this is reported as an
# open item (POSTING_RECEIPT_GRACE_DAYS overrides it).
POSTING_RECEIPT_GRACE_DEFAULT_DAYS = 7


def posting_receipt_grace_days() -> int:
    """The configured grace window for posting receipts, in days."""
    load_dotenv()
    return max(
        0,
        env_int(
            "POSTING_RECEIPT_GRACE_DAYS", POSTING_RECEIPT_GRACE_DEFAULT_DAYS
        ),
    )


# Default escalation threshold for open postings: an approved
# invoice still unposted this many days after its decision stops
# being a report line and joins the approver queue as an action
# item (POSTING_ESCALATION_DAYS overrides it). The effective
# threshold is never below the grace window — an item cannot
# escalate before it is even open.
POSTING_ESCALATION_DEFAULT_DAYS = 21


def posting_escalation_days() -> int:
    """The configured open-posting escalation threshold, in days."""
    load_dotenv()
    return max(
        1, env_int("POSTING_ESCALATION_DAYS", POSTING_ESCALATION_DEFAULT_DAYS)
    )


# Default lookback for the digest auto-store sweep: the sweep
# pins digests for the last N days, today included, wherever a
# morning was missed (DIGEST_STORE_LOOKBACK_DAYS overrides it).
# A week on purpose: long enough to survive a long weekend plus
# a forgotten Monday, short enough that a scheduler running the
# sweep cannot quietly rewrite ancient history — and
# first-computed wins in any case, so stored days are immune.
DIGEST_STORE_LOOKBACK_DEFAULT_DAYS = 7


def digest_store_lookback_days() -> int:
    """The configured lookback for the digest auto-store sweep,
    in days including the anchor day."""
    load_dotenv()
    return max(
        1,
        env_int(
            "DIGEST_STORE_LOOKBACK_DAYS", DIGEST_STORE_LOOKBACK_DEFAULT_DAYS
        ),
    )


def _handoff_profile_summary(profiles: list[HandoffProfile]) -> dict:
    """The governed hand-off profiles in force, counted for
    /metrics: how much routing is configured, where it points,
    and how many payees' banks reject at-import markers (the
    population the proposal-time flags watch)."""
    by_scope: dict[str, int] = {}
    by_format: dict[str, int] = {}
    for profile in profiles:
        by_scope[profile.scope] = by_scope.get(profile.scope, 0) + 1
        by_format[profile.format] = by_format.get(profile.format, 0) + 1
    return {
        "count": len(profiles),
        "by_scope": by_scope,
        "by_format": by_format,
        "markers_rejected": sum(
            1 for profile in profiles if profile.markers_rejected
        ),
    }


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


def _post_signed_webhook(payload: dict) -> str | None:
    """POST one JSON event payload to ``ACTION_WEBHOOK_URL`` — the
    shared wire for the agent's event webhooks. Same contract as
    the approval dispatch: ``None`` when no URL is configured (off
    by default), HMAC-SHA256 signed (``X-Trida-Signature``) when
    ``ACTION_WEBHOOK_SECRET`` is set, ``"sent"`` on 2xx, ``"failed"``
    on any error or non-2xx — a failure is a return value, never an
    exception, so a down receiver never breaks the caller's work.
    """
    load_dotenv()
    url = env_str("ACTION_WEBHOOK_URL")
    if not url:
        return None
    body = json.dumps(payload).encode("utf-8")
    headers = {"Content-Type": "application/json"}
    secret = env_str("ACTION_WEBHOOK_SECRET")
    if secret:
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


def dispatch_recovery_escalation_webhook(
    pack: RecoveryPack,
    *,
    sla_days: int,
    status_age_days: int | None,
    age_days: int | None,
) -> str | None:
    """POST the ``recovery_escalation`` event for one SLA-breached
    recovery case (off by default — the same opt-in seam, secret,
    and signature as the approval webhook).

    The payload is the case's aging snapshot at the breach: which
    case, which supplier, how much money is stuck, the status it is
    stuck in, and how far past the SLA it has run. It carries no
    drafts and no bank details — it is a pointer at the case in the
    queue, for the customer's own paging/ticketing system to act
    on. Returns ``None`` when unconfigured, else ``"sent"`` /
    ``"failed"`` (see :func:`_post_signed_webhook`)."""
    payload = {
        "event": "recovery_escalation",
        "pack_id": pack.pack_id,
        "duplicate_invoice_id": pack.duplicate_invoice_id,
        "original_invoice_id": pack.original_invoice_id,
        "supplier_name": pack.supplier_name,
        "invoice_number": pack.invoice_number,
        "currency": pack.currency,
        "amount": pack.amount,
        "exposure_status": pack.exposure_status,
        "case_status": pack.case_status,
        "age_days": age_days,
        "status_age_days": status_age_days,
        "sla_days": sla_days,
        "sla_breached": True,
    }
    return _post_signed_webhook(payload)


def dispatch_posting_escalation_webhook(
    item: OpenPostingItem,
    *,
    escalation_days: int,
    level: int = 1,
) -> str | None:
    """POST one rung of the posting-escalation ladder for one open
    posting (off by default — the same opt-in seam, secret, and
    signature as the approval webhook).

    Level 1 is the ``posting_escalation`` event, fired past the
    escalation threshold; level 2 is ``posting_escalation_urgent``,
    fired past twice the threshold for a posting the first shout
    did not close — the payload names its rung (``level``,
    ``urgent``) so one receiver can route the two differently.
    ``escalation_days`` is the firing rung's own threshold.

    The payload is the posting's snapshot at the crossing: which
    invoice, which supplier, how much, who approved it, how long
    the ERP loop has been open, and what the ERP said when a
    receipt failed. It carries no drafts and no bank details — it
    is a pointer at the open item in the queue, for the customer's
    own paging/ticketing system to act on. Returns ``None`` when
    unconfigured, else ``"sent"`` / ``"failed"`` (see
    :func:`_post_signed_webhook`)."""
    payload = {
        "event": (
            "posting_escalation" if level < 2 else "posting_escalation_urgent"
        ),
        "invoice_id": item.invoice_id,
        "supplier_name": item.supplier_name,
        "invoice_number": item.invoice_number,
        "currency": item.currency,
        "total_amount": item.total_amount,
        "decided_by": item.decided_by,
        "decided_at": item.decided_at,
        "age_days": item.age_days,
        "age_basis": item.age_basis,
        "receipt_state": item.receipt_state,
        "receipt_detail": item.receipt_detail,
        "escalation_days": escalation_days,
        "escalated": True,
    }
    if level >= 2:
        payload["level"] = level
        payload["urgent"] = True
    return _post_signed_webhook(payload)


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
    # The supplier master-data port (bank details on file): the fraud
    # screen checks document-stated details against it. None = resolve
    # from the environment on first use.
    supplier_master: object | None = None
    _resolved_store: Store | None = field(default=None, repr=False)
    _resolved_supplier_master: object | None = field(default=None, repr=False)
    _resolved_checkpointer: Checkpointer | None = field(default=None, repr=False)
    _checkpointer_resolved: bool = field(default=False, repr=False)

    def _get_store(self) -> Store:
        if self._resolved_store is None:
            self._resolved_store = self.store or default_store()
        return self._resolved_store

    def _get_supplier_master(self):
        if self._resolved_supplier_master is None:
            if self.supplier_master is not None:
                self._resolved_supplier_master = self.supplier_master
            else:
                from .supplier_master import get_supplier_master

                self._resolved_supplier_master = get_supplier_master()
        return self._resolved_supplier_master

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
        # Governed master changes whose effective date has arrived
        # land in the supplier master before screening runs, so the
        # bank check below compares against the record in force
        # today — never against a merely-proposed one.
        self.apply_due_master_changes()
        # Pending changes past their decision window expire here
        # too, so a stale bank-detail request never lingers
        # decidable between operator visits.
        self.expire_stale_master_changes()
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
            supplier_master=self._get_supplier_master(),
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
        record.result.decided_at = datetime.now(timezone.utc).isoformat(
            timespec="seconds"
        )
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
        record.result.decided_at = datetime.now(timezone.utc).isoformat(
            timespec="seconds"
        )
        record.result.decision_reason = reason or None
        record.result.external_action_taken = False
        store.save(record)
        self._resume_thread(
            invoice_id,
            {"decision": "rejected", "actor": reviewer, "reason": reason},
        )
        return record.result

    def record_posting_receipt(
        self,
        invoice_id: str,
        *,
        status: str,
        actor: str,
        external_reference: str = "",
        idempotency_key: str = "",
        detail: str = "",
    ) -> AgentResult:
        """Record the ERP's posting outcome on an approved invoice.

        The receipt loop closes the output seam: the example adapter
        (``examples/erp_posting.py``) shapes the posting, the
        customer's ERP posts it (or refuses it), and its answer —
        ``posted`` + the ERP document number, or ``failed`` + the
        reason — is recorded here, on the invoice record and in the
        audit export. Recording is bookkeeping: the agent posted
        nothing, and no decision changes.

        The guards are the loop's integrity: only an approved
        invoice can carry a receipt (only approvals post); a posted
        receipt must name its ERP reference and a failed one its
        reason; and once an invoice is recorded as posted, a second
        receipt is refused unless it replays the same idempotency
        key — a double posting is an ERP-side correction, not a
        second receipt. A failed receipt may be superseded (the
        retry's outcome replaces it).
        """
        store = self._get_store()
        record = store.get(invoice_id)
        if record is None:
            raise KeyError(f"Unknown invoice_id: {invoice_id} (analyze it first)")
        if status not in ("posted", "failed"):
            raise ValueError(
                f"Unknown posting outcome: {status!r} — a receipt is "
                "'posted' or 'failed'"
            )
        result = record.result
        if result.status != "approved":
            raise ValueError(
                f"Invoice {invoice_id} is {result.status} — only an "
                "approved invoice can be posted, so only an approved "
                "invoice can carry a posting receipt"
            )
        if status == "posted" and not external_reference.strip():
            raise ValueError(
                "a 'posted' receipt must carry the ERP's external "
                "reference (its document number) — that reference is "
                "what closes the loop"
            )
        if status == "failed" and not detail.strip():
            raise ValueError(
                "a 'failed' receipt must carry the ERP's reason — a "
                "failure without a reason closes nothing"
            )
        existing = result.posting_receipt
        if existing is not None and existing.status == "posted":
            if (
                idempotency_key
                and existing.idempotency_key
                and idempotency_key == existing.idempotency_key
            ):
                return result  # the same posting reported twice: one receipt
            raise ValueError(
                f"Invoice {invoice_id} is already recorded as posted "
                f"(ERP reference {existing.external_reference}) — a "
                "second, different posting receipt would double-post "
                "it; correct the posting in the ERP, not here"
            )
        result.posting_receipt = PostingReceipt(
            invoice_id=invoice_id,
            status=status,
            external_reference=external_reference,
            idempotency_key=idempotency_key,
            detail=detail,
            recorded_by=actor,
            recorded_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        )
        store.save(record)
        return result

    def get(self, invoice_id: str) -> AgentResult | None:
        record = self._get_store().get(invoice_id)
        return record.result if record else None

    def approved_not_posted_report(
        self, as_of: str = "", grace_days: int | None = None
    ) -> ApprovedNotPostedReport:
        """The ERP loop's open items: approved invoices with no
        posted receipt after the grace window
        (``POSTING_RECEIPT_GRACE_DAYS``, default 7; ``grace_days``
        overrides for one call). A pure read over the store — it
        posts and chases nothing; see ``posting.py``."""
        from .posting import approved_not_posted

        anchor = _parse_date(as_of) if as_of else None
        if as_of and anchor is None:
            raise ValueError(f"as_of {as_of!r} is not an ISO date")
        anchor = anchor or datetime.now(timezone.utc).date()
        return approved_not_posted(
            self._get_store().list_records(),
            as_of=anchor,
            grace_days=(
                posting_receipt_grace_days() if grace_days is None else grace_days
            ),
        )

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
        from .posting import approved_not_posted
        from .supplier_master import master_coverage

        coverage = master_coverage(
            records,
            self._get_supplier_master(),
            pending_changes=sum(
                1
                for change in self._get_store().list_master_changes()
                if change.status == "pending"
            ),
        )
        packs = self._get_store().list_recovery_packs()
        sla_days = recovery_case_sla_days()
        case_items, case_buckets, case_breaches = recovery_case_aging(
            packs,
            as_of=datetime.now(timezone.utc).date(),
            sla_days=sla_days,
        )
        cases_by_status: dict[str, int] = {}
        for pack in packs:
            cases_by_status[pack.case_status] = (
                cases_by_status.get(pack.case_status, 0) + 1
            )
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
            "recovery_cases": {
                "total": len(packs),
                "open": len(case_items),
                "by_status": cases_by_status,
                "sla_days": sla_days,
                "sla_breaches": case_breaches,
                "age_buckets": case_buckets,
            },
            "supplier_master": coverage,
            "posting_open_items": {
                "count": self.approved_not_posted_report().count,
                "grace_days": posting_receipt_grace_days(),
                "escalated_count": len(
                    approved_not_posted(
                        records,
                        as_of=datetime.now(timezone.utc).date(),
                        grace_days=max(
                            posting_escalation_days(),
                            posting_receipt_grace_days(),
                        ),
                    ).items
                ),
                "escalation_days": max(
                    posting_escalation_days(), posting_receipt_grace_days()
                ),
                # posting escalation webhooks recorded on the
                # invoice records (the sweep's ledger): pushes
                # fired, as opposed to items merely past the line —
                # split by ladder rung, so the urgent second shouts
                # are countable apart from the first crossings.
                "escalations_recorded": sum(
                    len(r.result.posting_escalations) for r in records
                ),
                "escalations_recorded_urgent": sum(
                    1
                    for r in records
                    for entry in r.result.posting_escalations
                    if entry.level >= 2
                ),
            },
            "payment_runs": handoff_metrics(self._get_store().list_payment_runs()),
            "handoff_profiles": _handoff_profile_summary(
                self._get_store().list_handoff_profiles()
            ),
            # The priced four-eyes wait against its monthly
            # budgets (FOUR_EYES_WAIT_BUDGETS): this month's
            # destroyed discounts per currency, the ceiling in
            # force, and which currencies are over it — the
            # control's cost, owned.
            "four_eyes_wait_budget": wait_cost_budget_status(
                self._get_store().list_payment_runs(),
                four_eyes_wait_budgets(),
                month=datetime.now(timezone.utc).strftime("%Y-%m"),
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
                    "posting_status": (
                        result.posting_receipt.status
                        if result.posting_receipt
                        else ""
                    ),
                    "posting_reference": (
                        result.posting_receipt.external_reference
                        if result.posting_receipt
                        else ""
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
            four_eyes_threshold=payment_run_four_eyes_threshold(),
            four_eyes_thresholds=payment_run_four_eyes_thresholds(),
        )
        stamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
        # Hand-off profile flags, stamped at proposal: a member
        # whose supplier's bank rejects at-import markers while
        # the master holds no approved bank details for them
        # would export a file its bank refuses — flag it now,
        # while onboarding is cheap (handoff_profiles.py). The
        # master check is the completion's own has-details rule.
        from .handoff_profiles import marker_flags as _marker_flags
        from .supplier_master import _iban_norm

        profiles = store.list_handoff_profiles()
        master_covered: set[str] = set()
        if profiles:
            master = self._get_supplier_master()
            for name in {
                line.supplier_name for run in runs for line in run.lines
            }:
                record = master.get_supplier(name) if master else None
                if record is not None and (
                    (record.iban and _iban_norm(record.iban))
                    or (record.account_number or "").strip()
                ):
                    master_covered.add(name)
        saved: list[PaymentRun] = []
        for run in runs:
            if run.run_id in confirmed_ids:
                continue  # a confirmed run is the record; never refreshed
            run.proposed_at = stamp
            run.handoff_profile_flags = _marker_flags(
                run, profiles, master_covered=master_covered
            )
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

    def transition_recovery_case(
        self, invoice_id: str, to_status: str, actor: str, note: str = ""
    ) -> RecoveryPack:
        """Walk a recovery case one step forward in its lifecycle
        (opened → supplier_contacted → credit_received → closed).

        Operator bookkeeping on the pack record only: the transition
        and its actor/note are appended to the case's audit history.
        The agent still recovers nothing itself — the operator is
        recording work they did outside the system (sent the letter,
        received the credit). A blocked pack (guardrails failed) has
        no workable case: its letter may not be sent, so it cannot
        be walked.
        """
        from .schemas import RECOVERY_CASE_FLOW, RecoveryCaseEvent

        store = self._get_store()
        pack = store.get_recovery_pack(invoice_id)
        if pack is None:
            raise KeyError(
                f"No recovery pack for invoice {invoice_id} "
                "(draft it first)"
            )
        if pack.status != "ready":
            raise ValueError(
                f"Recovery case for {invoice_id} is blocked — the "
                "pack's drafts failed the guardrails, so there is no "
                "letter to work until the pack is redrafted clean"
            )
        current = pack.case_status
        if to_status not in RECOVERY_CASE_FLOW:
            raise ValueError(
                f"Unknown case status: {to_status!r} — the lifecycle is "
                + " → ".join(RECOVERY_CASE_FLOW)
            )
        expected = (
            RECOVERY_CASE_FLOW[RECOVERY_CASE_FLOW.index(current) + 1]
            if current in RECOVERY_CASE_FLOW
            and RECOVERY_CASE_FLOW.index(current) + 1 < len(RECOVERY_CASE_FLOW)
            else None
        )
        if to_status != expected:
            raise ValueError(
                f"Recovery case for {invoice_id} is {current} — the "
                f"next step is {expected or 'none (the case is closed)'}, "
                f"not {to_status}"
            )
        pack.case_status = to_status
        pack.case_history.append(
            RecoveryCaseEvent(
                status=to_status,
                actor=actor,
                at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
                note=note,
            )
        )
        store.save_recovery_pack(pack)
        return pack

    def escalate_breached_recovery_cases(
        self, as_of: str = ""
    ) -> list[RecoveryPack]:
        """Sweep the open recovery cases and escalate the breached.

        The queue has always *shown* a case whose current status
        outlasted the SLA (``RECOVERY_CASE_SLA_DAYS``); this is the
        push counterpart — for each breached case, fire the signed
        ``recovery_escalation`` webhook (the opt-in
        ``ACTION_WEBHOOK_URL`` seam) with the case's aging snapshot,
        so the customer's paging/ticketing system hears about money
        stuck in recovery without anyone watching the queue.

        One escalation per crossing: the dispatch is recorded in
        the pack's own escalation ledger, deduped by (case status,
        the date that status began) — re-running the sweep does not
        re-fire, and a case that moves and stalls in a later status
        escalates again for that crossing. The ledger lives outside
        ``case_history`` because the status-age clock runs from the
        last history event. With no webhook configured the sweep is
        a no-op that records nothing — there is nothing to dedupe
        against a dispatch that never happened. Returns the packs
        escalated by this sweep.
        """
        store = self._get_store()
        anchor = _parse_date(as_of) if as_of else None
        if as_of and anchor is None:
            raise ValueError(f"as_of {as_of!r} is not an ISO date")
        anchor = anchor or datetime.now(timezone.utc).date()
        sla_days = recovery_case_sla_days()
        packs = store.list_recovery_packs()
        items, _, _ = recovery_case_aging(packs, as_of=anchor, sla_days=sla_days)
        by_id = {pack.pack_id: pack for pack in packs}
        escalated: list[RecoveryPack] = []
        for item in items:
            if not item.sla_breached:
                continue
            pack = by_id.get(item.pack_id)
            if pack is None:
                continue
            last_at = (
                pack.case_history[-1].at if pack.case_history else pack.created_at
            )
            since = _parse_date(last_at)
            status_since = since.isoformat() if since is not None else ""
            if any(
                entry.case_status == pack.case_status
                and entry.status_since == status_since
                for entry in pack.escalations
            ):
                continue  # this crossing already escalated — once is the rule
            outcome = dispatch_recovery_escalation_webhook(
                pack,
                sla_days=sla_days,
                status_age_days=item.status_age_days,
                age_days=item.age_days,
            )
            if outcome is None:
                continue  # webhook off: nothing fired, nothing recorded
            pack.escalations.append(
                RecoveryEscalation(
                    case_status=pack.case_status,
                    status_since=status_since,
                    sla_days=sla_days,
                    status_age_days=item.status_age_days,
                    age_days=item.age_days,
                    dispatched_at=datetime.now(timezone.utc).isoformat(
                        timespec="seconds"
                    ),
                    dispatch_status=outcome,
                )
            )
            store.save_recovery_pack(pack)
            escalated.append(pack)
        return escalated

    def escalate_open_postings(self, as_of: str = "") -> list[OpenPostingItem]:
        """Sweep the open postings and escalate the crossed.

        The queue has always *shown* an approved invoice still
        unposted past the escalation threshold
        (``POSTING_ESCALATION_DAYS``, never below the receipt
        grace window); this is the push counterpart — for each
        crossed posting, fire the signed escalation webhook (the
        opt-in ``ACTION_WEBHOOK_URL`` seam) with the posting's
        snapshot, so the customer's paging/ticketing system hears
        that the ERP loop is open without anyone watching the
        queue. The recovery cases got this treatment first; an
        unposted approval is the same shape of silence.

        The escalation is a ladder of two rungs, so a posting
        nobody acts on does not shout once and then go quiet
        forever: past the threshold, level 1 fires
        (``posting_escalation``); past *twice* the threshold —
        the first shout unanswered — level 2 fires
        (``posting_escalation_urgent``) in the same shape with
        the rung named. Each rung fires once per episode: the
        dispatch is recorded in the invoice record's own
        escalation ledger (``result.posting_escalations``, beside
        the receipt fields), deduped by (receipt state, episode
        start, level) — an episode runs from the approval while
        no receipt exists, or from the latest failed receipt,
        because a failure is new information worth a ladder of
        its own once its episode crosses. A sweep climbs at most
        one rung per posting — a posting first *seen* past the
        second threshold still hears level 1 first, and level 2
        on the next sweep — so the two events stay distinguishable
        in the receiver's log instead of arriving as a pair. A
        posted receipt closes the item outright: the reset. With
        no webhook configured the sweep is a no-op that records
        nothing — there is nothing to dedupe against a dispatch
        that never happened. Returns the items escalated by this
        sweep, at either rung.
        """
        from .posting import approved_not_posted

        store = self._get_store()
        anchor = _parse_date(as_of) if as_of else None
        if as_of and anchor is None:
            raise ValueError(f"as_of {as_of!r} is not an ISO date")
        anchor = anchor or datetime.now(timezone.utc).date()
        threshold = max(posting_escalation_days(), posting_receipt_grace_days())
        rungs = [(1, threshold), (2, 2 * threshold)]
        report = approved_not_posted(
            store.list_records(), as_of=anchor, grace_days=threshold
        )
        escalated: list[OpenPostingItem] = []
        for item in report.items:
            record = store.get(item.invoice_id)
            if record is None:
                continue
            result = record.result
            receipt = result.posting_receipt
            if receipt is not None:
                since = _parse_date(receipt.recorded_at)
                episode_since = since.isoformat() if since is not None else ""
            else:
                # No receipt: the episode runs from the item's age
                # anchor (decision, else submission/invoice date —
                # posting._anchor's rule), recoverable exactly from
                # the age the report computed.
                episode_since = (
                    anchor - timedelta(days=item.age_days or 0)
                ).isoformat()
            fired = {
                (entry.receipt_state, entry.episode_since, entry.level)
                for entry in result.posting_escalations
            }
            age = item.age_days or 0
            due = next(
                (
                    (level, rung_days)
                    for level, rung_days in rungs
                    if age > rung_days
                    and (item.receipt_state, episode_since, level) not in fired
                ),
                None,
            )
            if due is None:
                continue  # every due rung already fired — once per rung is the rule
            level, rung_days = due
            outcome = dispatch_posting_escalation_webhook(
                item, escalation_days=rung_days, level=level
            )
            if outcome is None:
                continue  # webhook off: nothing fired, nothing recorded
            result.posting_escalations.append(
                PostingEscalation(
                    receipt_state=item.receipt_state,
                    episode_since=episode_since,
                    escalation_days=rung_days,
                    level=level,
                    age_days=item.age_days,
                    dispatched_at=datetime.now(timezone.utc).isoformat(
                        timespec="seconds"
                    ),
                    dispatch_status=outcome,
                )
            )
            store.save(record)
            escalated.append(item)
        return escalated

    # --- Governed supplier-master changes ---------------------------

    def propose_master_change(
        self,
        proposed: SupplierMasterRecord,
        *,
        actor: str,
        source: str = "operator",
        source_invoice_id: str = "",
        reason: str = "",
        effective_from: str = "",
    ) -> MasterChangeRequest:
        """Propose a change to a supplier's master record.

        The proposal changes nothing the fraud screen reads: the
        request is stored as ``pending``, snapshotting the record it
        was made against, and the master keeps returning the old
        record — so documents stating the new details keep reading
        as a mismatch — until a second operator approves the change
        and its effective date arrives.
        """
        if source not in ("operator", "document"):
            raise ValueError(
                f"Unknown change source: {source!r} — a change is "
                "proposed by an operator or from a document"
            )
        if source == "document" and not source_invoice_id:
            raise ValueError(
                "a document-sourced change must name the invoice it "
                "came from (source_invoice_id)"
            )
        if not proposed.supplier_name.strip():
            raise ValueError("the proposed record names no supplier")
        if not (proposed.account_number or proposed.iban or proposed.sort_code):
            raise ValueError(
                "a master change must state at least one bank "
                "identifier (account number, sort code, or IBAN)"
            )
        if effective_from and _parse_date(effective_from) is None:
            raise ValueError(
                f"effective_from {effective_from!r} is not an ISO date"
            )
        store = self._get_store()
        current = self._get_supplier_master().get_supplier(proposed.supplier_name)
        stamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
        change = MasterChangeRequest(
            request_id=next_request_id(
                [c.request_id for c in store.list_master_changes()]
            ),
            supplier_name=proposed.supplier_name,
            proposed=proposed,
            current=current,
            source=source,
            source_invoice_id=source_invoice_id,
            reason=reason,
            proposed_by=actor,
            proposed_at=stamp,
            effective_from=effective_from,
            history=[
                MasterChangeEvent(
                    status="proposed", actor=actor, at=stamp, note=reason
                )
            ],
        )
        store.save_master_change(change)
        return change

    def propose_master_change_from_invoice(
        self, invoice_id: str, *, actor: str, reason: str = ""
    ) -> MasterChangeRequest:
        """Propose a master change from a processed invoice: the bank
        details its documents state become the proposal (overlaid on
        the current record by ``master_changes.proposed_from_invoice``).

        This is the governed path for the fraud case itself — a
        supplier's "our bank has changed" note becomes a pending
        request for out-of-band verification, not an update."""
        store = self._get_store()
        record = store.get(invoice_id)
        if record is None:
            raise KeyError(f"Unknown invoice_id: {invoice_id} (analyze it first)")
        invoice = InvoiceInput.model_validate(record.invoice or {})
        current = self._get_supplier_master().get_supplier(invoice.supplier_name)
        proposed, notes = proposed_from_invoice(invoice, current)
        full_reason = reason or "bank details stated in the supplier's document"
        if notes:
            full_reason = f"{full_reason} — {'; '.join(notes)}"
        return self.propose_master_change(
            proposed,
            actor=actor,
            source="document",
            source_invoice_id=invoice_id,
            reason=full_reason,
        )

    def list_master_changes(self, status: str = "") -> list[MasterChangeRequest]:
        changes = self._get_store().list_master_changes()
        return [c for c in changes if not status or c.status == status]

    def get_master_change(self, request_id: str) -> MasterChangeRequest | None:
        return self._get_store().get_master_change(request_id)

    def decide_master_change(
        self,
        request_id: str,
        *,
        actor: str,
        approve: bool,
        verification_ref: str = "",
        note: str = "",
    ) -> MasterChangeRequest:
        """Record the second operator's decision on a master change.

        Four-eyes is enforced here: the proposer cannot decide their
        own change. Approval additionally requires the out-of-band
        verification reference — the document (or colleague) asking
        for new bank details is never itself the evidence. An
        approved change applies immediately when its effective date
        has arrived; a future-dated one waits for
        :meth:`apply_due_master_changes`. Rejection leaves the
        master exactly as it was. A request that sat pending past
        the expiry window (``MASTER_CHANGE_EXPIRY_DAYS``) is expired
        by the attempt itself and refused — it must be re-proposed.
        """
        store = self._get_store()
        change = store.get_master_change(request_id)
        if change is None:
            raise KeyError(
                f"Unknown master change: {request_id} (propose it first)"
            )
        if change.status == "pending" and is_change_expired(
            change,
            datetime.now(timezone.utc).date(),
            master_change_expiry_days(),
        ):
            # The apply-time check: a decision attempted after the
            # window expires the request on the spot (audited) and
            # is refused — the change must be re-proposed, which
            # restarts the governed path with fresh eyes.
            self._expire_master_change(change)
            raise ValueError(
                f"Master change {request_id} sat pending past the "
                f"{master_change_expiry_days()}-day decision window "
                "and has expired — it can no longer be decided; "
                "re-propose the change to restart it"
            )
        if change.status != "pending":
            detail = (
                f" (decided by {change.decided_by})"
                if change.decided_by
                else " — it can no longer be decided; re-propose to restart it"
            )
            raise ValueError(
                f"Master change {request_id} is already {change.status}{detail}"
            )
        if actor == change.proposed_by:
            raise ValueError(
                "four-eyes on master data: "
                f"{change.proposed_by} proposed this change and cannot "
                "also decide it — a second operator must approve or "
                "reject a bank-detail change"
            )
        if approve and not verification_ref.strip():
            raise ValueError(
                "approving a bank-detail change requires the out-of-band "
                "verification reference (e.g. 'call-back to the number "
                "on file, confirmed with the supplier's AP contact') — "
                "the document that asked for the change is not evidence"
            )
        now = datetime.now(timezone.utc)
        stamp = now.isoformat(timespec="seconds")
        change.decided_by = actor
        change.decided_at = stamp
        change.decision_note = note
        if not approve:
            change.status = "rejected"
            change.history.append(
                MasterChangeEvent(
                    status="rejected", actor=actor, at=stamp, note=note
                )
            )
            store.save_master_change(change)
            return change
        change.status = "approved"
        change.verification_ref = verification_ref
        if not change.effective_from:
            change.effective_from = now.date().isoformat()
        change.history.append(
            MasterChangeEvent(
                status="approved",
                actor=actor,
                at=stamp,
                note=note or f"verified out-of-band: {verification_ref}",
            )
        )
        store.save_master_change(change)
        effective = _parse_date(change.effective_from)
        if effective is not None and effective <= now.date():
            self._apply_master_change(
                change,
                actor=actor,
                note="applied on approval — the effective date has arrived",
            )
        return change

    def _expire_master_change(
        self, change: MasterChangeRequest, *, as_of=None
    ) -> MasterChangeRequest:
        """Close a pending change that outlived its decision window.

        Expiry is an audit event, not a decision: the request keeps
        its proposer, its proposal, and its history, gains an
        ``expired`` entry, and can never be decided — the change
        must be re-proposed, restarting the governed path. The
        master record is untouched, exactly as a rejection leaves
        it."""
        window = master_change_expiry_days()
        anchor = as_of or datetime.now(timezone.utc).date()
        age = change_age_days(change, anchor)
        stamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
        change.status = "expired"
        change.history.append(
            MasterChangeEvent(
                status="expired",
                actor="system",
                at=stamp,
                note=(
                    f"pending {age} days without a decision — expired "
                    f"after the {window}-day window "
                    "(MASTER_CHANGE_EXPIRY_DAYS); re-propose the "
                    "change to restart it"
                ),
            )
        )
        self._get_store().save_master_change(change)
        return change

    def expire_stale_master_changes(self, as_of: str = "") -> list[MasterChangeRequest]:
        """Expire every pending master change past its decision
        window (expiry sweep). Called at the top of each analysis,
        so a stale request never lingers decidable, and exposed as
        an endpoint for operators. Returns the changes this call
        expired."""
        anchor = (
            _parse_date(as_of)
            if as_of
            else datetime.now(timezone.utc).date()
        )
        if anchor is None:
            raise ValueError(f"as_of {as_of!r} is not an ISO date")
        window = master_change_expiry_days()
        expired: list[MasterChangeRequest] = []
        for change in self._get_store().list_master_changes():
            if is_change_expired(change, anchor, window):
                expired.append(self._expire_master_change(change, as_of=anchor))
        return expired

    def _apply_master_change(
        self, change: MasterChangeRequest, *, actor: str, note: str
    ) -> MasterChangeRequest:
        """Land an approved change in the supplier master and record
        the application on the request's audit history. This is the
        only write path from the change workflow into the master."""
        self._get_supplier_master().upsert_supplier(change.proposed)
        stamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
        change.applied_at = stamp
        change.history.append(
            MasterChangeEvent(status="applied", actor=actor, at=stamp, note=note)
        )
        self._get_store().save_master_change(change)
        return change

    def apply_due_master_changes(self, as_of: str = "") -> list[MasterChangeRequest]:
        """Apply every approved master change whose effective date
        has arrived (promotion sweep). Called at the top of each
        analysis, so screening always compares against the record in
        force today, and exposed as an endpoint for operators.
        Returns the changes applied by this call."""
        anchor = (
            _parse_date(as_of)
            if as_of
            else datetime.now(timezone.utc).date()
        )
        if anchor is None:
            raise ValueError(f"as_of {as_of!r} is not an ISO date")
        applied: list[MasterChangeRequest] = []
        for change in self._get_store().list_master_changes():
            if change.status != "approved" or change.applied_at:
                continue
            effective = _parse_date(change.effective_from)
            if effective is None or effective > anchor:
                continue
            applied.append(
                self._apply_master_change(
                    change,
                    actor="system",
                    note=(
                        f"effective date {change.effective_from} reached — "
                        "applied to the supplier master"
                    ),
                )
            )
        return applied

    def queue_view(self, as_of: str = "") -> QueueView:
        """The AP aging queue: invoices still at the gate, aged by
        due date, overdue first, with the triage mix — plus the open
        recovery cases with their per-status aging and SLA flags,
        the open postings that have outlasted the escalation
        threshold (``POSTING_ESCALATION_DAYS``), and the four-eyes
        payment runs whose second confirmation has outlasted its
        aging threshold (``PAYMENT_RUN_FOUR_EYES_AGING_DAYS``), so
        one worklist shows everything the building still owes an
        answer for. A pure read over the store — it decides and
        writes nothing."""
        from .posting import approved_not_posted

        anchor = _parse_date(as_of) if as_of else None
        store = self._get_store()
        records = store.list_records()
        escalation = max(posting_escalation_days(), posting_receipt_grace_days())
        escalated = approved_not_posted(
            records,
            as_of=anchor or datetime.now(timezone.utc).date(),
            grace_days=escalation,
        )
        return build_queue(
            records,
            as_of=anchor,
            recovery_packs=store.list_recovery_packs(),
            recovery_sla_days=recovery_case_sla_days(),
            open_postings=escalated.items,
            posting_escalation_days=escalation,
            payment_runs=store.list_payment_runs(),
            four_eyes_aging_days=four_eyes_aging_days(),
        )

    def escalation_digest(self, as_of: str = "") -> EscalationDigest:
        """The day's escalation digest: the queue's action items
        (recovery SLA breaches, escalated open postings, overdue
        invoices, four-eyes stalls) plus the day's payment-run news
        (forecast lapses on runs confirmed today, drift realised on
        runs exported today), composed into one operator summary
        by ``digest.build_digest``. Computed at read time from the
        stored records and sent nowhere — the push counterpart to
        the queue's pull, left for a human to read or forward."""
        from .digest import build_digest
        from .schemas import DigestWaitBudgetEntry

        anchor = _parse_date(as_of) if as_of else None
        if as_of and anchor is None:
            raise ValueError(f"as_of {as_of!r} is not an ISO date")
        anchor = anchor or datetime.now(timezone.utc).date()
        runs = self._get_store().list_payment_runs()
        budget_status = wait_cost_budget_status(
            runs,
            four_eyes_wait_budgets(),
            month=anchor.strftime("%Y-%m"),
        )
        breaches = [
            DigestWaitBudgetEntry(
                currency=row["currency"],
                month=budget_status["month"],
                wait_cost=row["wait_cost"],
                budget=row["budget"],
                excess=row["excess"],
            )
            for row in budget_status["currencies"]
            if row["over_budget"]
        ]
        return build_digest(
            as_of=anchor,
            queue=self.queue_view(as_of=anchor.isoformat()),
            runs=runs,
            wait_budget_breaches=breaches,
        )

    def store_digest(self, as_of: str = "", actor: str = "") -> EscalationDigest:
        """Store a day's digest: the audit-file copy.

        The computed digest describes the records as they are at
        read time, so yesterday's digest cannot be re-derived once
        the records move — a case transitions, a posting closes,
        and the day re-composes differently. Storing is an
        operator action that pins the day's first-computed digest
        in the store (migration 0010), stamped with who stored it
        and when, provenance ``stored``.

        First-computed wins: when a copy is already stored for the
        date, it is returned unchanged — a later store attempt
        never overwrites the copy a human may already have
        forwarded. Recomputation stays available through
        :meth:`escalation_digest`, labelled ``computed``.
        """
        store = self._get_store()
        anchor = _parse_date(as_of) if as_of else None
        if as_of and anchor is None:
            raise ValueError(f"as_of {as_of!r} is not an ISO date")
        anchor = anchor or datetime.now(timezone.utc).date()
        existing = store.get_digest(anchor.isoformat())
        if existing is not None:
            return existing  # the first-computed copy is the record
        digest = self.escalation_digest(as_of=anchor.isoformat())
        digest.provenance = "stored"
        digest.stored_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
        digest.stored_by = actor
        store.save_digest(digest)
        return digest

    def get_stored_digest(self, digest_date: str) -> EscalationDigest | None:
        """Read back the stored digest for a date, verbatim — or
        None when nobody stored that day. The returned digest is
        the copy as stored (provenance ``stored``); it is never a
        recomputation wearing the label."""
        anchor = _parse_date(digest_date) if digest_date else None
        if anchor is None:
            raise ValueError(f"digest_date {digest_date!r} is not an ISO date")
        return self._get_store().get_digest(anchor.isoformat())

    def digest_history(self) -> DigestHistory:
        """The stored digests as a day-over-day series: headline
        counts per stored day, each with its changes against the
        previous stored day (see ``digest.build_digest_history``).
        Days nobody stored are absent, not interpolated."""
        from .digest import build_digest_history

        return build_digest_history(self._get_store().list_digests())

    def store_due_digests(
        self, lookback_days: int | None = None, as_of: str = ""
    ) -> DigestStoreSweep:
        """Sweep: pin the digest for every recent day nobody
        stored, over a bounded lookback ending at ``as_of``
        (default today).

        The digest history is only as good as the mornings
        somebody remembered to press Store — a holiday, a sick
        day, a busy Monday, and the trend grows a hole exactly
        where an auditor will look. This sweep is the backfill:
        for each day in the window (oldest first) with no stored
        digest, it stores that day's first-computed digest
        through the same :meth:`store_digest` path a human uses
        — first-computed still wins, so a day a human already
        stored is reported ``already_stored`` and its copy and
        attribution are never touched.

        Sweep-stored days are attributed to ``system`` in the
        stored copy's own provenance: the backfill is honest
        about being a backfill — a day pinned by the sweep,
        computed from the records as they stand now and aged to
        its own date, never masquerades as a digest somebody
        composed that morning. The lookback is
        ``DIGEST_STORE_LOOKBACK_DAYS`` (default 7) unless the
        caller passes one; it is bounded on purpose, so wiring
        the sweep to a scheduler cannot rewrite ancient history.
        Returns the per-day report: stored vs already stored.
        """
        store = self._get_store()
        anchor = _parse_date(as_of) if as_of else None
        if as_of and anchor is None:
            raise ValueError(f"as_of {as_of!r} is not an ISO date")
        anchor = anchor or datetime.now(timezone.utc).date()
        window = (
            digest_store_lookback_days()
            if lookback_days is None
            else max(1, int(lookback_days))
        )
        entries: list[DigestStoreSweepEntry] = []
        for offset in range(window - 1, -1, -1):
            day = (anchor - timedelta(days=offset)).isoformat()
            existing = store.get_digest(day)
            if existing is not None:
                entries.append(
                    DigestStoreSweepEntry(
                        date=day,
                        outcome="already_stored",
                        stored_by=existing.stored_by,
                    )
                )
                continue
            digest = self.store_digest(as_of=day, actor="system")
            entries.append(
                DigestStoreSweepEntry(
                    date=day, outcome="stored", stored_by=digest.stored_by
                )
            )
        return DigestStoreSweep(
            through_date=anchor.isoformat(),
            lookback_days=window,
            stored_count=sum(1 for e in entries if e.outcome == "stored"),
            already_stored_count=sum(
                1 for e in entries if e.outcome == "already_stored"
            ),
            entries=entries,
        )

    def marker_worklist(self) -> MarkerWorklist:
        """The supplier-onboarding worklist: suppliers ranked by
        the export value that still carried the at-import marker,
        from the stored runs' completion receipts (see
        ``worklist.marker_worklist``).

        A supplier drops off the list when the master now holds
        approved bank details for them — checked here against the
        live master with the same has-details rule the export
        completion uses (a record exists AND carries an IBAN or an
        account number): onboarding is the governed master-change
        path, and the moment it lands, the marked exports become
        history rather than work. Listed rows are joined with the
        store's master-change requests, so a supplier already
        being onboarded reads as in flight (its live change named,
        with its age — not open work to chase a second time) and
        a supplier whose change was rejected wears the rejection.
        A pure read — it onboards nobody."""
        from .supplier_master import _iban_norm
        from .worklist import marker_worklist as _marker_worklist

        store = self._get_store()
        runs = store.list_payment_runs()
        master = self._get_supplier_master()
        candidates = {
            entry.supplier_name
            for run in runs
            for entry in run.handoff_creditor_entries
            if entry.source != "supplier_master" and entry.supplier_name
        }
        onboarded = set()
        for name in candidates:
            record = master.get_supplier(name) if master else None
            if record is not None and (
                (record.iban and _iban_norm(record.iban))
                or (record.account_number or "").strip()
            ):
                onboarded.add(name)
        return _marker_worklist(
            runs,
            onboarded_suppliers=onboarded,
            changes=store.list_master_changes(),
            as_of=datetime.now(timezone.utc).date(),
            expiry_days=master_change_expiry_days(),
        )

    def set_handoff_profile(
        self,
        *,
        actor: str,
        scope: str = "currency",
        supplier_name: str = "",
        currency: str = "",
        format: str = "csv",
        lead_days: int = 0,
        markers_rejected: bool = False,
        note: str = "",
    ) -> HandoffProfile:
        """Set (create or update) a governed hand-off routing
        profile — the payee side's say in how its runs are
        exported (see ``handoff_profiles.py``).

        Governed means attributable: the set names its actor,
        and the profile record keeps an append-only audit trail
        of every set with the values it carried. One profile
        exists per scope key (supplier name or currency code);
        setting it again updates the same record. The profile
        changes no run by itself — it is read at proposal time
        (marker-rejection flags) and at export time (format
        resolution and the pain.001 lead days)."""
        from .handoff import HANDOFF_FORMATS
        from .handoff_profiles import profile_detail, profile_id_for

        if not (actor or "").strip():
            raise ValueError(
                "Provide 'actor' — a hand-off profile is governed "
                "configuration; an unattributed set is refused"
            )
        if scope not in ("supplier", "currency"):
            raise ValueError(
                f"Unknown profile scope {scope!r} — expected 'supplier' "
                "or 'currency'"
            )
        if format not in HANDOFF_FORMATS:
            raise ValueError(
                f"Unknown hand-off format {format!r} — expected one of "
                f"{HANDOFF_FORMATS}"
            )
        lead_days = int(lead_days)
        if lead_days < 0 or lead_days > 30:
            raise ValueError(
                f"lead_days {lead_days} is outside 0–30 — a routing "
                "profile buys processing days, not a month of float"
            )
        supplier_name = (supplier_name or "").strip()
        currency = (currency or "").strip().upper()
        if scope == "supplier":
            if not supplier_name:
                raise ValueError(
                    "A supplier-scoped profile needs a supplier_name"
                )
            currency = ""
        else:
            if len(currency) != 3 or not currency.isalpha():
                raise ValueError(
                    f"A currency-scoped profile needs a 3-letter "
                    f"currency code, got {currency!r}"
                )
            supplier_name = ""
        profile_id = profile_id_for(
            scope, supplier_name=supplier_name, currency=currency
        )
        store = self._get_store()
        existing = store.get_handoff_profile(profile_id)
        stamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
        profile = HandoffProfile(
            profile_id=profile_id,
            scope=scope,
            supplier_name=supplier_name,
            currency=currency,
            format=format,
            lead_days=lead_days,
            markers_rejected=bool(markers_rejected),
            note=(note or "").strip(),
            updated_by=actor,
            updated_at=stamp,
            history=list(existing.history) if existing else [],
        )
        profile.history.append(
            HandoffProfileEvent(
                at=stamp,
                actor=actor,
                action="updated" if existing else "created",
                detail=profile_detail(profile),
            )
        )
        store.save_handoff_profile(profile)
        return profile

    def list_handoff_profiles(self) -> list[HandoffProfile]:
        """The governed hand-off routing profiles in force."""
        return self._get_store().list_handoff_profiles()

    def get_handoff_profile(self, profile_id: str) -> HandoffProfile | None:
        return self._get_store().get_handoff_profile(profile_id)

    def get_payment_run(self, run_id: str) -> PaymentRun | None:
        return self._get_store().get_payment_run(run_id)

    def confirm_payment_run(self, run_id: str, actor: str) -> PaymentRun:
        """Record a finance lead's confirmation of a proposed run.

        Confirmation touches the run record only: no invoice changes
        state and no payment is executed — the confirmed run is the
        pay list the customer's AP system of record acts on.

        Confirmation is also when the drift forecast is computed
        (``paymentruns.drift_forecast``): each discounted member's
        deadline is set against the run's expected export date —
        its proposal date plus the measured mean proposal→export
        lag over the runs already exported — and the members
        projected to lapse are flagged on the run, with the basis
        stated. A forecast, labelled as one: it warns the
        confirmer while they can still hurry the export; it
        changes no figures, and the export-time drift check
        remains the authority on what actually lapsed.
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
        lag_stats = handoff_metrics(store.list_payment_runs())["export_lag"][
            "proposal_to_export"
        ]
        forecast, basis = _drift_forecast(
            run,
            as_of=datetime.now(timezone.utc).date(),
            lag_stats=lag_stats,
        )
        run.drift_forecast = forecast
        run.drift_forecast_basis = basis
        if forecast:
            flagged = "; ".join(
                f"{entry.invoice_id} ({run.currency} "
                f"{entry.discount_amount:.2f}, deadline "
                f"{entry.discount_deadline})"
                for entry in forecast
            )
            run.note += (
                " Drift forecast at confirmation — a projection "
                "from the measured proposal→export lag, not a "
                f"measurement: {flagged} projected to lapse before "
                "the expected export date; see drift_forecast."
            )
        store.save_payment_run(run)
        return run

    def confirm_payment_run_second(self, run_id: str, actor: str) -> PaymentRun:
        """Record the second confirmation a large run requires.

        Four-eyes: the second confirmer must be a different person
        from the first. Only a run stamped ``four_eyes_required``
        (net total above its currency's four-eyes threshold when
        proposed) takes a second confirmation, and only after the
        first — the confirmations are a pair, in order, both named
        on the run record. Like the first confirmation, this pays
        nothing; it unlocks the hand-off export. Landing the
        second confirmation also closes the wait, so the run is
        priced for it: the discounts capturable at the first
        confirmation that lapsed during the wait are stamped on
        the run as its four-eyes wait cost.
        """
        store = self._get_store()
        run = store.get_payment_run(run_id)
        if run is None:
            raise KeyError(f"Unknown run_id: {run_id} (propose it first)")
        if not run.four_eyes_required:
            raise ValueError(
                f"Payment run {run_id} totals {run.currency} "
                f"{run.total_net:.2f} — not above the four-eyes "
                "threshold"
                + (
                    f" for {run.currency} runs ({run.currency} "
                    f"{run.four_eyes_threshold:,.2f})"
                    if run.four_eyes_threshold
                    else ""
                )
                + ", so no second confirmation is required"
            )
        if run.status != "confirmed":
            raise ValueError(
                f"Payment run {run_id} is {run.status} — the first "
                "confirmation comes before the second"
            )
        if run.second_confirmed_by:
            raise ValueError(
                f"Payment run {run_id} was already second-confirmed by "
                f"{run.second_confirmed_by}"
            )
        if actor == run.confirmed_by:
            raise ValueError(
                "four-eyes on payment runs: the second confirmer must "
                f"differ from the first confirmer ({run.confirmed_by})"
            )
        run.second_confirmed_by = actor
        run.second_confirmed_at = datetime.now(timezone.utc).isoformat(
            timespec="seconds"
        )
        # The wait is now over — price it before the record is
        # saved: the member discounts that were capturable at the
        # first confirmation and lapsed while the run sat waiting
        # are the control's measured cost, stamped on the run
        # (paymentruns.price_four_eyes_wait) and aggregated in
        # /metrics beside the forecast accuracy. The export still
        # re-prices every member itself; this stamp never changes
        # what gets paid — it prices the delay, nothing else.
        run.four_eyes_wait_cost, run.four_eyes_wait_lapses = _price_four_eyes_wait(
            run
        )
        store.save_payment_run(run)
        return run

    def export_payment_run_handoff(
        self, run_id: str, actor: str, as_of: str = "", format: str = ""
    ) -> PaymentRun:
        """Render a confirmed run's hand-off file and store it
        through the object-store port.

        The file (``handoff.py`` — CSV layout v2, or ISO 20022
        pain.001; the mapping is in ``docs/handoff-formats.md``)
        is the routing adapter to the customer's banking/ERP
        system. Generating it changes no payment state — the run
        stays confirmed, its invoices stay approved; the run
        record gains only the export receipt (object key +
        sha256 of the stored bytes, and the format). Re-exporting
        renders the same bytes and refreshes the receipt;
        exporting in the other format stores that format's file
        alongside and moves the receipt to it.

        The format is resolved, not assumed (``handoff_profiles.
        resolve_export_format``): a format named on the call
        wins; else the governing hand-off profile's format —
        the run's currency profile, or a supplier profile the
        run's profiled members agree on; else the CSV default.
        The governing profile's lead days also set a pain.001
        export's requested execution date. The run record
        stamps which way it went (``handoff_profile_id``,
        ``handoff_execution_date``).

        The gates are the same in either format: the run must be
        confirmed, and a four-eyes run must carry its second
        confirmation. Before rendering, every member's discount is
        re-verified as of the export date (``as_of``, default
        today): a window that closed since confirmation means the
        confirmed net figure is stale, so the member is exported at
        its export-time figure — full gross once the window has
        closed, because the full total is what is owed — and the
        drift is itemised on the run record (``handoff_drift``).
        The export proceeds for the clean members; drift is
        flagged, never silently exported and never a reason to
        short-pay.

        Creditor accounts are completed at the same moment from
        the supplier master: a member whose supplier has approved
        bank details on file exports with them (IBAN where
        recorded); every other member keeps the documented
        at-import marker. Which way each member went is the
        export's completion receipt (``handoff_creditor_entries``,
        masked labels only).
        """
        from .handoff import HANDOFF_FORMATS

        if format and format not in HANDOFF_FORMATS:
            raise ValueError(
                f"Unknown hand-off format {format!r} — expected one of "
                f"{HANDOFF_FORMATS}, or no format to let the run's "
                "hand-off profile decide"
            )
        store = self._get_store()
        run = store.get_payment_run(run_id)
        if run is None:
            raise KeyError(f"Unknown run_id: {run_id} (propose it first)")
        if run.status != "confirmed":
            raise ValueError(
                f"Payment run {run_id} is {run.status} — only a confirmed "
                "run is handed off; confirm it first"
            )
        if run.four_eyes_required and not run.second_confirmed_by:
            raise ValueError(
                f"Payment run {run_id} totals {run.currency} "
                f"{run.total_net:.2f} — above the four-eyes threshold "
                f"for {run.currency} runs"
                + (
                    f" ({run.currency} {run.four_eyes_threshold:,.2f})"
                    if run.four_eyes_threshold
                    else ""
                )
                + f" — and has only been confirmed by {run.confirmed_by}: "
                "a second confirmer, different from the first, must "
                "confirm the run before the hand-off file can be "
                "generated"
            )
        object_store = self.object_store
        if object_store is None:
            from .storage import get_object_store

            object_store = get_object_store()
        if object_store is None:
            raise ValueError(
                "Object storage is not configured (set S3_BUCKET, or "
                "inject an object store): the hand-off file has nowhere "
                "to live"
            )
        from .handoff import (
            creditor_completions,
            handoff_content_type,
            handoff_key,
            render_handoff,
        )

        anchor = _parse_date(as_of) if as_of else None
        if as_of and anchor is None:
            raise ValueError(f"as_of {as_of!r} is not an ISO date")
        anchor = anchor or datetime.now(timezone.utc).date()
        records = {
            record.result.invoice_id: record for record in store.list_records()
        }
        adjusted, drift = discount_drift(run, records, as_of=anchor)
        if drift:
            # The exported figures become the run's figures: the
            # file the bank receives is the run of record. The
            # confirmation itself (who, when, four-eyes) is
            # untouched, and the drift entries preserve what the
            # run was confirmed with.
            run.lines = adjusted.lines
            run.total_discount = adjusted.total_discount
            run.total_net = adjusted.total_net
            run.handoff_drift = drift
            run.handoff_drift_total = round(
                sum(entry.discount_lost for entry in drift), 2
            )
            lost = ", ".join(
                f"{entry.invoice_id} lost {run.currency} "
                f"{entry.discount_lost:.2f} (window closed "
                f"{entry.discount_deadline})"
                for entry in drift
            )
            run.note += (
                f" Discount drift at export ({anchor.isoformat()}): "
                f"{lost} — the drifted member(s) are exported at "
                "their full gross amounts; see handoff_drift."
            )
        # Creditor accounts complete from the supplier master where
        # (and only where) it holds approved bank details for the
        # member's supplier; the rest keep the at-import marker.
        # The completion receipt lands on the run either way.
        completions, creditor_entries = creditor_completions(
            run, self._get_supplier_master()
        )
        run.handoff_creditor_entries = creditor_entries
        # The format resolves through the governed hand-off
        # profiles: an explicit format on the call wins, else
        # the governing profile's, else the CSV default. The
        # governing profile's lead days set a pain.001 export's
        # requested execution date — the payee's processing
        # runway is a routing fact, whoever picked the format.
        from .handoff_profiles import (
            requested_execution_date,
            resolve_export_format,
        )

        resolved_format, profile, basis = resolve_export_format(
            run, store.list_handoff_profiles(), format
        )
        execution_date = (
            requested_execution_date(run, profile, export_date=anchor)
            if resolved_format == "pain001"
            else ""
        )
        data = render_handoff(
            run,
            resolved_format,
            creditor_accounts=completions,
            requested_execution_date=execution_date,
        )
        key = handoff_key(run.run_id, resolved_format)
        object_store.put(
            key, data, content_type=handoff_content_type(resolved_format)
        )
        run.handoff_key = key
        run.handoff_sha256 = hashlib.sha256(data).hexdigest()
        run.handoff_generated_at = datetime.now(timezone.utc).isoformat(
            timespec="seconds"
        )
        run.handoff_generated_by = actor
        run.handoff_format = resolved_format
        run.handoff_profile_id = profile.profile_id if profile else ""
        run.handoff_execution_date = execution_date
        if profile is not None and basis.startswith("profile"):
            run.note += (
                f" Hand-off format {resolved_format} from {basis}"
                + (
                    f"; requested execution date {execution_date} "
                    f"under the profile's {profile.lead_days}d lead."
                    if execution_date and execution_date != run.window_end
                    else "."
                )
            )
        store.save_payment_run(run)
        return run

    def get_payment_run_handoff(self, run_id: str) -> bytes | None:
        """The stored hand-off bytes for a run, or None when no
        hand-off has been generated (or the object is gone)."""
        run = self._get_store().get_payment_run(run_id)
        if run is None or not run.handoff_key:
            return None
        object_store = self.object_store
        if object_store is None:
            from .storage import get_object_store

            object_store = get_object_store()
        if object_store is None:
            return None
        try:
            return object_store.get(run.handoff_key)
        except Exception:
            return None

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
