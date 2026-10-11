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
from .clarify import reanalysis_analytics
from .config import env_float, env_int, env_str, load_dotenv
from .creditnotes import evaluate_credit_note
from .creditnotes import payable_position as _payable_position
from .matching import _parse_date
from .paymentruns import (
    discount_drift,
    grade_lag_statistics,
    handoff_metrics,
    lag_grade_drift_watch,
    lag_statistics,
    wait_cost_budget_status,
)
from .paymentruns import price_four_eyes_wait as _price_four_eyes_wait
from .paymentruns import drift_forecast as _drift_forecast
from .paymentruns import propose_payment_runs as _propose_runs
from .paymentruns import run_member_ids
from .queue import (
    build_queue,
    information_escalation_follow_through,
    recovery_case_aging,
)
from .recovery import build_recovery_pack
from .reconciliation import reconcile_statement as _reconcile
from .statement_intake import extract_statement_document
from .supplier_positions import supplier_positions as _supplier_positions
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
    CreditApplication,
    CreditNoteInput,
    CreditNoteMatch,
    PayablePosition,
    DigestHistory,
    DigestStoreSweep,
    DigestStoreSweepEntry,
    DocumentInput,
    EscalationDigest,
    GoodsReceipt,
    HandoffGroupReceipt,
    HandoffGroupReexport,
    HandoffProfile,
    HandoffProfileEvent,
    InformationEscalation,
    InformationWaitItem,
    MarkerWorklist,
    InvoiceBundleInput,
    InvoiceInput,
    LagSegmentProposal,
    SupplierContactProposal,
    SupplierContactProposalEvent,
    LagSegmentProposalEvent,
    MasterChangeEvent,
    MasterChangeRequest,
    OpenPostingItem,
    PaymentRun,
    PostingEscalation,
    PostingReceipt,
    ProfileChangeProposal,
    ProfileProposalEvent,
    PurchaseOrder,
    QueueView,
    RecoveryEscalation,
    RecoveryPack,
    StatementReconciliation,
    SupplierMasterRecord,
    SupplierPositions,
    SupplierStatement,
)
from .store import ApprovalRecord, default_store, supplier_summary
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
    "duplicate_of",
    "arrival_shape",
    "twin_arrival_shape",
    "twin_cluster",
    "twin_cluster_size",
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


# Default twin-cluster escalation size: the same invoice on file
# this many times stops reading as a resend and starts reading
# as a pattern the approver works as one case (TWIN_CLUSTER_MIN_
# SIZE overrides it; never below 2 — a cluster is at least a pair).
TWIN_CLUSTER_MIN_SIZE_DEFAULT = 3


def twin_cluster_min_size() -> int:
    """The configured twin-cluster escalation size."""
    load_dotenv()
    return max(2, env_int("TWIN_CLUSTER_MIN_SIZE", TWIN_CLUSTER_MIN_SIZE_DEFAULT))


# Default twin-cluster value threshold: a cluster whose members
# total this much in one currency is material double-payment
# exposure even as a pair (TWIN_CLUSTER_VALUE_THRESHOLD overrides
# it). Totals are summed per currency — currencies never mix.
TWIN_CLUSTER_VALUE_THRESHOLD_DEFAULT = 10000.0


def twin_cluster_value_threshold() -> float:
    """The configured twin-cluster combined-value threshold."""
    load_dotenv()
    return max(
        0.0,
        env_float("TWIN_CLUSTER_VALUE_THRESHOLD", TWIN_CLUSTER_VALUE_THRESHOLD_DEFAULT),
    )


# Default recovery-case SLA: a case may sit in one status this many
# days before the queue flags it as stuck (RECOVERY_CASE_SLA_DAYS
# overrides it).
RECOVERY_CASE_SLA_DEFAULT_DAYS = 14


def recovery_case_sla_days() -> int:
    """The configured recovery-case SLA, in days per status."""
    load_dotenv()
    return max(1, env_int("RECOVERY_CASE_SLA_DAYS", RECOVERY_CASE_SLA_DEFAULT_DAYS))


# Default aging threshold for an information wait: an invoice
# blocked on a missing PO / receipt / legible copy waits on
# someone outside the building, and past this many days without
# a complete reply the wait joins the approver queue as work
# (INFORMATION_WAIT_AGING_DAYS overrides it).
INFORMATION_WAIT_AGING_DEFAULT_DAYS = 3


def information_wait_aging_days() -> int:
    """The configured information-wait aging threshold, in days
    since the request (or since a partial reply restarted it)."""
    load_dotenv()
    return max(
        0,
        env_int(
            "INFORMATION_WAIT_AGING_DAYS",
            INFORMATION_WAIT_AGING_DEFAULT_DAYS,
        ),
    )


# Default aging threshold for an unanswered batch group: the
# file went out, the bank answered the other groups, and this
# one has heard nothing. Banks are slower than suppliers, so
# the threshold sits above the information wait's
# (GROUP_RECEIPT_AGING_DAYS overrides it).
GROUP_RECEIPT_AGING_DEFAULT_DAYS = 5


def group_receipt_aging_days() -> int:
    """The configured group-receipt aging threshold, in days
    since the file the group is waiting on went out (the whole
    export, or the group's solo re-export when re-presented)."""
    load_dotenv()
    return max(
        0,
        env_int(
            "GROUP_RECEIPT_AGING_DAYS",
            GROUP_RECEIPT_AGING_DEFAULT_DAYS,
        ),
    )


def information_wait_escalation_days() -> int:
    """The information-wait escalation threshold: twice the aging
    threshold — the ladder's second rung, in the posting
    ladder's idiom (its urgent rung is likewise 2× its first).
    A wait that has been on the worklist for as long again has
    stopped being a list line: the sweep pushes it."""
    return 2 * information_wait_aging_days()


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


# Default warning band for a pending master change's expiry:
# with this many days (or fewer) of its decision window left,
# the worklist flags the change as expiring soon and the digest
# names it — the rescue has to happen while a decision can
# still land (MASTER_CHANGE_EXPIRY_WARNING_DAYS overrides it).
MASTER_CHANGE_EXPIRY_WARNING_DEFAULT_DAYS = 7


def master_change_expiry_warning_days() -> int:
    """The configured expiry warning band for pending master
    changes, in days of window remaining."""
    load_dotenv()
    return max(
        0,
        env_int(
            "MASTER_CHANGE_EXPIRY_WARNING_DAYS",
            MASTER_CHANGE_EXPIRY_WARNING_DEFAULT_DAYS,
        ),
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


def dispatch_information_wait_webhook(
    item: InformationWaitItem,
    *,
    escalation_days: int,
) -> str | None:
    """POST the ``information_wait_escalation`` event for one
    information wait (off by default — the same opt-in seam,
    secret, and signature as the approval webhook).

    The payload is the wait's snapshot at the crossing: which
    invoice is blocked, on whom, what is still missing, how long
    the current episode has run, and where the loop stands (no
    reply yet, or a partial one that restarted the wait). It
    carries no drafts and no bank details — it is a pointer at
    the wait in the queue, for the customer's own paging/
    ticketing system to chase. Returns ``None`` when
    unconfigured, else ``"sent"`` / ``"failed"`` (see
    :func:`_post_signed_webhook`)."""
    payload = {
        "event": "information_wait_escalation",
        "invoice_id": item.invoice_id,
        "supplier_name": item.supplier_name,
        "invoice_number": item.invoice_number,
        "outcome": item.outcome,
        "requested_at": item.requested_at,
        "reply_status": item.reply_status,
        "reply_received_at": item.reply_received_at,
        "still_missing": item.still_missing,
        "waiting_days": item.waiting_days,
        "episode_since": item.episode_since,
        "escalation_days": escalation_days,
        "escalated": True,
    }
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
        arrival_shape: str = "single",
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
        # number + total as a prior analysed invoice. The pipeline
        # computes the verdict itself from the priors handed over
        # below — the screen lives in run_invoice, not here, so no
        # run mode can reach the pipeline with priors and skip it.
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
        )
        self._get_store().save(
            ApprovalRecord(
                result=result,
                invoice=model.model_dump(mode="json"),
                arrival_shape=arrival_shape,
            )
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

    def record_information_reply(
        self,
        invoice_id: str,
        *,
        actor: str,
        supplied_items: list[str],
        note: str = "",
        received_at: str = "",
        purchase_order: PurchaseOrder | None = None,
        goods_receipt: GoodsReceipt | None = None,
        documents: list[DocumentInput] | None = None,
    ) -> AgentResult:
        """Record a reply to the invoice's open information
        request, and analyse it against the gap.

        The request the agent composed asked for computed items
        (a PO, a receipt, a legible copy); the answer arrives
        outside the system and used to live in an inbox, with a
        human re-linking it to the case by memory. Here the
        operator records what arrived — the canonical item keys
        it supplied, when, and a note — and the analysis is
        arithmetic (clarify.analyse_information_reply): the
        supplied items close against the missing list, the
        resolution (answered / partially answered / still
        awaiting) and the still-missing items are named on the
        record, and replies accumulate across answers.

        The reply may also carry the supplied documents
        themselves. They accumulate on the request as its
        overlay (validated by clarify.merge_supplied_facts:
        a payload must be accounted for by a supplied key, fills
        only a gap the case actually has, and a legible copy
        replaces only a document the case holds). When a reply
        closes the gap AND the accumulated overlay carries
        documents AND the case is still at the gate, the case is
        re-analysed — :meth:`_reanalyse_from_reply` — and the
        new outcome supersedes the blocked one. A keys-only
        completion stays bookkeeping, as it always was: the
        assertion is recorded, but there is nothing new to run
        the match against. A decided case's reply is bookkeeping
        too — re-analysis never unmakes a human's decision.

        Guards: the invoice must be carrying an open request
        (a case that never asked, or whose request is already
        fully answered, has no loop to close); supplied keys must
        be the request's vocabulary — an unknown key is refused
        by name; and a reply must carry something (an item or a
        note) — an empty recording is a keystroke, not an answer.
        An overlay refusal refuses the whole recording: the
        record is left exactly as it stood.
        """
        from .clarify import (
            INFORMATION_ITEM_KEYS,
            analyse_information_reply,
            merge_supplied_facts,
        )
        from .schemas import ReanalysisEntry

        unknown = [key for key in supplied_items if key not in INFORMATION_ITEM_KEYS]
        if unknown:
            raise ValueError(
                f"Unknown information item(s): {', '.join(unknown)} — "
                f"a reply supplies one of: {', '.join(INFORMATION_ITEM_KEYS)}"
            )
        carried = (
            purchase_order is not None
            or goods_receipt is not None
            or bool(documents)
        )
        if not supplied_items and not note.strip() and not carried:
            raise ValueError(
                "a reply must record something — the item(s) it "
                "supplied, or a note saying what arrived"
            )
        store = self._get_store()
        record = store.get(invoice_id)
        if record is None:
            raise KeyError(f"Unknown invoice_id: {invoice_id} (analyze it first)")
        result = record.result
        request = result.information_request
        if not result.needs_information or request is None:
            raise ValueError(
                f"Invoice {invoice_id} carries no information "
                "request — there is no open loop for a reply to close"
            )
        if request.reply_status == "answered":
            raise ValueError(
                f"Invoice {invoice_id}'s information request is "
                "already fully answered — the loop is closed"
            )
        if received_at:
            try:
                datetime.fromisoformat(received_at)
            except ValueError:
                raise ValueError(
                    f"received_at {received_at!r} is not an ISO "
                    "date or datetime"
                ) from None
            received = received_at
        else:
            received = datetime.now(timezone.utc).isoformat(timespec="seconds")
        updated = analyse_information_reply(
            request,
            list(supplied_items),
            received_at=received,
            note=note.strip(),
            recorded_by=actor,
        )
        if carried:
            # Validate this reply's documents against the case as
            # it effectively stands — stored invoice, earlier
            # chain overlays, this request's accumulated overlay —
            # then fold them into the request's overlay. A refusal
            # here refuses the recording whole (nothing saved yet).
            original = InvoiceInput.model_validate(record.invoice)
            working = self._overlay_invoice(result, updated, original)
            _merged, applied = merge_supplied_facts(
                working,
                supplied_keys=list(supplied_items),
                purchase_order=purchase_order,
                goods_receipt=goods_receipt,
                documents=list(documents or []),
            )
            fold: dict = {}
            if "purchase_order" in applied:
                fold["overlay_purchase_order"] = purchase_order
            if "goods_receipt" in applied:
                fold["overlay_goods_receipt"] = goods_receipt
            if "legible_invoice" in applied:
                by_id = {d.document_id: d for d in updated.overlay_documents}
                for doc in documents or []:
                    by_id[doc.document_id] = doc
                fold["overlay_documents"] = list(by_id.values())
            if fold:
                updated = updated.model_copy(update=fold)
        overlay_ready = bool(
            updated.overlay_purchase_order
            or updated.overlay_goods_receipt
            or updated.overlay_documents
        )
        if not (
            updated.reply_status == "answered"
            and overlay_ready
            and result.status == "awaiting_approver"
        ):
            result.information_request = updated
            store.save(record)
            return result
        # The gap is closed and the documents are in hand: run
        # the case again on the overlay. The re-run happens
        # BEFORE the reply is saved — if the pipeline refuses,
        # the record still stands as it was and the operator can
        # retry the recording whole.
        original = InvoiceInput.model_validate(record.invoice)
        merged = self._overlay_invoice(result, updated, original)
        previous_outcome = result.triage.outcome
        previous_history = list(result.reanalysis_history)
        arrival_shape = record.arrival_shape
        cluster_decisions = list(record.cluster_decisions)
        backend = self.backend or get_backend()
        if self.retriever is None:
            self.retriever = get_retriever()
        new_result = self._analyze_model(merged, backend, self.retriever)
        overlay_keys: list[str] = []
        if updated.overlay_purchase_order is not None:
            overlay_keys.append("purchase_order")
        if updated.overlay_goods_receipt is not None:
            overlay_keys.append("goods_receipt")
        if updated.overlay_documents:
            overlay_keys.append("legible_invoice")
        entry = ReanalysisEntry(
            trigger="information_reply",
            previous_outcome=previous_outcome,
            new_outcome=new_result.triage.outcome,
            supplied_items=list(updated.reply_supplied_items),
            overlay=overlay_keys,
            overlay_purchase_order=updated.overlay_purchase_order,
            overlay_goods_receipt=updated.overlay_goods_receipt,
            overlay_documents=list(updated.overlay_documents),
            closed_request=updated,
            at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
            by=actor,
            note=note.strip(),
        )
        new_result.reanalysis_history = [*previous_history, entry]
        if new_result.information_request is None:
            # The re-run needs nothing further: the closed loop
            # stays on the case, so the record reads as one
            # account — request, replies, resolution, re-run.
            new_result.information_request = updated
            new_result.needs_information = True
        new_record = store.get(invoice_id)
        if new_record is not None:
            # The re-run saved its own record (over the merged
            # input): restore the case's identity — the ORIGINAL
            # stored invoice, its arrival shape, its cluster
            # stamps. The overlay lives in the chain entry, not
            # in the stored invoice.
            new_record.invoice = record.invoice
            new_record.arrival_shape = arrival_shape
            new_record.cluster_decisions = cluster_decisions
            new_record.result = new_result
            store.save(new_record)
        return new_result

    @staticmethod
    def _overlay_invoice(result, request, original):
        """The invoice a re-analysis runs on: the stored original
        with every overlay the case has accumulated applied in
        order — earlier re-analysis chain entries first (their
        overlays are part of the case's effective facts now),
        then the open request's own accumulated overlay. The
        overlays were each validated when recorded
        (clarify.merge_supplied_facts); this is the pure fold."""
        from .clarify import apply_supplied_overlay

        merged = original
        for entry in result.reanalysis_history:
            merged = apply_supplied_overlay(
                merged,
                purchase_order=entry.overlay_purchase_order,
                goods_receipt=entry.overlay_goods_receipt,
                documents=entry.overlay_documents,
            )
        return apply_supplied_overlay(
            merged,
            purchase_order=request.overlay_purchase_order,
            goods_receipt=request.overlay_goods_receipt,
            documents=request.overlay_documents,
        )

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
        from .digest import digest_store_coverage
        from .handoff_profiles import profile_effectiveness
        from .posting import approved_not_posted, posting_ladder_analytics
        from .supplier_contact_proposals import (
            documents_upfront_effectiveness,
        )
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
                # The ladder, graded: which rung a posting
                # followed (a posted receipt after that rung,
                # before any higher one), which episodes closed
                # without the ladder resolving them, and which
                # stand open exactly as they stood when the
                # last rung fired — the wait follow-through's
                # question, asked of the ERP loop.
                "escalation_ladder": posting_ladder_analytics(records),
            },
            # Information waits: the aging threshold that puts a
            # wait on the worklist, the escalation threshold (2×)
            # that pushes it, and the pushes actually recorded on
            # the records — escalations fired, as opposed to waits
            # merely old.
            "information_waits": {
                "aging_days": information_wait_aging_days(),
                "escalation_days": information_wait_escalation_days(),
                "escalations_recorded": sum(
                    len(r.result.information_escalations) for r in records
                ),
                # The pushes, graded: a reply that followed the
                # firing (the chase landed), a wait closed without
                # one (the push claims no credit), and the waits
                # still open exactly as they were — plus how long
                # the answered ones took to answer.
                "escalation_follow_through": (
                    information_escalation_follow_through(records)
                ),
            },
            # The re-analysis chains, read back: how often a
            # completing reply actually moved the case it blocked,
            # which outcome transitions the replies bought, which
            # overlay item does the unblocking — and the suppliers
            # whose replies most often change nothing.
            "reanalysis": reanalysis_analytics(records),
            "payment_runs": handoff_metrics(
                self._get_store().list_payment_runs(),
                as_of=datetime.now(timezone.utc).date(),
                group_aging_days=group_receipt_aging_days(),
            ),
            "handoff_profiles": _handoff_profile_summary(
                self._get_store().list_handoff_profiles()
            ),
            # The profiles, graded: exports routed, marker
            # incidents against the ungoverned population,
            # proposal-flag follow-through, and lead promises
            # kept — configuration that governs nothing, or
            # governs no safer than nothing, says so here.
            "handoff_profile_effectiveness": profile_effectiveness(
                self._get_store().list_handoff_profiles(),
                self._get_store().list_payment_runs(),
            ),
            # The grades' governed follow-through: profile-change
            # proposals drafted from them, by state — a pending
            # count that never moves is a drafting sweep nobody
            # decides, and it should read that way here.
            "handoff_profile_proposals": {
                state: sum(
                    1
                    for p in self._get_store().list_profile_change_proposals()
                    if p.status == state
                )
                for state in ("pending", "approved", "rejected")
            },
            # The drift watch's governed follow-through:
            # lag-segment proposals by state, and the segments
            # standing excluded from the forecast's candidate set
            # under approved decisions — an exclusion nobody can
            # see is a forecast nobody can explain.
            "lag_segment_proposals": {
                **{
                    state: sum(
                        1
                        for p in self._get_store().list_lag_segment_proposals()
                        if p.status == state
                    )
                    for state in ("pending", "approved", "rejected")
                },
                "excluded_segments": self.excluded_lag_segments(),
            },
            # The unblocking pattern's governed follow-through:
            # supplier-contact proposals by state, and the
            # suppliers standing on the documents-up-front list
            # under approved decisions — a checklist change
            # nobody can see is a pattern nobody can check.
            "supplier_contact_proposals": {
                **{
                    state: sum(
                        1
                        for p in self._get_store().list_supplier_contact_proposals()
                        if p.status == state
                    )
                    for state in ("pending", "approved", "rejected")
                },
                "documents_upfront_suppliers": (
                    self.documents_upfront_suppliers()
                ),
            },
            # The documents-up-front listings, graded: each
            # listed supplier's information waits and
            # clarification loops per case, before vs after the
            # listing moment — a listing nobody grades is
            # configuration theatre, so the grade rides the
            # same read as the proposals it can draft.
            "documents_upfront_effectiveness": (
                documents_upfront_effectiveness(
                    records,
                    self._get_store().list_supplier_contact_proposals(),
                )
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
            # The stored digests' completeness: stored days by
            # attribution, the stored range, and the gaps inside
            # it — split into gaps the sweep can still backfill
            # and gaps that predate its lookback and never will.
            "digest_store": digest_store_coverage(
                self._get_store().list_digests(),
                as_of=datetime.now(timezone.utc).date(),
                lookback_days=digest_store_lookback_days(),
            ),
        }

    def audit_rows(self) -> list[dict]:
        """The audit trail: one flat row per processed invoice —
        what the pipeline concluded, what the human decided, and what
        left the building (dispatch). Column order is stable; the
        API renders it as JSON or CSV.

        The twin columns close each row's duplicate story: the
        record it duplicates (``duplicate_of``), the shape this
        record arrived in and the shape its twin arrived in
        (``arrival_shape`` / ``twin_arrival_shape``), and the twin
        cluster it belongs to (``twin_cluster`` / ``twin_cluster_size``
        from :mod:`ap_agent.twins`) — the whole cluster is one
        filter away, without opening each record."""
        from .twins import cluster_membership, record_arrival_shape, twin_clusters

        records = self._get_store().list_records()
        clusters = twin_clusters(records)
        membership = cluster_membership(clusters)
        shapes = {
            record.result.invoice_id: record_arrival_shape(record)
            for record in records
        }
        rows: list[dict] = []
        for record in records:
            result = record.result
            invoice = record.invoice or {}
            telemetry = result.telemetry
            duplicate_of = result.facts.duplicate_of or ""
            cluster = membership.get(result.invoice_id, {})
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
                    "duplicate_of": duplicate_of,
                    "arrival_shape": shapes[result.invoice_id],
                    "twin_arrival_shape": (
                        shapes.get(duplicate_of, "") if duplicate_of else ""
                    ),
                    "twin_cluster": cluster.get("cluster_id", ""),
                    "twin_cluster_size": cluster.get("size", 0),
                }
            )
        return rows

    def duplicate_twins(self) -> dict:
        """The duplicate-twin graph over the stored records
        (``twins.twin_clusters``): every cluster of records the
        cross-shape screen links as the same invoice — members
        with their arrival shapes, the screened edges, and the
        canonical-key evidence — for a reviewer who wants the
        whole cluster at once (``GET /audit/twins``)."""
        from .twins import twin_clusters

        clusters = twin_clusters(self._get_store().list_records())
        return {"clusters": clusters, "count": len(clusters)}

    def match_credit_note(
        self, note: CreditNoteInput | dict
    ) -> CreditNoteMatch:
        """Match a supplier credit note against the processed invoices.

        A pure query over the store: the note is tied to the invoice
        it references (supplier + supplier invoice number), that
        invoice's overbilled amount is recomputed from its stored
        match facts, and the report says whether the credit settles
        the variance that remains after any credits already recorded
        against the invoice. Nothing is written — an approver acts on
        the report (``apply_credit_note``).
        """
        model = (
            note
            if isinstance(note, CreditNoteInput)
            else CreditNoteInput.model_validate(note)
        )
        store = self._get_store()
        return evaluate_credit_note(
            model,
            store.list_records(),
            applied=store.list_credit_applications(),
        )

    def apply_credit_note(
        self, note: CreditNoteInput | dict, actor: str
    ) -> CreditApplication:
        """Record a matched credit note against its invoice — the
        human-gated write path of credit-note matching.

        The match is recomputed at apply time, applications included,
        and only a credit that settles a remaining variance in full or
        in part may be recorded. Two caps are enforced here in code:
        recorded credits never exceed the invoice's remaining computed
        variance (the match status itself), and never exceed its open
        amount. Recording changes no invoice's gate state and executes
        no payment — the credit nets into the invoice's payable
        position and into future payment-run proposals."""
        model = (
            note
            if isinstance(note, CreditNoteInput)
            else CreditNoteInput.model_validate(note)
        )
        store = self._get_store()
        if store.get_credit_application(model.credit_note_id) is not None:
            raise ValueError(
                f"Credit note {model.credit_note_id} was already recorded "
                "against its invoice — a credit is never applied twice"
            )
        match = evaluate_credit_note(
            model,
            store.list_records(),
            applied=store.list_credit_applications(),
        )
        if match.status not in ("settles_variance", "partial_settlement"):
            raise ValueError(
                f"Credit note {model.credit_note_id} cannot be recorded: "
                f"its match status is {match.status} ({match.note})"
            )
        record = store.get(match.matched_invoice_id)
        if record is None:  # pragma: no cover - the match came from the store
            raise KeyError(f"Unknown invoice_id: {match.matched_invoice_id}")
        position = _payable_position(record, store.list_credit_applications())
        if match.credit_amount > position.open_amount + 0.005:
            raise ValueError(
                f"Credit note {model.credit_note_id} is for "
                f"{match.credit_amount:.2f} but invoice "
                f"{match.matched_invoice_id} has only "
                f"{position.open_amount:.2f} open — a credit never "
                "exceeds its invoice's open amount"
            )
        application = CreditApplication(
            credit_note_id=model.credit_note_id,
            credit_note_number=model.credit_note_number,
            supplier_name=model.supplier_name or match.supplier_name,
            invoice_id=match.matched_invoice_id,
            invoice_number=match.referenced_invoice_number,
            currency=model.currency or position.currency,
            amount=match.credit_amount,
            match_status=match.status,
            remaining_variance_after=match.remaining_variance,
            applied_by=actor,
            applied_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        )
        store.save_credit_application(application)
        return application

    def payable_position(self, invoice_id: str) -> PayablePosition:
        """One invoice's payable position: its processed total less
        the credits an approver recorded against it."""
        store = self._get_store()
        record = store.get(invoice_id)
        if record is None:
            raise KeyError(f"Unknown invoice_id: {invoice_id}")
        return _payable_position(record, store.list_credit_applications())

    def list_credit_applications(
        self, invoice_id: str | None = None
    ) -> list[CreditApplication]:
        return self._get_store().list_credit_applications(invoice_id)

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

    def reconcile_statement_document(
        self, document: DocumentInput | dict
    ) -> StatementReconciliation:
        """Reconcile a supplier statement that arrived as a document.

        The document runs through the intake ports — OCR for the
        read, vision for the classification — and deterministic
        code parses it into the same ``SupplierStatement`` the
        structured path reconciles (``statement_intake.py``), so a
        document statement and its structured twin produce the same
        report, the document's reads riding on the record with
        their confidences. A document that is not a statement is
        refused: the stored record carries the refusal and its
        reason, and nothing is reconciled. Either way the report
        is stored for confirmation, and no invoice record changes.
        """
        model = (
            document
            if isinstance(document, DocumentInput)
            else DocumentInput.model_validate(document)
        )
        statement, extraction = extract_statement_document(
            model,
            ocr_engine=self.ocr_engine or get_ocr_engine(),
            vision_engine=self.vision_engine or get_vision_engine(),
        )
        store = self._get_store()
        if statement is None:
            report = StatementReconciliation(
                statement_id=model.document_id or "STMT-DOCUMENT",
                supplier_name=model.fields.get("supplier_name", ""),
                source="document",
                refused=True,
                refusal_reason=extraction.refusal_reason,
                summary=extraction.refusal_reason,
                extraction=extraction,
            )
            store.save_reconciliation(report)
            return report
        history = store.prior_invoices(exclude_invoice_id="")
        report = _reconcile(statement, history)
        report.source = "document"
        report.extraction = extraction
        store.save_reconciliation(report)
        return report

    def supplier_positions(self, as_of: str = "") -> SupplierPositions:
        """The per-supplier open-position roll-up: one row per
        supplier per currency — gross invoiced, credits recorded,
        open payable, the disputed amount from the supplier's
        latest reconciliation, and the oldest open invoice's age.

        A pure read over the store (``supplier_positions.py``):
        each row is the sum of its invoices' payable positions,
        so it cannot disagree with ``GET /invoices/{id}/payable``.
        Nothing is written and no payment state changes."""
        store = self._get_store()
        anchor = _parse_date(as_of) if as_of else None
        return _supplier_positions(
            store.list_records(),
            store.list_credit_applications(),
            store.list_reconciliations(),
            as_of=anchor,
        )

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
        credits_by_invoice: dict[str, float] = {}
        for application in store.list_credit_applications():
            credits_by_invoice[application.invoice_id] = round(
                credits_by_invoice.get(application.invoice_id, 0.0)
                + application.amount,
                2,
            )
        runs = _propose_runs(
            store.list_records(),
            window_days=window_days,
            as_of=anchor,
            exclude_invoice_ids=confirmed_members,
            four_eyes_threshold=payment_run_four_eyes_threshold(),
            four_eyes_thresholds=payment_run_four_eyes_thresholds(),
            credits_by_invoice=credits_by_invoice,
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
                result = self._analyze_model(
                    invoice, backend, retriever, arrival_shape="bundle_segment"
                )
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

    def decide_twin_cluster(
        self,
        cluster_id: str,
        *,
        actor: str,
        action: str,
        reason: str = "",
    ) -> dict:
        """Hand an escalated twin cluster off as one case.

        The queue works the cluster whole — its graph, its
        combined exposure — and the hand-off is one act with two
        shapes:

        - ``approve_all``: every member still at the gate is
          approved THROUGH the cluster, each through the standard
          approval path (the guardrail gate included: a member
          whose drafts failed validation is refused by name in
          the report, never waved through by the bulk act).
          Members already decided keep their decision; the
          report says so per member.
        - ``split``: the grouping itself is judged wrong — these
          are not the same invoice — and the cluster dissolves
          as a case: no status changes, and the queue stops
          escalating it; the members are worked individually.

        Either way the act is recorded PER MEMBER: every member's
        record carries the cluster id, the action, the actor, the
        reason, the time, and that member's own outcome under the
        act. A reason is required for both shapes — approving
        suspected duplicates in bulk is the double-payment shape
        itself, and a split that cannot say why is a shrug, not a
        judgement. A cluster already split refuses further
        cluster acts (its members are individual cases now); a
        cluster that no longer exists over the stored records is
        unknown. Approving a duplicate whose twin is approved
        leaves the recovery path (draft_duplicate_recovery)
        exactly as available as before — the act records the
        approver's call, it does not relitigate it.
        """
        if action not in ("approve_all", "split"):
            raise ValueError(
                f"Unknown cluster action: {action!r} — a cluster "
                "is handed off as 'approve_all' or 'split'"
            )
        if not reason.strip():
            raise ValueError(
                "a cluster hand-off carries its reason: it is "
                "stamped on every member's record beside the act"
            )
        from .twins import twin_clusters

        store = self._get_store()
        records = store.list_records()
        cluster = next(
            (c for c in twin_clusters(records) if c["cluster_id"] == cluster_id),
            None,
        )
        if cluster is None:
            raise KeyError(
                f"Unknown twin cluster: {cluster_id} (no cluster "
                "over the stored records carries that id)"
            )
        members = cluster["members"]
        already_split = all(
            any(
                decision.get("cluster_id") == cluster_id
                and decision.get("action") == "split"
                for decision in member["cluster_decisions"]
            )
            for member in members
        )
        if already_split:
            stamp = next(
                decision
                for member in members
                for decision in member["cluster_decisions"]
                if decision.get("cluster_id") == cluster_id
                and decision.get("action") == "split"
            )
            raise ValueError(
                f"Cluster {cluster_id} was already split by "
                f"{stamp.get('actor') or 'an approver'} "
                f"({stamp.get('decided_at', '')}) — its members "
                "are worked individually now"
            )
        decided_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
        outcomes: list[dict] = []
        for member in members:
            invoice_id = member["invoice_id"]
            record = store.get(invoice_id)
            if record is None:
                outcomes.append({"invoice_id": invoice_id, "outcome": "not_on_record"})
                continue
            if action == "split":
                outcome = "split"
            elif record.result.status == "awaiting_approver":
                try:
                    self.approve(invoice_id, approver=actor, reason=reason)
                    outcome = "approved"
                except ValueError as exc:
                    outcome = f"blocked: {exc}"
                record = store.get(invoice_id) or record
            else:
                outcome = f"already_{record.result.status}"
            record.cluster_decisions = [
                *record.cluster_decisions,
                {
                    "cluster_id": cluster_id,
                    "action": action,
                    "actor": actor,
                    "reason": reason.strip(),
                    "decided_at": decided_at,
                    "outcome": outcome,
                },
            ]
            store.save(record)
            outcomes.append({"invoice_id": invoice_id, "outcome": outcome})
        return {
            "cluster_id": cluster_id,
            "action": action,
            "actor": actor,
            "reason": reason.strip(),
            "decided_at": decided_at,
            "members": outcomes,
        }

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

    def escalate_information_waits(self, as_of: str = "") -> list[InformationWaitItem]:
        """Sweep the information waits and escalate the crossed.

        The queue has always *shown* a wait past its aging
        threshold (``INFORMATION_WAIT_AGING_DAYS``); this is the
        push counterpart — for each wait whose current episode
        has run past twice that threshold with the gap still
        open, fire the signed ``information_wait_escalation``
        webhook (the opt-in ``ACTION_WEBHOOK_URL`` seam) with
        the wait's snapshot, so the chase of purchasing or the
        supplier does not depend on someone reading the worklist.

        One escalation per episode: the dispatch is recorded in
        the invoice record's own ledger
        (``result.information_escalations``, beside the request),
        deduped by the episode's start — re-running the sweep
        does not re-fire, and a partial reply starts a new
        episode (the remaining items have been owed only since
        that answer landed) with an escalation of its own once IT
        crosses. A complete reply closes the wait outright: the
        reset. With no webhook configured the sweep is a no-op
        that records nothing — there is nothing to dedupe
        against a dispatch that never happened. Returns the waits
        escalated by this sweep.
        """
        from .queue import information_wait_items

        store = self._get_store()
        anchor = _parse_date(as_of) if as_of else None
        if as_of and anchor is None:
            raise ValueError(f"as_of {as_of!r} is not an ISO date")
        anchor = anchor or datetime.now(timezone.utc).date()
        escalation_days = information_wait_escalation_days()
        items = information_wait_items(
            store.list_records(), as_of=anchor, aging_days=escalation_days
        )
        escalated: list[InformationWaitItem] = []
        for item in items:
            if item.escalated:
                continue  # this episode already fired — once is the rule
            record = store.get(item.invoice_id)
            if record is None:
                continue
            outcome = dispatch_information_wait_webhook(
                item, escalation_days=escalation_days
            )
            if outcome is None:
                continue  # webhook off: nothing fired, nothing recorded
            record.result.information_escalations.append(
                InformationEscalation(
                    episode_since=item.episode_since,
                    escalation_days=escalation_days,
                    waiting_days=item.waiting_days,
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

    def repropose_master_change(
        self,
        request_id: str,
        *,
        actor: str,
        response: str,
        effective_from: str = "",
    ) -> MasterChangeRequest:
        """Re-propose a REJECTED master change, answering the refusal.

        A rejection is information: the reviewer found something
        the first attempt did not answer. The assist carries that
        record into the second attempt — the new proposal's reason
        quotes the rejection (decider and decision note) beside
        the proposer's response to it, and ``reproposes`` links
        the lineage — and the evidence is re-verified, not copied:
        a document-sourced original is re-derived from the stored
        invoice against the CURRENT master (the master may have
        moved since), while an operator-sourced original restates
        the operator's own proposed record against a fresh
        snapshot of the master. The response is required: a
        re-proposal that does not say what it addresses is a
        resubmission, and the governed path is not a retry loop.
        Only a rejected change is re-proposed this way — a pending
        one is decidable, an approved one is done, and an expired
        one is re-proposed plainly, with fresh eyes, through
        :meth:`propose_master_change`. Four-eyes is untouched: the
        re-proposer still cannot decide their own change.
        """
        if not response.strip():
            raise ValueError(
                "a re-proposal must answer the rejection: say what "
                "the new attempt addresses that the refused one "
                "did not"
            )
        store = self._get_store()
        original = store.get_master_change(request_id)
        if original is None:
            raise KeyError(
                f"Unknown master change: {request_id} (propose it first)"
            )
        if original.status != "rejected":
            raise ValueError(
                f"Master change {request_id} is {original.status} — "
                "only a rejected change is re-proposed with its "
                "refusal answered"
            )
        rejection = f"rejected by {original.decided_by or 'the reviewer'}" + (
            f": {original.decision_note}"
            if original.decision_note
            else " (no reason recorded)"
        )
        reason = (
            f"Re-proposal of {original.request_id} — {rejection}. "
            f"Response: {response.strip()}"
        )
        if original.source == "document":
            record = store.get(original.source_invoice_id)
            if record is None:
                raise ValueError(
                    f"Master change {request_id} was proposed from "
                    f"invoice {original.source_invoice_id}, which is "
                    "no longer on record — its evidence cannot be "
                    "re-verified, so the change cannot be re-proposed "
                    "from it; propose the change directly instead"
                )
            invoice = InvoiceInput.model_validate(record.invoice or {})
            current = self._get_supplier_master().get_supplier(
                invoice.supplier_name
            )
            proposed, notes = proposed_from_invoice(invoice, current)
            if notes:
                reason = f"{reason} — {'; '.join(notes)}"
        else:
            proposed = original.proposed.model_copy(deep=True)
        change = self.propose_master_change(
            proposed,
            actor=actor,
            source=original.source,
            source_invoice_id=original.source_invoice_id,
            reason=reason,
            effective_from=effective_from,
        )
        change.reproposes = original.request_id
        store.save_master_change(change)
        return change

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
        threshold (``POSTING_ESCALATION_DAYS``), the four-eyes
        payment runs whose second confirmation has outlasted its
        aging threshold (``PAYMENT_RUN_FOUR_EYES_AGING_DAYS``), and
        the twin clusters past their escalation triggers
        (``TWIN_CLUSTER_MIN_SIZE`` / ``TWIN_CLUSTER_VALUE_THRESHOLD``,
        or any approved+awaiting span), and the information waits
        past their aging threshold (``INFORMATION_WAIT_AGING_DAYS``)
        whose requests no complete reply has closed, so one worklist
        shows everything the building still owes an answer for. A
        pure read over the store — it decides and writes nothing."""
        from .posting import approved_not_posted
        from .queue import (
            information_wait_items,
            twin_escalation_items,
            unanswered_group_items,
        )

        anchor = _parse_date(as_of) if as_of else None
        store = self._get_store()
        records = store.list_records()
        runs = store.list_payment_runs()
        escalation = max(posting_escalation_days(), posting_receipt_grace_days())
        escalated = approved_not_posted(
            records,
            as_of=anchor or datetime.now(timezone.utc).date(),
            grace_days=escalation,
        )
        min_size = twin_cluster_min_size()
        value_threshold = twin_cluster_value_threshold()
        wait_aging = information_wait_aging_days()
        group_aging = group_receipt_aging_days()
        return build_queue(
            records,
            as_of=anchor,
            recovery_packs=store.list_recovery_packs(),
            recovery_sla_days=recovery_case_sla_days(),
            open_postings=escalated.items,
            posting_escalation_days=escalation,
            payment_runs=runs,
            four_eyes_aging_days=four_eyes_aging_days(),
            twin_items=twin_escalation_items(
                records, min_size=min_size, value_threshold=value_threshold
            ),
            twin_min_size=min_size,
            twin_value_threshold=value_threshold,
            information_waits=information_wait_items(
                records,
                as_of=anchor or datetime.now(timezone.utc).date(),
                aging_days=wait_aging,
                documents_upfront=self._documents_upfront_suppliers(),
            ),
            information_wait_aging_days=wait_aging,
            group_receipt_waits=unanswered_group_items(
                runs,
                as_of=anchor or datetime.now(timezone.utc).date(),
                aging_days=group_aging,
            ),
            group_receipt_aging_days=group_aging,
        )

    def escalation_digest(self, as_of: str = "") -> EscalationDigest:
        """The day's escalation digest: the queue's action items
        (recovery SLA breaches, escalated open postings, overdue
        invoices, four-eyes stalls) plus the day's payment-run news
        (forecast lapses on runs confirmed today, drift realised on
        runs exported today) and the standing governance lines —
        among them the lag drift watch's announcements: a
        statistic quietly degrading, or standing excluded, is an
        ops event and the digest names it — composed into one
        operator summary by ``digest.build_digest``. Computed at
        read time from the stored records and sent nowhere — the
        push counterpart to the queue's pull, left for a human
        to read or forward."""
        from .digest import build_digest
        from .schemas import DigestOnboardingEntry, DigestWaitBudgetEntry

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
        # The onboarding early warning, from the marker
        # worklist aged against the digest's own day: pending
        # changes inside their expiry warning band, whose owed
        # decision can still rescue them.
        onboarding = [
            DigestOnboardingEntry(
                supplier_name=item.supplier_name,
                currency=item.currency,
                change_id=item.in_flight_change_id,
                proposed_by=item.in_flight_proposed_by,
                age_days=item.in_flight_age_days,
                days_to_expiry=item.in_flight_days_to_expiry,
                marked_value=item.marked_value,
            )
            for item in self.marker_worklist(as_of=anchor.isoformat()).items
            if item.in_flight_expiring_soon
        ]
        # The lag watch's announcements: the drift watch demotes
        # a decaying statistic silently, inside the forecast's
        # selection — the digest is where that becomes an event.
        # Entries are the segments with something to announce
        # (degrading, or standing excluded by governed decision);
        # the demotion count is the day's confirmations whose
        # forecast basis the watch actually shaped.
        from .paymentruns import lag_grade_drift_watch
        from .schemas import DigestLagWatchEntry

        excluded = self._excluded_lag_segments()
        watch_entries = [
            DigestLagWatchEntry(
                segment=segment,
                earlier_recall=row["earlier_recall"],
                recent_recall=row["recent_recall"],
                earlier_graded_members=row["earlier_graded_members"],
                recent_graded_members=row["recent_graded_members"],
                degrading=bool(row["degrading"]),
                excluded=segment in excluded,
            )
            for segment, row in lag_grade_drift_watch(runs).items()
            if row["degrading"] or segment in excluded
        ]
        # An excluded segment the watch has never graded still
        # stands excluded — announced with no readings rather
        # than silently absent.
        announced = {entry.segment for entry in watch_entries}
        watch_entries += [
            DigestLagWatchEntry(segment=segment, excluded=True)
            for segment in sorted(excluded - announced)
        ]
        demotions_today = sum(
            1
            for run in runs
            if _parse_date(run.confirmed_at or "") == anchor
            and "was demoted by" in (run.drift_forecast_basis or "")
        )
        return build_digest(
            as_of=anchor,
            queue=self.queue_view(as_of=anchor.isoformat()),
            runs=runs,
            wait_budget_breaches=breaches,
            onboarding_expiring=onboarding,
            lag_watch=watch_entries,
            lag_watch_excluded_segments=sorted(excluded),
            lag_watch_demotions_today=demotions_today,
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

    def marker_worklist(self, as_of: str = "") -> MarkerWorklist:
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
        anchor = _parse_date(as_of) if as_of else None
        if as_of and anchor is None:
            raise ValueError(f"as_of {as_of!r} is not an ISO date")
        return _marker_worklist(
            runs,
            onboarded_suppliers=onboarded,
            changes=store.list_master_changes(),
            as_of=anchor or datetime.now(timezone.utc).date(),
            expiry_days=master_change_expiry_days(),
            expiry_warning_days=master_change_expiry_warning_days(),
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
        batch_booking: bool = False,
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
            batch_booking=bool(batch_booking),
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

    # --- Effectiveness-driven profile-change proposals -------------

    def draft_profile_change_proposals(
        self, *, actor: str
    ) -> list[ProfileChangeProposal]:
        """Draft profile-change proposals from the effectiveness
        grades (see ``profile_proposals.py``).

        The grades name configuration theatre — a profile that
        governs no exports, or one whose files are no safer than
        ungoverned ones — and this sweep turns each such grade
        into a governed proposal: drafted from the evidence,
        carrying the grade's account as its basis, pending a
        second operator's decision. Drafting applies NOTHING:
        no profile changes, retires, or tightens until an
        approval lands. One open question per profile and kind —
        a profile with a pending proposal of the same kind is
        not re-drafted, so re-running the sweep is a no-op until
        the open questions are decided. Returns the proposals
        this sweep drafted (not the pending backlog)."""
        from .handoff_profiles import profile_effectiveness
        from .profile_proposals import next_proposal_id, proposals_from_grades

        if not (actor or "").strip():
            raise ValueError(
                "Provide 'actor' — a profile-change proposal is "
                "governed configuration work; an unattributed "
                "draft is refused"
            )
        store = self._get_store()
        profiles = store.list_handoff_profiles()
        grades = profile_effectiveness(profiles, store.list_payment_runs())
        specs = proposals_from_grades(profiles, grades)
        existing = store.list_profile_change_proposals()
        open_keys = {
            (p.profile_id, p.kind) for p in existing if p.status == "pending"
        }
        ids = [p.proposal_id for p in existing]
        stamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
        drafted: list[ProfileChangeProposal] = []
        for spec in specs:
            if (spec["profile_id"], spec["kind"]) in open_keys:
                continue  # the question is already open — one per profile+kind
            proposal_id = next_proposal_id(ids)
            ids.append(proposal_id)
            proposal = ProfileChangeProposal(
                proposal_id=proposal_id,
                profile_id=spec["profile_id"],
                kind=spec["kind"],
                basis=spec["basis"],
                current=spec["current"],
                proposed=spec["proposed"],
                proposed_by=actor,
                proposed_at=stamp,
                history=[
                    ProfileProposalEvent(
                        status="proposed",
                        actor=actor,
                        at=stamp,
                        note="drafted from the effectiveness grade",
                    )
                ],
            )
            store.save_profile_change_proposal(proposal)
            drafted.append(proposal)
        return drafted

    def list_profile_change_proposals(
        self, status: str = ""
    ) -> list[ProfileChangeProposal]:
        """The profile-change proposals on file, oldest first —
        or only those in one state (pending | approved |
        rejected)."""
        proposals = self._get_store().list_profile_change_proposals()
        if status:
            proposals = [p for p in proposals if p.status == status]
        return proposals

    def get_profile_change_proposal(
        self, proposal_id: str
    ) -> ProfileChangeProposal | None:
        return self._get_store().get_profile_change_proposal(proposal_id)

    def decide_profile_change_proposal(
        self,
        proposal_id: str,
        *,
        actor: str,
        approve: bool,
        note: str = "",
    ) -> ProfileChangeProposal:
        """Record the second operator's decision on a
        profile-change proposal, and apply an approval.

        Four-eyes, as on master changes: the drafter cannot
        decide their own proposal. Rejection leaves the profile
        exactly as it was. Approval applies the change the
        proposal carries — a retirement deletes the profile
        record; a marker tightening sets ``markers_rejected``
        through the profile's own governed path (the set names
        this decision and cites the proposal, so the profile's
        audit trail and the proposal's agree). The application
        is guarded against the world having moved since the
        draft: a retirement whose profile has started governing
        exports, and a tightening whose flag is already on, are
        refused — the evidence the proposal answered no longer
        stands, so the proposal must be rejected and re-drafted
        against fresh grades, not applied on stale ones."""
        from .handoff_profiles import profile_effectiveness

        store = self._get_store()
        proposal = store.get_profile_change_proposal(proposal_id)
        if proposal is None:
            raise KeyError(
                f"Unknown profile-change proposal: {proposal_id} "
                "(draft it first)"
            )
        if proposal.status != "pending":
            detail = (
                f" (decided by {proposal.decided_by})"
                if proposal.decided_by
                else ""
            )
            raise ValueError(
                f"Profile-change proposal {proposal_id} is already "
                f"{proposal.status}{detail}"
            )
        if actor == proposal.proposed_by:
            raise ValueError(
                "four-eyes on governed configuration: "
                f"{proposal.proposed_by} drafted this proposal and "
                "cannot also decide it — a second operator must "
                "approve or reject a profile change"
            )
        profile = store.get_handoff_profile(proposal.profile_id)
        if profile is None:
            raise ValueError(
                f"Profile {proposal.profile_id} no longer exists — "
                "the proposal has nothing to apply to; reject it "
                "and re-draft against the profiles in force"
            )
        now = datetime.now(timezone.utc)
        stamp = now.isoformat(timespec="seconds")
        if not approve:
            proposal.status = "rejected"
            proposal.decided_by = actor
            proposal.decided_at = stamp
            proposal.decision_note = note
            proposal.history.append(
                ProfileProposalEvent(
                    status="rejected", actor=actor, at=stamp, note=note
                )
            )
            store.save_profile_change_proposal(proposal)
            return proposal
        if proposal.kind == "retire":
            grades = profile_effectiveness([profile], store.list_payment_runs())
            if grades and grades[0]["exports_governed"] > 0:
                raise ValueError(
                    f"Profile {proposal.profile_id} has started "
                    "governing exports since this retirement was "
                    "drafted — the grade it answered no longer "
                    "stands; reject the proposal and re-draft "
                    "against fresh grades"
                )
            store.delete_handoff_profile(proposal.profile_id)
            applied = "profile retired (record deleted)"
        elif proposal.kind == "tighten_markers":
            if profile.markers_rejected:
                raise ValueError(
                    f"Profile {proposal.profile_id} already records "
                    "markers_rejected — the tightening this proposal "
                    "asks for is already in force; reject it as moot "
                    "rather than applying it twice"
                )
            self.set_handoff_profile(
                actor=actor,
                scope=profile.scope,
                supplier_name=profile.supplier_name,
                currency=profile.currency,
                format=profile.format,
                lead_days=profile.lead_days,
                markers_rejected=True,
                batch_booking=profile.batch_booking,
                note=(
                    f"applied from proposal {proposal_id} "
                    f"({proposal.proposed_by}'s draft, decided by "
                    f"{actor}): the effectiveness grade's marker "
                    "finding"
                ),
            )
            applied = "markers_rejected set through the governed profile path"
        else:  # a kind this build does not draft — refuse, never guess
            raise ValueError(
                f"Unknown profile-change proposal kind: {proposal.kind!r}"
            )
        proposal.status = "approved"
        proposal.decided_by = actor
        proposal.decided_at = stamp
        proposal.decision_note = note
        proposal.applied_at = stamp
        proposal.history.append(
            ProfileProposalEvent(
                status="approved", actor=actor, at=stamp,
                note=note or applied,
            )
        )
        store.save_profile_change_proposal(proposal)
        return proposal

    # --- Governed lag-segment proposals (the drift watch's
    # standing decisions) --------------------------------------

    def _excluded_lag_segments(self) -> set[str]:
        """The lag segments standing excluded from the forecast's
        candidate set, derived from the lag-segment proposal
        ledger (the latest approved decision per segment rules —
        see ``lag_segment_proposals.excluded_segments``)."""
        from .lag_segment_proposals import excluded_segments

        return excluded_segments(
            self._get_store().list_lag_segment_proposals()
        )

    def excluded_lag_segments(self) -> list[str]:
        """The excluded segments, sorted — the read the console
        and /metrics name."""
        return sorted(self._excluded_lag_segments())

    def draft_lag_segment_proposals(
        self, *, actor: str
    ) -> list[LagSegmentProposal]:
        """Draft lag-segment proposals from the drift watch (see
        ``lag_segment_proposals.py``).

        The watch demotes a decaying statistic inside the
        selection, run by run; this sweep turns its readings
        into governed questions: each degrading statistic that
        still stands admitted gets a retirement proposal, and
        each excluded segment whose recent record has recovered
        gets a re-grade proposal — each carrying the watch's
        account as its basis and its readings as evidence,
        pending a second operator's decision. Drafting applies
        NOTHING: no segment leaves or re-enters the candidate
        set until an approval lands. One open question per
        segment and kind — a segment with a pending proposal of
        the same kind is not re-drafted, so re-running the sweep
        is a no-op until the open questions are decided. Returns
        the proposals this sweep drafted."""
        from .lag_segment_proposals import (
            excluded_segments,
            next_lag_proposal_id,
            proposals_from_watch,
        )

        if not (actor or "").strip():
            raise ValueError(
                "Provide 'actor' — a lag-segment proposal is "
                "governed forecast work; an unattributed draft "
                "is refused"
            )
        store = self._get_store()
        runs = store.list_payment_runs()
        watch = lag_grade_drift_watch(runs)
        grades = grade_lag_statistics(runs)
        existing = store.list_lag_segment_proposals()
        specs = proposals_from_watch(
            watch, grades, excluded_segments(existing)
        )
        open_keys = {
            (p.segment, p.kind) for p in existing if p.status == "pending"
        }
        ids = [p.proposal_id for p in existing]
        stamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
        drafted: list[LagSegmentProposal] = []
        for spec in specs:
            if (spec["segment"], spec["kind"]) in open_keys:
                continue  # the question is already open — one per segment+kind
            proposal_id = next_lag_proposal_id(ids)
            ids.append(proposal_id)
            proposal = LagSegmentProposal(
                proposal_id=proposal_id,
                segment=spec["segment"],
                kind=spec["kind"],
                basis=spec["basis"],
                evidence=spec["evidence"],
                proposed_by=actor,
                proposed_at=stamp,
                history=[
                    LagSegmentProposalEvent(
                        status="proposed",
                        actor=actor,
                        at=stamp,
                        note="drafted from the drift watch",
                    )
                ],
            )
            store.save_lag_segment_proposal(proposal)
            drafted.append(proposal)
        return drafted

    def list_lag_segment_proposals(
        self, status: str = ""
    ) -> list[LagSegmentProposal]:
        """The lag-segment proposals on file, oldest first — or
        only those in one state (pending | approved | rejected)."""
        proposals = self._get_store().list_lag_segment_proposals()
        if status:
            proposals = [p for p in proposals if p.status == status]
        return proposals

    def get_lag_segment_proposal(
        self, proposal_id: str
    ) -> LagSegmentProposal | None:
        return self._get_store().get_lag_segment_proposal(proposal_id)

    def decide_lag_segment_proposal(
        self,
        proposal_id: str,
        *,
        actor: str,
        approve: bool,
        note: str = "",
    ) -> LagSegmentProposal:
        """Record the second operator's decision on a
        lag-segment proposal.

        Four-eyes, as on profile changes: the drafter cannot
        decide their own proposal. Rejection leaves the
        selection exactly as it was (the watch's own demotion
        still applies, run by run). Approval applies the
        decision — and because the exclusion state is DERIVED
        from this ledger, applying is the recording itself: an
        approved retirement takes the segment out of the
        forecast's candidate set, an approved re-grade lets it
        back in. The application is guarded against the watch
        having moved since the draft: a retirement whose
        statistic no longer reads degrading, and a re-grade
        whose segment is not excluded or still reads degrading,
        are refused — the evidence the proposal answered no
        longer stands, so it must be rejected and re-drafted
        against a fresh watch, not applied on a stale one."""
        from .lag_segment_proposals import segment_label

        store = self._get_store()
        proposal = store.get_lag_segment_proposal(proposal_id)
        if proposal is None:
            raise KeyError(
                f"Unknown lag-segment proposal: {proposal_id} "
                "(draft it first)"
            )
        if proposal.status != "pending":
            detail = (
                f" (decided by {proposal.decided_by})"
                if proposal.decided_by
                else ""
            )
            raise ValueError(
                f"Lag-segment proposal {proposal_id} is already "
                f"{proposal.status}{detail}"
            )
        if actor == proposal.proposed_by:
            raise ValueError(
                "four-eyes on governed forecast decisions: "
                f"{proposal.proposed_by} drafted this proposal "
                "and cannot also decide it — a second operator "
                "must approve or reject a lag-segment decision"
            )
        stamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
        if not approve:
            proposal.status = "rejected"
            proposal.decided_by = actor
            proposal.decided_at = stamp
            proposal.decision_note = note
            proposal.history.append(
                LagSegmentProposalEvent(
                    status="rejected", actor=actor, at=stamp, note=note
                )
            )
            store.save_lag_segment_proposal(proposal)
            return proposal
        runs = store.list_payment_runs()
        watch = lag_grade_drift_watch(runs)
        entry = watch.get(proposal.segment) or {}
        label = segment_label(proposal.segment)
        excluded = self._excluded_lag_segments()
        if proposal.kind == "retire_segment":
            if proposal.segment in excluded:
                raise ValueError(
                    f"The {label} statistic already stands "
                    "excluded — this retirement has nothing to "
                    "apply to; reject it and let the ledger stand"
                )
            if not entry.get("degrading"):
                raise ValueError(
                    f"The {label} statistic no longer reads "
                    "degrading on the current watch — the decay "
                    "this retirement answered has recovered since "
                    "the draft; reject the proposal (the watch's "
                    "own demotion lifts with the record) rather "
                    "than retiring a statistic on stale evidence"
                )
            applied = (
                "segment retired from the forecast's candidate set"
            )
        elif proposal.kind == "regrade_segment":
            if proposal.segment not in excluded:
                raise ValueError(
                    f"The {label} statistic does not stand "
                    "excluded — a re-grade re-admits an excluded "
                    "segment, and this one is not out; reject the "
                    "proposal as moot"
                )
            if entry.get("degrading") or not entry.get("assessed"):
                raise ValueError(
                    f"The {label} statistic does not read "
                    "recovered on the current watch — re-admitting "
                    "it now would crown a record the watch has not "
                    "cleared; reject the proposal and re-draft "
                    "when the recent record stands"
                )
            applied = "segment re-admitted to the forecast's candidate set"
        else:  # a kind this build does not draft — refuse, never guess
            raise ValueError(
                f"Unknown lag-segment proposal kind: {proposal.kind!r}"
            )
        proposal.status = "approved"
        proposal.decided_by = actor
        proposal.decided_at = stamp
        proposal.decision_note = note
        proposal.applied_at = stamp
        proposal.history.append(
            LagSegmentProposalEvent(
                status="approved", actor=actor, at=stamp,
                note=note or applied,
            )
        )
        store.save_lag_segment_proposal(proposal)
        return proposal

    # --- Governed supplier-contact proposals (the unblocking
    # pattern's onboarding decisions) ---------------------------

    def _documents_upfront_suppliers(self) -> set[str]:
        """The suppliers standing on the documents-up-front
        list, derived from the supplier-contact proposal ledger
        (the latest approved decision per supplier rules — see
        ``supplier_contact_proposals.documents_upfront_suppliers``)."""
        from .supplier_contact_proposals import documents_upfront_suppliers

        return documents_upfront_suppliers(
            self._get_store().list_supplier_contact_proposals()
        )

    def documents_upfront_suppliers(self) -> list[str]:
        """The listed suppliers, sorted — the read the console
        and /metrics name."""
        return sorted(self._documents_upfront_suppliers())

    def draft_supplier_contact_proposals(
        self, *, actor: str
    ) -> list[SupplierContactProposal]:
        """Draft supplier-contact proposals from the re-analysis
        analytics (see ``supplier_contact_proposals.py``).

        The analytics name the suppliers re-analysis cannot
        unblock; this sweep turns the pattern into governed
        questions: each supplier repeatedly landing unchanged
        gets a collect-documents-up-front proposal (their
        onboarding checklist should have asked for the settling
        documents first), and each listed supplier whose every
        re-analysis since listing moved its case gets a release
        proposal — as does each listed supplier the
        effectiveness grade reads no-effect, on the grade's
        evidence — each carrying the analytics' account as its
        basis and its counts as evidence, pending a second
        operator's decision. Drafting applies NOTHING: no
        supplier joins or leaves the list until an approval
        lands. One open question per supplier and kind — a
        supplier with a pending proposal of the same kind is
        not re-drafted, so re-running the sweep is a no-op until
        the open questions are decided. Returns the proposals
        this sweep drafted."""
        from .clarify import supplier_reanalysis_counts
        from .supplier_contact_proposals import (
            documents_upfront_effectiveness,
            documents_upfront_suppliers,
            next_supplier_proposal_id,
            proposals_from_analytics,
        )

        if not (actor or "").strip():
            raise ValueError(
                "Provide 'actor' — a supplier-contact proposal "
                "is governed onboarding work; an unattributed "
                "draft is refused"
            )
        store = self._get_store()
        records = store.list_records()
        existing = store.list_supplier_contact_proposals()
        analytics = reanalysis_analytics(records)
        # The listings, graded: a listing the effectiveness
        # grade reads no-effect drafts its own release on the
        # grade's evidence, beside the analytics' questions.
        grades = {
            row["supplier_name"]: row
            for row in documents_upfront_effectiveness(records, existing)[
                "suppliers"
            ]
        }
        # The analytics' worst list is capped at five; a listed
        # supplier can be crowded off it by worse ones without
        # their record changing. Their release question must
        # not depend on the cap: merge their uncapped counts in.
        listed = documents_upfront_suppliers(existing)
        named = {
            row["supplier_name"]
            for row in analytics["suppliers_not_unblocked"]
        }
        merged = dict(analytics)
        merged["suppliers_not_unblocked"] = list(
            analytics["suppliers_not_unblocked"]
        ) + [
            {
                "supplier_name": supplier,
                **supplier_reanalysis_counts(records, supplier),
            }
            for supplier in sorted(listed - named)
        ]
        specs = proposals_from_analytics(merged, existing, effectiveness=grades)
        open_keys = {
            (p.supplier_name, p.kind)
            for p in existing
            if p.status == "pending"
        }
        ids = [p.proposal_id for p in existing]
        stamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
        drafted: list[SupplierContactProposal] = []
        for spec in specs:
            if (spec["supplier_name"], spec["kind"]) in open_keys:
                continue  # the question is already open — one per supplier+kind
            proposal_id = next_supplier_proposal_id(ids)
            ids.append(proposal_id)
            proposal = SupplierContactProposal(
                proposal_id=proposal_id,
                supplier_name=spec["supplier_name"],
                kind=spec["kind"],
                basis=spec["basis"],
                evidence=spec["evidence"],
                proposed_by=actor,
                proposed_at=stamp,
                history=[
                    SupplierContactProposalEvent(
                        status="proposed",
                        actor=actor,
                        at=stamp,
                        note="drafted from the re-analysis analytics",
                    )
                ],
            )
            store.save_supplier_contact_proposal(proposal)
            drafted.append(proposal)
        return drafted

    def list_supplier_contact_proposals(
        self, status: str = ""
    ) -> list[SupplierContactProposal]:
        """The supplier-contact proposals on file, oldest first
        — or only those in one state (pending | approved |
        rejected)."""
        proposals = self._get_store().list_supplier_contact_proposals()
        if status:
            proposals = [p for p in proposals if p.status == status]
        return proposals

    def get_supplier_contact_proposal(
        self, proposal_id: str
    ) -> SupplierContactProposal | None:
        return self._get_store().get_supplier_contact_proposal(proposal_id)

    def decide_supplier_contact_proposal(
        self,
        proposal_id: str,
        *,
        actor: str,
        approve: bool,
        note: str = "",
    ) -> SupplierContactProposal:
        """Record the second operator's decision on a
        supplier-contact proposal.

        Four-eyes, as on the other proposals: the drafter
        cannot decide their own proposal. Rejection leaves the
        list exactly as it was. Approval applies the decision —
        and because the list is DERIVED from this ledger,
        applying is the recording itself: an approved collect
        puts the supplier on the documents-up-front list, an
        approved release takes them off. The application is
        guarded against the record having moved since the
        draft: a collect for a supplier who already stands
        listed is moot, and a release whose supplier has landed
        a re-analysis unchanged since the draft is refused —
        the pattern the release says is over has recurred, so
        it must be rejected and the listing left standing, not
        applied on stale evidence."""
        from .clarify import supplier_reanalysis_counts

        store = self._get_store()
        proposal = store.get_supplier_contact_proposal(proposal_id)
        if proposal is None:
            raise KeyError(
                f"Unknown supplier-contact proposal: {proposal_id} "
                "(draft it first)"
            )
        if proposal.status != "pending":
            detail = (
                f" (decided by {proposal.decided_by})"
                if proposal.decided_by
                else ""
            )
            raise ValueError(
                f"Supplier-contact proposal {proposal_id} is "
                f"already {proposal.status}{detail}"
            )
        if actor == proposal.proposed_by:
            raise ValueError(
                "four-eyes on governed onboarding decisions: "
                f"{proposal.proposed_by} drafted this proposal "
                "and cannot also decide it — a second operator "
                "must approve or reject a checklist change"
            )
        stamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
        if not approve:
            proposal.status = "rejected"
            proposal.decided_by = actor
            proposal.decided_at = stamp
            proposal.decision_note = note
            proposal.history.append(
                SupplierContactProposalEvent(
                    status="rejected", actor=actor, at=stamp, note=note
                )
            )
            store.save_supplier_contact_proposal(proposal)
            return proposal
        listed = self._documents_upfront_suppliers()
        if proposal.kind == "collect_documents_upfront":
            if proposal.supplier_name in listed:
                raise ValueError(
                    f"{proposal.supplier_name} already stands on "
                    "the documents-up-front list — this collect "
                    "has nothing to apply to; reject it and let "
                    "the ledger stand"
                )
            applied = (
                "supplier added to the documents-up-front list — "
                "their onboarding checklist collects the settling "
                "documents up front"
            )
        elif proposal.kind == "release_documents_upfront":
            if proposal.supplier_name not in listed:
                raise ValueError(
                    f"{proposal.supplier_name} does not stand on "
                    "the documents-up-front list — a release "
                    "unlists a listed supplier, and this one is "
                    "not listed; reject the proposal as moot"
                )
            if proposal.evidence.get("drafted_from") == "effectiveness_grade":
                # The grade's release is guarded in the grade's
                # own terms: re-grade on the current record, and
                # release only while the listing still reads as
                # one that changed nothing.
                from .supplier_contact_proposals import (
                    documents_upfront_effectiveness,
                )

                grades = documents_upfront_effectiveness(
                    store.list_records(),
                    store.list_supplier_contact_proposals(),
                )
                row = next(
                    (
                        r
                        for r in grades["suppliers"]
                        if r["supplier_name"] == proposal.supplier_name
                    ),
                    None,
                )
                verdict = (row or {}).get("verdict") or "unlisted"
                if verdict != "no_effect":
                    raise ValueError(
                        f"{proposal.supplier_name}'s listing no "
                        f"longer grades as no-effect (it now "
                        f"reads {verdict}) — the evidence this "
                        "release rests on has moved; reject the "
                        "proposal and leave the supplier listed, "
                        "rather than releasing a checklist on a "
                        "stale grade"
                    )
                applied = (
                    "supplier released from the documents-up-front "
                    "list — the listing was graded and changed "
                    "nothing it was bought to change"
                )
            else:
                current = supplier_reanalysis_counts(
                    store.list_records(), proposal.supplier_name
                )
                if current["unchanged"] > int(
                    proposal.evidence.get("unchanged") or 0
                ):
                    raise ValueError(
                        f"{proposal.supplier_name} has landed a "
                        "re-analysis unchanged since this release was "
                        "drafted — the pattern it says is over has "
                        "recurred; reject the proposal and leave the "
                        "supplier listed, rather than releasing a "
                        "checklist on stale evidence"
                    )
                applied = (
                    "supplier released from the documents-up-front "
                    "list — their re-analyses since listing all moved "
                    "their cases"
                )
        else:  # a kind this build does not draft — refuse, never guess
            raise ValueError(
                f"Unknown supplier-contact proposal kind: {proposal.kind!r}"
            )
        proposal.status = "approved"
        proposal.decided_by = actor
        proposal.decided_at = stamp
        proposal.decision_note = note
        proposal.applied_at = stamp
        proposal.history.append(
            SupplierContactProposalEvent(
                status="approved", actor=actor, at=stamp,
                note=note or applied,
            )
        )
        store.save_supplier_contact_proposal(proposal)
        return proposal

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
        its proposal date plus a measured proposal→export lag —
        and the members projected to lapse are flagged on the run,
        with the basis stated. The lag is a selected statistic
        (the governing profile's segment, the currency's, or the
        global pool), graded against the recorded export outcomes
        in the accuracy ledger; the basis names the statistic and
        why it was chosen. A forecast, labelled as one: it warns
        the confirmer while they can still hurry the export; it
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
        # The forecast's lag statistic is selected, not assumed:
        # the run's governing profile segment, its currency
        # segment, or the global pool — graded against the
        # accuracy ledger (paymentruns._select_lag_statistic).
        # The governing profile is resolved the way the export
        # will resolve it, so the segment the forecast names is
        # the segment the export will stamp.
        from .handoff_profiles import resolve_export_format

        runs = store.list_payment_runs()
        _format, profile, _basis = resolve_export_format(
            run, store.list_handoff_profiles(), ""
        )
        forecast, basis = _drift_forecast(
            run,
            as_of=datetime.now(timezone.utc).date(),
            lag_segments=lag_statistics(runs),
            lag_grades=grade_lag_statistics(runs),
            lag_drift_watch=lag_grade_drift_watch(runs),
            profile_id=profile.profile_id if profile else "",
            excluded_segments=self._excluded_lag_segments(),
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
        ``handoff_execution_date``). A governing profile with
        batch booking groups the pain.001 into one ``PmtInf``
        per requested execution date (its lead days flooring
        every group's date), and each completion entry is
        stamped with the execution date its member exported
        under.

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
        # Batch booking is the governing profile's call, and it
        # is a pain.001 shape: the file groups the run's lines
        # into one PmtInf per requested execution date. The
        # profile's lead days become the floor every group's date
        # respects (batch_groups states which date wins per
        # group); the run stamps every group's date.
        batch_booking = bool(
            profile is not None
            and profile.batch_booking
            and resolved_format == "pain001"
        )
        batch_floor = ""
        groups: list = []
        if batch_booking:
            from .handoff import batch_groups

            if profile.lead_days > 0:
                batch_floor = (
                    anchor + timedelta(days=profile.lead_days)
                ).isoformat()
            groups = batch_groups(run, batch_floor)
            execution_date = ", ".join(date for date, _ in groups)
        else:
            execution_date = (
                requested_execution_date(run, profile, export_date=anchor)
                if resolved_format == "pain001"
                else ""
            )
        data = render_handoff(
            run,
            resolved_format,
            creditor_accounts=completions,
            requested_execution_date=execution_date if not batch_booking else "",
            batch_booking=batch_booking,
            batch_floor_date=batch_floor,
        )
        if resolved_format == "pain001":
            # Each completion entry records the execution date its
            # member was exported under — the file's single
            # ReqExctnDt, or its own PmtInf group's date under
            # batch booking — so completion coverage can split by
            # PmtInf grouping (paymentruns.creditor_completion_metrics).
            if batch_booking:
                group_of = {
                    line.invoice_id: date
                    for date, group_lines in groups
                    for line in group_lines
                }
                for entry in run.handoff_creditor_entries:
                    entry.execution_date = group_of.get(entry.invoice_id, "")
            else:
                for entry in run.handoff_creditor_entries:
                    entry.execution_date = execution_date
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
        run.handoff_batch_booking = batch_booking
        # A fresh export is a fresh file: the bank's per-group
        # answers belonged to the previous file's groups, and a
        # receipt must never read against a file it did not
        # answer. The new file starts unanswered.
        run.handoff_group_receipts = []
        run.handoff_profile_id = profile.profile_id if profile else ""
        run.handoff_execution_date = execution_date
        if batch_booking:
            run.note += (
                f" Hand-off pain.001 batch-booked under profile "
                f"{profile.profile_id}: {len(groups)} PmtInf group(s) "
                f"by requested execution date ({execution_date})"
                + (
                    f"; the profile's {profile.lead_days}d lead floors "
                    "every group date."
                    if profile.lead_days > 0
                    else "."
                )
            )
        if profile is not None and basis.startswith("profile") and not batch_booking:
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

    def record_handoff_group_receipts(
        self,
        run_id: str,
        *,
        actor: str,
        receipts: list,
    ) -> PaymentRun:
        """Record the bank's per-group answers for a batch-booked
        export (``HandoffGroupReceipt``).

        The loop the batch booking left open: the file went out
        as one ``PmtInf`` per requested execution date, and the
        bank answers per group — one books, another bounces. The
        answers are recorded against their groups here, each
        receipt stamped with the group's ``PmtInf`` id and its
        members (from the export receipt — the file the bank
        actually saw), so every member carries its group's fate
        on the run. Recording is bookkeeping, like the posting
        receipt: no payment state changes, no member moves.

        The guards are the loop's integrity, and the recording
        is validated WHOLE before anything is written (a refusal
        refuses it entire, the clarify idiom): the run must have
        a batch-booked pain.001 export in force (a flat file has
        no groups to answer); every answer must name a group the
        export actually made, carry a known status, and a
        rejection must carry the bank's reason; one answer per
        group per recording. A group the bank already ACCEPTED
        is final — its money booked, a second answer is a
        bank-side correction, not a receipt — while a REJECTED
        group's later answer supersedes the first (the group was
        corrected and re-presented), the old receipt staying on
        the run marked superseded."""
        from .handoff import batch_group_members

        if not (actor or "").strip():
            raise ValueError(
                "Provide 'actor' — bank feedback is recorded "
                "against a name; an unattributed recording is "
                "refused"
            )
        if not receipts:
            raise ValueError(
                "Provide at least one group receipt — an empty "
                "recording answers nothing"
            )
        store = self._get_store()
        run = store.get_payment_run(run_id)
        if run is None:
            raise KeyError(f"Unknown run_id: {run_id} (propose it first)")
        if not run.handoff_generated_at:
            raise ValueError(
                f"Payment run {run_id} has no hand-off export — "
                "the bank cannot answer a file that was never "
                "generated; export it first"
            )
        if not (run.handoff_format == "pain001" and run.handoff_batch_booking):
            raise ValueError(
                f"Payment run {run_id} was exported as "
                f"{run.handoff_format or 'unknown'}"
                + (
                    " without batch booking"
                    if run.handoff_format == "pain001"
                    else ""
                )
                + " — per-group bank answers belong to a "
                "batch-booked pain.001 file, whose PmtInf groups "
                "are what the bank answers"
            )
        groups = batch_group_members(run)
        if not groups:
            raise ValueError(
                f"Payment run {run_id}'s export receipt names no "
                "batch groups — there is nothing for the bank's "
                "answers to attach to"
            )
        standing = {
            receipt.execution_date: receipt
            for receipt in run.handoff_group_receipts
            if not receipt.superseded
        }
        # Validate the whole recording before writing any of it.
        seen: set[str] = set()
        for receipt in receipts:
            date = (receipt.execution_date or "").strip()
            if date in seen:
                raise ValueError(
                    f"Two answers for the {date} group in one "
                    "recording — the bank answers a group once "
                    "per file; record them in separate recordings"
                )
            seen.add(date)
            if date not in groups:
                names = ", ".join(sorted(groups))
                raise ValueError(
                    f"No group with requested execution date "
                    f"{date!r} in run {run_id}'s export — its "
                    f"groups are: {names}. An answer for a group "
                    "the file never made is refused, not filed "
                    "against the nearest date"
                )
            if receipt.status not in ("accepted", "rejected"):
                raise ValueError(
                    f"Unknown group receipt status "
                    f"{receipt.status!r} for the {date} group — "
                    "the bank accepts a group or rejects it"
                )
            if receipt.status == "rejected" and not (
                receipt.detail or ""
            ).strip():
                raise ValueError(
                    f"The {date} group was rejected with no "
                    "reason recorded — a rejection without the "
                    "bank's reason cannot be acted on; record "
                    "the reason the bank gave"
                )
            current = standing.get(date)
            if current is not None and current.status == "accepted":
                raise ValueError(
                    f"The {date} group already stands accepted "
                    f"(recorded by {current.recorded_by}) — its "
                    "money is booked; a second answer is a "
                    "bank-side correction, not a receipt"
                )
        stamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
        for receipt in receipts:
            date = (receipt.execution_date or "").strip()
            current = standing.get(date)
            if current is not None:
                current.superseded = True
            run.handoff_group_receipts.append(
                HandoffGroupReceipt(
                    pmt_inf_id=f"{run.run_id}-{date}",
                    execution_date=date,
                    status=receipt.status,
                    detail=(receipt.detail or "").strip(),
                    bank_reference=(receipt.bank_reference or "").strip(),
                    member_invoice_ids=list(groups[date]),
                    recorded_by=actor,
                    recorded_at=stamp,
                )
            )
        store.save_payment_run(run)
        return run

    def reexport_handoff_group(
        self, run_id: str, execution_date: str, *, actor: str, as_of: str = ""
    ) -> PaymentRun:
        """Re-export one bounced batch group as its own file.

        The bank rejected the group; the whole-file answer —
        re-export the run — would re-present every group,
        including the ones that already booked, and invite the
        bank to answer money already moving. So the rejected
        group goes out ALONE: a single-``PmtInf`` pain.001
        carrying the group's own ``PmtInf`` id and only its
        members, at their export-time figures with the discount
        drift re-verified as of the re-export date
        (``discount_drift``, itemised on the re-export record —
        a member whose window closed since the first file is
        re-priced for the new one, never silently).

        The act is stamped on the run (``HandoffGroupReexport``):
        the file's key and content hash, who re-exported it and
        when, the group's new requested execution date — the
        later of the group's original date and the re-export
        date, since re-presenting on a date already past asks
        the bank for yesterday — and the bank's rejection
        reason, carried from the standing rejected receipt: the
        corrected file exists because of that reason, and the
        record says so.

        The re-export resets the group's receipts: the standing
        rejected receipt is marked superseded (it answered the
        first file), so the group stands unanswered against the
        new file and the bank's answer to it records through the
        same endpoint, against the group's original execution
        date — the group's identity in both files. The other
        groups' receipts are untouched, and no payment state
        moves: the re-export is a file in object storage, like
        the export it corrects.
        """
        from .handoff import (
            batch_group_members,
            creditor_completions,
            current_group_receipts,
            group_reexports,
            handoff_content_type,
            handoff_group_reexport_key,
            render_handoff_pain001,
        )

        if not (actor or "").strip():
            raise ValueError(
                "Provide 'actor' — a group re-export is recorded "
                "against a name; an unattributed re-presentation "
                "is refused"
            )
        date = (execution_date or "").strip()
        store = self._get_store()
        run = store.get_payment_run(run_id)
        if run is None:
            raise KeyError(f"Unknown run_id: {run_id} (propose it first)")
        if not run.handoff_generated_at:
            raise ValueError(
                f"Payment run {run_id} has no hand-off export — "
                "there is no group to re-export; export it first"
            )
        if not (run.handoff_format == "pain001" and run.handoff_batch_booking):
            raise ValueError(
                f"Payment run {run_id} was exported as "
                f"{run.handoff_format or 'unknown'}"
                + (
                    " without batch booking"
                    if run.handoff_format == "pain001"
                    else ""
                )
                + " — solo group re-exports belong to a "
                "batch-booked pain.001 file, whose PmtInf groups "
                "are what the bank answers"
            )
        groups = batch_group_members(run)
        if date not in groups:
            names = ", ".join(sorted(groups)) or "none"
            raise ValueError(
                f"No group with requested execution date "
                f"{date!r} in run {run_id}'s export — its "
                f"groups are: {names}. A re-export for a group "
                "the file never made is refused, not filed "
                "against the nearest date"
            )
        standing = current_group_receipts(run).get(date)
        if standing is None:
            prior = group_reexports(run, date)
            if prior:
                raise ValueError(
                    f"The {date} group of run {run_id} was "
                    f"already re-exported on "
                    f"{prior[-1].reexported_at[:10]} by "
                    f"{prior[-1].reexported_by} and is awaiting "
                    "the bank's answer to that file — record "
                    "the answer when it lands instead of sending "
                    "the group a third time"
                )
            raise ValueError(
                f"The {date} group of run {run_id} has no "
                "bank answer recorded — nothing has bounced. "
                "Re-exporting an unanswered group would fork "
                "the file: the bank could answer either copy. "
                "Record the group's answer first; only a "
                "rejected group is re-presented"
            )
        if standing.status == "accepted":
            raise ValueError(
                f"The {date} group of run {run_id} stands "
                f"accepted (recorded by {standing.recorded_by}) "
                "— its money is booked; there is nothing to "
                "re-present"
            )
        object_store = self.object_store
        if object_store is None:
            from .storage import get_object_store

            object_store = get_object_store()
        if object_store is None:
            raise ValueError(
                "Object storage is not configured (set S3_BUCKET, or "
                "inject an object store): the re-export file has "
                "nowhere to live"
            )
        anchor = _parse_date(as_of) if as_of else None
        if as_of and anchor is None:
            raise ValueError(f"as_of {as_of!r} is not an ISO date")
        anchor = anchor or datetime.now(timezone.utc).date()
        members = list(groups[date])
        member_set = set(members)
        # The group's own run: its members only, at the figures
        # the first file carried (the run's lines ARE the
        # export-time figures) — then drift re-verified as of
        # the re-export date, so the new file never nets off a
        # discount the bank run can no longer take.
        sub = run.model_copy(deep=True)
        sub.lines = [
            line for line in sub.lines if line.invoice_id in member_set
        ]
        sub.invoice_count = len(sub.lines)
        sub.total_gross = round(
            sum(line.gross_amount for line in sub.lines), 2
        )
        sub.total_discount = round(
            sum(line.discount_amount for line in sub.lines), 2
        )
        sub.total_net = round(
            sum(line.net_amount for line in sub.lines), 2
        )
        records = {
            record.result.invoice_id: record for record in store.list_records()
        }
        sub, drift = discount_drift(sub, records, as_of=anchor)
        completions, _entries = creditor_completions(
            sub, self._get_supplier_master()
        )
        pmt_inf_id = f"{run.run_id}-{date}"
        sub.run_id = pmt_inf_id
        group_date = _parse_date(date)
        reexport_date = (
            max(anchor, group_date) if group_date is not None else anchor
        ).isoformat()
        data = render_handoff_pain001(
            sub, completions, requested_execution_date=reexport_date
        )
        sequence = len(group_reexports(run, date)) + 1
        key = handoff_group_reexport_key(run.run_id, date, sequence)
        object_store.put(
            key, data, content_type=handoff_content_type("pain001")
        )
        stamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
        # The rejected receipt answered the first file; the new
        # file starts the group unanswered.
        standing.superseded = True
        run.handoff_group_reexports.append(
            HandoffGroupReexport(
                pmt_inf_id=pmt_inf_id,
                execution_date=date,
                reexport_execution_date=reexport_date,
                rejection_reason=standing.detail,
                member_invoice_ids=members,
                key=key,
                sha256=hashlib.sha256(data).hexdigest(),
                reexported_by=actor,
                reexported_at=stamp,
                drift=drift,
                drift_total=round(
                    sum(entry.discount_lost for entry in drift), 2
                ),
            )
        )
        run.note += (
            f" Group {date} re-exported alone on "
            f"{anchor.isoformat()} by {actor}: the bank rejected "
            f"it ('{standing.detail}'), so the group was "
            f"re-presented as a single-PmtInf pain.001 "
            f"({pmt_inf_id}) for execution {reexport_date}; the "
            "group's receipts reset for the new file."
        )
        store.save_payment_run(run)
        return run

    def get_handoff_group_reexport(
        self, run_id: str, execution_date: str
    ) -> bytes | None:
        """The stored bytes of a group's latest solo re-export,
        or None when the group has never been re-exported (or
        the object is gone)."""
        from .handoff import latest_group_reexport

        run = self._get_store().get_payment_run(run_id)
        if run is None:
            return None
        reexport = latest_group_reexport(
            run, (execution_date or "").strip()
        )
        if reexport is None or not reexport.key:
            return None
        object_store = self.object_store
        if object_store is None:
            from .storage import get_object_store

            object_store = get_object_store()
        if object_store is None:
            return None
        try:
            return object_store.get(reexport.key)
        except Exception:
            return None

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
