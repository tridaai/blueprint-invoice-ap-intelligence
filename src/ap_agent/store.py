"""Approval store: where analyses and approver decisions live.

**PostgreSQL is the system of record.** ``PostgresStore`` (selected
whenever ``DATABASE_URL`` is set — the production and compose path)
keeps one row per invoice in the ``approvals`` table created by
``migrations/0001_core.sql``; schema migrations run on construction,
so a fresh deployment needs no separate step. Analyses and decisions
survive restarts and are shared across processes and batch workers,
as a production AP queue requires.

``InMemoryStore`` is a test double: it implements the same ``Store``
protocol for the hermetic test suite and for ephemeral single-process
runs where no database is configured. It persists nothing and is
never the documented runtime story.

The store is also the memory source: ``prior_invoices`` feeds the
diagnosis evidence (same supplier) and duplicate detection, and the
feedback loop (decisions with their reasons teach later runs).
"""

from __future__ import annotations

import json
import threading
from dataclasses import dataclass, field

from .config import env_str, load_dotenv
from .schemas import AgentResult


@dataclass
class ApprovalRecord:
    result: AgentResult
    invoice: dict = field(default_factory=dict)
    approved: bool = False
    approver: str = ""
    approve_reason: str = ""
    rejected_by: str = ""
    reject_reason: str = ""
    dispatch_status: str | None = None


def history_entry(record: ApprovalRecord) -> dict:
    """The compact history projection of one analysed invoice.

    Invoice number / totals come from the stored invoice payload (the
    result object itself carries only the supplier + triage)."""
    result = record.result
    sub = record.invoice or {}
    return {
        "invoice_id": result.invoice_id,
        "supplier_name": result.supplier_name or sub.get("supplier_name", ""),
        "invoice_number": sub.get("invoice_number", ""),
        "total_amount": sub.get("total_amount"),
        "outcome": result.triage.outcome,
        "severity": result.triage.severity,
    }


def feedback_entry(record: ApprovalRecord) -> dict | None:
    """A decided record as a feedback entry, or None when undecided
    (or decided without a reason — feedback without a reason teaches
    nothing)."""
    result = record.result
    if result.status not in ("approved", "rejected") or not result.decision_reason:
        return None
    sub = record.invoice or {}
    return {
        "invoice_id": result.invoice_id,
        "supplier_name": result.supplier_name or sub.get("supplier_name", ""),
        "decision": result.status,
        "reason": result.decision_reason,
    }


def supplier_summary(priors: list[dict], supplier: str) -> dict:
    """Outcome counts for one supplier across prior invoices."""
    hits = [e for e in priors if supplier and e.get("supplier_name") == supplier]
    counts: dict[str, int] = {}
    for entry in hits:
        counts[entry["outcome"]] = counts.get(entry["outcome"], 0) + 1
    return {
        "supplier": supplier,
        "supplier_count": len(hits),
        "supplier_outcome_counts": counts,
    }


def find_duplicate(priors: list[dict], invoice) -> str:
    """The invoice_id of a prior invoice that looks like a duplicate
    of this one — same supplier, same supplier invoice number, same
    total (AP-DUP-01) — or "" when none matches."""
    if not invoice.invoice_number:
        return ""
    for entry in priors:
        if (
            entry.get("supplier_name") == invoice.supplier_name
            and entry.get("invoice_number") == invoice.invoice_number
            and invoice.total_amount is not None
            and entry.get("total_amount") is not None
            and abs(float(entry["total_amount"]) - float(invoice.total_amount)) < 0.005
        ):
            return entry["invoice_id"]
    return ""


def format_outcome_counts(counts: dict[str, int]) -> str:
    """{'clean_match': 2, 'price_variance': 1} -> '2 clean_match, 1 price_variance'."""
    return ", ".join(f"{n} {outcome}" for outcome, n in sorted(counts.items())) or "none"


def _dump_record(record: ApprovalRecord) -> str:
    return json.dumps(
        {
            "result": record.result.model_dump(mode="json"),
            "invoice": record.invoice,
            "approved": record.approved,
            "approver": record.approver,
            "approve_reason": record.approve_reason,
            "rejected_by": record.rejected_by,
            "reject_reason": record.reject_reason,
            "dispatch_status": record.dispatch_status,
        }
    )


def _load_record(payload: str | dict) -> ApprovalRecord:
    data = json.loads(payload) if isinstance(payload, str) else payload
    return ApprovalRecord(
        result=AgentResult.model_validate(data["result"]),
        invoice=data.get("invoice", {}),
        approved=data.get("approved", False),
        approver=data.get("approver", ""),
        approve_reason=data.get("approve_reason", ""),
        rejected_by=data.get("rejected_by", ""),
        reject_reason=data.get("reject_reason", ""),
        dispatch_status=data.get("dispatch_status"),
    )


class InMemoryStore:
    """Process-local store — a test double (and the ephemeral default
    when no DATABASE_URL is configured). Thread-safe; persists nothing."""

    name = "memory"

    def __init__(self) -> None:
        self._records: dict[str, ApprovalRecord] = {}
        self._lock = threading.Lock()

    def save(self, record: ApprovalRecord) -> None:
        with self._lock:
            self._records[record.result.invoice_id] = record

    def get(self, invoice_id: str) -> ApprovalRecord | None:
        with self._lock:
            return self._records.get(invoice_id)

    def _all(self) -> list[ApprovalRecord]:
        with self._lock:
            return list(self._records.values())

    def prior_invoices(self, exclude_invoice_id: str) -> list[dict]:
        return [
            history_entry(r)
            for r in self._all()
            if r.result.invoice_id != exclude_invoice_id
        ]

    def decision_feedback(self, exclude_invoice_id: str) -> list[dict]:
        entries = []
        for record in reversed(self._all()):
            if record.result.invoice_id == exclude_invoice_id:
                continue
            entry = feedback_entry(record)
            if entry is not None:
                entries.append(entry)
        return entries

    def ping(self) -> bool:
        return True

    def close(self) -> None:
        pass


class PostgresStore:
    """PostgreSQL-backed store: one short-lived connection per call,
    so concurrent batch workers and multiple API processes share state
    through the database — the production shape. Migrations run on
    construction (idempotent)."""

    name = "postgres"

    def __init__(self, database_url: str) -> None:
        try:
            import psycopg
            from psycopg.rows import dict_row
        except ImportError as exc:
            raise RuntimeError(
                "DATABASE_URL is set but the psycopg driver is not installed. "
                "Install the postgres extra: uv sync --extra postgres "
                "(pip fallback: pip install -e \".[postgres]\"), then re-run."
            ) from exc
        self._url = database_url
        self._psycopg = psycopg
        self._row_factory = dict_row
        from .migrations import run_migrations

        with self._connect() as conn:
            run_migrations(conn)

    def _connect(self):
        return self._psycopg.connect(self._url, row_factory=self._row_factory)

    def save(self, record: ApprovalRecord) -> None:
        decided = record.result.status in ("approved", "rejected")
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "INSERT INTO approvals (invoice_id, payload, decided, updated_at) "
                    "VALUES (%s, %s::jsonb, %s, now()) "
                    "ON CONFLICT (invoice_id) DO UPDATE SET "
                    "payload = EXCLUDED.payload, decided = EXCLUDED.decided, "
                    "updated_at = now()",
                    (record.result.invoice_id, _dump_record(record), decided),
                )
            conn.commit()

    def get(self, invoice_id: str) -> ApprovalRecord | None:
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT payload FROM approvals WHERE invoice_id = %s",
                    (invoice_id,),
                )
                row = cur.fetchone()
        return _load_record(row["payload"]) if row else None

    def _all_payloads(self) -> list:
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT payload FROM approvals ORDER BY seq")
                return [row["payload"] for row in cur.fetchall()]

    def prior_invoices(self, exclude_invoice_id: str) -> list[dict]:
        return [
            history_entry(_load_record(payload))
            for payload in self._all_payloads()
            if _load_record(payload).result.invoice_id != exclude_invoice_id
        ]

    def decision_feedback(self, exclude_invoice_id: str) -> list[dict]:
        entries = []
        for payload in reversed(self._all_payloads()):
            record = _load_record(payload)
            if record.result.invoice_id == exclude_invoice_id:
                continue
            entry = feedback_entry(record)
            if entry is not None:
                entries.append(entry)
        return entries

    def ping(self) -> bool:
        """Readiness probe: True when the database answers."""
        try:
            with self._connect() as conn:
                with conn.cursor() as cur:
                    cur.execute("SELECT 1")
                    cur.fetchone()
            return True
        except Exception:
            return False

    def close(self) -> None:
        pass  # connections are per-call; nothing pooled to release


def default_store() -> InMemoryStore | PostgresStore:
    """The store from the environment: PostgreSQL when ``DATABASE_URL``
    is set (production), otherwise the ephemeral in-memory double."""
    load_dotenv()
    url = env_str("DATABASE_URL")
    if url:
        return PostgresStore(url)
    return InMemoryStore()
