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
from .schemas import (
    AgentResult,
    BundleResult,
    EscalationDigest,
    HandoffProfile,
    MasterChangeRequest,
    PaymentRun,
    RecoveryPack,
    StatementReconciliation,
)


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
        "duplicate_of": result.facts.duplicate_of,
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


def duplicate_key(supplier_name: str, invoice_number: str) -> tuple[str, str]:
    """The canonical identity a duplicate is judged on.

    The same invoice arrives in different shapes — a structured
    portal submission, a segment cut out of a combined bundle scan,
    a line on a supplier statement — and each shape writes the
    identity slightly differently: supplier case and spacing vary,
    and invoice numbers pick up or lose their separators
    (``NB-4471`` vs ``NB 4471`` vs ``nb4471``). Judging duplicates
    on the raw strings would make the control depend on the arrival
    shape, so both sides are reduced to a canonical key first:
    the supplier name casefolded, punctuation dropped and
    whitespace collapsed (``Acme, Inc.`` and ``Acme Inc`` are the
    same supplier), the invoice number reduced to its uppercased
    alphanumerics. The screen only ever flags for a human — a
    generous key costs a glance, a strict one costs a double
    payment."""
    supplier = " ".join(
        "".join(
            ch if ch.isalnum() else " " for ch in str(supplier_name or "").casefold()
        ).split()
    )
    number = "".join(
        ch for ch in str(invoice_number or "").upper() if ch.isalnum()
    )
    return supplier, number


def find_duplicate(priors: list[dict], invoice) -> str:
    """The invoice_id of a prior invoice that looks like a duplicate
    of this one — same supplier, same supplier invoice number, same
    total (AP-DUP-01) — or "" when none matches. The identity is
    compared on :func:`duplicate_key`, so a resend that arrives in
    a different shape (a bundle segment after the portal single, or
    the reverse) is still caught, with the earlier record as the
    twin."""
    if not invoice.invoice_number:
        return ""
    key = duplicate_key(invoice.supplier_name, invoice.invoice_number)
    if not key[1]:
        return ""
    for entry in priors:
        if (
            duplicate_key(
                str(entry.get("supplier_name", "")), str(entry.get("invoice_number", ""))
            )
            == key
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
        self._reconciliations: dict[str, str] = {}
        self._payment_runs: dict[str, str] = {}
        self._recovery_packs: dict[str, str] = {}
        self._master_changes: dict[str, str] = {}
        self._bundles: dict[str, str] = {}
        self._digests: dict[str, str] = {}
        self._handoff_profiles: dict[str, str] = {}
        self._lock = threading.Lock()

    def save(self, record: ApprovalRecord) -> None:
        with self._lock:
            self._records[record.result.invoice_id] = record

    def save_reconciliation(self, report: StatementReconciliation) -> None:
        with self._lock:
            self._reconciliations[report.statement_id] = report.model_dump_json()

    def get_reconciliation(
        self, statement_id: str
    ) -> StatementReconciliation | None:
        with self._lock:
            payload = self._reconciliations.get(statement_id)
        return (
            StatementReconciliation.model_validate_json(payload)
            if payload
            else None
        )

    def save_payment_run(self, run: PaymentRun) -> None:
        with self._lock:
            self._payment_runs[run.run_id] = run.model_dump_json()

    def get_payment_run(self, run_id: str) -> PaymentRun | None:
        with self._lock:
            payload = self._payment_runs.get(run_id)
        return PaymentRun.model_validate_json(payload) if payload else None

    def list_payment_runs(self) -> list[PaymentRun]:
        with self._lock:
            payloads = [self._payment_runs[k] for k in sorted(self._payment_runs)]
        return [PaymentRun.model_validate_json(p) for p in payloads]

    def save_recovery_pack(self, pack: RecoveryPack) -> None:
        with self._lock:
            self._recovery_packs[pack.duplicate_invoice_id] = pack.model_dump_json()

    def get_recovery_pack(self, invoice_id: str) -> RecoveryPack | None:
        with self._lock:
            payload = self._recovery_packs.get(invoice_id)
        return RecoveryPack.model_validate_json(payload) if payload else None

    def list_recovery_packs(self) -> list[RecoveryPack]:
        with self._lock:
            payloads = [
                self._recovery_packs[k] for k in sorted(self._recovery_packs)
            ]
        return [RecoveryPack.model_validate_json(p) for p in payloads]

    def save_master_change(self, change: MasterChangeRequest) -> None:
        with self._lock:
            self._master_changes[change.request_id] = change.model_dump_json()

    def get_master_change(self, request_id: str) -> MasterChangeRequest | None:
        with self._lock:
            payload = self._master_changes.get(request_id)
        return MasterChangeRequest.model_validate_json(payload) if payload else None

    def list_master_changes(self) -> list[MasterChangeRequest]:
        with self._lock:
            payloads = [
                self._master_changes[k] for k in sorted(self._master_changes)
            ]
        return [MasterChangeRequest.model_validate_json(p) for p in payloads]

    def save_bundle(self, bundle: BundleResult) -> None:
        with self._lock:
            self._bundles[bundle.bundle_id] = bundle.model_dump_json()

    def get_bundle(self, bundle_id: str) -> BundleResult | None:
        with self._lock:
            payload = self._bundles.get(bundle_id)
        return BundleResult.model_validate_json(payload) if payload else None

    def save_digest(self, digest: EscalationDigest) -> None:
        with self._lock:
            self._digests[digest.as_of] = digest.model_dump_json()

    def get_digest(self, digest_date: str) -> EscalationDigest | None:
        with self._lock:
            payload = self._digests.get(digest_date)
        return EscalationDigest.model_validate_json(payload) if payload else None

    def list_digests(self) -> list[EscalationDigest]:
        with self._lock:
            payloads = [self._digests[k] for k in sorted(self._digests)]
        return [EscalationDigest.model_validate_json(p) for p in payloads]

    def save_handoff_profile(self, profile: HandoffProfile) -> None:
        with self._lock:
            self._handoff_profiles[profile.profile_id] = (
                profile.model_dump_json()
            )

    def get_handoff_profile(self, profile_id: str) -> HandoffProfile | None:
        with self._lock:
            payload = self._handoff_profiles.get(profile_id)
        return HandoffProfile.model_validate_json(payload) if payload else None

    def list_handoff_profiles(self) -> list[HandoffProfile]:
        with self._lock:
            payloads = [
                self._handoff_profiles[k] for k in sorted(self._handoff_profiles)
            ]
        return [HandoffProfile.model_validate_json(p) for p in payloads]

    def get(self, invoice_id: str) -> ApprovalRecord | None:
        with self._lock:
            return self._records.get(invoice_id)

    def list_records(self) -> list[ApprovalRecord]:
        return self._all()

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

    def list_records(self) -> list[ApprovalRecord]:
        return [_load_record(payload) for payload in self._all_payloads()]

    def save_reconciliation(self, report: StatementReconciliation) -> None:
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "INSERT INTO reconciliations "
                    "(statement_id, payload, confirmed, updated_at) "
                    "VALUES (%s, %s::jsonb, %s, now()) "
                    "ON CONFLICT (statement_id) DO UPDATE SET "
                    "payload = EXCLUDED.payload, "
                    "confirmed = EXCLUDED.confirmed, updated_at = now()",
                    (
                        report.statement_id,
                        report.model_dump_json(),
                        report.confirmed_by is not None,
                    ),
                )
            conn.commit()

    def get_reconciliation(
        self, statement_id: str
    ) -> StatementReconciliation | None:
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT payload FROM reconciliations WHERE statement_id = %s",
                    (statement_id,),
                )
                row = cur.fetchone()
        if not row:
            return None
        payload = row["payload"]
        return StatementReconciliation.model_validate(
            json.loads(payload) if isinstance(payload, str) else payload
        )

    # --- Round-3 artefacts: payment runs, recovery packs, bundles.
    # Same pattern as reconciliations: one row per id, the full
    # object in the JSONB payload; the artefact tables never join
    # back into approvals.

    def _save_blob(self, table: str, key_col: str, key: str, payload: str,
                   confirmed: bool | None = None) -> None:
        with self._connect() as conn:
            with conn.cursor() as cur:
                if confirmed is None:
                    cur.execute(
                        f"INSERT INTO {table} ({key_col}, payload, updated_at) "
                        "VALUES (%s, %s::jsonb, now()) "
                        f"ON CONFLICT ({key_col}) DO UPDATE SET "
                        "payload = EXCLUDED.payload, updated_at = now()",
                        (key, payload),
                    )
                else:
                    cur.execute(
                        f"INSERT INTO {table} ({key_col}, payload, confirmed, updated_at) "
                        "VALUES (%s, %s::jsonb, %s, now()) "
                        f"ON CONFLICT ({key_col}) DO UPDATE SET "
                        "payload = EXCLUDED.payload, "
                        "confirmed = EXCLUDED.confirmed, updated_at = now()",
                        (key, payload, confirmed),
                    )
            conn.commit()

    def _get_blob(self, table: str, key_col: str, key: str):
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    f"SELECT payload FROM {table} WHERE {key_col} = %s", (key,)
                )
                row = cur.fetchone()
        if not row:
            return None
        payload = row["payload"]
        return json.loads(payload) if isinstance(payload, str) else payload

    def _all_blobs(self, table: str) -> list:
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(f"SELECT payload FROM {table} ORDER BY seq")
                rows = cur.fetchall()
        return [
            json.loads(r["payload"]) if isinstance(r["payload"], str) else r["payload"]
            for r in rows
        ]

    def save_payment_run(self, run: PaymentRun) -> None:
        self._save_blob(
            "payment_runs", "run_id", run.run_id, run.model_dump_json(),
            confirmed=run.status == "confirmed",
        )

    def get_payment_run(self, run_id: str) -> PaymentRun | None:
        data = self._get_blob("payment_runs", "run_id", run_id)
        return PaymentRun.model_validate(data) if data is not None else None

    def list_payment_runs(self) -> list[PaymentRun]:
        runs = [PaymentRun.model_validate(d) for d in self._all_blobs("payment_runs")]
        return sorted(runs, key=lambda r: r.run_id)

    def save_recovery_pack(self, pack: RecoveryPack) -> None:
        self._save_blob(
            "recovery_packs", "invoice_id", pack.duplicate_invoice_id,
            pack.model_dump_json(),
        )

    def get_recovery_pack(self, invoice_id: str) -> RecoveryPack | None:
        data = self._get_blob("recovery_packs", "invoice_id", invoice_id)
        return RecoveryPack.model_validate(data) if data is not None else None

    def list_recovery_packs(self) -> list[RecoveryPack]:
        packs = [
            RecoveryPack.model_validate(d)
            for d in self._all_blobs("recovery_packs")
        ]
        return sorted(packs, key=lambda p: p.duplicate_invoice_id)

    def save_master_change(self, change: MasterChangeRequest) -> None:
        self._save_blob(
            "master_change_requests", "request_id", change.request_id,
            change.model_dump_json(),
        )

    def get_master_change(self, request_id: str) -> MasterChangeRequest | None:
        data = self._get_blob("master_change_requests", "request_id", request_id)
        return MasterChangeRequest.model_validate(data) if data is not None else None

    def list_master_changes(self) -> list[MasterChangeRequest]:
        changes = [
            MasterChangeRequest.model_validate(d)
            for d in self._all_blobs("master_change_requests")
        ]
        return sorted(changes, key=lambda c: c.request_id)

    def save_bundle(self, bundle: BundleResult) -> None:
        self._save_blob(
            "bundles", "bundle_id", bundle.bundle_id, bundle.model_dump_json()
        )

    def get_bundle(self, bundle_id: str) -> BundleResult | None:
        data = self._get_blob("bundles", "bundle_id", bundle_id)
        return BundleResult.model_validate(data) if data is not None else None

    # --- Round-10 artefacts: stored digests. Same blob pattern,
    # keyed by the digest's date; the first-computed copy is the
    # one kept (the service enforces first-wins on save).

    def save_digest(self, digest: EscalationDigest) -> None:
        self._save_blob(
            "digests", "digest_date", digest.as_of, digest.model_dump_json()
        )

    def get_digest(self, digest_date: str) -> EscalationDigest | None:
        data = self._get_blob("digests", "digest_date", digest_date)
        return EscalationDigest.model_validate(data) if data is not None else None

    def list_digests(self) -> list[EscalationDigest]:
        digests = [
            EscalationDigest.model_validate(d)
            for d in self._all_blobs("digests")
        ]
        return sorted(digests, key=lambda d: d.as_of)

    # --- Round-12 artefacts: governed hand-off routing profiles
    # (migration 0011). Same blob pattern, keyed by profile id;
    # the audit trail rides inside the payload, append-only.

    def save_handoff_profile(self, profile: HandoffProfile) -> None:
        self._save_blob(
            "handoff_profiles",
            "profile_id",
            profile.profile_id,
            profile.model_dump_json(),
        )

    def get_handoff_profile(self, profile_id: str) -> HandoffProfile | None:
        data = self._get_blob("handoff_profiles", "profile_id", profile_id)
        return HandoffProfile.model_validate(data) if data is not None else None

    def list_handoff_profiles(self) -> list[HandoffProfile]:
        profiles = [
            HandoffProfile.model_validate(d)
            for d in self._all_blobs("handoff_profiles")
        ]
        return sorted(profiles, key=lambda p: p.profile_id)

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
