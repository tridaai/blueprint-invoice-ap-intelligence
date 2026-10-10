"""Supplier master data: the out-of-band record the fraud screen
checks bank details against (AP-BANK-01).

Until now the bank-change screen was pattern-only: change language
in a document ("remit to our new account …") was the signal. That
catches the wording, but it cannot tell a fraudster's new account
from the account AP already has on file. This module is the port's
implementations plus the comparison the screen node runs:

- Bank identifiers are extracted from the invoice's text **only in
  bank context** (sentences that talk about bank / account / remit /
  sort code / IBAN — an invoice total or a date is not a bank
  detail), normalised, and compared with the master record's
  identifiers, kind by kind.
- A stated identifier that is **not** the master record is the fraud
  signal (``checked_mismatch``) — it blocks the case like any
  bank-change hit.
- Stated identifiers that all match are **evidence**
  (``checked_match``): a pattern hit whose details turn out to be
  the ones already on file is downgraded by the screen — the change
  language pointed at the approved account.
- No master record, or no bank details stated: the screen keeps its
  pattern-only behaviour, and the check says which case it was.

Comparisons report masked labels only (kind + last four). The full
identifiers live in the master record and in the supplier's own
documents — never in flags, traces, or drafts.

Implementations:

- ``PostgresSupplierMaster`` — production: one row per supplier in
  the ``supplier_master`` table (migration 0008), JSONB payload,
  psycopg per-call connections like the store. Starts empty: the
  customer loads their verified master data.
- ``InMemorySupplierMaster`` — the test double. The factory seeds it
  with the synthetic sample suppliers so the offline demo/tests
  exercise the comparison; that seed is demo data, labelled as such.
"""

from __future__ import annotations

import json
import re
import threading
from typing import Iterable

from .config import env_str, load_dotenv
from .schemas import BankMasterCheck, InvoiceInput, SupplierMasterRecord
from .screening import _all_text_sources

_BANK_CONTEXT = re.compile(
    r"\b(bank|account|remit|remittance|iban|sort code|routing)\b", re.I
)
_IBAN = re.compile(r"\b[A-Z]{2}\d{2}[A-Z0-9 ]{10,34}\b")
_SORT_CODE = re.compile(r"\b\d{2}-\d{2}-\d{2}\b")
_ACCOUNT = re.compile(r"(?<!\d)\d{8,17}(?!\d)")
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+|\n+")


def _digits(value: str) -> str:
    return re.sub(r"\D", "", value or "")


def _iban_norm(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9]", "", value or "").upper()


def _key(supplier_name: str) -> str:
    return " ".join((supplier_name or "").split()).casefold()


def master_identifiers(record: SupplierMasterRecord) -> dict[str, set[str]]:
    """The master record's bank identifiers, normalised, by kind."""
    ids: dict[str, set[str]] = {}
    if record.account_number and _digits(record.account_number):
        ids.setdefault("account", set()).add(_digits(record.account_number))
    if record.sort_code and _digits(record.sort_code):
        ids.setdefault("sort_code", set()).add(_digits(record.sort_code))
    if record.iban and _iban_norm(record.iban):
        ids.setdefault("iban", set()).add(_iban_norm(record.iban))
    return ids


def extract_bank_identifiers(text: str) -> dict[str, set[str]]:
    """Bank identifiers stated in ``text``, normalised, by kind.

    Only bank-context sentences are read — the sentence must talk
    about bank details, so an invoice number, a date, or a total in
    ordinary prose is never mistaken for an account number.
    """
    found: dict[str, set[str]] = {}
    for sentence in _SENTENCE_SPLIT.split(text or ""):
        if not _BANK_CONTEXT.search(sentence):
            continue
        for match in _IBAN.finditer(sentence):
            iban = _iban_norm(match.group(0))
            if 15 <= len(iban) <= 34:
                found.setdefault("iban", set()).add(iban)
        for match in _SORT_CODE.finditer(sentence):
            found.setdefault("sort_code", set()).add(_digits(match.group(0)))
        for match in _ACCOUNT.finditer(sentence):
            found.setdefault("account", set()).add(match.group(0))
    # A sort code's digit run ("20-44-18") is not an account number,
    # and an IBAN's numeric tail is not one either: drop account
    # candidates that are just a fragment of a longer identifier.
    accounts = found.get("account", set())
    longer = set().union(*[v for k, v in found.items() if k != "account"]) if len(found) > 1 else set()
    if accounts and longer:
        found["account"] = {
            a for a in accounts if not any(a in other and a != other for other in longer)
        }
        if not found["account"]:
            del found["account"]
    return found


def _masked(kind: str, value: str) -> str:
    return f"{kind} ••••{value[-4:]}"


def compare_bank_details(
    invoice: InvoiceInput,
    record: SupplierMasterRecord | None,
    *,
    source: str = "",
) -> BankMasterCheck:
    """Compare the bank identifiers the invoice's text states with
    the supplier's master record. Pure — the screen node owns what
    the verdict does to the fraud flags."""
    if record is None:
        return BankMasterCheck(
            status="no_master_record",
            supplier_name=invoice.supplier_name,
            source=source,
        )
    stated: dict[str, set[str]] = {}
    for _field, text in _all_text_sources(invoice):
        for kind, values in extract_bank_identifiers(text).items():
            stated.setdefault(kind, set()).update(values)
    if not stated:
        return BankMasterCheck(
            status="no_details_in_documents",
            supplier_name=invoice.supplier_name,
            source=source,
        )
    master = master_identifiers(record)
    matched: list[str] = []
    mismatched: list[str] = []
    for kind in sorted(stated):
        for value in sorted(stated[kind]):
            if value in master.get(kind, set()):
                matched.append(_masked(kind, value))
            else:
                mismatched.append(_masked(kind, value))
    return BankMasterCheck(
        status="checked_mismatch" if mismatched else "checked_match",
        supplier_name=invoice.supplier_name,
        source=source,
        matched=matched,
        mismatched=mismatched,
    )


def mismatch_flag(check: BankMasterCheck) -> str:
    """The fraud flag for a master mismatch. Masked labels only —
    the full identifiers stay in the master record and the case file."""
    return (
        "bank_details_mismatch vs supplier master: document states "
        + ", ".join(check.mismatched)
        + " — not the approved master record [AP-BANK-01]"
    )


def sample_supplier_master() -> list[SupplierMasterRecord]:
    """The synthetic sample suppliers' master records (demo seed).

    Harborline's on-file account is deliberately NOT the account its
    INV-1010 note asks AP to remit to — that note is the fraud case,
    and with master data present it now resolves as a checked
    mismatch instead of a pattern-only hit."""
    return [
        SupplierMasterRecord(
            supplier_name="Northbeam Office Supplies",
            bank_name="First Charter Bank",
            account_number="1029384756",
            sort_code="30-12-44",
            payment_terms="2/10 net 30",
            tax_id="US-TAX-4471",
        ),
        SupplierMasterRecord(
            supplier_name="Harborline Packaging",
            bank_name="Harbor Bank",
            account_number="5520109873",
            sort_code="20-44-18",
            payment_terms="net 30",
            tax_id="US-TAX-0885",
        ),
    ]


class InMemorySupplierMaster:
    """Process-local supplier master — the test double, and the
    offline default the factory seeds with the sample records."""

    name = "memory"

    def __init__(self, records: Iterable[SupplierMasterRecord] = ()) -> None:
        self._records: dict[str, SupplierMasterRecord] = {}
        self._lock = threading.Lock()
        for record in records:
            self._records[_key(record.supplier_name)] = record

    def get_supplier(self, supplier_name: str) -> SupplierMasterRecord | None:
        with self._lock:
            return self._records.get(_key(supplier_name))

    def upsert_supplier(self, record: SupplierMasterRecord) -> None:
        with self._lock:
            self._records[_key(record.supplier_name)] = record


class PostgresSupplierMaster:
    """PostgreSQL-backed supplier master: one row per supplier in
    ``supplier_master`` (migration 0008), the full record in the
    JSONB payload. Per-call connections, like the store. Migrations
    run on construction (idempotent)."""

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

    def get_supplier(self, supplier_name: str) -> SupplierMasterRecord | None:
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT payload FROM supplier_master WHERE supplier_key = %s",
                    (_key(supplier_name),),
                )
                row = cur.fetchone()
        if not row:
            return None
        payload = row["payload"]
        return SupplierMasterRecord.model_validate(
            json.loads(payload) if isinstance(payload, str) else payload
        )

    def upsert_supplier(self, record: SupplierMasterRecord) -> None:
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "INSERT INTO supplier_master (supplier_key, payload, updated_at) "
                    "VALUES (%s, %s::jsonb, now()) "
                    "ON CONFLICT (supplier_key) DO UPDATE SET "
                    "payload = EXCLUDED.payload, updated_at = now()",
                    (_key(record.supplier_name), record.model_dump_json()),
                )
            conn.commit()


def get_supplier_master() -> InMemorySupplierMaster | PostgresSupplierMaster:
    """The supplier master from the environment: PostgreSQL when
    ``DATABASE_URL`` is set (production; starts empty — the customer
    loads their verified records), otherwise the in-memory double
    seeded with the synthetic sample suppliers, so the offline
    pipeline exercises the master comparison too."""
    load_dotenv()
    url = env_str("DATABASE_URL")
    if url:
        return PostgresSupplierMaster(url)
    return InMemorySupplierMaster(sample_supplier_master())
