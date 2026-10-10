"""Governed supplier-master changes: the proposal half.

The fraud screen (AP-BANK-01) trusts the supplier master — which
makes the master's change path a control in its own right. This
module holds the pure pieces of that path; the service layer owns
the workflow (propose → second-operator decision → apply):

- Request ids are sequential and readable (``MCR-0001`` …), minted
  from the ids already in the store.
- A change can be proposed **from a document**: the bank identifiers
  a processed invoice's text states are extracted the same way the
  screen extracts them (bank-context sentences only), and overlaid
  on the supplier's current record — terms, tax id, and unstated
  identifiers carry over, so the proposal is exactly the delta the
  document asks for. What the document states never applies itself:
  it becomes a pending request a second operator must approve with
  an out-of-band verification reference.
- Display and audit notes use masked labels (kind + last four),
  like every other master comparison in the codebase — full
  identifiers live in the master record and the change request,
  never in prose.
"""

from __future__ import annotations

import re

from .schemas import InvoiceInput, SupplierMasterRecord
from .screening import _all_text_sources
from .supplier_master import _masked, extract_bank_identifiers, master_identifiers

_ID_SUFFIX = re.compile(r"(\d+)$")


def next_request_id(existing_ids: list[str]) -> str:
    """The next sequential change-request id: ``MCR-0001`` …"""
    highest = 0
    for request_id in existing_ids:
        match = _ID_SUFFIX.search(request_id or "")
        if match:
            highest = max(highest, int(match.group(1)))
    return f"MCR-{highest + 1:04d}"


def record_labels(record: SupplierMasterRecord | None) -> list[str]:
    """A master record's bank identifiers as masked labels."""
    if record is None:
        return []
    ids = master_identifiers(record)
    return [
        _masked(kind, value)
        for kind in sorted(ids)
        for value in sorted(ids[kind])
    ]


def proposed_from_invoice(
    invoice: InvoiceInput,
    current: SupplierMasterRecord | None,
) -> tuple[SupplierMasterRecord, list[str]]:
    """Build the proposed record from what an invoice's documents
    state, overlaid on the supplier's current master record.

    Returns ``(proposed, notes)``. Raises ``ValueError`` when the
    documents state no bank identifiers at all — there is then no
    change to govern. When a document states several identifiers of
    one kind (an old account and a new one), the first in sorted
    order is proposed and the note says so: telling old from new is
    exactly what the approver's out-of-band verification is for.
    """
    stated: dict[str, set[str]] = {}
    for _field, text in _all_text_sources(invoice):
        for kind, values in extract_bank_identifiers(text).items():
            stated.setdefault(kind, set()).update(values)
    if not stated:
        raise ValueError(
            f"Invoice {invoice.invoice_id} states no bank identifiers "
            "in bank context — there is no bank-detail change to propose"
        )
    proposed = (
        current.model_copy(deep=True)
        if current is not None
        else SupplierMasterRecord(supplier_name=invoice.supplier_name)
    )
    proposed.supplier_name = invoice.supplier_name or proposed.supplier_name
    notes: list[str] = []
    by_kind = {
        "account": "account_number",
        "sort_code": "sort_code",
        "iban": "iban",
    }
    for kind in sorted(stated):
        values = sorted(stated[kind])
        if len(values) > 1:
            notes.append(
                f"the document states {len(values)} {kind} identifiers; "
                f"{_masked(kind, values[0])} is proposed — the out-of-band "
                "verification must confirm which is the supplier's"
            )
        setattr(proposed, by_kind[kind], values[0])
    return proposed, notes
