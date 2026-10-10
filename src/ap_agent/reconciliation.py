"""Supplier statement reconciliation — computed in code, report only.

A supplier statement is the supplier's view of what is open. AP's
view is the processed invoices in the store. Reconciliation lines
the two up by supplier invoice number and classifies every line:

- ``matched`` — on the statement and processed, amounts agree.
- ``amount_differs`` — both sides have it; the amounts do not agree.
- ``missing_from_ap`` — the supplier lists it; AP never processed it
  (never received, or received and not yet analysed).
- ``missing_from_statement`` — AP processed it; the supplier does not
  list it (already paid, credited, or the statement is stale).

The comparison is arithmetic over stored history entries — the model
computes none of it. The summary prose is composed deterministically
from the computed lines (``summary_source: template``): in this
surface the agent drafts nothing free-form, because a reconciliation
summary that invents a figure is worse than a plain one. A human
confirms the report (service layer); confirming changes the report
record only — never an invoice.
"""

from __future__ import annotations

from .schemas import (
    StatementLineResult,
    StatementReconciliation,
    SupplierStatement,
)

_AMOUNT_TOLERANCE = 0.01


def reconcile_statement(
    statement: SupplierStatement, history: list[dict]
) -> StatementReconciliation:
    """Reconcile one statement against history entries (store.py's
    ``history_entry`` projection) for the statement's supplier."""
    entries = [
        e
        for e in history
        if not statement.supplier_name
        or e.get("supplier_name") == statement.supplier_name
    ]
    by_number: dict[str, dict] = {}
    for entry in entries:
        number = entry.get("invoice_number") or ""
        if number and number not in by_number:
            by_number[number] = entry

    lines: list[StatementLineResult] = []
    matched = differs = missing_ap = 0
    statement_total = 0.0
    seen_numbers: set[str] = set()
    for line in statement.lines:
        seen_numbers.add(line.invoice_number)
        if line.amount is not None:
            statement_total += float(line.amount)
        entry = by_number.get(line.invoice_number)
        if entry is None:
            missing_ap += 1
            lines.append(
                StatementLineResult(
                    invoice_number=line.invoice_number,
                    status="missing_from_ap",
                    statement_amount=line.amount,
                )
            )
            continue
        ap_amount = entry.get("total_amount")
        difference = None
        if line.amount is not None and ap_amount is not None:
            difference = round(float(line.amount) - float(ap_amount), 2)
        if difference is not None and abs(difference) > _AMOUNT_TOLERANCE:
            differs += 1
            status = "amount_differs"
        else:
            matched += 1
            status = "matched"
        lines.append(
            StatementLineResult(
                invoice_number=line.invoice_number,
                status=status,
                statement_amount=line.amount,
                ap_amount=float(ap_amount) if ap_amount is not None else None,
                ap_invoice_id=entry.get("invoice_id", ""),
                difference=difference,
            )
        )

    missing_from_statement = [
        {
            "invoice_id": e.get("invoice_id", ""),
            "invoice_number": e.get("invoice_number", ""),
            "total_amount": e.get("total_amount"),
            "outcome": e.get("outcome", ""),
        }
        for e in entries
        if (e.get("invoice_number") or "") not in seen_numbers
        and e.get("invoice_number")
    ]
    ap_total = round(
        sum(
            float(e["total_amount"])
            for e in entries
            if e.get("total_amount") is not None and e.get("invoice_number")
        ),
        2,
    )

    report = StatementReconciliation(
        statement_id=statement.statement_id,
        supplier_name=statement.supplier_name,
        statement_date=statement.statement_date,
        currency=statement.currency,
        lines=lines,
        missing_from_statement=missing_from_statement,
        matched_count=matched,
        differs_count=differs,
        missing_from_ap_count=missing_ap,
        missing_from_statement_count=len(missing_from_statement),
        statement_total=round(statement_total, 2),
        ap_total=ap_total,
    )
    report.summary = _compose_summary(report)
    return report


def _compose_summary(report: StatementReconciliation) -> str:
    """The drafted summary a human confirms — composed from the
    computed lines, with every exception named."""
    parts = [
        f"Statement {report.statement_id} from {report.supplier_name or 'the supplier'}"
        + (f" dated {report.statement_date}" if report.statement_date else "")
        + f": {len(report.lines)} lines totalling {report.statement_total:.2f} "
        f"{report.currency} against {report.ap_total:.2f} processed by AP. "
        f"{report.matched_count} matched, {report.differs_count} differ in "
        f"amount, {report.missing_from_ap_count} are on the statement but "
        f"were never processed by AP, and {report.missing_from_statement_count} "
        "processed invoices do not appear on the statement."
    ]
    for line in report.lines:
        if line.status == "amount_differs":
            parts.append(
                f"{line.invoice_number}: statement {line.statement_amount:.2f} "
                f"vs AP {line.ap_amount:.2f} (difference {line.difference:+.2f})."
            )
        elif line.status == "missing_from_ap":
            parts.append(
                f"{line.invoice_number}: on the statement at "
                f"{(line.statement_amount or 0):.2f}, never processed by AP."
            )
    for entry in report.missing_from_statement:
        parts.append(
            f"{entry['invoice_number']}: processed by AP "
            f"({entry['invoice_id']}, {(entry['total_amount'] or 0):.2f}) but "
            "absent from the statement."
        )
    if (
        report.differs_count == 0
        and report.missing_from_ap_count == 0
        and report.missing_from_statement_count == 0
    ):
        parts.append("No exceptions: the statement and AP's records agree.")
    return " ".join(parts)
