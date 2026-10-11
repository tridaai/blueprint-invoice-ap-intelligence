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
- ``credit_line`` — a credit the supplier shows against the account
  on the statement. It is not an invoice and is never matched as
  one; it enters the recomputed total signed, and it is counted
  and named in the summary.

A line that arrived through document intake carries the confidence
of its read; a line read below the OCR trust threshold is flagged
on the report (``low_confidence``) — the figure stands and is
reconciled like any other, but the approver is told to verify it
against the document before acting on it.

Lines are also **duplicate-screened**: a statement is a classic
duplicate-payment path, so a line is flagged when AP holds two
records for its number (an original and its resend — whichever
shape each arrived in), when the matched record is itself a
suspected duplicate, or when the statement lists the same number
twice. The flag rides alongside the match status and names the
twin records; the recovery path for the underlying pair is the
same one the invoice gate uses.

The statement's own printed total is never trusted either: when a
statement carries one, it is compared against the recomputed sum of
its lines and any disagreement is reported as an exception in its
own right — and no line is ever reclassified to make the totals
agree. The report also carries a drafted supplier query letter for
the exceptions (``draft_supplier_query``): deterministic prose over
the computed lines, labelled a draft, sent by nobody — a human
reviews it alongside the report they confirm.

The comparison is arithmetic over stored history entries — the model
computes none of it. Numbers are matched on the store's canonical
duplicate key, so a number the statement writes as ``NB 4471``
still lines up with the ``NB-4471`` AP processed. The summary prose
is composed deterministically from the computed lines
(``summary_source: template``): in this surface the agent drafts
nothing free-form, because a reconciliation summary that invents a
figure is worse than a plain one. A human confirms the report
(service layer); confirming changes the report record only — never
an invoice.
"""

from __future__ import annotations

from .ocr import LOW_CONFIDENCE_THRESHOLD
from .schemas import (
    StatementLineResult,
    StatementReconciliation,
    SupplierStatement,
)
from .store import duplicate_key

_AMOUNT_TOLERANCE = 0.01


def _read_flag(line) -> tuple[float | None, bool]:
    """The (confidence, low_confidence) of a statement line's
    document read. Structured lines carry no read and claim none."""
    confidence = line.confidence
    if confidence is None:
        return None, False
    return float(confidence), float(confidence) < LOW_CONFIDENCE_THRESHOLD


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
    by_number: dict[str, list[dict]] = {}
    for entry in entries:
        number = entry.get("invoice_number") or ""
        if number:
            key = duplicate_key(
                str(entry.get("supplier_name", "")), number
            )[1]
            by_number.setdefault(key, []).append(entry)

    lines: list[StatementLineResult] = []
    matched = differs = missing_ap = duplicates = 0
    credit_lines = 0
    credit_total = 0.0
    low_confidence_lines = 0
    statement_total = 0.0
    seen_numbers: set[str] = set()
    for line in statement.lines:
        if line.amount is not None:
            statement_total += float(line.amount)
        confidence, low_confidence = _read_flag(line)
        if low_confidence:
            low_confidence_lines += 1
        if line.is_credit:
            # A credit the supplier shows against the account: it is
            # not an invoice, so it is never matched as one and never
            # marks an invoice as seen on the statement. Its signed
            # amount already entered the recomputed total above.
            credit_lines += 1
            if line.amount is not None:
                credit_total += float(line.amount)
            lines.append(
                StatementLineResult(
                    invoice_number=line.invoice_number,
                    status="credit_line",
                    statement_amount=line.amount,
                    confidence=confidence,
                    low_confidence=low_confidence,
                )
            )
            continue
        key = duplicate_key(statement.supplier_name, line.invoice_number)[1]
        repeated = key in seen_numbers and bool(key)
        seen_numbers.add(key)
        group = by_number.get(key) or []
        entry = group[0] if group else None
        if entry is None:
            missing_ap += 1
            lines.append(
                StatementLineResult(
                    invoice_number=line.invoice_number,
                    status="missing_from_ap",
                    statement_amount=line.amount,
                    confidence=confidence,
                    low_confidence=low_confidence,
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
        # Duplicate screening: the twin evidence rides on the line.
        duplicate_suspected = False
        duplicate_invoice_id = ""
        duplicate_of = ""
        if len(group) > 1:
            twin = next(
                (
                    e
                    for e in group[1:]
                    if e.get("duplicate_of") or e.get("outcome") == "duplicate_suspected"
                ),
                group[1],
            )
            duplicate_suspected = True
            duplicate_invoice_id = twin.get("invoice_id", "")
            duplicate_of = entry.get("invoice_id", "")
        elif entry.get("duplicate_of"):
            duplicate_suspected = True
            duplicate_invoice_id = entry.get("invoice_id", "")
            duplicate_of = str(entry.get("duplicate_of"))
        if repeated:
            duplicate_suspected = True
            if not duplicate_invoice_id:
                duplicate_invoice_id = entry.get("invoice_id", "")
        if duplicate_suspected:
            duplicates += 1
        lines.append(
            StatementLineResult(
                invoice_number=line.invoice_number,
                status=status,
                statement_amount=line.amount,
                ap_amount=float(ap_amount) if ap_amount is not None else None,
                ap_invoice_id=entry.get("invoice_id", ""),
                difference=difference,
                duplicate_suspected=duplicate_suspected,
                duplicate_invoice_id=duplicate_invoice_id,
                duplicate_of=duplicate_of,
                confidence=confidence,
                low_confidence=low_confidence,
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
        if e.get("invoice_number")
        and duplicate_key(str(e.get("supplier_name", "")), e["invoice_number"])[1]
        not in seen_numbers
    ]
    ap_total = round(
        sum(
            float(e["total_amount"])
            for e in entries
            if e.get("total_amount") is not None and e.get("invoice_number")
        ),
        2,
    )

    claimed_total = (
        round(float(statement.claimed_total), 2)
        if statement.claimed_total is not None
        else None
    )
    claimed_difference = (
        round(claimed_total - round(statement_total, 2), 2)
        if claimed_total is not None
        else None
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
        duplicate_count=duplicates,
        credit_line_count=credit_lines,
        credit_total=round(credit_total, 2),
        low_confidence_count=low_confidence_lines,
        statement_total=round(statement_total, 2),
        ap_total=ap_total,
        claimed_total=claimed_total,
        claimed_total_difference=claimed_difference,
    )
    report.summary = _compose_summary(report)
    report.supplier_query_draft = draft_supplier_query(report)
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
        + (
            f" {report.duplicate_count} line(s) are flagged as suspected "
            "duplicates."
            if report.duplicate_count
            else ""
        )
    ]
    if (
        report.claimed_total_difference is not None
        and abs(report.claimed_total_difference) > _AMOUNT_TOLERANCE
    ):
        parts.append(
            f"The statement's own total of {report.claimed_total:.2f} "
            f"disagrees with the sum of its lines "
            f"({report.statement_total:.2f}) by "
            f"{report.claimed_total_difference:+.2f}; the recomputed "
            "figure is used throughout this report."
        )
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
        if line.duplicate_suspected and line.duplicate_of:
            parts.append(
                f"{line.invoice_number}: AP holds two records for this "
                f"invoice — {line.duplicate_invoice_id} is a suspected "
                f"duplicate of {line.duplicate_of}; paying this line "
                "again would pay it twice."
            )
        elif line.duplicate_suspected:
            parts.append(
                f"{line.invoice_number}: flagged as a suspected duplicate "
                f"(see {line.duplicate_invoice_id or line.ap_invoice_id}) — "
                "hold the line until the twin is resolved."
            )
    for entry in report.missing_from_statement:
        parts.append(
            f"{entry['invoice_number']}: processed by AP "
            f"({entry['invoice_id']}, {(entry['total_amount'] or 0):.2f}) but "
            "absent from the statement."
        )
    if report.credit_line_count:
        parts.append(
            f"The statement also shows {report.credit_line_count} credit "
            f"line(s) totalling {report.credit_total:.2f} — credits "
            "against the account, reconciled as credits, not as invoices "
            "to pay."
        )
    if report.low_confidence_count:
        flagged = ", ".join(
            line.invoice_number
            for line in report.lines
            if line.low_confidence
        )
        parts.append(
            f"{report.low_confidence_count} line(s) were read from the "
            f"document at low confidence ({flagged}) — the figures are "
            "reconciled as read, but verify them against the document "
            "itself before acting on them."
        )
    if (
        report.differs_count == 0
        and report.missing_from_ap_count == 0
        and report.missing_from_statement_count == 0
        and report.duplicate_count == 0
        and (
            report.claimed_total_difference is None
            or abs(report.claimed_total_difference) <= _AMOUNT_TOLERANCE
        )
    ):
        parts.append("No exceptions: the statement and AP's records agree.")
    return " ".join(parts)


def draft_supplier_query(report: StatementReconciliation) -> str:
    """The supplier query letter for a reconciliation's exceptions —
    drafted deterministically from the computed lines, every figure
    traced to the report. It is a draft for the human who confirms
    the reconciliation: nothing in this blueprint sends it. A clean
    statement needs no query, so the draft is empty."""
    exceptions = (
        report.differs_count
        + report.missing_from_ap_count
        + report.missing_from_statement_count
        + report.duplicate_count
    )
    claimed_off = (
        report.claimed_total_difference is not None
        and abs(report.claimed_total_difference) > _AMOUNT_TOLERANCE
    )
    if exceptions == 0 and not claimed_off:
        return ""
    cur = report.currency
    out = [
        "DRAFT — supplier query letter (not sent)",
        f"To: {report.supplier_name or 'Supplier'}",
        f"Subject: Statement {report.statement_id}"
        + (f" dated {report.statement_date}" if report.statement_date else "")
        + " — items we could not reconcile",
        "",
        "We reconciled your statement against our processed invoices. "
        "The items below need your help before the affected amounts "
        "can be settled:",
        "",
    ]
    for line in report.lines:
        if line.status == "missing_from_ap":
            out.append(
                f"- {line.invoice_number}: listed on your statement at "
                f"{(line.statement_amount or 0):.2f} {cur}, but we have no "
                "processed invoice with this number. Please send a copy "
                "of the invoice."
            )
        elif line.status == "amount_differs":
            out.append(
                f"- {line.invoice_number}: your statement shows "
                f"{line.statement_amount:.2f} {cur}; the invoice we "
                f"processed ({line.ap_invoice_id}) totals "
                f"{line.ap_amount:.2f} {cur} — a difference of "
                f"{line.difference:+.2f}. Please confirm which figure "
                "is correct, or issue a credit note for the difference."
            )
        if line.duplicate_suspected:
            out.append(
                f"- {line.invoice_number}: this number appears more "
                "than once across your statement and our records, so "
                "the line is held until the duplicate is resolved — "
                "please confirm it was billed once only."
            )
    for entry in report.missing_from_statement:
        out.append(
            f"- {entry['invoice_number']}: we processed this invoice "
            f"({entry['invoice_id']}, {(entry['total_amount'] or 0):.2f} "
            f"{cur}) but it does not appear on your statement. Please "
            "confirm whether it is already settled or credited at your "
            "end."
        )
    if claimed_off:
        out.append(
            f"- Statement total: your statement prints a total of "
            f"{report.claimed_total:.2f} {cur}, but its lines sum to "
            f"{report.statement_total:.2f} {cur}. Please confirm the "
            "correct total."
        )
    out += [
        "",
        "Amounts not listed here reconciled exactly. Thank you for "
        "your help.",
        "",
        "Accounts Payable",
    ]
    return "\n".join(out)
