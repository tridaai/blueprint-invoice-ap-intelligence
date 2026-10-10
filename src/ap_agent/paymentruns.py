"""Payment-run proposals — computed in code over the approved records.

This is how AP actually pays: not invoice by invoice, but in a run —
the approved invoices grouped into a pay list a finance lead
confirms. The agent's part stops exactly where its authority stops:

- Eligibility is a fact from the store: the invoice's human decision
  is ``approved``. Nothing awaiting, rejected, or already inside a
  confirmed run is proposed again.
- Grouping is arithmetic: by currency, then by due window —
  consecutive ``window_days``-long windows anchored on ``as_of``,
  with overdue invoices swept into the first window.
- Totals are summed in code; the discount column reuses each run's
  stored discount evaluation, and counts only while its deadline is
  still reachable inside the window.
- A run is a draft object. Confirming it (service layer) records a
  human's acceptance of the proposal; it pays nothing, moves no
  money, and changes no invoice's state — the customer's AP system
  of record executes the run, as with everything this agent drafts.
- Large runs carry a four-eyes rule: a net total above the
  configured threshold stamps the run ``four_eyes_required``, and
  the hand-off export (service layer) refuses it until a second,
  different confirmer has also confirmed the run.
"""

from __future__ import annotations

from datetime import date, timedelta

from .matching import _parse_date
from .schemas import PaymentRun, PaymentRunLine
from .store import ApprovalRecord

DEFAULT_WINDOW_DAYS = 7


def _window_index(due: date, as_of: date, window_days: int) -> int:
    """Which due window a date falls in. Overdue and in-window dates
    share window 0; later dates step forward a window at a time."""
    delta = (due - as_of).days
    if delta < window_days:
        return 0
    return delta // window_days


def _line_for(record: ApprovalRecord, as_of: date) -> PaymentRunLine:
    result = record.result
    invoice = record.invoice or {}
    gross = float(invoice.get("total_amount") or 0.0)
    discount_amount = 0.0
    discount_deadline = ""
    evaluation = result.discount
    if evaluation is not None and evaluation.worthwhile:
        deadline = _parse_date(evaluation.discount_deadline)
        if deadline is not None and deadline >= as_of:
            discount_amount = round(float(evaluation.gross_discount), 2)
            discount_deadline = evaluation.discount_deadline
    return PaymentRunLine(
        invoice_id=result.invoice_id,
        supplier_name=result.supplier_name or str(invoice.get("supplier_name", "")),
        invoice_number=str(invoice.get("invoice_number", "")),
        due_date=str(invoice.get("due_date", "")),
        gross_amount=gross,
        discount_amount=discount_amount,
        discount_deadline=discount_deadline,
        net_amount=round(gross - discount_amount, 2),
    )


def propose_payment_runs(
    records: list[ApprovalRecord],
    *,
    window_days: int = DEFAULT_WINDOW_DAYS,
    as_of: date | None = None,
    exclude_invoice_ids: set[str] | None = None,
    four_eyes_threshold: float | None = None,
) -> list[PaymentRun]:
    """Group the approved, not-yet-run records into proposed runs.

    Pure over the records it is given — the service layer supplies
    the store's records and the membership of confirmed runs. When
    ``four_eyes_threshold`` is given, a run whose net total is
    *above* it is stamped ``four_eyes_required``: the hand-off then
    waits for a second confirmer, different from the first (the
    service enforces it). A total exactly at the threshold is not
    above it.
    """
    as_of = as_of or date.today()
    window_days = max(1, int(window_days))
    excluded = exclude_invoice_ids or set()
    groups: dict[tuple[str, int], list[ApprovalRecord]] = {}
    for record in records:
        result = record.result
        if result.status != "approved":
            continue
        if result.invoice_id in excluded:
            continue
        invoice = record.invoice or {}
        currency = str(invoice.get("currency") or result.facts.currency or "USD")
        due = _parse_date(str(invoice.get("due_date", ""))) or as_of
        index = _window_index(due, as_of, window_days)
        groups.setdefault((currency, index), []).append(record)

    runs: list[PaymentRun] = []
    for (currency, index), members in sorted(groups.items()):
        lines = [_line_for(record, as_of) for record in members]
        lines.sort(key=lambda line: (line.due_date or "9999", line.invoice_id))
        dues = [
            parsed
            for line in lines
            if (parsed := _parse_date(line.due_date)) is not None
        ]
        window_end = as_of + timedelta(days=(index + 1) * window_days - 1)
        window_start = min(dues) if dues and index == 0 else as_of + timedelta(
            days=index * window_days
        )
        run = PaymentRun(
            run_id=f"RUN-{currency}-{window_start.isoformat()}-{window_end.isoformat()}",
            currency=currency,
            window_start=window_start.isoformat(),
            window_end=window_end.isoformat(),
            window_days=window_days,
            lines=lines,
            invoice_count=len(lines),
            total_gross=round(sum(line.gross_amount for line in lines), 2),
            total_discount=round(sum(line.discount_amount for line in lines), 2),
            total_net=round(sum(line.net_amount for line in lines), 2),
        )
        if four_eyes_threshold is not None and run.total_net > four_eyes_threshold:
            run.four_eyes_required = True
        run.note = (
            f"{run.invoice_count} approved invoice(s), {currency} "
            f"{run.total_net:.2f} net payable in the window ending "
            f"{run.window_end}"
            + (
                f" — includes {run.total_discount:.2f} of early-payment "
                "discounts still capturable"
                if run.total_discount
                else ""
            )
            + ". A proposal only: a finance lead confirms the run; "
            "payment itself happens in the AP system of record."
            + (
                " This run is above the four-eyes threshold: a second "
                "confirmer, different from the first, must also confirm "
                "it before the hand-off file can be generated."
                if run.four_eyes_required
                else ""
            )
        )
        runs.append(run)
    return runs


def run_member_ids(run: PaymentRun) -> set[str]:
    return {line.invoice_id for line in run.lines}
