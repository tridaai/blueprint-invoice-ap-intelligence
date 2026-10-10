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
- Discounts are re-verified at export: a window can close between
  confirmation and hand-off, so :func:`discount_drift` re-prices
  every discounted member as of the export date and itemises any
  drift on the run record — the exported file never promises a
  discount the bank run can no longer take.
"""

from __future__ import annotations

from datetime import date, timedelta

from .discounts import evaluate_discount
from .matching import _parse_date
from .schemas import InvoiceInput, PaymentRun, PaymentRunDrift, PaymentRunLine
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


def discount_drift(
    run: PaymentRun,
    records: dict[str, ApprovalRecord],
    *,
    as_of: date,
) -> tuple[PaymentRun, list[PaymentRunDrift]]:
    """Re-verify a confirmed run's discounts at export time.

    A run is confirmed against the discounts capturable on proposal
    day; the hand-off can be generated days later, after a member's
    discount window has closed — and a file that still nets off a
    lapsed discount short-pays the supplier (the full total is what
    is owed once the window closes) while telling the bank a figure
    nobody can take. So export re-prices every member that carries
    a discount, from its stored invoice, as of the export date:

    - Still capturable at the confirmed figure → the line stands.
    - No longer capturable (or capturable at a different figure) →
      the line is adjusted to the export-time figure — full gross
      when the window has closed — and the drift is itemised in
      the returned entries: which invoice, which deadline, how
      much discount was lost, and why.

    The export proceeds with the adjusted figures; the drift is
    never silent and never blocks the clean members. Pure: the
    input run is not mutated — the service layer applies the
    adjusted copy to the stored run alongside the export receipt.
    """
    adjusted = run.model_copy(deep=True)
    drift: list[PaymentRunDrift] = []
    for line in adjusted.lines:
        if line.discount_amount <= 0:
            continue
        record = records.get(line.invoice_id)
        capturable = 0.0
        if record is None:
            reason = (
                "the invoice record is no longer in the store, so "
                "its discount cannot be re-verified"
            )
        else:
            invoice = InvoiceInput.model_validate(record.invoice or {})
            evaluation = evaluate_discount(invoice, as_of=as_of)
            if evaluation is not None and evaluation.worthwhile:
                capturable = round(float(evaluation.gross_discount), 2)
            if capturable > 0:
                reason = (
                    "the discount re-priced at export differs from "
                    "the figure the run was confirmed with"
                )
            else:
                reason = (
                    f"the discount window closed on "
                    f"{line.discount_deadline} — the discount is no "
                    "longer capturable, so the member is exported at "
                    "its full gross amount"
                )
        if abs(capturable - line.discount_amount) <= 0.005:
            continue  # still capturable at the confirmed figure
        drift.append(
            PaymentRunDrift(
                invoice_id=line.invoice_id,
                supplier_name=line.supplier_name,
                discount_deadline=line.discount_deadline,
                discount_at_proposal=line.discount_amount,
                discount_at_export=capturable,
                discount_lost=round(line.discount_amount - capturable, 2),
                reason=reason,
            )
        )
        line.discount_amount = capturable
        line.net_amount = round(line.gross_amount - capturable, 2)
    if drift:
        adjusted.total_discount = round(
            sum(line.discount_amount for line in adjusted.lines), 2
        )
        adjusted.total_net = round(
            sum(line.net_amount for line in adjusted.lines), 2
        )
    return adjusted, drift
