"""The payment-run control report — what the approver reads
before a run is confirmed.

A proposed run is arithmetic over stored records; this report
re-derives that arithmetic at read time and puts the three things
a bare total hides in front of the approver, per payment line:

- **Traceability** — every payment names its invoice, the stored
  three-way match verdict behind it, and the credits the
  applications ledger holds against it. The credit figure is
  recomputed from the ledger, not copied from the run line: a
  credit recorded after the run was proposed moves the line here
  and is *named* as a disagreement with the line's stamp, never
  smoothed over.
- **Disputes** — an invoice whose amount the supplier's latest
  statement disagrees with is excluded from the payable totals,
  with the reason (statement, both figures, the difference) on
  the line. It stays visible — excluded is not deleted.
- **Recovery conflicts** — a payment to a supplier who owes us
  money back (an open duplicate-payment recovery claim in the
  run's currency) is flagged ``ACK_REQUIRED``. Confirmation
  refuses until a named approver has acknowledged the claims
  against that line on the run record: paying a supplier who
  owes us a confirmed recovery is a decision, and the report
  makes it an explicit one.

The totals reconciliation closes the report the way the statement
claimed-total check closes a reconciliation: the recomputed lines
are set against the totals stamped on the run, and a disagreement
is reported in words. Nothing here writes, confirms, or pays —
the report is a read model over the run and the records.
"""

from __future__ import annotations

from .recovery import open_claims_for_supplier
from .schemas import (
    CreditApplication,
    PaymentRun,
    PaymentRunControlLine,
    PaymentRunControlReport,
    RecoveryPack,
    StatementReconciliation,
)
from .store import ApprovalRecord
from .supplier_positions import _latest_reconciliations

_TOLERANCE = 0.005


def _disputed_lines(
    reconciliations: list[StatementReconciliation],
) -> dict[tuple[str, str], dict[str, tuple]]:
    """Per (supplier, currency), the invoice numbers in dispute on
    the latest reconciliation, mapped to (line result, report) —
    the same dispute definition the supplier roll-up uses: a line
    whose stated amount differs from what AP processed."""
    latest = _latest_reconciliations(reconciliations)
    disputed: dict[tuple[str, str], dict[str, tuple]] = {}
    for key, report in latest.items():
        lines = {}
        for line in report.lines:
            if line.status == "amount_differs":
                lines[line.invoice_number] = (line, report)
        if lines:
            disputed[key] = lines
    return disputed


def build_control_report(
    run: PaymentRun,
    records: list[ApprovalRecord],
    applications: list[CreditApplication],
    packs: list[RecoveryPack],
    reconciliations: list[StatementReconciliation],
) -> PaymentRunControlReport:
    """Compose the control report over one proposed run. Pure over
    the records it is given — the service supplies the store's."""
    by_id = {r.result.invoice_id: r for r in records}
    credits_by_invoice: dict[str, float] = {}
    for application in applications:
        credits_by_invoice[application.invoice_id] = round(
            credits_by_invoice.get(application.invoice_id, 0.0)
            + application.amount,
            2,
        )
    disputed = _disputed_lines(reconciliations)
    acks = {a.invoice_id: a for a in run.recovery_acknowledgements}
    claims_cache: dict[str, list] = {}

    lines: list[PaymentRunControlLine] = []
    notes: list[str] = []
    for line in run.lines:
        record = by_id.get(line.invoice_id)
        credits = credits_by_invoice.get(line.invoice_id, 0.0)
        credits_agree = abs(credits - line.credit_amount) <= _TOLERANCE
        if not credits_agree:
            notes.append(
                f"{line.invoice_id}: the run line carries "
                f"{line.credit_amount:.2f} of credits but the "
                f"applications ledger holds {credits:.2f} — the "
                "report nets the ledger figure"
            )
        net = round(
            max(0.0, line.gross_amount - line.discount_amount - credits), 2
        )
        control = PaymentRunControlLine(
            invoice_id=line.invoice_id,
            invoice_number=line.invoice_number,
            supplier_name=line.supplier_name,
            due_date=line.due_date,
            gross_amount=line.gross_amount,
            discount_amount=line.discount_amount,
            credit_amount=credits,
            line_credit_amount=line.credit_amount,
            credits_agree=credits_agree,
            net_amount=net,
        )
        if record is None:
            control.excluded = True
            control.exclusion_reason = (
                "no stored record for this invoice — the payment "
                "cannot be traced to a processed invoice and is "
                "excluded until it can"
            )
            lines.append(control)
            continue
        result = record.result
        control.record_status = result.status
        control.match_status = result.facts.match_status
        control.triage_outcome = result.triage.outcome
        dispute = disputed.get((line.supplier_name, run.currency), {}).get(
            line.invoice_number
        )
        if dispute is not None:
            disputed_line, report = dispute
            control.excluded = True
            control.exclusion_reason = (
                f"invoice {line.invoice_number} is in dispute on "
                f"{line.supplier_name}'s latest statement "
                f"{report.statement_id}: the statement shows "
                f"{(disputed_line.statement_amount or 0.0):.2f} "
                f"against {(disputed_line.ap_amount or 0.0):.2f} "
                f"processed (difference "
                f"{(disputed_line.difference or 0.0):+.2f}) — "
                "excluded from the payable total until the "
                "difference is resolved"
            )
        if line.supplier_name not in claims_cache:
            claims_cache[line.supplier_name] = open_claims_for_supplier(
                packs, line.supplier_name, run.currency
            )
        claims = claims_cache[line.supplier_name]
        control.recovery_claims = claims
        control.recovery_claimed_amount = round(
            sum(c.amount_outstanding for c in claims), 2
        )
        ack = acks.get(line.invoice_id)
        if ack is not None:
            control.acknowledged_by = ack.acknowledged_by
        control.ack_required = bool(
            not control.excluded
            and control.recovery_claimed_amount > _TOLERANCE
            and ack is None
        )
        lines.append(control)

    payable = [line for line in lines if not line.excluded]
    excluded = [line for line in lines if line.excluded]
    total_gross = round(sum(line.gross_amount for line in payable), 2)
    total_discount = round(sum(line.discount_amount for line in payable), 2)
    total_credit = round(sum(line.credit_amount for line in payable), 2)
    total_net = round(sum(line.net_amount for line in payable), 2)
    excluded_net = round(sum(line.net_amount for line in excluded), 2)
    lines_net_total = round(total_net + excluded_net, 2)
    lines_gross_total = round(
        sum(line.gross_amount for line in lines), 2
    )
    totals_agree = (
        abs(lines_net_total - run.total_net) <= _TOLERANCE
        and abs(lines_gross_total - run.total_gross) <= _TOLERANCE
        and not notes
    )
    if abs(lines_net_total - run.total_net) > _TOLERANCE:
        notes.append(
            f"the run's stamped net total is {run.total_net:.2f} but "
            f"its lines net {lines_net_total:.2f} as the records "
            "stand now — the stamp is the proposal's; the lines are "
            "recomputed (credits recorded after proposal move the "
            "lines, never the stamp)"
        )
    if abs(lines_gross_total - run.total_gross) > _TOLERANCE:
        notes.append(
            f"the run's stamped gross total is {run.total_gross:.2f} "
            f"but its lines gross {lines_gross_total:.2f}"
        )
    ack_required = [line for line in lines if line.ack_required]
    summary_parts = [
        f"{len(payable)} payment(s) totalling {run.currency} "
        f"{total_net:.2f} payable"
    ]
    if excluded:
        summary_parts.append(
            f"{len(excluded)} excluded in dispute "
            f"({run.currency} {excluded_net:.2f} held back)"
        )
    if ack_required:
        names = ", ".join(
            f"{line.invoice_id} ({line.supplier_name}, "
            f"{run.currency} {line.recovery_claimed_amount:.2f} "
            "claimed)"
            for line in ack_required
        )
        summary_parts.append(
            f"{len(ack_required)} payment(s) need a recovery "
            f"acknowledgement before this run can be confirmed: "
            f"{names}"
        )
    acknowledged = [line for line in lines if line.acknowledged_by]
    if acknowledged:
        summary_parts.append(
            f"{len(acknowledged)} recovery acknowledgement(s) on "
            "record"
        )
    summary_parts.append(
        "the lines add up to the run's stamped totals"
        if totals_agree
        else "the totals do not add up — see the totals note"
    )
    return PaymentRunControlReport(
        run_id=run.run_id,
        currency=run.currency,
        run_status=run.status,
        lines=lines,
        payable_count=len(payable),
        excluded_count=len(excluded),
        ack_required_count=len(ack_required),
        total_gross=total_gross,
        total_discount=total_discount,
        total_credit=total_credit,
        total_net=total_net,
        excluded_net=excluded_net,
        lines_net_total=lines_net_total,
        run_total_gross=run.total_gross,
        run_total_net=run.total_net,
        totals_agree=totals_agree,
        totals_note="; ".join(notes),
        acknowledgements=list(run.recovery_acknowledgements),
        summary="; ".join(summary_parts) + ".",
    )
