"""Re-export outcome analytics: the corrected files, graded.

A bounced group re-presented alone is a stamped record, but
stamps are not a read: nothing said whether the solo files
fared better than the files that bounced, which suppliers'
groups keep bouncing, or what the bank's reasons add up to.
This analytics reads the re-export population back — each
re-presented group's eventual standing, the acceptance rate
of the corrections, the rejection reasons as patterns, the
repeat-rejection suppliers, and the drift the second files
re-priced — into /metrics beside the feedback loop's
coverage counts.
"""

from __future__ import annotations

from ap_agent.paymentruns import group_reexport_analytics, handoff_metrics
from ap_agent.schemas import (
    HandoffCreditorEntry,
    HandoffGroupReceipt,
    HandoffGroupReexport,
    PaymentRun,
    PaymentRunDrift,
    PaymentRunLine,
)

NORTHBEAM = "Northbeam Office Supplies"
HARBORLIGHT = "Harborlight Ltd"


def _run(run_id: str, currency: str, groups: list[dict]) -> PaymentRun:
    lines, entries = [], []
    for group in groups:
        lines.append(
            PaymentRunLine(
                invoice_id=group["invoice"],
                supplier_name=group["supplier"],
                due_date=group["date"],
                gross_amount=1000.00,
                net_amount=1000.00,
            )
        )
        entries.append(
            HandoffCreditorEntry(
                invoice_id=group["invoice"],
                source="supplier_master",
                supplier_name=group["supplier"],
                execution_date=group["date"],
            )
        )
    return PaymentRun(
        run_id=run_id,
        currency=currency,
        window_start="2026-10-05",
        window_end="2026-10-11",
        window_days=7,
        status="confirmed",
        lines=lines,
        invoice_count=len(lines),
        total_gross=1000.00 * len(lines),
        total_net=1000.00 * len(lines),
        proposed_at="2026-10-05T09:00:00+00:00",
        confirmed_by="Tomas Reid",
        confirmed_at="2026-10-05T09:30:00+00:00",
        handoff_format="pain001",
        handoff_batch_booking=True,
        handoff_generated_at="2026-10-11T09:00:00+00:00",
        handoff_key=f"payment-runs/{run_id}/handoff.pain001.xml",
        handoff_creditor_entries=entries,
    )


def _receipt(run_id: str, group: str, status: str, detail: str = "", **kw):
    return HandoffGroupReceipt(
        pmt_inf_id=f"{run_id}-{group}",
        execution_date=group,
        status=status,
        detail=detail,
        member_invoice_ids=kw.pop("members", []),
        recorded_by="Tomas Reid",
        recorded_at="2026-10-19T09:00:00+00:00",
        **kw,
    )


def _reexport(run_id: str, group: str, invoice: str, reason: str, **kw):
    return HandoffGroupReexport(
        pmt_inf_id=f"{run_id}-{group}",
        execution_date=group,
        reexport_execution_date=group,
        rejection_reason=reason,
        member_invoice_ids=[invoice],
        reexported_by="Tomas Reid",
        reexported_at="2026-10-19T15:00:00+00:00",
        **kw,
    )


def _population() -> list[PaymentRun]:
    """Northbeam: group A corrected and accepted; group B
    bounced, corrected, and bounced again on the same reason.
    Harborlight: one group corrected, the bank yet to answer."""
    eur = _run(
        "RUN-EUR-1",
        "EUR",
        [
            {"invoice": "INV-1001", "supplier": NORTHBEAM, "date": "2026-10-20"},
            {"invoice": "INV-1008", "supplier": NORTHBEAM, "date": "2026-10-21"},
        ],
    )
    eur.handoff_group_receipts = [
        _receipt("RUN-EUR-1", "2026-10-20", "rejected", "account closed",
                 members=["INV-1001"], superseded=True),
        _receipt("RUN-EUR-1", "2026-10-20", "accepted",
                 members=["INV-1001"]),
        _receipt("RUN-EUR-1", "2026-10-21", "rejected", "account closed",
                 members=["INV-1008"], superseded=True),
        _receipt("RUN-EUR-1", "2026-10-21", "rejected", "account closed",
                 members=["INV-1008"]),
    ]
    eur.handoff_group_reexports = [
        _reexport(
            "RUN-EUR-1", "2026-10-20", "INV-1001", "account closed",
            drift=[
                PaymentRunDrift(
                    invoice_id="INV-1001",
                    supplier_name=NORTHBEAM,
                    discount_lost=25.0,
                )
            ],
            drift_total=25.0,
        ),
        _reexport("RUN-EUR-1", "2026-10-21", "INV-1008", "account closed"),
    ]
    usd = _run(
        "RUN-USD-1",
        "USD",
        [{"invoice": "INV-2001", "supplier": HARBORLIGHT, "date": "2026-10-22"}],
    )
    usd.handoff_group_receipts = [
        _receipt("RUN-USD-1", "2026-10-22", "rejected",
                 "value date not a business day",
                 members=["INV-2001"], superseded=True),
    ]
    usd.handoff_group_reexports = [
        _reexport("RUN-USD-1", "2026-10-22", "INV-2001",
                  "value date not a business day"),
    ]
    return [eur, usd]


# -- the analytics ------------------------------------------------------------------


def test_the_population_reads_its_eventual_standings():
    analytics = group_reexport_analytics(_population())
    assert analytics["reexports_total"] == 3
    assert analytics["groups_reexported"] == 3
    assert analytics["accepted_after_reexport"] == 1
    assert analytics["rejected_again"] == 1
    assert analytics["awaiting_answer"] == 1
    assert analytics["acceptance_rate"] == 0.5


def test_groups_that_bounce_twice_and_the_reason_patterns():
    analytics = group_reexport_analytics(_population())
    assert analytics["groups_bounced_twice"] == 1
    assert analytics["rejection_reasons"] == [
        {"reason": "account closed", "count": 3},
        {"reason": "value date not a business day", "count": 1},
    ]


def test_repeat_rejection_suppliers_are_named_with_their_counts():
    analytics = group_reexport_analytics(_population())
    assert analytics["repeat_rejection_suppliers"] == [
        {"supplier_name": NORTHBEAM, "rejections": 3}
    ]


def test_the_drift_the_second_files_repriced():
    analytics = group_reexport_analytics(_population())
    assert analytics["drift_repriced_by_currency"] == {"EUR": 25.0}
    assert analytics["drift_members_repriced"] == 1


def test_no_reexports_is_an_empty_population_not_a_zero_rate():
    analytics = group_reexport_analytics([_run("RUN-X", "USD", [
        {"invoice": "INV-1", "supplier": HARBORLIGHT, "date": "2026-10-22"}
    ])])
    assert analytics["reexports_total"] == 0
    assert analytics["groups_reexported"] == 0
    assert analytics["acceptance_rate"] is None
    assert analytics["rejection_reasons"] == []
    assert analytics["repeat_rejection_suppliers"] == []


def test_the_analytics_ride_the_handoff_metrics():
    section = handoff_metrics(_population())
    assert section["group_reexports"]["groups_reexported"] == 3
    assert section["group_reexports"]["acceptance_rate"] == 0.5
    assert section["group_answers"]["reexports"] == 3
