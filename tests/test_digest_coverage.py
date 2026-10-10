"""Digest-store coverage in /metrics: the audit file, audited.

The digest history is only as trustworthy as its completeness:
a missing day reads like a quiet day unless the coverage says
otherwise. /metrics now reports the stored days by attribution
(human vs the sweep's system backfill vs unattributed), the
stored range, and the gaps inside it — split at the sweep's
reach, so the gaps that predate the lookback and can never
backfill are named apart from the ones a sweep run still heals.
"""

from __future__ import annotations

from datetime import date

from ap_agent.digest import digest_store_coverage
from ap_agent.schemas import EscalationDigest
from ap_agent.service import InvoiceService
from ap_agent.store import InMemoryStore
from ap_agent.supplier_master import InMemorySupplierMaster


def _digest(day: str, stored_by: str) -> EscalationDigest:
    return EscalationDigest(
        as_of=day, provenance="stored", stored_by=stored_by
    )


def test_empty_store_reports_an_empty_coverage():
    coverage = digest_store_coverage(
        [], as_of=date(2026, 10, 20), lookback_days=7
    )
    assert coverage["stored_days"] == 0
    assert coverage["by_attribution"] == {
        "human": 0,
        "system": 0,
        "unattributed": 0,
    }
    assert coverage["oldest_stored_date"] == ""
    assert coverage["gap_dates"] == []
    assert coverage["sweep_reach_start"] == "2026-10-14"


def test_attribution_and_range_and_gaps():
    digests = [
        _digest("2026-10-01", "Ana Ruiz"),     # human
        _digest("2026-10-02", "system"),       # sweep backfill
        _digest("2026-10-05", "system"),       # gap: 03, 04
        _digest("2026-10-19", "Tomas Reid"),   # gap: 06..18
        _digest("2026-10-20", ""),             # unattributed
    ]
    coverage = digest_store_coverage(
        digests, as_of=date(2026, 10, 20), lookback_days=7
    )
    assert coverage["stored_days"] == 5
    assert coverage["by_attribution"] == {
        "human": 2,
        "system": 2,
        "unattributed": 1,
    }
    assert coverage["oldest_stored_date"] == "2026-10-01"
    assert coverage["latest_stored_date"] == "2026-10-20"
    assert coverage["gap_count"] == 15  # 03, 04 + 06..18
    assert coverage["gap_dates"][0] == "2026-10-03"
    assert coverage["gap_dates"][-1] == "2026-10-18"
    # The sweep anchored 2026-10-20 reaches back to 2026-10-14:
    # gaps before that can never backfill now; 14..18 still can.
    assert coverage["unrecoverable_gap_count"] == 10  # 03, 04 + 06..13
    assert coverage["unrecoverable_gap_dates"][-1] == "2026-10-13"
    healable = [
        g for g in coverage["gap_dates"]
        if g not in coverage["unrecoverable_gap_dates"]
    ]
    assert healable == [
        "2026-10-14",
        "2026-10-15",
        "2026-10-16",
        "2026-10-17",
        "2026-10-18",
    ]


def test_a_contiguous_series_has_no_gaps():
    digests = [_digest(f"2026-10-{d:02d}", "system") for d in (18, 19, 20)]
    coverage = digest_store_coverage(
        digests, as_of=date(2026, 10, 20), lookback_days=7
    )
    assert coverage["gap_count"] == 0
    assert coverage["unrecoverable_gap_count"] == 0


def test_metrics_carry_the_coverage():
    svc = InvoiceService(
        store=InMemoryStore(),
        checkpointer=False,
        supplier_master=InMemorySupplierMaster(),
    )
    store = svc._get_store()
    store.save_digest(_digest("2026-10-01", "Ana Ruiz"))
    store.save_digest(_digest("2026-10-03", "system"))
    section = svc.metrics_snapshot()["digest_store"]
    assert section["stored_days"] == 2
    assert section["by_attribution"]["human"] == 1
    assert section["by_attribution"]["system"] == 1
    assert section["gap_dates"] == ["2026-10-02"]
    assert section["lookback_days"] == 7
