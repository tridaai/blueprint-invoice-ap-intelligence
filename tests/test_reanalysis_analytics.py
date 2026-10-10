"""Re-analysis outcome analytics: the chain is a ledger, and
/metrics reads it back.

Round 17 made a completing reply re-run its case and recorded
each supersession in the case's re-analysis chain — but a ledger
nobody reads is just storage. The analytics aggregate the
chains: how many cases were re-run, how often the re-run's
outcome actually changed (a reply that informed) against how
often it landed exactly where the blocked case already stood
(the supplier's claim tested and found wanting — a
supplier-quality signal, named per supplier), which outcome
transitions the replies bought, which overlay item does the
unblocking, and the loop's shape (loops per case, overlays
accumulated per case).
"""

from __future__ import annotations

import pytest

from ap_agent.clarify import reanalysis_analytics
from ap_agent.samples import sample_invoice_models
from ap_agent.schemas import AgentResult, ReanalysisEntry, Triage
from ap_agent.service import InvoiceService
from ap_agent.store import ApprovalRecord, InMemoryStore
from test_reply_reanalysis import _grn_1005, _grn_1006, _po_1005

SAMPLES = {s.invoice_id: s for s in sample_invoice_models()}


@pytest.fixture()
def service() -> InvoiceService:
    return InvoiceService(store=InMemoryStore(), checkpointer=False)


def _record(
    invoice_id: str, supplier: str, chain: list[ReanalysisEntry]
) -> ApprovalRecord:
    result = AgentResult(
        invoice_id=invoice_id,
        supplier_name=supplier,
        triage=Triage(outcome=chain[-1].new_outcome if chain else "clean_match"),
        reanalysis_history=chain,
    )
    return ApprovalRecord(result=result, invoice={"invoice_id": invoice_id})


def _entry(previous: str, new: str, overlay: list[str]) -> ReanalysisEntry:
    return ReanalysisEntry(
        previous_outcome=previous, new_outcome=new, overlay=overlay
    )


# -- the pure aggregate ------------------------------------------------------


def test_empty_population_reads_as_zeros_not_inventions():
    block = reanalysis_analytics([])
    assert block["cases_reanalysed"] == 0
    assert block["reanalyses"] == 0
    assert block["outcomes_changed"] == 0
    assert block["outcomes_unchanged"] == 0
    assert block["change_rate"] is None
    assert block["transitions"] == {}
    assert block["suppliers_not_unblocked"] == []


def test_records_without_chains_are_not_counted():
    block = reanalysis_analytics([_record("INV-A", "Acme", [])])
    assert block["cases_reanalysed"] == 0
    assert block["reanalyses"] == 0


def test_transitions_overlays_and_change_counts():
    records = [
        _record(
            "INV-A",
            "Acme",
            [
                _entry("missing_po", "missing_receipt", ["purchase_order"]),
                _entry("missing_receipt", "clean_match", ["goods_receipt"]),
            ],
        ),
        _record(
            "INV-B",
            "Berg",
            [_entry("missing_receipt", "clean_match", ["goods_receipt"])],
        ),
    ]
    block = reanalysis_analytics(records)
    assert block["cases_reanalysed"] == 2
    assert block["reanalyses"] == 3
    assert block["outcomes_changed"] == 3
    assert block["outcomes_unchanged"] == 0
    assert block["change_rate"] == 1.0
    assert block["transitions"] == {
        "missing_po->missing_receipt": 1,
        "missing_receipt->clean_match": 2,
    }
    # Each loop's overlay counts: the GRN unblocked two cases.
    assert block["overlay_items"] == {"goods_receipt": 2, "purchase_order": 1}
    assert block["loops_per_case"] == {"1": 1, "2": 1}
    assert block["overlays_per_case"] == {"1": 1, "2": 1}
    assert block["suppliers_not_unblocked"] == []


def test_same_outcome_rerun_is_the_supplier_quality_signal():
    records = [
        _record(
            "INV-A",
            "Acme",
            [
                _entry("missing_receipt", "missing_receipt", ["goods_receipt"]),
                _entry("missing_receipt", "missing_receipt", ["goods_receipt"]),
            ],
        ),
        _record(
            "INV-B",
            "Berg",
            [_entry("missing_po", "missing_po", ["purchase_order"])],
        ),
        _record(
            "INV-C",
            "Acme",
            [_entry("missing_receipt", "clean_match", ["goods_receipt"])],
        ),
    ]
    block = reanalysis_analytics(records)
    assert block["outcomes_unchanged"] == 3
    assert block["outcomes_changed"] == 1
    assert block["change_rate"] == 0.25
    # Worst first: Acme's replies twice failed to move the case;
    # the loop that DID move it does not clear the signal.
    assert block["suppliers_not_unblocked"] == [
        {"supplier_name": "Acme", "reanalyses": 3, "unchanged": 2},
        {"supplier_name": "Berg", "reanalyses": 1, "unchanged": 1},
    ]


def test_long_chains_bucket_at_three_plus():
    chain = [
        _entry("missing_po", "missing_receipt", ["purchase_order"]),
        _entry("missing_receipt", "missing_receipt", ["goods_receipt"]),
        _entry("missing_receipt", "missing_po", ["legible_invoice"]),
        _entry("missing_po", "clean_match", ["purchase_order", "goods_receipt"]),
    ]
    block = reanalysis_analytics([_record("INV-A", "Acme", chain)])
    assert block["loops_per_case"] == {"3+": 1}
    # Five overlay components accumulated across the chain.
    assert block["overlays_per_case"] == {"3+": 1}


# -- the metrics wiring, over real chains ------------------------------------


def test_metrics_read_the_real_chains_back(service):
    service.analyze(SAMPLES["INV-1005"])
    service.record_information_reply(
        "INV-1005",
        actor="Sam (AP)",
        supplied_items=["purchase_order"],
        purchase_order=_po_1005(),
    )
    service.record_information_reply(
        "INV-1005",
        actor="Jo (AP)",
        supplied_items=["goods_receipt"],
        goods_receipt=_grn_1005(),
    )
    service.analyze(SAMPLES["INV-1006"])
    service.record_information_reply(
        "INV-1006",
        actor="Sam (AP)",
        supplied_items=["goods_receipt"],
        goods_receipt=_grn_1006(),
    )
    block = service.metrics_snapshot()["reanalysis"]
    assert block["cases_reanalysed"] == 2
    assert block["reanalyses"] == 3
    assert block["outcomes_changed"] == 3
    assert block["outcomes_unchanged"] == 0
    assert block["transitions"] == {
        "missing_po->missing_receipt": 1,
        "missing_receipt->clean_match": 2,
    }
    assert block["overlay_items"] == {"goods_receipt": 2, "purchase_order": 1}
    assert block["loops_per_case"] == {"1": 1, "2": 1}
