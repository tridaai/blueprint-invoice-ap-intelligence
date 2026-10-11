"""The recovery ↔ payable seam: an open duplicate-payment claim
reaches the money views as its own line — the per-invoice payable
position and the per-supplier roll-up show what is claimed back,
what has been recorded received, and what is still outstanding —
and none of it is ever netted into what we owe. The receipt is a
figure, recorded on the credit_received transition; the roll-up
moves by exactly that figure and nothing else."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from ap_agent.api import create_app
from ap_agent.recovery import is_open_claim, recovery_totals
from ap_agent.samples import sample_invoice_models
from ap_agent.schemas import RecoveryPack
from ap_agent.service import InvoiceService
from ap_agent.store import InMemoryStore

SAMPLES = {s.invoice_id: s for s in sample_invoice_models()}


@pytest.fixture()
def service() -> InvoiceService:
    """INV-1001 approved, its duplicate INV-1013 analysed, plus
    INV-1008 (Northbeam) and INV-1002 (Cascade) processed."""
    svc = InvoiceService(store=InMemoryStore(), checkpointer=False)
    svc.analyze(SAMPLES["INV-1001"])
    svc.approve("INV-1001", approver="Priya Nair", reason="clean match")
    svc.analyze(SAMPLES["INV-1013"])
    svc.analyze(SAMPLES["INV-1008"])
    svc.analyze(SAMPLES["INV-1002"])
    svc.draft_duplicate_recovery("INV-1013")
    return svc


def _northbeam(service, as_of="2026-10-05"):
    roll = service.supplier_positions(as_of=as_of)
    return next(
        r for r in roll.suppliers if r.supplier_name == "Northbeam Office Supplies"
    )


# --- the per-invoice payable position -------------------------------------

def test_payable_position_carries_the_claim_as_its_own_line(service):
    position = service.payable_position("INV-1001")
    # The payable figure is untouched by the claim: money owed
    # back to us never shrinks money we owe.
    assert position.gross_amount == 1180.00
    assert position.open_amount == 1180.00
    assert position.recovery_open_amount == 1180.00
    assert len(position.recovery_claims) == 1
    claim = position.recovery_claims[0]
    assert claim.pack_id == "REC-INV-1013"
    assert claim.case_status == "opened"
    assert claim.amount_claimed == 1180.00
    assert claim.amount_received == 0.00
    assert claim.amount_outstanding == 1180.00
    assert claim.duplicate_invoice_id == "INV-1013"
    assert claim.original_invoice_id == "INV-1001"
    assert claim.evidence  # the computed duplicate evidence rides along


def test_the_duplicate_shows_the_same_claim(service):
    position = service.payable_position("INV-1013")
    assert [c.pack_id for c in position.recovery_claims] == ["REC-INV-1013"]
    assert position.recovery_open_amount == 1180.00


def test_an_unrelated_invoice_has_no_claim_line(service):
    position = service.payable_position("INV-1002")
    assert position.recovery_claims == []
    assert position.recovery_open_amount == 0.00


# --- the receipt is a figure ----------------------------------------------

def test_receipt_defaults_to_the_full_claim(service):
    service.transition_recovery_case(
        "INV-1013", to_status="supplier_contacted", actor="Priya Nair"
    )
    pack = service.transition_recovery_case(
        "INV-1013", to_status="credit_received", actor="Priya Nair"
    )
    assert pack.received_amount == 1180.00
    event = pack.case_history[-1]
    assert event.status == "credit_received" and event.amount == 1180.00
    # Nothing outstanding: the claim leaves the open lines.
    position = service.payable_position("INV-1001")
    assert position.recovery_claims == []
    assert position.recovery_open_amount == 0.00


def test_partial_receipt_leaves_the_remainder_outstanding(service):
    service.transition_recovery_case(
        "INV-1013", to_status="supplier_contacted", actor="Priya Nair"
    )
    pack = service.transition_recovery_case(
        "INV-1013", to_status="credit_received",
        actor="Priya Nair", amount=500.00,
    )
    assert pack.received_amount == 500.00
    claim = service.payable_position("INV-1001").recovery_claims[0]
    assert claim.amount_received == 500.00
    assert claim.amount_outstanding == 680.00
    assert claim.case_status == "credit_received"


def test_receipt_amount_is_validated(service):
    service.transition_recovery_case(
        "INV-1013", to_status="supplier_contacted", actor="Priya Nair"
    )
    with pytest.raises(ValueError, match="exceeds the recovery claim"):
        service.transition_recovery_case(
            "INV-1013", to_status="credit_received",
            actor="Priya Nair", amount=1180.01,
        )
    with pytest.raises(ValueError, match="above zero"):
        service.transition_recovery_case(
            "INV-1013", to_status="credit_received",
            actor="Priya Nair", amount=0.0,
        )
    # The refused attempts recorded nothing.
    pack = service.get_recovery_pack("INV-1013")
    assert pack.case_status == "supplier_contacted"
    assert pack.received_amount == 0.0


def test_an_amount_belongs_only_to_the_receipt_transition(service):
    with pytest.raises(ValueError, match="only on the move into"):
        service.transition_recovery_case(
            "INV-1013", to_status="supplier_contacted",
            actor="Priya Nair", amount=100.0,
        )


def test_receipt_amount_travels_the_api(service):
    client = TestClient(create_app(service))
    moved = client.post(
        "/invoices/INV-1013/recovery-case/transition",
        json={"actor": "Priya Nair", "to_status": "supplier_contacted"},
    )
    assert moved.status_code == 200
    moved = client.post(
        "/invoices/INV-1013/recovery-case/transition",
        json={
            "actor": "Priya Nair",
            "to_status": "credit_received",
            "amount": 1180.00,
        },
    )
    assert moved.status_code == 200
    assert moved.json()["received_amount"] == 1180.00


# --- the roll-up ------------------------------------------------------------

def test_rollup_shows_claims_received_and_outstanding(service):
    row = _northbeam(service)
    assert row.recovery_open_amount == 1180.00
    assert row.recovery_received_amount == 0.00
    assert [c.pack_id for c in row.recovery_claims] == ["REC-INV-1013"]
    # The payable side is exactly what it was before claims existed
    # as a concept: all four Northbeam/Cascade records processed.
    assert row.open_amount == 3540.00  # INV-1001 + INV-1008 + INV-1013
    roll = service.supplier_positions(as_of="2026-10-05")
    usd = roll.totals_by_currency["USD"]
    assert usd["recovery_open_amount"] == 1180.00
    assert usd["recovery_received_amount"] == 0.00
    cascade = next(
        r for r in roll.suppliers if r.supplier_name == "Cascade IT Hardware"
    )
    assert cascade.recovery_claims == []
    assert cascade.recovery_open_amount == 0.00


def test_rollup_moves_by_exactly_the_receipt_and_nothing_else(service):
    """The pinned identity: before/after a confirmed receipt, the
    roll-up figures differ by exactly the received amount in the
    recovery line — gross, credits, open payable and dispute are
    byte-identical."""
    before = _northbeam(service).model_dump()
    service.transition_recovery_case(
        "INV-1013", to_status="supplier_contacted", actor="Priya Nair"
    )
    service.transition_recovery_case(
        "INV-1013", to_status="credit_received",
        actor="Priya Nair", amount=700.00,
    )
    after = _northbeam(service).model_dump()
    for key in before:
        if key.startswith("recovery"):
            continue
        assert after[key] == before[key], key
    assert before["recovery_open_amount"] == 1180.00
    assert after["recovery_open_amount"] == 480.00  # 1180 − 700
    assert before["recovery_received_amount"] == 0.00
    assert after["recovery_received_amount"] == 700.00
    claim = after["recovery_claims"][0]
    assert claim["amount_received"] == 700.00
    assert claim["amount_outstanding"] == 480.00


def test_closing_a_case_ends_the_claim_but_keeps_the_receipt(service):
    for step in ("supplier_contacted", "credit_received"):
        service.transition_recovery_case(
            "INV-1013", to_status=step, actor="Priya Nair", amount=None
            if step != "credit_received" else 900.00,
        )
    service.transition_recovery_case(
        "INV-1013", to_status="closed", actor="Priya Nair",
        note="remainder written off by the finance lead",
    )
    row = _northbeam(service)
    assert row.recovery_claims == []
    assert row.recovery_open_amount == 0.00
    assert row.recovery_received_amount == 900.00  # the receipt stays on record


# --- claim semantics, unit level ---------------------------------------------

def _pack(**overrides) -> RecoveryPack:
    base = dict(
        pack_id="REC-X",
        duplicate_invoice_id="INV-D",
        original_invoice_id="INV-O",
        supplier_name="Supplier",
        currency="USD",
        amount=100.0,
        status="ready",
        case_status="opened",
    )
    base.update(overrides)
    return RecoveryPack(**base)


def test_blocked_and_closed_packs_claim_nothing():
    assert not is_open_claim(_pack(status="blocked"))
    assert not is_open_claim(_pack(case_status="closed"))
    assert not is_open_claim(_pack(received_amount=100.0))
    assert is_open_claim(_pack(received_amount=40.0))
    open_amt, received = recovery_totals(
        [
            _pack(),
            _pack(pack_id="REC-Y", case_status="closed", received_amount=100.0),
            _pack(pack_id="REC-Z", status="blocked"),
        ],
        "Supplier",
        "USD",
    )
    assert open_amt == 100.0
    assert received == 100.0  # the closed case's receipt; the blocked pack's nothing


def test_positions_endpoint_carries_the_recovery_lines(service):
    client = TestClient(create_app(service))
    payable = client.get("/invoices/INV-1001/payable")
    assert payable.status_code == 200
    body = payable.json()
    assert body["recovery_open_amount"] == 1180.00
    assert body["recovery_claims"][0]["pack_id"] == "REC-INV-1013"
    positions = client.get("/suppliers/positions", params={"as_of": "2026-10-05"})
    assert positions.status_code == 200
    row = next(
        r for r in positions.json()["suppliers"]
        if r["supplier_name"] == "Northbeam Office Supplies"
    )
    assert row["recovery_open_amount"] == 1180.00
    assert row["open_amount"] == 3540.00  # unnetted, as ever
