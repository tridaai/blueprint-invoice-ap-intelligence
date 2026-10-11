"""Credit-note application: the human-gated write path of credit-note
matching. Matching stays a pure query; an approver records a matched
credit against its invoice, and the credit then nets into the
invoice's payable position and into payment-run proposals. Caps are
enforced in code: never more than the remaining computed variance,
never more than the invoice's open amount, never twice."""

from __future__ import annotations

from datetime import date

import pytest
from fastapi.testclient import TestClient

from ap_agent.api import create_app
from ap_agent.paymentruns import discount_drift
from ap_agent.samples import sample_invoice_models
from ap_agent.schemas import CreditNoteInput, PaymentRun, PaymentRunLine
from ap_agent.service import InvoiceService
from ap_agent.store import InMemoryStore

SAMPLES = {s.invoice_id: s for s in sample_invoice_models()}


@pytest.fixture()
def service() -> InvoiceService:
    svc = InvoiceService(store=InMemoryStore(), checkpointer=False)
    # INV-1001 clean (1180.00), INV-1002 price variance 94.40 on
    # 1274.40, INV-1003 short-receipt variance 141.60 on 708.00.
    for invoice_id in ("INV-1001", "INV-1002", "INV-1003"):
        svc.analyze(SAMPLES[invoice_id])
    return svc


def _note(**overrides) -> CreditNoteInput:
    base = {
        "credit_note_id": "CN-T1",
        "supplier_name": "Cascade IT Hardware",
        "credit_note_number": "CN-T1",
        "currency": "USD",
        "amount": 94.40,
        "referenced_invoice_number": "CI-2210",
    }
    base.update(overrides)
    return CreditNoteInput(**base)


def _harborline_note(**overrides) -> CreditNoteInput:
    base = {
        "credit_note_id": "CN-T2",
        "supplier_name": "Harborline Packaging",
        "credit_note_number": "CN-T2",
        "currency": "USD",
        "amount": 100.00,
        "referenced_invoice_number": "HP-881",
    }
    base.update(overrides)
    return CreditNoteInput(**base)


def test_apply_records_the_credit_and_moves_the_payable_position(service):
    application = service.apply_credit_note(_note(), actor="Priya (AP lead)")
    assert application.invoice_id == "INV-1002"
    assert application.amount == 94.40
    assert application.match_status == "settles_variance"
    assert application.remaining_variance_after == 0.0
    assert application.applied_by == "Priya (AP lead)"
    assert application.applied_at

    position = service.payable_position("INV-1002")
    assert position.gross_amount == 1274.40
    assert position.total_credits == 94.40
    assert position.open_amount == 1180.00
    assert [c.credit_note_id for c in position.credits] == ["CN-T1"]


def test_matching_stays_a_pure_query(service):
    before = service.get("INV-1002")
    service.match_credit_note(_note())
    service.match_credit_note(_note())
    assert service.list_credit_applications() == []
    after = service.get("INV-1002")
    assert after.status == before.status == "awaiting_approver"
    # Applying is the write — and even it leaves the gate state alone.
    service.apply_credit_note(_note(), actor="Priya (AP lead)")
    assert service.get("INV-1002").status == "awaiting_approver"


def test_applied_credit_is_never_counted_twice(service):
    service.apply_credit_note(_note(), actor="Priya (AP lead)")
    match = service.match_credit_note(_note())
    assert match.status == "already_applied"
    assert match.applied_credits_total == 94.40
    with pytest.raises(ValueError, match="never applied twice"):
        service.apply_credit_note(_note(), actor="Priya (AP lead)")


def test_partial_credits_accumulate_against_the_remaining_variance(service):
    first = service.apply_credit_note(
        _harborline_note(), actor="Priya (AP lead)"
    )
    assert first.match_status == "partial_settlement"
    assert first.remaining_variance_after == 41.60

    # The next note is evaluated against what remains, not the
    # invoice's original variance.
    second = _harborline_note(
        credit_note_id="CN-T3", credit_note_number="CN-T3", amount=41.60
    )
    match = service.match_credit_note(second)
    assert match.status == "settles_variance"
    assert match.applied_credits_total == 100.00
    service.apply_credit_note(second, actor="Priya (AP lead)")

    position = service.payable_position("INV-1003")
    assert position.total_credits == 141.60
    assert position.open_amount == 566.40

    # Nothing remains: a third credit has no variance to settle.
    third = _harborline_note(
        credit_note_id="CN-T4", credit_note_number="CN-T4", amount=10.00
    )
    assert service.match_credit_note(third).status == "no_variance"
    with pytest.raises(ValueError, match="no_variance"):
        service.apply_credit_note(third, actor="Priya (AP lead)")


def test_unmatched_and_excess_credits_cannot_be_recorded(service):
    with pytest.raises(ValueError, match="unmatched"):
        service.apply_credit_note(
            _note(credit_note_id="CN-T5", referenced_invoice_number="CI-9999"),
            actor="Priya (AP lead)",
        )
    with pytest.raises(ValueError, match="exceeds_variance"):
        service.apply_credit_note(
            _note(credit_note_id="CN-T6", amount=500.00),
            actor="Priya (AP lead)",
        )
    assert service.list_credit_applications() == []


def test_payable_position_for_an_invoice_without_credits(service):
    position = service.payable_position("INV-1001")
    assert position.gross_amount == 1180.00
    assert position.total_credits == 0.0
    assert position.open_amount == 1180.00
    with pytest.raises(KeyError):
        service.payable_position("INV-9999")


def test_recorded_credit_nets_into_the_payment_run(service):
    service.approve(
        "INV-1002", approver="Priya (AP lead)", reason="variance accepted"
    )
    service.apply_credit_note(_note(), actor="Priya (AP lead)")
    runs = service.propose_payment_runs()
    lines = [line for run in runs for line in run.lines]
    line = next(l for l in lines if l.invoice_id == "INV-1002")
    assert line.credit_amount == 94.40
    assert line.net_amount == round(
        line.gross_amount - line.discount_amount - 94.40, 2
    )
    run = next(
        r for r in runs if any(l.invoice_id == "INV-1002" for l in r.lines)
    )
    assert run.total_credit == 94.40
    assert run.total_net == round(
        sum(l.net_amount for l in run.lines), 2
    )


def test_discount_drift_keeps_recorded_credits_netted():
    run = PaymentRun(
        run_id="RUN-USD-2026-10-01-2026-10-07",
        currency="USD",
        status="confirmed",
        lines=[
            PaymentRunLine(
                invoice_id="INV-1002",
                supplier_name="Cascade IT Hardware",
                invoice_number="CI-2210",
                gross_amount=1274.40,
                discount_amount=25.49,
                discount_deadline="2026-10-07",
                credit_amount=94.40,
                net_amount=1154.51,
            )
        ],
        total_gross=1274.40,
        total_discount=25.49,
        total_credit=94.40,
        total_net=1154.51,
    )
    adjusted, drift = discount_drift(run, {}, as_of=date(2026, 10, 20))
    assert len(drift) == 1  # the discount window closed
    line = adjusted.lines[0]
    assert line.discount_amount == 0.0
    assert line.credit_amount == 94.40  # the credit is not drift
    assert line.net_amount == 1180.00  # gross - lapsed discount - credit


def test_credit_application_endpoints(service):
    client = TestClient(create_app(service))
    payload = _note().model_dump()

    missing_actor = client.post(
        "/credit-notes/apply", json={"note": payload, "actor": ""}
    )
    assert missing_actor.status_code == 422

    applied = client.post(
        "/credit-notes/apply", json={"note": payload, "actor": "Priya (AP lead)"}
    )
    assert applied.status_code == 200
    assert applied.json()["invoice_id"] == "INV-1002"

    again = client.post(
        "/credit-notes/apply", json={"note": payload, "actor": "Priya (AP lead)"}
    )
    assert again.status_code == 422  # never applied twice

    payable = client.get("/invoices/INV-1002/payable")
    assert payable.status_code == 200
    assert payable.json()["open_amount"] == 1180.00

    applications = client.get("/credit-notes/applications")
    assert [a["credit_note_id"] for a in applications.json()] == ["CN-T1"]
    filtered = client.get("/credit-notes/applications?invoice_id=INV-1001")
    assert filtered.json() == []

    assert client.get("/invoices/INV-9999/payable").status_code == 404
