"""The payment-run control report: before a run is confirmed, the
approver sees every payment traced to its invoice and match
verdict, credits recomputed from the applications ledger,
disputed invoices excluded with reasons, and open recovery claims
against a supplier flagged on the payment line — and confirmation
refuses until a named approver has acknowledged those claims on
the run."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from ap_agent.api import create_app
from ap_agent.samples import (
    sample_credit_note_models,
    sample_invoice_models,
    sample_statement_models,
)
from ap_agent.service import InvoiceService
from ap_agent.store import InMemoryStore

SAMPLES = {s.invoice_id: s for s in sample_invoice_models()}
NOTES = {n.credit_note_id: n for n in sample_credit_note_models()}
STATEMENTS = {s.statement_id: s for s in sample_statement_models()}


@pytest.fixture()
def service() -> InvoiceService:
    """The seeded tangle: Northbeam's INV-1001 approved with its
    duplicate INV-1013 recovered against it (an open claim);
    INV-1008 approved but disputed on Northbeam's statement;
    Cascade's INV-1002 approved with CN-3001 recorded against it."""
    svc = InvoiceService(store=InMemoryStore(), checkpointer=False)
    for invoice_id in ("INV-1001", "INV-1008", "INV-1002", "INV-1013"):
        svc.analyze(SAMPLES[invoice_id])
    for invoice_id in ("INV-1001", "INV-1002", "INV-1008"):
        svc.approve(invoice_id, approver="Priya Nair", reason="eval seed")
    svc.draft_duplicate_recovery("INV-1013")
    svc.apply_credit_note(NOTES["CN-3001"], actor="Priya Nair")
    svc.reconcile_statement(STATEMENTS["STMT-NB-2026-10"])
    return svc


@pytest.fixture()
def run(service) -> object:
    runs = service.propose_payment_runs(as_of="2026-10-05")
    return next(r for r in runs if r.currency == "USD")


def _line(report, invoice_id):
    return next(l for l in report.lines if l.invoice_id == invoice_id)


# --- the report itself ------------------------------------------------------

def test_every_payment_is_traced(service, run):
    report = service.payment_run_control_report(run.run_id)
    inv1001 = _line(report, "INV-1001")
    assert inv1001.match_status == "full_match"
    assert inv1001.record_status == "approved"
    assert inv1001.gross_amount == 1180.00
    assert inv1001.discount_amount == 23.60  # 2/10 terms, still reachable
    assert inv1001.credit_amount == 0.00
    assert inv1001.net_amount == 1156.40
    inv1002 = _line(report, "INV-1002")
    assert inv1002.credit_amount == 94.40  # CN-3001, from the ledger
    assert inv1002.credits_agree is True
    assert inv1002.net_amount == 1180.00


def test_disputed_invoice_is_excluded_with_its_reason(service, run):
    report = service.payment_run_control_report(run.run_id)
    inv1008 = _line(report, "INV-1008")
    assert inv1008.excluded is True
    assert "STMT-NB-2026-10" in inv1008.exclusion_reason
    assert "1100.00" in inv1008.exclusion_reason
    assert "1180.00" in inv1008.exclusion_reason
    # Excluded from the payable totals, still counted in the lines.
    assert report.excluded_count == 1
    assert report.excluded_net == 1180.00
    assert report.payable_count == 2
    assert report.total_net == 2336.40  # 1156.40 + 1180.00
    assert report.lines_net_total == 3516.40
    assert report.run_total_net == run.total_net == 3516.40
    assert report.totals_agree is True


def test_recovery_claim_flags_the_payment_line(service, run):
    report = service.payment_run_control_report(run.run_id)
    inv1001 = _line(report, "INV-1001")
    assert inv1001.ack_required is True
    assert inv1001.recovery_claimed_amount == 1180.00
    assert [c.pack_id for c in inv1001.recovery_claims] == ["REC-INV-1013"]
    assert report.ack_required_count == 1
    # The excluded line needs no acknowledgement — it is not being paid.
    assert _line(report, "INV-1008").ack_required is False
    assert _line(report, "INV-1002").ack_required is False
    assert "ACK" not in report.summary  # the summary speaks plainly
    assert "acknowledgement" in report.summary


def test_credits_are_recomputed_from_the_ledger_not_the_stamp():
    """A credit recorded after proposal moves the report's line
    and is named as a disagreement — the stamp is never quietly
    followed, and the totals say they no longer add up."""
    svc = InvoiceService(store=InMemoryStore(), checkpointer=False)
    svc.analyze(SAMPLES["INV-1002"])
    svc.approve("INV-1002", approver="Priya Nair")
    run = svc.propose_payment_runs(as_of="2026-10-05")[0]
    assert run.total_credit == 0.00  # nothing recorded at proposal
    svc.apply_credit_note(NOTES["CN-3001"], actor="Priya Nair")
    report = svc.payment_run_control_report(run.run_id)
    line = _line(report, "INV-1002")
    assert line.line_credit_amount == 0.00  # the run line's stamp
    assert line.credit_amount == 94.40  # the ledger's figure
    assert line.credits_agree is False
    assert line.net_amount == 1180.00
    assert report.totals_agree is False
    assert "INV-1002" in report.totals_note
    assert "do not add up" in report.summary


# --- the acknowledgement gate -------------------------------------------------

def test_confirmation_refuses_until_the_acknowledgement(service, run):
    with pytest.raises(ValueError, match="cannot be confirmed yet"):
        service.confirm_payment_run(run.run_id, actor="Priya Nair")
    # The run is untouched by the refusal.
    assert service.get_payment_run(run.run_id).status == "proposed"


def test_acknowledgement_is_recorded_against_the_claims(service, run):
    updated = service.acknowledge_run_recovery(
        run.run_id, "INV-1001", actor="Tomas Reid"
    )
    assert len(updated.recovery_acknowledgements) == 1
    ack = updated.recovery_acknowledgements[0]
    assert ack.invoice_id == "INV-1001"
    assert ack.supplier_name == "Northbeam Office Supplies"
    assert ack.pack_ids == ["REC-INV-1013"]
    assert ack.claimed_amount == 1180.00
    assert ack.acknowledged_by == "Tomas Reid"
    assert ack.acknowledged_at
    report = service.payment_run_control_report(run.run_id)
    assert report.ack_required_count == 0
    assert _line(report, "INV-1001").acknowledged_by == "Tomas Reid"
    confirmed = service.confirm_payment_run(run.run_id, actor="Priya Nair")
    assert confirmed.status == "confirmed"


def test_acknowledging_the_wrong_things_is_refused(service, run):
    with pytest.raises(ValueError, match="no open recovery claim"):
        service.acknowledge_run_recovery(run.run_id, "INV-1002", actor="T")
    with pytest.raises(ValueError, match="excluded"):
        service.acknowledge_run_recovery(run.run_id, "INV-1008", actor="T")
    with pytest.raises(ValueError, match="not a member"):
        service.acknowledge_run_recovery(run.run_id, "INV-1007", actor="T")
    service.acknowledge_run_recovery(run.run_id, "INV-1001", actor="Tomas Reid")
    with pytest.raises(ValueError, match="already acknowledged"):
        service.acknowledge_run_recovery(run.run_id, "INV-1001", actor="Priya")


def test_no_acknowledgement_after_confirmation(service, run):
    service.acknowledge_run_recovery(run.run_id, "INV-1001", actor="Tomas Reid")
    service.confirm_payment_run(run.run_id, actor="Priya Nair")
    with pytest.raises(ValueError, match="already confirmed"):
        service.acknowledge_run_recovery(run.run_id, "INV-1001", actor="Tomas")


def test_a_claim_that_lands_after_proposal_still_gates():
    """The gate reads the records at confirmation time: a recovery
    case opened after the run was proposed still blocks it."""
    svc = InvoiceService(store=InMemoryStore(), checkpointer=False)
    svc.analyze(SAMPLES["INV-1001"])
    svc.approve("INV-1001", approver="Priya Nair")
    svc.analyze(SAMPLES["INV-1013"])
    run = svc.propose_payment_runs(as_of="2026-10-05")[0]
    svc.draft_duplicate_recovery("INV-1013")  # the claim opens now
    with pytest.raises(ValueError, match="cannot be confirmed yet"):
        svc.confirm_payment_run(run.run_id, actor="Priya Nair")


def test_a_clean_run_confirms_without_acknowledgements():
    svc = InvoiceService(store=InMemoryStore(), checkpointer=False)
    svc.analyze(SAMPLES["INV-1002"])
    svc.approve("INV-1002", approver="Priya Nair")
    run = svc.propose_payment_runs(as_of="2026-10-05")[0]
    report = svc.payment_run_control_report(run.run_id)
    assert report.ack_required_count == 0
    assert report.excluded_count == 0
    assert report.totals_agree is True
    assert svc.confirm_payment_run(run.run_id, actor="Priya").status == "confirmed"


def test_receipt_clears_the_flag():
    """Once the recovery is received in full, the supplier's line
    is no longer flagged — the gate follows the money."""
    svc = InvoiceService(store=InMemoryStore(), checkpointer=False)
    svc.analyze(SAMPLES["INV-1001"])
    svc.approve("INV-1001", approver="Priya Nair")
    svc.analyze(SAMPLES["INV-1013"])
    svc.draft_duplicate_recovery("INV-1013")
    run = svc.propose_payment_runs(as_of="2026-10-05")[0]
    for step in ("supplier_contacted", "credit_received"):
        svc.transition_recovery_case("INV-1013", to_status=step, actor="Priya")
    report = svc.payment_run_control_report(run.run_id)
    assert report.ack_required_count == 0
    assert svc.confirm_payment_run(run.run_id, actor="Priya").status == "confirmed"


# --- endpoints ------------------------------------------------------------------

def test_control_report_endpoints(service, run):
    client = TestClient(create_app(service))
    missing = client.get("/payment-runs/RUN-NOPE/control-report")
    assert missing.status_code == 404
    response = client.get(f"/payment-runs/{run.run_id}/control-report")
    assert response.status_code == 200
    body = response.json()
    assert body["ack_required_count"] == 1
    assert body["totals_agree"] is True

    refused = client.post(
        f"/payment-runs/{run.run_id}/confirm", json={"actor": "Priya Nair"}
    )
    assert refused.status_code == 422
    assert "acknowledgement" in refused.json()["detail"]

    no_actor = client.post(
        f"/payment-runs/{run.run_id}/recovery-acknowledgements",
        json={"invoice_id": "INV-1001"},
    )
    assert no_actor.status_code == 422

    acked = client.post(
        f"/payment-runs/{run.run_id}/recovery-acknowledgements",
        json={"invoice_id": "INV-1001", "actor": "Tomas Reid"},
    )
    assert acked.status_code == 200
    assert acked.json()["recovery_acknowledgements"][0]["pack_ids"] == [
        "REC-INV-1013"
    ]
    confirmed = client.post(
        f"/payment-runs/{run.run_id}/confirm", json={"actor": "Priya Nair"}
    )
    assert confirmed.status_code == 200
    assert confirmed.json()["status"] == "confirmed"
