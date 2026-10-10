"""Batch-booking bank feedback: the bank answers per PmtInf,
and the run records it per group.

A batch-booked pain.001 file is several payment instructions
in one message. The bank does not answer the file — it answers
each group: one books, another bounces. Until now that answer
lived in the bank portal and the run knew nothing. The receipts
close the loop: recorded against their groups (named by the
group's requested execution date), stamped with the group's
PmtInf id and its members from the export receipt, so every
member carries its group's fate on the run. The discipline is
the posting receipt's: recording is bookkeeping (no payment
state moves), a rejection must carry the bank's reason, an
accepted group is final, and a rejected group's later answer
supersedes the first — the recording validated whole, so one
bad answer refuses it entire.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from ap_agent.api import create_app
from ap_agent.handoff import (
    batch_group_members,
    current_group_receipts,
    group_receipt_by_member,
)
from ap_agent.schemas import HandoffGroupReceiptInput
from ap_agent.service import InvoiceService
from test_handoff_batch_booking import _service_with_run


def _batch_exported() -> tuple[InvoiceService, object]:
    """A batch-booked EUR export: groups 2026-10-20 (INV-1001)
    and 2026-10-21 (INV-1008), profile lead 3d floored below
    both due dates."""
    svc, run = _service_with_run()
    svc.set_handoff_profile(
        actor="Ana Ruiz", scope="currency", currency="EUR",
        format="pain001", lead_days=3, batch_booking=True,
    )
    exported = svc.export_payment_run_handoff(
        run.run_id, actor="Tomas Reid", as_of="2026-10-11"
    )
    assert exported.handoff_batch_booking is True
    return svc, exported


def _receipt(date: str, status: str, **kw) -> HandoffGroupReceiptInput:
    return HandoffGroupReceiptInput(execution_date=date, status=status, **kw)


# -- the recording --------------------------------------------------------------


def test_groups_read_back_from_the_export_receipt():
    _svc, exported = _batch_exported()
    assert batch_group_members(exported) == {
        "2026-10-20": ["INV-1001"],
        "2026-10-21": ["INV-1008"],
    }
    # Nothing answered yet: unmarked means unanswered.
    assert current_group_receipts(exported) == {}
    assert group_receipt_by_member(exported) == {}


def test_answers_are_recorded_per_group_and_mark_their_members():
    svc, exported = _batch_exported()
    run = svc.record_handoff_group_receipts(
        exported.run_id,
        actor="Tomas Reid",
        receipts=[
            _receipt("2026-10-20", "accepted", bank_reference="BNK-771"),
            _receipt(
                "2026-10-21",
                "rejected",
                detail="creditor account closed at the receiving bank",
            ),
        ],
    )
    receipts = {r.execution_date: r for r in run.handoff_group_receipts}
    accepted = receipts["2026-10-20"]
    assert accepted.pmt_inf_id == f"{exported.run_id}-2026-10-20"
    assert accepted.member_invoice_ids == ["INV-1001"]
    assert accepted.bank_reference == "BNK-771"
    assert accepted.recorded_by == "Tomas Reid"
    rejected = receipts["2026-10-21"]
    assert rejected.member_invoice_ids == ["INV-1008"]
    assert "account closed" in rejected.detail

    marked = group_receipt_by_member(run)
    assert marked["INV-1001"]["status"] == "accepted"
    assert marked["INV-1001"]["pmt_inf_id"] == accepted.pmt_inf_id
    assert marked["INV-1008"]["status"] == "rejected"
    assert marked["INV-1008"]["execution_date"] == "2026-10-21"


def test_a_rejected_group_can_be_answered_again_and_supersedes():
    svc, exported = _batch_exported()
    svc.record_handoff_group_receipts(
        exported.run_id,
        actor="Tomas Reid",
        receipts=[_receipt("2026-10-21", "rejected", detail="bad value date")],
    )
    run = svc.record_handoff_group_receipts(
        exported.run_id,
        actor="Tomas Reid",
        receipts=[
            _receipt("2026-10-21", "accepted", bank_reference="BNK-902")
        ],
    )
    assert len(run.handoff_group_receipts) == 2
    first, second = run.handoff_group_receipts
    assert first.superseded is True
    assert second.superseded is False
    standing = current_group_receipts(run)["2026-10-21"]
    assert standing.status == "accepted"
    assert group_receipt_by_member(run)["INV-1008"]["status"] == "accepted"


def test_a_fresh_export_clears_the_previous_files_answers():
    svc, exported = _batch_exported()
    svc.record_handoff_group_receipts(
        exported.run_id,
        actor="Tomas Reid",
        receipts=[_receipt("2026-10-20", "accepted")],
    )
    re_exported = svc.export_payment_run_handoff(
        exported.run_id, actor="Tomas Reid", as_of="2026-10-11"
    )
    assert re_exported.handoff_group_receipts == []


# -- the guards -------------------------------------------------------------------


def test_an_accepted_group_refuses_a_second_answer():
    svc, exported = _batch_exported()
    svc.record_handoff_group_receipts(
        exported.run_id,
        actor="Tomas Reid",
        receipts=[_receipt("2026-10-20", "accepted", bank_reference="BNK-1")],
    )
    with pytest.raises(ValueError, match="already stands accepted"):
        svc.record_handoff_group_receipts(
            exported.run_id,
            actor="Tomas Reid",
            receipts=[
                _receipt("2026-10-20", "rejected", detail="late reversal")
            ],
        )


def test_a_rejection_without_the_banks_reason_is_refused():
    svc, exported = _batch_exported()
    with pytest.raises(ValueError, match="no reason recorded"):
        svc.record_handoff_group_receipts(
            exported.run_id,
            actor="Tomas Reid",
            receipts=[_receipt("2026-10-20", "rejected")],
        )


def test_an_answer_for_a_group_the_file_never_made_is_refused():
    svc, exported = _batch_exported()
    with pytest.raises(ValueError, match="2026-10-20, 2026-10-21"):
        svc.record_handoff_group_receipts(
            exported.run_id,
            actor="Tomas Reid",
            receipts=[_receipt("2026-12-25", "accepted")],
        )


def test_one_bad_answer_refuses_the_whole_recording():
    svc, exported = _batch_exported()
    with pytest.raises(ValueError, match="Unknown group receipt status"):
        svc.record_handoff_group_receipts(
            exported.run_id,
            actor="Tomas Reid",
            receipts=[
                _receipt("2026-10-20", "accepted"),
                _receipt("2026-10-21", "queued"),
            ],
        )
    run = svc.get_payment_run(exported.run_id)
    assert run.handoff_group_receipts == []


def test_two_answers_for_one_group_in_one_recording_are_refused():
    svc, exported = _batch_exported()
    with pytest.raises(ValueError, match="answers a group once"):
        svc.record_handoff_group_receipts(
            exported.run_id,
            actor="Tomas Reid",
            receipts=[
                _receipt("2026-10-20", "rejected", detail="first word"),
                _receipt("2026-10-20", "accepted"),
            ],
        )


def test_a_flat_export_has_no_groups_to_answer():
    svc, run = _service_with_run()
    exported = svc.export_payment_run_handoff(
        run.run_id, actor="Tomas Reid", as_of="2026-10-11", format="csv"
    )
    assert exported.handoff_batch_booking is False
    with pytest.raises(ValueError, match="batch-booked pain.001"):
        svc.record_handoff_group_receipts(
            run.run_id,
            actor="Tomas Reid",
            receipts=[_receipt("2026-10-20", "accepted")],
        )


def test_a_single_pmtinf_pain001_export_refuses_group_answers():
    svc, run = _service_with_run()
    svc.set_handoff_profile(
        actor="Ana Ruiz", scope="currency", currency="EUR",
        format="pain001", lead_days=3, batch_booking=False,
    )
    svc.export_payment_run_handoff(run.run_id, actor="Tomas Reid", as_of="2026-10-11")
    with pytest.raises(ValueError, match="without batch booking"):
        svc.record_handoff_group_receipts(
            run.run_id,
            actor="Tomas Reid",
            receipts=[_receipt("2026-10-20", "accepted")],
        )


def test_an_unexported_run_refuses_answers():
    svc, run = _service_with_run()
    with pytest.raises(ValueError, match="no hand-off export"):
        svc.record_handoff_group_receipts(
            run.run_id,
            actor="Tomas Reid",
            receipts=[_receipt("2026-10-20", "accepted")],
        )


# -- the API -----------------------------------------------------------------------


def test_group_receipts_endpoint():
    svc, exported = _batch_exported()
    client = TestClient(create_app(svc))
    response = client.post(
        f"/payment-runs/{exported.run_id}/handoff/group-receipts",
        json={
            "actor": "Tomas Reid",
            "receipts": [
                {
                    "execution_date": "2026-10-21",
                    "status": "rejected",
                    "detail": "receiving bank refused the value date",
                }
            ],
        },
    )
    assert response.status_code == 200
    body = response.json()
    (receipt,) = body["handoff_group_receipts"]
    assert receipt["execution_date"] == "2026-10-21"
    assert receipt["member_invoice_ids"] == ["INV-1008"]
    refused = client.post(
        f"/payment-runs/{exported.run_id}/handoff/group-receipts",
        json={
            "actor": "Tomas Reid",
            "receipts": [{"execution_date": "2026-10-21", "status": "rejected"}],
        },
    )
    assert refused.status_code == 422
    missing = client.post(
        "/payment-runs/RUN-XXXX/handoff/group-receipts",
        json={
            "actor": "Tomas Reid",
            "receipts": [{"execution_date": "2026-10-21", "status": "accepted"}],
        },
    )
    assert missing.status_code == 404
