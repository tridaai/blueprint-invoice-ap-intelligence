"""The ERP posting example adapter (examples/erp_posting.py):
shapes an approved result into a balanced generic journal, refuses
anything unapproved or unreconciled, and derives a stable
idempotency key. The example is stdlib-only; it is loaded from the
examples directory the way a customer would copy it."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

from ap_agent.samples import sample_invoice_models
from ap_agent.service import InvoiceService
from ap_agent.store import InMemoryStore

_EXAMPLE = Path(__file__).resolve().parents[1] / "examples" / "erp_posting.py"


def _load_adapter():
    spec = importlib.util.spec_from_file_location("erp_posting_example", _EXAMPLE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture()
def approved_result() -> dict:
    svc = InvoiceService(store=InMemoryStore(), checkpointer=False)
    sample = {s.invoice_id: s for s in sample_invoice_models()}["INV-1001"]
    svc.analyze(sample)
    result = svc.approve("INV-1001", approver="Priya Nair", reason="clean match")
    return result.model_dump(mode="json")


def test_posting_is_a_balanced_journal_from_the_computed_facts(approved_result):
    adapter = _load_adapter()
    posting = adapter.build_posting(approved_result)
    assert posting["document_type"] == "vendor_invoice_posting"
    assert posting["external_reference"] == "NB-4471"
    assert posting["balanced"] is True
    assert posting["total_debit"] == posting["total_credit"] == 1180.0
    lines = {line["account"]: line for line in posting["journal_lines"]}
    assert lines["6100-PURCHASES"]["debit"] == 1000.0  # lines total, net of tax
    assert lines["1310-INPUT-TAX-RECOVERABLE"]["debit"] == 180.0
    assert lines["2000-ACCOUNTS-PAYABLE-CONTROL"]["credit"] == 1180.0


def test_idempotency_key_is_stable_and_invoice_specific(approved_result):
    adapter = _load_adapter()
    first = adapter.build_posting(approved_result)
    second = adapter.build_posting(approved_result)
    assert first["idempotency_key"] == second["idempotency_key"]
    other = dict(approved_result, invoice_id="INV-9999")
    assert adapter.idempotency_key(other) != first["idempotency_key"]


def test_unapproved_results_are_refused(approved_result):
    adapter = _load_adapter()
    awaiting = dict(approved_result, status="awaiting_approver")
    with pytest.raises(ValueError, match="only an approved result"):
        adapter.build_posting(awaiting)


def test_unreconciled_figures_are_refused(approved_result):
    adapter = _load_adapter()
    broken = dict(approved_result)
    broken["facts"] = dict(approved_result["facts"], invoice_total_stated=9999.0)
    with pytest.raises(ValueError, match="do not reconcile"):
        adapter.build_posting(broken)


# --- the receipt loop: the ERP's answer, shaped for the endpoint -------

def test_receipt_carries_the_postings_key_and_reference(approved_result):
    adapter = _load_adapter()
    posting = adapter.build_posting(approved_result)
    receipt = adapter.build_receipt(
        posting,
        outcome="posted",
        external_reference="ERP-DOC-1042",
        recorded_by="Tomas Reid",
    )
    assert receipt["invoice_id"] == "INV-1001"
    assert receipt["status"] == "posted"
    assert receipt["external_reference"] == "ERP-DOC-1042"
    assert receipt["idempotency_key"] == posting["idempotency_key"]
    assert receipt["actor"] == "Tomas Reid"


def test_receipt_guards_mirror_the_endpoint(approved_result):
    adapter = _load_adapter()
    posting = adapter.build_posting(approved_result)
    with pytest.raises(ValueError, match="document number"):
        adapter.build_receipt(posting, outcome="posted")
    with pytest.raises(ValueError, match="reason"):
        adapter.build_receipt(posting, outcome="failed")
    with pytest.raises(ValueError, match="'posted' or 'failed'"):
        adapter.build_receipt(posting, outcome="queued", external_reference="X")
    failed = adapter.build_receipt(
        posting, outcome="failed", detail="period 2026-09 is closed"
    )
    assert failed["detail"] == "period 2026-09 is closed"


def test_receipt_cli_mode(approved_result, tmp_path, capsys):
    import json as _json

    adapter = _load_adapter()
    posting_path = tmp_path / "posting.json"
    posting_path.write_text(
        _json.dumps(adapter.build_posting(approved_result)), encoding="utf-8"
    )
    assert adapter.main(
        ["erp_posting.py", "--receipt", str(posting_path), "posted", "ERP-DOC-1042"]
    ) == 0
    receipt = _json.loads(capsys.readouterr().out)
    assert receipt["status"] == "posted"
    assert receipt["external_reference"] == "ERP-DOC-1042"
    assert adapter.main(
        ["erp_posting.py", "--receipt", str(posting_path), "posted"]
    ) == 1  # no reference: refused before the endpoint ever sees it
