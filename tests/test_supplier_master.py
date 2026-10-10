"""Supplier master data: the bank-change fraud screen checks the
bank identifiers a document states against the approved master
record (the SupplierMaster port). A mismatch is the fraud signal;
a match is evidence that downgrades a pattern-only hit; no master
record (or no details stated) keeps the pattern-only behaviour."""

from __future__ import annotations

import pytest

from ap_agent.samples import sample_invoice_models
from ap_agent.schemas import SupplierMasterRecord
from ap_agent.service import InvoiceService
from ap_agent.store import InMemoryStore
from ap_agent.supplier_master import (
    InMemorySupplierMaster,
    compare_bank_details,
    extract_bank_identifiers,
    sample_supplier_master,
)

SAMPLES = {s.invoice_id: s for s in sample_invoice_models()}


@pytest.fixture()
def service() -> InvoiceService:
    return InvoiceService(store=InMemoryStore(), checkpointer=False)


# --- extraction: bank context only --------------------------------------

def test_extracts_account_and_sort_code_from_bank_context():
    found = extract_bank_identifiers(
        "From now on, remit all payments to account number 8839204716, "
        "sort code 20-44-18. Do not use the old account."
    )
    assert found["account"] == {"8839204716"}
    assert found["sort_code"] == {"204418"}


def test_ignores_numbers_outside_bank_context():
    found = extract_bank_identifiers(
        "Invoice total 12345678 units shipped on 20260929 per order."
    )
    assert found == {}


def test_ignores_invoice_numbers_and_totals_in_plain_prose():
    found = extract_bank_identifiers(SAMPLES["INV-1001"].documents[0].text_layer)
    assert found == {}


# --- comparison verdicts -------------------------------------------------

def _harborline_master(account: str = "5520109873") -> SupplierMasterRecord:
    return SupplierMasterRecord(
        supplier_name="Harborline Packaging",
        account_number=account,
        sort_code="20-44-18",
    )


def test_compare_flags_the_fraud_samples_account_as_mismatch():
    check = compare_bank_details(SAMPLES["INV-1010"], _harborline_master(), source="memory")
    assert check.status == "checked_mismatch"
    assert check.mismatched == ["account ••••4716"]
    assert check.matched == ["sort_code ••••4418"]  # the sort code IS on file
    # Masked labels only — the full stated account never propagates.
    assert "8839204716" not in " ".join(check.mismatched + check.matched)


def test_compare_match_when_document_states_the_master_details():
    check = compare_bank_details(SAMPLES["INV-1014"], _harborline_master())
    assert check.status == "checked_match"
    assert not check.mismatched


def test_compare_no_master_record():
    check = compare_bank_details(SAMPLES["INV-1010"], None)
    assert check.status == "no_master_record"


def test_compare_no_details_in_documents():
    check = compare_bank_details(SAMPLES["INV-1001"], sample_supplier_master()[0])
    assert check.status == "no_details_in_documents"


# --- pipeline integration -------------------------------------------------

def test_pipeline_mismatch_blocks_and_names_the_master(service):
    result = service.analyze(SAMPLES["INV-1010"])
    assert result.bank_master_check is not None
    assert result.bank_master_check.status == "checked_mismatch"
    assert result.bank_change_blocked is True
    assert any("supplier master" in f for f in result.fraud_flags)


def test_pipeline_match_is_evidence_not_a_block(service):
    result = service.analyze(SAMPLES["INV-1014"])
    assert result.bank_master_check is not None
    assert result.bank_master_check.status == "checked_match"
    assert result.bank_change_blocked is False
    assert result.fraud_flags == []
    assert result.triage.outcome == "clean_match"


def test_pipeline_downgrades_pattern_hit_when_details_are_on_file():
    # The fraud sample's note asks to remit to account 8839204716.
    # When the master record says that IS the approved account, the
    # pattern hit is downgraded to evidence — no block, no flags.
    master = InMemorySupplierMaster([_harborline_master(account="8839204716")])
    svc = InvoiceService(
        store=InMemoryStore(), checkpointer=False, supplier_master=master
    )
    result = svc.analyze(SAMPLES["INV-1010"])
    assert result.bank_master_check.status == "checked_match"
    assert result.bank_change_blocked is False
    assert result.fraud_flags == []


def test_pipeline_without_master_record_keeps_pattern_behaviour():
    svc = InvoiceService(
        store=InMemoryStore(),
        checkpointer=False,
        supplier_master=InMemorySupplierMaster(),  # empty master
    )
    result = svc.analyze(SAMPLES["INV-1010"])
    assert result.bank_master_check.status == "no_master_record"
    assert result.bank_change_blocked is True  # the pattern still blocks


def test_master_upsert_then_get_roundtrip():
    master = InMemorySupplierMaster()
    master.upsert_supplier(_harborline_master())
    got = master.get_supplier("harborline packaging")  # case-insensitive key
    assert got is not None and got.account_number == "5520109873"
    assert master.get_supplier("Unknown Supplier") is None
