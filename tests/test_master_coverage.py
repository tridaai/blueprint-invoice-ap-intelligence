"""Master-coverage metrics: the bank-change screen is only as
strong as the supplier master behind it. /metrics now reports
what share of processed invoices (and of spend) the screen could
actually check against master data, the verdict mix it recorded,
and how many governed changes are in flight to close the gap."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from ap_agent.api import create_app
from ap_agent.samples import sample_invoice_models
from ap_agent.schemas import SupplierMasterRecord
from ap_agent.service import InvoiceService
from ap_agent.store import InMemoryStore
from ap_agent.supplier_master import (
    InMemorySupplierMaster,
    master_coverage,
    sample_supplier_master,
)

SAMPLES = {s.invoice_id: s for s in sample_invoice_models()}


@pytest.fixture()
def service() -> InvoiceService:
    """Three processed invoices against the seeded master:
    Northbeam (master, no bank details stated), Harborline
    (master, document states different details — the fraud case),
    Bluepine (no master record at all)."""
    svc = InvoiceService(
        store=InMemoryStore(),
        checkpointer=False,
        supplier_master=InMemorySupplierMaster(sample_supplier_master()),
    )
    svc.analyze(SAMPLES["INV-1001"])  # Northbeam, 1180.00
    svc.analyze(SAMPLES["INV-1010"])  # Harborline, 708.00
    svc.analyze(SAMPLES["INV-1005"])  # Bluepine, 590.00
    return svc


def test_coverage_counts_invoices_spend_and_suppliers(service):
    coverage = service.metrics_snapshot()["supplier_master"]
    assert coverage["invoices_total"] == 3
    assert coverage["invoices_with_master"] == 2
    assert coverage["coverage_pct"] == 66.7
    assert coverage["spend_total"] == 2478.00
    assert coverage["spend_with_master"] == 1888.00
    assert coverage["spend_coverage_pct"] == 76.2
    assert coverage["suppliers_total"] == 3
    assert coverage["suppliers_with_master"] == 2


def test_coverage_reports_the_verdict_mix(service):
    coverage = service.metrics_snapshot()["supplier_master"]
    assert coverage["by_check_status"] == {
        "no_details_in_documents": 1,  # Northbeam stated no bank details
        "checked_mismatch": 1,  # Harborline's stated account is not on file
        "no_master_record": 1,  # Bluepine — the screen's blind spot
    }


def test_coverage_nets_in_pending_changes(service):
    assert service.metrics_snapshot()["supplier_master"]["pending_changes"] == 0
    service.propose_master_change(
        SupplierMasterRecord(
            supplier_name="Bluepine Catering Co", account_number="6677889900"
        ),
        actor="Priya Nair",
        reason="onboarding the caterer",
    )
    coverage = service.metrics_snapshot()["supplier_master"]
    assert coverage["pending_changes"] == 1
    # A pending change is not coverage yet — the gap is unchanged
    # until the change is approved and applied.
    assert coverage["invoices_with_master"] == 2


def test_coverage_over_an_empty_store_is_null_not_zero():
    coverage = master_coverage([], InMemorySupplierMaster())
    assert coverage["invoices_total"] == 0
    assert coverage["coverage_pct"] is None
    assert coverage["spend_coverage_pct"] is None
    assert coverage["by_check_status"] == {}


def test_metrics_endpoint_carries_the_coverage(service):
    client = TestClient(create_app(service))
    body = client.get("/metrics").json()
    assert body["supplier_master"]["coverage_pct"] == 66.7
    assert body["supplier_master"]["by_check_status"]["checked_mismatch"] == 1
