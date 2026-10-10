"""Persistence seams: the in-memory test double + migration hygiene.

The production stores (Postgres) are covered by the gated integration
tests; here we pin the store contract every implementation shares and
that the packaged migrations match the repo-root copies — production
runs them from the package, humans review them at the root.
"""

from __future__ import annotations

from pathlib import Path

from ap_agent.samples import sample_invoice_models
from ap_agent.service import InvoiceService
from ap_agent.store import InMemoryStore, find_duplicate

ROOT = Path(__file__).resolve().parents[1]


def test_inmemory_store_roundtrip_and_decision():
    store = InMemoryStore()
    service = InvoiceService(store=store, checkpointer=False)
    inv = sample_invoice_models()[0]
    service.analyze(inv)
    record = store.get("INV-1001")
    assert record is not None and record.result.invoice_id == "INV-1001"
    assert [h["invoice_id"] for h in store.prior_invoices(exclude_invoice_id="")] == ["INV-1001"]
    service.approve("INV-1001", approver="priya")
    assert store.get("INV-1001").result.status == "approved"
    assert store.ping() is True


def test_find_duplicate_requires_same_supplier_number_and_amount():
    priors = [
        {
            "invoice_id": "INV-1001",
            "supplier_name": "Northbeam Office Supplies",
            "invoice_number": "NB-4471",
            "total_amount": 1180.0,
        }
    ]
    dup = sample_invoice_models()[3]  # INV-1004
    assert find_duplicate(priors, dup) == "INV-1001"
    different_amount = sample_invoice_models()[3]
    different_amount.total_amount = 999.0
    assert find_duplicate(priors, different_amount) == ""


def test_migrations_packaged_and_root_copies_agree():
    root_dir = ROOT / "migrations"
    packaged_dir = ROOT / "src" / "ap_agent" / "migrations"
    root_files = sorted(p.name for p in root_dir.glob("*.sql"))
    packaged_files = sorted(p.name for p in packaged_dir.glob("*.sql"))
    assert root_files == packaged_files
    assert root_files, "migrations must exist"
    for name in root_files:
        assert (root_dir / name).read_text() == (packaged_dir / name).read_text(), name


def test_migrations_are_numbered_and_create_the_core_tables():
    from ap_agent.migrations import migration_files

    files = migration_files()
    versions = [version for version, _ in files]
    assert versions == sorted(versions)
    body = "\n".join(path.read_text() for _, path in files)
    assert "CREATE TABLE IF NOT EXISTS approvals" in body
    assert "CREATE TABLE IF NOT EXISTS policy_embeddings" in body
    assert "CREATE TABLE IF NOT EXISTS checkpoints" in body
    assert "CREATE EXTENSION" in body  # pgvector
