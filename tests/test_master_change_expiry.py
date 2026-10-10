"""Master-change expiry: a pending bank-detail change that sits
undecided past the configured window (MASTER_CHANGE_EXPIRY_DAYS,
default 30) expires — recorded in its audit history, decidable no
longer, and re-proposable in one step. A live credential change
must not wait indefinitely for a careless approval."""

from __future__ import annotations

from datetime import date, timedelta

import pytest
from fastapi.testclient import TestClient

from ap_agent.api import create_app
from ap_agent.matching import _parse_date
from ap_agent.schemas import SupplierMasterRecord
from ap_agent.service import InvoiceService
from ap_agent.store import InMemoryStore
from ap_agent.supplier_master import InMemorySupplierMaster, sample_supplier_master

PROPOSED = SupplierMasterRecord(
    supplier_name="Northbeam Office Supplies",
    account_number="6677889900",
    sort_code="30-12-44",
)


@pytest.fixture()
def store() -> InMemoryStore:
    return InMemoryStore()


@pytest.fixture()
def master() -> InMemorySupplierMaster:
    return InMemorySupplierMaster(sample_supplier_master())


@pytest.fixture()
def service(store, master) -> InvoiceService:
    return InvoiceService(store=store, checkpointer=False, supplier_master=master)


def _propose(service) -> object:
    return service.propose_master_change(
        PROPOSED, actor="Priya Nair", reason="supplier moved banks"
    )


def _backdate(store, change, days: int) -> None:
    """Move a change's proposal timestamp into the past."""
    proposed = _parse_date(change.proposed_at)
    change.proposed_at = (proposed - timedelta(days=days)).isoformat()
    store.save_master_change(change)


def _plus(change, days: int) -> str:
    return (_parse_date(change.proposed_at) + timedelta(days=days)).isoformat()


def test_sweep_expires_a_stale_pending_change(service, store, master):
    change = _propose(service)
    _backdate(store, change, 31)
    expired = service.expire_stale_master_changes()
    assert [c.request_id for c in expired] == ["MCR-0001"]
    after = service.get_master_change("MCR-0001")
    assert after.status == "expired"
    assert [e.status for e in after.history] == ["proposed", "expired"]
    event = after.history[-1]
    assert event.actor == "system"
    assert "expired" in event.note and "re-propose" in event.note
    # Expiry applied nothing and decided nothing.
    assert after.decided_by is None
    assert master.get_supplier("Northbeam Office Supplies").account_number == "1029384756"
    # The sweep is idempotent: an expired change does not re-expire.
    assert service.expire_stale_master_changes() == []


def test_the_window_boundary_is_exact(service):
    change = _propose(service)
    # On the window's last day the request is still decidable...
    assert service.expire_stale_master_changes(as_of=_plus(change, 30)) == []
    assert service.get_master_change("MCR-0001").status == "pending"
    # ...one day later it is not.
    expired = service.expire_stale_master_changes(as_of=_plus(change, 31))
    assert [c.request_id for c in expired] == ["MCR-0001"]


def test_the_window_is_configurable(service, store, monkeypatch):
    monkeypatch.setenv("MASTER_CHANGE_EXPIRY_DAYS", "5")
    change = _propose(service)
    assert service.expire_stale_master_changes(as_of=_plus(change, 5)) == []
    expired = service.expire_stale_master_changes(as_of=_plus(change, 6))
    assert [c.request_id for c in expired] == ["MCR-0001"]


def test_an_expired_change_cannot_be_decided(service, store):
    change = _propose(service)
    _backdate(store, change, 45)
    service.expire_stale_master_changes()
    with pytest.raises(ValueError, match="already expired"):
        service.decide_master_change(
            "MCR-0001",
            actor="Tomas Reid",
            approve=True,
            verification_ref="call-back confirmed",
        )
    with pytest.raises(ValueError, match="already expired"):
        service.decide_master_change(
            "MCR-0001", actor="Tomas Reid", approve=False
        )


def test_a_decision_attempt_after_the_window_expires_it_on_the_spot(
    service, store, master
):
    change = _propose(service)
    _backdate(store, change, 40)  # stale, but nobody ran the sweep
    with pytest.raises(ValueError, match="has expired"):
        service.decide_master_change(
            "MCR-0001",
            actor="Tomas Reid",
            approve=True,
            verification_ref="call-back confirmed",
        )
    after = service.get_master_change("MCR-0001")
    assert after.status == "expired"
    assert [e.status for e in after.history] == ["proposed", "expired"]
    assert master.get_supplier("Northbeam Office Supplies").account_number == "1029384756"


def test_an_expired_change_can_be_reproposed(service, store):
    change = _propose(service)
    _backdate(store, change, 40)
    service.expire_stale_master_changes()
    fresh = _propose(service)
    assert fresh.request_id == "MCR-0002"
    assert fresh.status == "pending"
    decided = service.decide_master_change(
        "MCR-0002",
        actor="Tomas Reid",
        approve=True,
        verification_ref="call-back to the number on file",
    )
    assert decided.status == "approved" and decided.applied_at


def test_decided_changes_never_expire(service):
    change = _propose(service)
    service.decide_master_change(
        change.request_id,
        actor="Tomas Reid",
        approve=True,
        verification_ref="call-back confirmed",
    )
    expired = service.expire_stale_master_changes(as_of="2099-01-01")
    assert expired == []
    assert service.get_master_change(change.request_id).status == "approved"


def test_expiry_endpoints(service, store):
    change = _propose(service)
    client = TestClient(create_app(service))
    swept = client.post(
        f"/suppliers/master-changes/expire-stale?as_of={_plus(change, 31)}"
    )
    assert swept.status_code == 200
    assert swept.json() == {"expired": ["MCR-0001"]}
    body = client.get("/suppliers/master-changes/MCR-0001").json()
    assert body["status"] == "expired"
    assert body["history"][-1]["status"] == "expired"
    decided = client.post(
        "/suppliers/master-changes/MCR-0001/decide",
        json={"actor": "Tomas Reid", "approve": False},
    )
    assert decided.status_code == 422
    assert "already expired" in decided.json()["detail"]
    bad = client.post("/suppliers/master-changes/expire-stale?as_of=nope")
    assert bad.status_code == 422
