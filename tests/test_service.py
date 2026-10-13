"""Service layer: decisions, the gate, batch, feedback, readiness."""

from __future__ import annotations

import pytest

from ap_agent.samples import sample_invoice_models
from ap_agent.service import InvoiceService
from ap_agent.store import InMemoryStore

SAMPLES = sample_invoice_models()


@pytest.fixture()
def service():
    return InvoiceService(store=InMemoryStore(), checkpointer=False)


def test_approve_records_decision(service):
    result = service.analyze(SAMPLES[0])
    decided = service.approve(result.invoice_id, approver="priya.finance", reason="match verified")
    assert decided.status == "approved"
    assert decided.decided_by == "priya.finance"
    assert decided.decision_reason == "match verified"
    assert decided.external_action_taken is False  # no webhook configured


def test_reject_records_decision(service):
    result = service.analyze(SAMPLES[1])
    decided = service.reject(result.invoice_id, reviewer="omar.finance", reason="price increase not agreed")
    assert decided.status == "rejected"
    assert decided.decided_by == "omar.finance"


def test_approve_unknown_invoice_404s(service):
    with pytest.raises(KeyError):
        service.approve("INV-9999", approver="x", reason="approved in test")


def test_double_decision_refused(service):
    result = service.analyze(SAMPLES[0])
    service.approve(result.invoice_id, approver="a", reason="approved in test")
    with pytest.raises(ValueError):
        service.approve(result.invoice_id, approver="a", reason="approved in test")


def test_guardrail_failed_draft_cannot_be_approved(service):
    result = service.analyze(SAMPLES[11])  # INV-1012, the guardrail trap
    assert result.validation.passed is False
    with pytest.raises(ValueError, match="guardrail"):
        service.approve(result.invoice_id, approver="a", reason="approved in test")


def test_batch_preserves_order_and_isolates_errors(service):
    bad = {"invoice_id": "BAD-1", "total_amount": "not-a-number"}
    items = service.analyze_batch([SAMPLES[0], bad, SAMPLES[1]])
    assert [i.invoice_id for i in items] == ["INV-1001", "BAD-1", "INV-1002"]
    assert items[0].result is not None
    assert items[1].error is not None
    assert items[2].result is not None


def test_batch_concurrent_matches_sequential(service):
    # Samples 0-2 carry no cross-invoice dependencies; the duplicate
    # pair (INV-1001/INV-1004) is deliberately excluded — duplicate
    # detection reads store state, so those two must not race.
    sequential = service.analyze_batch(SAMPLES[:3], concurrency=1)
    other = InvoiceService(store=InMemoryStore(), checkpointer=False)
    concurrent = other.analyze_batch(SAMPLES[:3], concurrency=3)
    assert [i.result.triage.outcome for i in sequential] == [
        i.result.triage.outcome for i in concurrent
    ]


def test_feedback_loop_teaches_later_runs(service):
    first = service.analyze(SAMPLES[1])  # Cascade IT, price variance
    service.reject(first.invoice_id, reviewer="omar.finance", reason="supplier raised prices without agreement")
    second = service.analyze(SAMPLES[8])  # Cascade IT again (INV-1009)
    evidence = " ".join(second.diagnosis.evidence)
    assert "approver feedback" in evidence
    assert "supplier raised prices" in evidence


def test_history_evidence_appears_for_repeat_supplier(service):
    service.analyze(SAMPLES[0])
    again = service.analyze(SAMPLES[7])  # Northbeam again (INV-1008)
    assert any("supplier history" in line for line in again.diagnosis.evidence)


def test_readiness_reports_store(service):
    report = service.readiness()
    assert report["ready"] is True
    assert report["store"] == "memory"


def test_webhook_dispatch_when_configured(service, monkeypatch):
    import ap_agent.service as service_module

    captured = {}

    class _Response:
        status = 200

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

    def fake_urlopen(request, timeout=0):
        captured["url"] = request.full_url
        return _Response()

    monkeypatch.setattr(service_module.urllib.request, "urlopen", fake_urlopen)
    monkeypatch.setenv("ACTION_WEBHOOK_URL", "https://example.invalid/hook")
    result = service.analyze(SAMPLES[0])
    decided = service.approve(result.invoice_id, approver="a", reason="approved in test")
    assert decided.dispatch_status == "sent"
    assert decided.external_action_taken is True
    assert captured["url"] == "https://example.invalid/hook"
