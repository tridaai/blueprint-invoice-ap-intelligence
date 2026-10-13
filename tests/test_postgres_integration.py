"""PostgreSQL integration tests — gated on DATABASE_URL.

These exercise the production persistence path end to end:
migrations, the Postgres approvals store, the Postgres LangGraph
checkpointer (pause → resume across saver instances), and the
pgvector policy retriever. They SKIP unless DATABASE_URL points at a
reachable PostgreSQL with the pgvector extension available (e.g. the
compose stack: `docker compose up -d postgres`).

    DATABASE_URL=postgresql://ap:ap@localhost:5432/ap pytest tests/test_postgres_integration.py
"""

from __future__ import annotations

import os

import pytest

DATABASE_URL = os.environ.get("DATABASE_URL", "")

pytestmark = pytest.mark.skipif(
    not DATABASE_URL.startswith(("postgres://", "postgresql://")),
    reason="DATABASE_URL not set — Postgres integration tests are opt-in",
)


def test_postgres_store_roundtrip():
    from ap_agent.samples import sample_invoice_models
    from ap_agent.service import InvoiceService
    from ap_agent.store import PostgresStore

    store = PostgresStore(DATABASE_URL)
    try:
        assert store.ping() is True
        service = InvoiceService(store=store, checkpointer=False)
        result = service.analyze(sample_invoice_models()[0])
        assert result.triage.outcome == "clean_match"
        stored = store.get_result("INV-1001")
        assert stored is not None and stored.invoice_id == "INV-1001"
        service.approve("INV-1001", approver="integration-test", reason="approved in test")
        assert store.get_result("INV-1001").status == "approved"
    finally:
        store.close()


def test_postgres_checkpointer_pause_and_resume():
    from ap_agent.checkpoints import PostgresCheckpointSaver
    from ap_agent.graph import resume_approval, run_invoice
    from ap_agent.model_backends import MockModelBackend
    from ap_agent.retriever import KeywordRetriever
    from ap_agent.samples import sample_invoice_models

    saver = PostgresCheckpointSaver(DATABASE_URL)
    invoice = sample_invoice_models()[0]
    paused = run_invoice(
        invoice,
        backend=MockModelBackend(),
        retriever=KeywordRetriever(),
        checkpointer=saver,
        thread_id="integration-thread-1",
    )
    assert paused.status == "awaiting_approver"
    # A fresh saver instance (a "restart") resumes from Postgres state.
    saver2 = PostgresCheckpointSaver(DATABASE_URL)
    resumed = resume_approval(
        "INV-1001",
        {
            "decision": "approved",
            "decided_by": "integration-test",
            "reason": "verified in the integration test",
        },
        saver2,
    )
    assert resumed is True


def test_pgvector_retriever_with_fake_embeddings(monkeypatch):
    import ap_agent.retriever as retriever_module

    class _Embeddings:
        def __init__(self):
            self.calls = 0

        def create(self, *, model, input):
            import hashlib

            self.calls += 1

            class _Item:
                def __init__(self, vec):
                    self.embedding = vec

            class _Resp:
                def __init__(self, vec):
                    self.data = [_Item(vec)]

            digest = hashlib.sha256(input.encode()).digest()
            vec = [b / 255.0 for b in digest[:16]]
            return _Resp(vec)

    class _Client:
        def __init__(self):
            self.embeddings = _Embeddings()

    monkeypatch.setattr(
        retriever_module, "_build_embeddings_client", lambda mode_label: _Client()
    )
    monkeypatch.setenv("DATABASE_URL", DATABASE_URL)
    retriever = retriever_module.PgVectorRetriever("semantic")
    results = retriever.retrieve("three-way match tolerance", top_k=3)
    assert results, "pgvector retrieval must return policies once indexed"
    assert all(r.sources == ["semantic"] for r in results)
