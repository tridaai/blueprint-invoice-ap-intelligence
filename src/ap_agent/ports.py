"""Ports: the seams of the agent, declared in one place.

The pipeline depends on replaceable collaborators, each a
``typing.Protocol`` (structural — an implementation qualifies by
having the methods, not by inheriting):

- :class:`ModelBackend` — the language model. Real providers
  (NVIDIA NIM as the lead profile, OpenAI / Anthropic / Ollama) in
  provider mode; the deterministic template renderer as the offline
  fallback. Implementations live in ``model_backends.py``.
- :class:`OcrEngine` — document text extraction. The NIM profile
  calls an OCR endpoint; the offline implementation returns the
  document's provided text layer. Implementations live in ``ocr.py``.
- :class:`DocVision` — document type + legibility verification via a
  vision model, with a deterministic metadata-based offline
  equivalent. Implementations live in ``vision.py``.
- :class:`Retriever` — policy retrieval (keyword / semantic over
  pgvector / hybrid with RRF fusion). Implementations live in
  ``retriever.py``.
- :class:`Store` — where analysis records and human decisions live.
  The production implementation is PostgreSQL (``store.py``); the
  in-memory implementation is a test double.
- :class:`ObjectStore` — document blobs (scans, attachments) in
  S3-compatible object storage. Documents reference object keys,
  never local paths. Implementations live in ``storage.py``.
- :class:`EventSink` — where structured run events go (streaming,
  tracing). Implementations live in ``events.py``.
- :class:`Checkpointer` — LangGraph graph-state persistence behind
  the human-approval gate: the saver interface (the checkpoint
  library's ``BaseCheckpointSaver`` surface), implemented over
  PostgreSQL with psycopg in ``checkpoints.py``. The checkpointer
  holds *graph state*; the :class:`Store` remains the record of
  *decisions*.

``wiring.py`` is the composition root that builds the concrete
collaborators from configuration; the rest of the codebase programs
against these protocols. This module is deliberately a leaf: it
imports nothing from the package at runtime, so anything may import
it without a cycle.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Iterator, Protocol, Sequence

if TYPE_CHECKING:  # annotation-only imports (no runtime cycles)
    from langgraph.checkpoint.base import Checkpoint, CheckpointMetadata, CheckpointTuple
    from langchain_core.runnables import RunnableConfig

    from .events import RunEvent
    from .model_backends import DraftContext
    from .schemas import (
        DocumentInput,
        InvoiceInput,
        OcrResult,
        RetrievedPolicy,
        VisionResult,
    )
    from .store import ApprovalRecord


class ModelBackend(Protocol):
    """The language-model seam. Only drafting is mandatory: every other
    capability (extraction, triage cross-check, diagnosis, options,
    verification, review) is optional and discovered with ``getattr`` —
    a backend that lacks one simply engages that node's deterministic
    path. The offline fallback implements drafting only."""

    name: str

    def draft_documents(self, context: "DraftContext") -> tuple[str, str, str]:
        """Return (subject, supplier letter, AP packet) drafts."""
        ...


class OcrEngine(Protocol):
    """The OCR seam: one document in, its text layer + confidence out."""

    name: str

    def read(self, document: "DocumentInput") -> "OcrResult": ...


class DocVision(Protocol):
    """The document-vision seam: is this an invoice, and is it legible?"""

    name: str

    def verify(
        self, invoice: "InvoiceInput", ocr_results: list["OcrResult"]
    ) -> "VisionResult": ...


class Retriever(Protocol):
    """The policy-retrieval seam."""

    def retrieve(self, query: str, top_k: int = 3) -> list["RetrievedPolicy"]: ...


class Store(Protocol):
    """The record seam: analyses in, decisions and history out."""

    def save(self, record: "ApprovalRecord") -> None: ...

    def get(self, invoice_id: str) -> "ApprovalRecord | None": ...

    def prior_invoices(self, exclude_invoice_id: str | None = None) -> list[dict]: ...

    def decision_feedback(self, exclude_invoice_id: str | None = None) -> list[dict]: ...


class ObjectStore(Protocol):
    """The document-blob seam: bytes in/out by object key (S3-compatible)."""

    def put(self, key: str, data: bytes, content_type: str = "application/octet-stream") -> str: ...

    def get(self, key: str) -> bytes: ...

    def exists(self, key: str) -> bool: ...


class EventSink(Protocol):
    """The run-events seam: one ``emit`` per structured event."""

    def emit(self, event: "RunEvent") -> None: ...


class Checkpointer(Protocol):
    """The graph-state seam behind the approval gate.

    Mirrors the sync surface of LangGraph's ``BaseCheckpointSaver``;
    the shipped implementation (``checkpoints.py``) subclasses that
    base over PostgreSQL (psycopg), so it satisfies this protocol and
    LangGraph's own expectations at once.
    """

    def get_tuple(self, config: "RunnableConfig") -> "CheckpointTuple | None": ...

    def list(
        self,
        config: "RunnableConfig | None",
        *,
        filter: dict[str, Any] | None = None,
        before: "RunnableConfig | None" = None,
        limit: int | None = None,
    ) -> Iterator["CheckpointTuple"]: ...

    def put(
        self,
        config: "RunnableConfig",
        checkpoint: "Checkpoint",
        metadata: "CheckpointMetadata",
        new_versions: dict[str, Any],
    ) -> "RunnableConfig": ...

    def put_writes(
        self,
        config: "RunnableConfig",
        writes: Sequence[tuple[str, Any]],
        task_id: str,
        task_path: str = "",
    ) -> None: ...

    def delete_thread(self, thread_id: str) -> None: ...
