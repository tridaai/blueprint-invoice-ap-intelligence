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
- :class:`SupplierMaster` — supplier master data: the approved bank
  details + terms AP verified out-of-band. The bank-change fraud
  screen checks document-stated details against this record — a
  mismatch is the fraud signal, a match is evidence. Implementations
  live in ``supplier_master.py`` (PostgreSQL in production, an
  in-memory double for tests/offline).
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
    from .schemas import BundleResult, MasterChangeRequest, PaymentRun, RecoveryPack
    from .schemas import EscalationDigest, StatementReconciliation, SupplierMasterRecord
    from .schemas import (
        HandoffProfile,
        LagSegmentProposal,
        ProfileChangeProposal,
        SupplierContactProposal,
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

    def list_records(self) -> list["ApprovalRecord"]: ...

    def save_reconciliation(self, report: "StatementReconciliation") -> None: ...

    def get_reconciliation(
        self, statement_id: str
    ) -> "StatementReconciliation | None": ...

    def list_reconciliations(self) -> list["StatementReconciliation"]: ...

    def save_payment_run(self, run: "PaymentRun") -> None: ...

    def get_payment_run(self, run_id: str) -> "PaymentRun | None": ...

    def list_payment_runs(self) -> list["PaymentRun"]: ...

    def save_recovery_pack(self, pack: "RecoveryPack") -> None: ...

    def get_recovery_pack(self, invoice_id: str) -> "RecoveryPack | None": ...

    def list_recovery_packs(self) -> list["RecoveryPack"]: ...

    def save_master_change(self, change: "MasterChangeRequest") -> None: ...

    def get_master_change(self, request_id: str) -> "MasterChangeRequest | None": ...

    def list_master_changes(self) -> list["MasterChangeRequest"]: ...

    def save_bundle(self, bundle: "BundleResult") -> None: ...

    def get_bundle(self, bundle_id: str) -> "BundleResult | None": ...

    def save_digest(self, digest: "EscalationDigest") -> None: ...

    def get_digest(self, digest_date: str) -> "EscalationDigest | None": ...

    def list_digests(self) -> list["EscalationDigest"]: ...

    def save_handoff_profile(self, profile: "HandoffProfile") -> None: ...

    def get_handoff_profile(self, profile_id: str) -> "HandoffProfile | None": ...

    def list_handoff_profiles(self) -> list["HandoffProfile"]: ...

    def delete_handoff_profile(self, profile_id: str) -> None: ...

    def save_profile_change_proposal(self, proposal: "ProfileChangeProposal") -> None: ...

    def get_profile_change_proposal(self, proposal_id: str) -> "ProfileChangeProposal | None": ...

    def list_profile_change_proposals(self) -> list["ProfileChangeProposal"]: ...

    def save_lag_segment_proposal(self, proposal: "LagSegmentProposal") -> None: ...

    def get_lag_segment_proposal(self, proposal_id: str) -> "LagSegmentProposal | None": ...

    def list_lag_segment_proposals(self) -> list["LagSegmentProposal"]: ...

    def save_supplier_contact_proposal(self, proposal: "SupplierContactProposal") -> None: ...

    def get_supplier_contact_proposal(self, proposal_id: str) -> "SupplierContactProposal | None": ...

    def list_supplier_contact_proposals(self) -> list["SupplierContactProposal"]: ...

    def prior_invoices(self, exclude_invoice_id: str | None = None) -> list[dict]: ...

    def decision_feedback(self, exclude_invoice_id: str | None = None) -> list[dict]: ...


class SupplierMaster(Protocol):
    """The supplier master-data seam: the out-of-band source of truth
    for a supplier's approved bank details and terms (AP-BANK-01).
    Documents state details; this port says what is on file."""

    name: str

    def get_supplier(self, supplier_name: str) -> "SupplierMasterRecord | None": ...

    def upsert_supplier(self, record: "SupplierMasterRecord") -> None: ...


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
