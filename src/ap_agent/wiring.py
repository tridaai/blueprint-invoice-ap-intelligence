"""Composition root: build the runtime collaborators from the environment.

Every surface (API, CLI, demo, evals) constructs its service through
here, so configuration means the same thing everywhere: the factories
in ``model_backends`` / ``retriever`` / ``store`` / ``checkpoints`` /
``ocr`` / ``vision`` / ``storage`` read the environment (with the
repo-root ``.env`` loaded), and tests inject doubles by passing them
explicitly.
"""

from __future__ import annotations

from .checkpoints import get_checkpointer
from .model_backends import get_backend
from .ocr import get_ocr_engine
from .ports import Checkpointer, DocVision, ModelBackend, OcrEngine, Retriever, Store
from .retriever import get_retriever
from .service import InvoiceService
from .storage import get_object_store
from .store import default_store
from .vision import get_vision_engine


def build_backend() -> ModelBackend:
    return get_backend()


def build_retriever() -> Retriever:
    return get_retriever()


def build_store() -> Store:
    return default_store()


def build_checkpointer() -> Checkpointer | None:
    return get_checkpointer()


def build_ocr_engine() -> OcrEngine:
    return get_ocr_engine()


def build_vision_engine() -> DocVision:
    return get_vision_engine()


def build_object_store():
    return get_object_store()


def build_service_from_env(
    *,
    backend: ModelBackend | None = None,
    retriever: Retriever | None = None,
    store: Store | None = None,
    checkpointer: Checkpointer | bool | None = None,
    ocr_engine: OcrEngine | None = None,
    vision_engine: DocVision | None = None,
    object_store=None,
) -> InvoiceService:
    """A service wired from the environment, with any collaborator
    overridable (tests and the CLI batch path inject doubles here).

    ``checkpointer=False`` forces the gate off regardless of the
    environment (the CLI batch path uses this). ``object_store=False``
    disables object storage; omitted means "from the environment".
    """
    return InvoiceService(
        backend=backend,
        retriever=retriever,
        store=store,
        checkpointer=checkpointer,
        ocr_engine=ocr_engine or build_ocr_engine(),
        vision_engine=vision_engine or build_vision_engine(),
        object_store=build_object_store() if object_store is None else (object_store or None),
    )
