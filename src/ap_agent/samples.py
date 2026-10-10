"""Bundled synthetic sample invoices.

The samples ship inside the package (``data/sample_invoices.json``)
so the CLI, demo, API, and console work identically from a source
checkout and a pip install. Every invoice is synthetic — invented
suppliers, suppliers, and loss histories, written for this blueprint.
"""

from __future__ import annotations

import json
from pathlib import Path

from .schemas import CreditNoteInput, InvoiceBundleInput, InvoiceInput

_DATA = Path(__file__).parent / "data" / "sample_invoices.json"
_CREDIT_DATA = Path(__file__).parent / "data" / "sample_credit_notes.json"
_BUNDLE_DATA = Path(__file__).parent / "data" / "sample_bundles.json"


def load_sample_invoices() -> list[dict]:
    return json.loads(_DATA.read_text(encoding="utf-8"))


def sample_invoice_models() -> list[InvoiceInput]:
    return [InvoiceInput.model_validate(item) for item in load_sample_invoices()]


def load_sample_credit_notes() -> list[dict]:
    return json.loads(_CREDIT_DATA.read_text(encoding="utf-8"))


def sample_credit_note_models() -> list[CreditNoteInput]:
    return [CreditNoteInput.model_validate(item) for item in load_sample_credit_notes()]


def load_sample_bundles() -> list[dict]:
    return json.loads(_BUNDLE_DATA.read_text(encoding="utf-8"))


def sample_bundle_models() -> list[InvoiceBundleInput]:
    return [InvoiceBundleInput.model_validate(item) for item in load_sample_bundles()]
