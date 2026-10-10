"""Document-vision port — the second capability port this blueprint adds.

Before an invoice is processed, someone has to answer two questions a
text pipeline cannot: *is this document actually an invoice*, and *is
it legible enough to trust the numbers*? This port answers them:

- ``MetadataVision`` — the deterministic offline equivalent. It reads
  the signals the submitting system and the OCR pass already provide:
  the document's (normalised) type, its scan-quality metadata, and the
  OCR confidence. Deterministic, hermetic, labelled.
- ``NimVision`` — the NVIDIA NIM profile: a Nemotron vision-language
  model call (OpenAI-compatible chat shape, ``VISION_MODEL`` /
  ``VISION_BASE_URL``) asked for a structured verdict over the
  document's text layer and metadata. (A production deployment would
  pass the page image; the adapter seam is this one class.)

Live NIM verification is a documented follow-up, as with the OCR
adapter; both implementations are fake-tested here.
"""

from __future__ import annotations

from .config import env_float, env_str, load_dotenv
from .errors import translate_provider_error
from .ocr import LOW_CONFIDENCE_THRESHOLD
from .schemas import DocumentInput, InvoiceInput, OcrResult, VisionResult

_VISION_SYSTEM = (
    "You verify supplier documents for an accounts-payable pipeline. "
    "Given a document's text and metadata, respond with a single JSON "
    "object: {\"document_type\": your best classification (invoice, "
    "purchase_order, goods_receipt, supplier_email, credit_note, "
    "remittance_note, statement, other), \"is_invoice\": true/false, "
    "\"legible\": true/false, \"quality\": good|poor|unreadable, "
    "\"confidence\": 0..1}. Judge only what you are given."
)


class MetadataVision:
    """Deterministic verdict from provided metadata + OCR results."""

    name = "metadata"

    def verify(self, invoice: InvoiceInput, ocr_results: list[OcrResult]) -> VisionResult:
        invoice_docs = [d for d in invoice.documents if d.doc_type == "invoice"]
        primary = invoice_docs[0] if invoice_docs else (
            invoice.documents[0] if invoice.documents else None
        )
        if primary is None:
            return VisionResult(
                document_type_detected="none",
                is_invoice=False,
                legible=False,
                quality="unreadable",
                confidence=0.9,
                source=self.name,
                note="no documents on the invoice",
            )
        ocr = next(
            (r for r in ocr_results if r.document_id == primary.document_id),
            None,
        )
        mean_conf = ocr.mean_confidence if ocr else 1.0
        quality_hint = primary.scan_quality
        if quality_hint in {"good", "poor", "unreadable"}:
            quality = quality_hint
        elif mean_conf < LOW_CONFIDENCE_THRESHOLD:
            quality = "unreadable"
        elif mean_conf < 0.85:
            quality = "poor"
        else:
            quality = "good"
        return VisionResult(
            document_type_detected=primary.doc_type,
            is_invoice=primary.doc_type == "invoice",
            legible=quality != "unreadable",
            quality=quality,
            confidence=round(mean_conf, 3),
            source=self.name,
            note="deterministic verdict from document metadata + OCR confidence",
        )


class NimVision:
    """NVIDIA NIM vision-language profile (chat shape)."""

    name = "vlm"

    def __init__(self) -> None:
        load_dotenv()
        api_key = env_str("NVIDIA_API_KEY") or env_str("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError(
                "VISION_ENGINE=nim but no API key is set. Add NVIDIA_API_KEY="
                "<your nvapi- key> to .env (from build.nvidia.com), or set "
                "VISION_ENGINE=metadata."
            )
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError(
                "VISION_ENGINE=nim needs the OpenAI SDK (the llm extra): "
                "uv sync --extra llm"
            ) from exc
        self.base_url = env_str("VISION_BASE_URL", "https://integrate.api.nvidia.com/v1")
        self._model = env_str("VISION_MODEL", "nvidia/llama-3.1-nemotron-nano-vl-8b-v1")
        self._client = OpenAI(
            api_key=api_key,
            base_url=self.base_url,
            timeout=env_float("LLM_TIMEOUT_SECONDS", 60.0),
            max_retries=0,
        )

    def verify(self, invoice: InvoiceInput, ocr_results: list[OcrResult]) -> VisionResult:
        from .model_backends import parse_json_object

        primary = next(
            (d for d in invoice.documents if d.doc_type == "invoice"),
            invoice.documents[0] if invoice.documents else DocumentInput(doc_type=""),
        )
        user = (
            f"Declared type: {primary.doc_type}\n"
            f"Scan quality metadata: {primary.scan_quality or 'not provided'}\n"
            f"Document text:\n---\n{primary.text_layer[:3000]}\n---"
        )
        try:
            response = self._client.chat.completions.create(
                model=self._model,
                max_tokens=300,
                messages=[
                    {"role": "system", "content": _VISION_SYSTEM},
                    {"role": "user", "content": user},
                ],
            )
        except Exception as exc:
            raise translate_provider_error(
                exc, backend="nim-vision", base_url=self.base_url
            ) from exc
        data = parse_json_object(response.choices[0].message.content or "")
        if data is None:
            fallback = MetadataVision().verify(invoice, ocr_results)
            fallback.note = "VLM reply unusable — metadata verdict used"
            return fallback
        quality = str(data.get("quality", "good")).lower()
        if quality not in {"good", "poor", "unreadable"}:
            quality = "good"
        try:
            confidence = float(data.get("confidence", 0.0))
        except (TypeError, ValueError):
            confidence = 0.0
        return VisionResult(
            document_type_detected=str(data.get("document_type", primary.doc_type)),
            is_invoice=bool(data.get("is_invoice", False)),
            legible=bool(data.get("legible", quality != "unreadable")),
            quality=quality,
            confidence=min(max(confidence, 0.0), 1.0),
            source=self.name,
            note=f"VLM verdict via {self._model}",
        )


def get_vision_engine(name: str | None = None) -> MetadataVision | NimVision:
    """Select the vision engine: ``VISION_ENGINE`` = metadata (default) | nim."""
    load_dotenv()
    selected = (name or env_str("VISION_ENGINE") or "metadata").lower()
    if selected == "metadata":
        return MetadataVision()
    if selected == "nim":
        return NimVision()
    raise ValueError(f"Unknown VISION_ENGINE: {selected!r} (expected metadata | nim)")


def run_vision(invoice: InvoiceInput, ocr_results: list[OcrResult], engine) -> VisionResult:
    """Verify the invoice's primary document. A failing engine
    degrades to the metadata verdict — verification is evidence,
    never a run failure."""
    try:
        return engine.verify(invoice, ocr_results)
    except Exception as exc:
        fallback = MetadataVision().verify(invoice, ocr_results)
        fallback.note = f"vision engine failed ({exc}) — metadata verdict used"
        return fallback
