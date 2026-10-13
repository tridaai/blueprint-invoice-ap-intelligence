"""OCR port — the first of the two capability ports this blueprint adds.

Accounts payable starts with paper and scans: the invoice is an image
long before it is data. This port turns a document into text plus a
confidence, behind one interface:

- ``OfflineOcrEngine`` — the hermetic default. It returns the
  document's own text layer (``DocumentInput.text_layer``), i.e. the
  text the sample or the submitting system already carries, scored
  with the document's provided ``ocr_confidence`` metadata when
  present. No network, fully deterministic — the labelled offline
  fallback, exactly like the mock model backend.
- ``NimOcrEngine`` — the NVIDIA NIM profile. It calls an OCR NIM
  endpoint (``OCR_BASE_URL``, default the hosted catalog) with the
  document text layer as the payload.

ADAPTER ASSUMPTION (documented seam): OCR NIMs ship in several shapes
— dedicated OCR endpoints and vision-chat shapes. This adapter posts
an OpenAI-compatible chat request whose user message carries the
document text layer and asks for a verbatim transcription plus a
confidence, because every NIM in the catalog speaks that shape. If
your OCR NIM exposes a dedicated transcription endpoint instead,
replace this one class — the rest of the pipeline programs against
``OcrEngine`` and never sees the difference. This adapter was
verified live against the hosted catalog on 2026-10-10 with a
text-native Nemotron model (the default below): the read came back
with a verbatim transcription and a calibrated confidence. One
catalog lesson is baked into that default: NVIDIA's image-native
parse models (e.g. ``nvidia/nemotron-parse``) reject text input
outright, so they belong behind the image variant of this seam,
not this text-layer adapter.
"""

from __future__ import annotations

from .config import completion_budget, env_float, env_str, load_dotenv
from .errors import translate_provider_error
from .resilience import call_with_transient_retry
from .schemas import DocumentInput, OcrResult

LOW_CONFIDENCE_THRESHOLD = 0.6

# The reply is the document's transcription verbatim, in JSON —
# its size scales with the document, and a reasoning reader thinks
# inside the same budget. The pre-audit 1200 could truncate a long
# read mid-document; 2000 is the audited default, tunable via
# OCR_MAX_TOKENS (docs/architecture.md §10).
DEFAULT_OCR_MAX_TOKENS = 2000

_OCR_SYSTEM = (
    "You are an OCR engine. Transcribe the document text verbatim. "
    "Respond with a single JSON object: {\"text\": the transcription, "
    "\"confidence\": a number between 0 and 1 for the overall read quality}."
)


class OfflineOcrEngine:
    """Deterministic offline OCR: the document's own text layer."""

    name = "offline"

    def read(self, document: DocumentInput) -> OcrResult:
        text = document.text_layer
        confidence = (
            float(document.ocr_confidence)
            if document.ocr_confidence is not None
            else (0.35 if document.scan_quality == "unreadable"
                  else 0.7 if document.scan_quality == "poor" else 0.98)
        )
        return OcrResult(
            document_id=document.document_id,
            doc_type=document.doc_type,
            engine=self.name,
            mean_confidence=confidence,
            char_count=len(text),
            low_confidence=confidence < LOW_CONFIDENCE_THRESHOLD,
            note="offline engine: document text layer used as the OCR read",
        )


class NimOcrEngine:
    """NVIDIA NIM OCR profile (OpenAI-compatible chat shape — see the
    module docstring for the adapter assumption)."""

    name = "nim"

    def __init__(self) -> None:
        load_dotenv()
        api_key = env_str("NVIDIA_API_KEY") or env_str("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError(
                "OCR_ENGINE=nim but no API key is set. Add NVIDIA_API_KEY="
                "<your nvapi- key> to .env (from build.nvidia.com), or set "
                "OCR_ENGINE=offline."
            )
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError(
                "OCR_ENGINE=nim needs the OpenAI SDK (the llm extra): "
                "uv sync --extra llm"
            ) from exc
        self.base_url = env_str("OCR_BASE_URL", "https://integrate.api.nvidia.com/v1")
        self._model = env_str("OCR_MODEL", "nvidia/nemotron-3-super-120b-a12b")
        self._client = OpenAI(
            api_key=api_key,
            base_url=self.base_url,
            timeout=env_float("LLM_TIMEOUT_SECONDS", 60.0),
            max_retries=0,
        )

    def read(self, document: DocumentInput) -> OcrResult:
        from .model_backends import parse_json_object

        try:
            response = call_with_transient_retry(lambda: self._client.chat.completions.create(
                model=self._model,
                temperature=0,
                max_tokens=completion_budget(
                    "OCR_MAX_TOKENS", DEFAULT_OCR_MAX_TOKENS
                ),
                messages=[
                    {"role": "system", "content": _OCR_SYSTEM},
                    {"role": "user", "content": document.text_layer},
                ],
            ))
        except Exception as exc:
            raise translate_provider_error(
                exc, backend="nim-ocr", base_url=self.base_url
            ) from exc
        text = response.choices[0].message.content or ""
        data = parse_json_object(text)
        if data is not None:
            read_text = str(data.get("text") or text)
            try:
                confidence = float(data.get("confidence", 0.0))
            except (TypeError, ValueError):
                confidence = 0.0
        else:
            read_text, confidence = text, 0.0
        return OcrResult(
            document_id=document.document_id,
            doc_type=document.doc_type,
            engine=self.name,
            mean_confidence=min(max(confidence, 0.0), 1.0),
            char_count=len(read_text),
            low_confidence=confidence < LOW_CONFIDENCE_THRESHOLD,
            note=f"NIM OCR read via {self._model}",
        )


def get_ocr_engine(name: str | None = None) -> OfflineOcrEngine | NimOcrEngine:
    """Select the OCR engine: ``OCR_ENGINE`` = offline (default) | nim."""
    load_dotenv()
    selected = (name or env_str("OCR_ENGINE") or "offline").lower()
    if selected == "offline":
        return OfflineOcrEngine()
    if selected == "nim":
        return NimOcrEngine()
    raise ValueError(f"Unknown OCR_ENGINE: {selected!r} (expected offline | nim)")


def run_ocr(invoice, engine) -> list[OcrResult]:
    """OCR every document on the invoice. A failing engine degrades
    per-document to the offline read — OCR is evidence, never a run
    failure."""
    offline = OfflineOcrEngine()
    results: list[OcrResult] = []
    for document in invoice.documents:
        try:
            results.append(engine.read(document))
        except Exception as exc:
            fallback = offline.read(document)
            fallback.note = f"OCR engine failed ({exc}) — offline text layer used"
            results.append(fallback)
    return results
