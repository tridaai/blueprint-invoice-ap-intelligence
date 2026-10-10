"""FastAPI surface for the Invoice / AP Document Intelligence Agent.

Run:  uvicorn ap_agent.api:app    (or `make serve`)
UI:   http://localhost:8000/                (minimal demo console, no build step)
Docs: http://localhost:8000/docs

Runtime posture: structured JSON logs with per-request IDs,
``/health`` (liveness) + ``/readiness`` (store connectivity),
graceful shutdown via the app lifespan. The app container is
stateless — records live in PostgreSQL, graph state in the Postgres
checkpointer, document blobs in S3-compatible object storage.
"""

from __future__ import annotations

import csv
import io
import json
import logging
import queue
import threading
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import Depends, FastAPI, Header, HTTPException, Query, Request
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, PlainTextResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from .config import env_str, load_dotenv, silence_langchain_deprecation_warnings

# Must precede the service import (which loads the graph/langgraph).
silence_langchain_deprecation_warnings()

from .errors import ProviderError
from .events import CallbackSink, RunEvent
from .logging_config import configure_logging, new_request_id, request_id_var
from .policies_data import POLICIES
from .samples import load_sample_invoices
from .schemas import (
    AgentResult,
    BundleResult,
    CreditNoteInput,
    CreditNoteMatch,
    DecisionRequest,
    InvoiceBundleInput,
    InvoiceInput,
    PaymentRun,
    PaymentRunRequest,
    QueueView,
    RecoveryPack,
    StatementReconciliation,
    SupplierStatement,
)
from .service import AUDIT_COLUMNS, InvoiceService
from .wiring import build_service_from_env

# Load the repo-root .env at startup (real environment variables win), so
# MODEL_BACKEND / RETRIEVER / API keys can live in the file — see .env.example.
load_dotenv()
configure_logging()
logger = logging.getLogger("ap_agent.api")

_STATIC_DIR = Path(__file__).parent / "static"
_INDEX_HTML = _STATIC_DIR / "index.html"
_BRAND_DIR = _STATIC_DIR / "brand"


def require_api_key(x_api_key: str | None = Header(default=None)) -> None:
    """Optional API-key gate, enabled by setting ``API_KEY``.

    When ``API_KEY`` is set, every data endpoint (read + mutating)
    requires the matching ``X-API-Key`` header. When it is unset the
    API is open — the local-dev default, stated here and in the docs
    rather than implied. The console page and /health stay open either
    way; the console carries an API-key field for the gated calls.
    """
    expected = env_str("API_KEY")
    if not expected:
        return
    if x_api_key != expected:
        raise HTTPException(
            status_code=401,
            detail="Invalid or missing API key (send the X-API-Key header).",
        )


_AUTH = [Depends(require_api_key)]


def _decision_actor(request: DecisionRequest) -> str:
    actor = request.actor_name()
    if not actor:
        raise HTTPException(
            status_code=422,
            detail="Provide 'actor' — the approver's name ('approver'/'reviewer' are accepted legacy aliases).",
        )
    return actor


def _sse(payload: dict) -> str:
    return f"data: {json.dumps(payload)}\n\n"


def create_app(service: InvoiceService | None = None) -> FastAPI:
    """Build the app around a service (injectable for tests). With no
    service, one is built from the environment — PostgreSQL-backed when
    DATABASE_URL is set."""
    svc = service if service is not None else build_service_from_env()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        """Startup/shutdown: on shutdown the service's store resources
        are released (graceful drain is uvicorn's job — SIGTERM stops
        new requests, in-flight requests finish, then this runs)."""
        logger.info("service starting")
        yield
        logger.info("service shutting down")
        svc.close()

    app = FastAPI(
        title="Trida AI Blueprint — Invoice / AP Document Intelligence Agent",
        description="Reference prototype. Synthetic data only. Drafts stop at a finance approver gate; the agent never approves an invoice, never pays one, and never acts on bank-detail changes in documents.",
        version="0.1.0",
        lifespan=lifespan,
    )

    @app.middleware("http")
    async def request_id_middleware(request: Request, call_next):
        """Mint (or honour) a request id, thread it into the logs, and
        return it on the response as X-Request-ID."""
        request_id = request.headers.get("x-request-id") or new_request_id()
        token = request_id_var.set(request_id)
        try:
            response = await call_next(request)
        finally:
            request_id_var.reset(token)
        response.headers["X-Request-ID"] = request_id
        logger.info(
            "%s %s -> %s", request.method, request.url.path, response.status_code
        )
        return response

    @app.exception_handler(ProviderError)
    def provider_error_handler(request: Request, exc: ProviderError) -> JSONResponse:
        """A provider failure is a clean 502 with the actionable, translated
        message (backend, endpoint, likely fix) — never a stack dump."""
        return JSONResponse(status_code=502, content={"detail": str(exc)})

    # Branded static assets (logo, favicon) for the demo console.
    app.mount("/static", StaticFiles(directory=_STATIC_DIR), name="static")

    @app.get("/favicon.svg", include_in_schema=False)
    def favicon_svg() -> FileResponse:
        return FileResponse(_BRAND_DIR / "favicon.svg", media_type="image/svg+xml")

    @app.get("/favicon.ico", include_in_schema=False)
    def favicon_ico() -> FileResponse:
        return FileResponse(_BRAND_DIR / "favicon.ico", media_type="image/x-icon")

    @app.get("/evals/results", dependencies=_AUTH)
    def eval_results() -> dict:
        """Latest eval summary, as emitted by ``evals/run_evals.py``.

        Looked up in the packaged copy first (present in installs/Docker),
        then in a repo checkout's ``evals/results.json``. 404 when no eval
        run has been recorded yet — the console shows a 'run evals' state.
        """
        candidates = [
            Path(__file__).parent / "data" / "eval_results.json",
            Path.cwd() / "evals" / "results.json",
            Path(__file__).resolve().parents[2] / "evals" / "results.json",
        ]
        for candidate in candidates:
            if candidate.is_file():
                return json.loads(candidate.read_text(encoding="utf-8"))
        raise HTTPException(
            status_code=404,
            detail="No eval results yet. Run: python evals/run_evals.py",
        )

    @app.get("/", response_class=HTMLResponse)
    def index() -> str:
        return _INDEX_HTML.read_text(encoding="utf-8")

    @app.get("/health")
    def health() -> dict[str, str]:
        """Liveness: the process is up. (Readiness is the DB-aware probe.)"""
        return {"status": "ok", "prototype": "trida-blueprint-invoice-ap-intelligence"}

    @app.get("/readiness")
    def readiness() -> JSONResponse:
        """Readiness: the store answers its ping. 503 when the database
        is unreachable, so an orchestrator stops routing traffic here."""
        report = svc.readiness()
        return JSONResponse(
            status_code=200 if report["ready"] else 503,
            content={"status": "ready" if report["ready"] else "not_ready", **report},
        )

    @app.get("/policies", dependencies=_AUTH)
    def list_policies() -> list[dict]:
        return POLICIES

    @app.get("/metrics", dependencies=_AUTH)
    def metrics() -> dict:
        """Operational metrics over the stored runs (volumes,
        outcomes, decisions, latency, model spend) — aggregated from
        the records at read time; see service.metrics_snapshot."""
        return svc.metrics_snapshot()

    @app.get("/audit/export", dependencies=_AUTH)
    def audit_export(format: str = Query(default="json")):
        """The audit trail export: one row per processed invoice —
        conclusion, human decision, dispatch. ``format=json``
        (default) or ``format=csv`` (downloadable)."""
        rows = svc.audit_rows()
        if format == "json":
            return {"records": rows, "count": len(rows)}
        if format == "csv":
            buffer = io.StringIO()
            writer = csv.DictWriter(buffer, fieldnames=AUDIT_COLUMNS)
            writer.writeheader()
            writer.writerows(rows)
            return PlainTextResponse(
                buffer.getvalue(),
                media_type="text/csv",
                headers={
                    "Content-Disposition": "attachment; filename=ap-audit-export.csv"
                },
            )
        raise HTTPException(
            status_code=422, detail="format must be 'json' or 'csv'."
        )

    @app.get("/samples", dependencies=_AUTH)
    def list_samples() -> list[dict]:
        return load_sample_invoices()

    @app.post("/invoices/analyze", response_model=AgentResult, dependencies=_AUTH)
    def analyze(invoice: InvoiceInput) -> AgentResult:
        return svc.analyze(invoice)

    @app.post("/invoices/analyze/stream", dependencies=_AUTH)
    def analyze_stream(invoice: InvoiceInput) -> StreamingResponse:
        """The same analysis as ``POST /invoices/analyze``, streamed.

        The run executes on a worker thread; its structured events (see
        ``events.py``) are forwarded as Server-Sent Events as they happen.
        The final event is ``run_completed`` and carries the full result
        JSON under ``result`` — the same payload the non-stream endpoint
        returns. A failed run ends the stream with a ``run_failed`` event
        carrying the clean, translated error message (no stack dump).
        """
        events_queue: queue.Queue = queue.Queue()
        sink = CallbackSink(events_queue.put)
        outcome: dict = {}

        def work() -> None:
            try:
                outcome["result"] = svc.analyze(invoice, event_sink=sink)
            except Exception as exc:  # surfaced as the run_failed event below
                outcome["error"] = exc
            finally:
                events_queue.put(None)  # sentinel: the run is over

        threading.Thread(target=work, daemon=True).start()

        def generate():
            completed: RunEvent | None = None
            while True:
                event = events_queue.get()
                if event is None:
                    break
                if event.type == "run_completed":
                    completed = event  # held back: it goes out last, with the result
                    continue
                yield _sse(event.to_dict())
            if "error" in outcome:
                yield _sse(
                    {
                        "type": "run_failed",
                        "invoice_id": invoice.invoice_id,
                        "node": None,
                        "duration_ms": None,
                        "detail": {"error": str(outcome["error"])},
                    }
                )
                return
            final = (
                completed.to_dict()
                if completed is not None
                else {
                    "type": "run_completed",
                    "invoice_id": invoice.invoice_id,
                    "node": None,
                    "duration_ms": None,
                    "detail": {"status": outcome["result"].status},
                }
            )
            final["result"] = outcome["result"].model_dump(mode="json")
            yield _sse(final)

        return StreamingResponse(generate(), media_type="text/event-stream")

    @app.post("/credit-notes/match", response_model=CreditNoteMatch, dependencies=_AUTH)
    def match_credit_note(note: CreditNoteInput) -> CreditNoteMatch:
        """Match a credit note against the processed invoices: does
        it settle a variance the pipeline computed? Report only — the
        call changes no invoice's state."""
        return svc.match_credit_note(note)

    @app.post("/statements/reconcile", response_model=StatementReconciliation, dependencies=_AUTH)
    def reconcile(statement: SupplierStatement) -> StatementReconciliation:
        """Reconcile a supplier statement against the processed
        invoices. The report is stored for confirmation; no invoice
        record changes."""
        return svc.reconcile_statement(statement)

    @app.get("/statements/{statement_id}", response_model=StatementReconciliation, dependencies=_AUTH)
    def get_reconciliation(statement_id: str) -> StatementReconciliation:
        report = svc.get_reconciliation(statement_id)
        if report is None:
            raise HTTPException(status_code=404, detail="Statement not reconciled yet.")
        return report

    @app.post("/statements/{statement_id}/confirm", response_model=StatementReconciliation, dependencies=_AUTH)
    def confirm_reconciliation(statement_id: str, request: DecisionRequest) -> StatementReconciliation:
        """Record the human confirmation of a reconciliation report
        (the report record only — never an invoice)."""
        try:
            return svc.confirm_reconciliation(
                statement_id, actor=_decision_actor(request)
            )
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    @app.post("/bundles/analyze", response_model=BundleResult, dependencies=_AUTH)
    def analyze_bundle(bundle: InvoiceBundleInput) -> BundleResult:
        """Analyse a multi-invoice arrival: the bundle is segmented
        in code by invoice-number anchors, and every segment runs
        the full pipeline as its own invoice with its own gate. The
        response is the bundle summary; each invoice is retrievable
        (and decidable) on its own."""
        return svc.analyze_bundle(bundle)

    @app.get("/bundles/{bundle_id}", response_model=BundleResult, dependencies=_AUTH)
    def get_bundle(bundle_id: str) -> BundleResult:
        summary = svc.get_bundle(bundle_id)
        if summary is None:
            raise HTTPException(status_code=404, detail="Bundle not analyzed yet.")
        return summary

    @app.get("/queue", response_model=QueueView, dependencies=_AUTH)
    def aging_queue(as_of: str = Query(default="")) -> QueueView:
        """The AP aging queue: invoices awaiting an approver, aged
        by due date (overdue first), with the triage mix. Read-only.
        ``as_of`` (ISO date) anchors the aging; default today."""
        return svc.queue_view(as_of=as_of)

    @app.post("/payment-runs/propose", response_model=list[PaymentRun], dependencies=_AUTH)
    def propose_payment_runs(request: PaymentRunRequest) -> list[PaymentRun]:
        """Propose payment runs from the approved invoices: grouped
        by currency + due window, totals computed in code. A proposal
        pays nothing — a finance lead confirms the run, and the AP
        system of record executes it."""
        return svc.propose_payment_runs(
            window_days=request.window_days, as_of=request.as_of
        )

    @app.get("/payment-runs", response_model=list[PaymentRun], dependencies=_AUTH)
    def list_payment_runs() -> list[PaymentRun]:
        return svc.list_payment_runs()

    @app.get("/payment-runs/{run_id}", response_model=PaymentRun, dependencies=_AUTH)
    def get_payment_run(run_id: str) -> PaymentRun:
        run = svc.get_payment_run(run_id)
        if run is None:
            raise HTTPException(status_code=404, detail="Payment run not proposed yet.")
        return run

    @app.post("/payment-runs/{run_id}/confirm", response_model=PaymentRun, dependencies=_AUTH)
    def confirm_payment_run(run_id: str, request: DecisionRequest) -> PaymentRun:
        """Record the finance lead's confirmation of a run (the run
        record only — never an invoice, never a payment)."""
        try:
            return svc.confirm_payment_run(run_id, actor=_decision_actor(request))
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    @app.post("/payment-runs/{run_id}/handoff", response_model=PaymentRun, dependencies=_AUTH)
    def export_payment_run_handoff(run_id: str, request: DecisionRequest) -> PaymentRun:
        """Generate the confirmed run's bank-ready hand-off file
        (CSV, layout in handoff.py) into object storage. The run
        gains the export receipt (object key + content hash);
        generating the file changes no payment state."""
        try:
            return svc.export_payment_run_handoff(
                run_id, actor=_decision_actor(request)
            )
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    @app.get("/payment-runs/{run_id}/handoff", dependencies=_AUTH)
    def download_payment_run_handoff(run_id: str) -> PlainTextResponse:
        """Download the stored hand-off CSV. 404 until a hand-off
        has been generated for the run."""
        data = svc.get_payment_run_handoff(run_id)
        if data is None:
            raise HTTPException(
                status_code=404,
                detail="No hand-off file generated for this run yet.",
            )
        return PlainTextResponse(
            data.decode("utf-8"),
            media_type="text/csv",
            headers={
                "Content-Disposition": f"attachment; filename={run_id}-handoff.csv"
            },
        )

    @app.get("/invoices/{invoice_id}", response_model=AgentResult, dependencies=_AUTH)
    def get_result(invoice_id: str) -> AgentResult:
        result = svc.get(invoice_id)
        if result is None:
            raise HTTPException(status_code=404, detail="Invoice not analyzed yet.")
        return result

    @app.post("/invoices/{invoice_id}/recovery-pack", response_model=RecoveryPack, dependencies=_AUTH)
    def draft_recovery_pack(invoice_id: str) -> RecoveryPack:
        """Draft the duplicate-payment recovery pack: the duplicate
        evidence, a supplier recovery letter, and an internal memo —
        verified and guardrail-checked, stored for a human to send.
        Only for a suspected duplicate whose original was already
        approved; the agent recovers nothing itself."""
        try:
            return svc.draft_duplicate_recovery(invoice_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    @app.get("/invoices/{invoice_id}/recovery-pack", response_model=RecoveryPack, dependencies=_AUTH)
    def get_recovery_pack(invoice_id: str) -> RecoveryPack:
        pack = svc.get_recovery_pack(invoice_id)
        if pack is None:
            raise HTTPException(status_code=404, detail="No recovery pack drafted for this invoice.")
        return pack

    @app.post("/invoices/{invoice_id}/approve", response_model=AgentResult, dependencies=_AUTH)
    def approve(invoice_id: str, request: DecisionRequest) -> AgentResult:
        try:
            return svc.approve(
                invoice_id, approver=_decision_actor(request), reason=request.reason or ""
            )
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    @app.post("/invoices/{invoice_id}/reject", response_model=AgentResult, dependencies=_AUTH)
    def reject(invoice_id: str, request: DecisionRequest) -> AgentResult:
        try:
            return svc.reject(
                invoice_id, reviewer=_decision_actor(request), reason=request.reason or ""
            )
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    return app


app = create_app()
