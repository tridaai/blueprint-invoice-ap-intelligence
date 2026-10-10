"""LLM-mode eval pack (opt-in): the golden set against a REAL provider.

The deterministic pack (``run_evals.py``) gates the rules path. This
pack answers the other question: with a live model in the loop
(extraction, triage cross-check, agentic diagnosis, drafting), do the
triage outcomes hold and are the drafts grounded and clean?

    MODEL_BACKEND=openai OPENAI_API_KEY=... python evals/run_llm_evals.py

For the NVIDIA NIM profile, point the OpenAI backend at the hosted
NIM endpoint (``OPENAI_BASE_URL=https://integrate.api.nvidia.com/v1``,
an ``nvapi-`` key, and a Nemotron model in ``OPENAI_MODEL``).

The judge is the same backend (optionally a different model via
``--judge-model`` / ``REVIEWER_MODEL`` conventions do not apply here).
Results land in ``evals/llm_results.json`` with token/cost totals.

Cases run through ONE ``InvoiceService`` over a shared in-memory
store, in file order — the same path as the API and the
deterministic pack. Order and accumulation are the point: the
duplicate outcome only exists against priors the earlier cases
left in the store. An earlier version of this runner called
``run_invoice`` directly per case, with no store at all: the
deterministic duplicate screen never saw a prior, and a live
slice scored the INV-1004 resend as a clean match (2026-10-10).
The pipeline now computes the screen itself from whatever priors
it is handed — but a harness that hands it none is grading a
pipeline nobody runs.

Exit codes: 0 = pack passed, 1 = pack ran and failed, 2 = NOT RUN —
no real provider configured. The refusal is loud on purpose: an
LLM-mode score produced by the offline fallback would be fiction.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ap_agent.matching import format_match_summary  # noqa: E402
from ap_agent.model_backends import (  # noqa: E402
    estimate_cost_usd,
    get_backend,
    parse_json_object,
)
from ap_agent.prompts import JUDGE_SYSTEM_PROMPT, JUDGE_USER_TEMPLATE  # noqa: E402
from ap_agent.retriever import get_retriever  # noqa: E402
from ap_agent.samples import load_sample_invoices  # noqa: E402
from ap_agent.schemas import InvoiceInput  # noqa: E402
from ap_agent.service import InvoiceService  # noqa: E402
from ap_agent.store import InMemoryStore  # noqa: E402

GOLDEN = Path(__file__).resolve().parent / "golden.jsonl"
RESULTS = Path(__file__).resolve().parent / "llm_results.json"


def judge_draft(backend, result, judge_model: str | None) -> dict:
    """One LLM-judge verdict for a produced draft pair. Never raises: a
    failed judge call is a ``judge_error`` verdict, which fails the pack."""
    user = JUDGE_USER_TEMPLATE.format(
        invoice_id=result.invoice_id,
        supplier_name=result.supplier_name,
        invoice_number=result.invoice_number or "not stated",
        outcome=result.triage.outcome,
        severity=result.triage.severity,
        match_summary=format_match_summary(result.facts),
        policies=", ".join(g.policy_id for g in result.policies) or "none",
        subject=result.drafts.subject if result.drafts else "",
        supplier_letter=result.drafts.supplier_letter if result.drafts else "",
        ap_packet=result.drafts.ap_packet if result.drafts else "",
    )
    try:
        text, _usage = backend.complete_with_usage(
            JUDGE_SYSTEM_PROMPT, user, max_tokens=300, model=judge_model
        )
    except Exception as exc:  # a broken judge must not look like a pass
        return {"judge_error": f"{type(exc).__name__}: {exc}"}
    data = parse_json_object(text)
    if data is None:
        return {"judge_error": "judge reply was not a JSON object"}
    try:
        return {
            "grounded": bool(data["grounded"]),
            "clean": bool(data["clean"]),
            "notes": str(data.get("notes", ""))[:300],
        }
    except (KeyError, TypeError, ValueError) as exc:
        return {"judge_error": f"judge reply missing/invalid fields: {exc}"}


def _load_cases(limit: int | None, outcomes: str | None) -> list[dict]:
    cases = [
        json.loads(line)
        for line in GOLDEN.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if outcomes:
        wanted = {o.strip() for o in outcomes.split(",") if o.strip()}
        cases = [c for c in cases if c["expected_outcome"] in wanted]
    if limit:
        cases = cases[:limit]
    return cases


def main() -> int:
    parser = argparse.ArgumentParser(description="LLM-mode eval pack (opt-in)")
    parser.add_argument("--limit", type=int, default=None, help="Run only the first N cases")
    parser.add_argument("--outcomes", type=str, default=None, help="Comma-separated expected outcomes to include")
    parser.add_argument("--judge-model", type=str, default=None, help="Model override for the judge calls")
    args = parser.parse_args()

    backend = get_backend()
    if backend.name == "mock":
        print(
            "error: MODEL_BACKEND is not set to a real provider — this pack "
            "evaluates the LLM paths and refuses to run against the offline "
            "fallback. Configure a provider first (see .env.example), e.g. "
            "MODEL_BACKEND=openai with the NVIDIA NIM endpoint + a Nemotron model.",
            file=sys.stderr,
        )
        return 2

    samples = {s["invoice_id"]: s for s in load_sample_invoices()}
    cases = _load_cases(args.limit, args.outcomes)
    retriever = get_retriever()
    # One service, one shared store, file order — the production
    # path (see the module docstring). The store accumulating
    # across cases is what makes the duplicate case a duplicate.
    service = InvoiceService(
        backend=backend,
        retriever=retriever,
        store=InMemoryStore(),
        checkpointer=False,
    )
    backend.reset_usage()
    started = time.perf_counter()

    rows: list[dict] = []
    outcome_hits = 0
    for case in cases:
        payload = case.get("invoice") or samples[case["sample_id"]]
        invoice = InvoiceInput.model_validate(payload)
        result = service.analyze(invoice)
        verdict = judge_draft(backend, result, args.judge_model)
        outcome_ok = result.triage.outcome == case["expected_outcome"]
        outcome_hits += outcome_ok
        rows.append(
            {
                "case_id": case["case_id"],
                "expected_outcome": case["expected_outcome"],
                "outcome": result.triage.outcome,
                "outcome_ok": outcome_ok,
                "resolution": result.cross_check.resolution if result.cross_check else "not_run",
                "validation_passed": bool(result.validation and result.validation.passed),
                "judge": verdict,
            }
        )

    usage = backend.usage_totals()
    model = getattr(backend, "_model", "")
    total = len(rows)
    judged = [r for r in rows if "judge_error" not in r["judge"]]
    grounded = sum(1 for r in judged if r["judge"]["grounded"])
    clean = sum(1 for r in judged if r["judge"]["clean"])
    errors = total - len(judged)
    passed = (
        total > 0
        and errors == 0
        and outcome_hits == total
        and grounded == total
        and clean == total
    )
    payload = {
        "backend": backend.name,
        "model": model,
        "judge_model": args.judge_model or model,
        "cases": total,
        "outcome_match": f"{outcome_hits}/{total}",
        "drafts_grounded": f"{grounded}/{total}",
        "drafts_clean": f"{clean}/{total}",
        "judge_errors": errors,
        "resolutions": dict(Counter(r["resolution"] for r in rows)),
        "usage": usage,
        "estimated_cost_usd": estimate_cost_usd(
            model, usage.get("input_tokens"), usage.get("output_tokens")
        ),
        "latency_seconds": round(time.perf_counter() - started, 2),
        "passed": passed,
        "rows": rows,
    }
    RESULTS.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    print(f"LLM eval pack — backend {backend.name} · model {model}")
    print(f"Cases: {total} · outcome match {outcome_hits}/{total}")
    print(f"Drafts grounded {grounded}/{total} · clean {clean}/{total} · judge errors {errors}")
    for row in rows:
        if not row["outcome_ok"] or "judge_error" in row["judge"] or not (
            row["judge"].get("grounded") and row["judge"].get("clean")
        ):
            print(f"  - {row['case_id']}: {row}")
    print("PASS" if passed else "FAIL")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
