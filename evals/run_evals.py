"""Golden-dataset evaluation for the Invoice / AP Document Intelligence Agent.

Runs the full pipeline (deterministic rules path, offline fallback
backend) over ``golden.jsonl`` through ONE service instance in file
order — order matters: the duplicate case only duplicates because the
original invoice was analysed earlier in the same run, exactly like a
real AP queue. Reports triage outcome accuracy and match-status
accuracy. All cases are synthetic. Run from the repo root:

    python evals/run_evals.py

Exit code is 0 when both accuracies >= 95%, 1 otherwise (the gate).
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ap_agent.samples import load_sample_invoices  # noqa: E402
from ap_agent.schemas import InvoiceInput  # noqa: E402
from ap_agent.service import InvoiceService  # noqa: E402
from ap_agent.store import InMemoryStore  # noqa: E402

GOLDEN = Path(__file__).resolve().parent / "golden.jsonl"
THRESHOLD = 0.95


def _write_results(payload: dict) -> None:
    """Emit results.json next to this script (and into the package, when
    the source tree is present) so the demo console can display it."""
    targets = [Path(__file__).resolve().parent / "results.json"]
    packaged = (
        Path(__file__).resolve().parents[1]
        / "src"
        / "ap_agent"
        / "data"
        / "eval_results.json"
    )
    if packaged.parent.is_dir():
        targets.append(packaged)
    for target in targets:
        target.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def evaluate(cases: list[dict]) -> dict:
    """Run the pipeline over the cases and score outcome + match status."""
    samples = {s["invoice_id"]: s for s in load_sample_invoices()}
    service = InvoiceService(store=InMemoryStore(), checkpointer=False)
    outcome_correct = 0
    match_correct = 0
    severity_correct = 0
    severity_total = 0
    per_outcome_total: dict[str, int] = defaultdict(int)
    per_outcome_correct: dict[str, int] = defaultdict(int)
    failures: list[str] = []

    for case in cases:
        payload = case.get("invoice") or samples[case["sample_id"]]
        invoice = InvoiceInput.model_validate(payload)
        result = service.analyze(invoice)
        expected_outcome = case["expected_outcome"]
        per_outcome_total[expected_outcome] += 1
        if result.triage.outcome == expected_outcome:
            outcome_correct += 1
            per_outcome_correct[expected_outcome] += 1
        else:
            failures.append(
                f"{case['case_id']}: expected outcome {expected_outcome}, "
                f"got {result.triage.outcome}"
            )
        if result.facts.match_status == case["expected_match_status"]:
            match_correct += 1
        elif result.triage.outcome == expected_outcome:
            failures.append(
                f"{case['case_id']}: outcome ok, expected match status "
                f"{case['expected_match_status']}, got {result.facts.match_status}"
            )
        if "expected_severity" in case:
            severity_total += 1
            if result.triage.severity == case["expected_severity"]:
                severity_correct += 1
            elif result.triage.outcome == expected_outcome:
                failures.append(
                    f"{case['case_id']}: outcome ok, expected severity "
                    f"{case['expected_severity']}, got {result.triage.severity}"
                )

    total = len(cases)
    return {
        "total": total,
        "outcome_correct": outcome_correct,
        "outcome_accuracy": round(outcome_correct / total, 4) if total else 0.0,
        "match_correct": match_correct,
        "match_accuracy": round(match_correct / total, 4) if total else 0.0,
        "severity_correct": severity_correct,
        "severity_total": severity_total,
        "per_outcome": {
            outcome: {
                "correct": per_outcome_correct[outcome],
                "total": per_outcome_total[outcome],
                "accuracy": round(per_outcome_correct[outcome] / per_outcome_total[outcome], 4),
            }
            for outcome in sorted(per_outcome_total)
        },
        "failures": failures,
    }


def main() -> int:
    cases = [
        json.loads(line)
        for line in GOLDEN.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    scored = evaluate(cases)
    total = scored["total"]
    passed = (
        scored["outcome_accuracy"] >= THRESHOLD
        and scored["match_accuracy"] >= THRESHOLD
    )

    _write_results(
        {
            "dataset": "golden.jsonl (synthetic)",
            "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "total": total,
            "outcome_match": f"{scored['outcome_correct']}/{total}",
            "match_status_match": f"{scored['match_correct']}/{total}",
            "outcome_correct": scored["outcome_correct"],
            "outcome_accuracy": scored["outcome_accuracy"],
            "match_correct": scored["match_correct"],
            "match_accuracy": scored["match_accuracy"],
            "per_outcome": scored["per_outcome"],
            "threshold": THRESHOLD,
            "passed": passed,
            "framing": (
                "Synthetic golden set used as a regression gate. "
                "It is not a real-world accuracy claim."
            ),
        }
    )

    print(f"Golden dataset: {total} synthetic cases")
    print(
        f"Triage outcome accuracy : {scored['outcome_correct']}/{total} "
        f"= {scored['outcome_accuracy']:.1%}"
    )
    print(
        f"Match-status accuracy   : {scored['match_correct']}/{total} "
        f"= {scored['match_accuracy']:.1%}"
    )
    if scored["severity_total"]:
        print(
            f"Severity accuracy       : {scored['severity_correct']}/"
            f"{scored['severity_total']} (where labelled)"
        )
    for outcome, row in scored["per_outcome"].items():
        print(f"  {outcome:<20} {row['correct']}/{row['total']}")
    if scored["failures"]:
        print("Failures:")
        for failure in scored["failures"]:
            print(f"  - {failure}")
    print("PASS" if passed else f"FAIL (threshold {THRESHOLD:.0%})")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
