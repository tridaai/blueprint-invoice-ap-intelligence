"""Golden-dataset evaluation for the Invoice / AP Document Intelligence Agent.

Runs the full pipeline (deterministic rules path, offline fallback
backend) over ``golden.jsonl`` through ONE service instance in file
order — order matters: the duplicate case only duplicates because the
original invoice was analysed earlier in the same run, exactly like a
real AP queue. Reports triage outcome accuracy and match-status
accuracy, then matches the bundled sample credit notes against the
populated store (a credit note only settles because its invoice was
processed earlier in the run). All cases are synthetic. Run from the
repo root:

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

from ap_agent.samples import load_sample_credit_notes, load_sample_invoices  # noqa: E402
from ap_agent.schemas import CreditNoteInput, InvoiceInput  # noqa: E402
from ap_agent.service import InvoiceService  # noqa: E402
from ap_agent.store import InMemoryStore  # noqa: E402

GOLDEN = Path(__file__).resolve().parent / "golden.jsonl"
THRESHOLD = 0.95

# The credit-note expectations: each bundled sample credit note,
# matched against the store the case loop above has populated (the
# referenced invoice is one of the samples, analysed earlier in the
# same run — the order dependency is the point).
EXPECTED_CREDIT_NOTES = {
    "CN-3001": {
        "matched_invoice_id": "INV-1002",
        "status": "settles_variance",
        "settles_variance": True,
        "overbilled_amount": 94.40,
    },
}


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

    # Credit-note matching runs against the same store, after the
    # invoices it references have been processed.
    credit_total = 0
    credit_correct = 0
    for payload in load_sample_credit_notes():
        note = CreditNoteInput.model_validate(payload)
        expected = EXPECTED_CREDIT_NOTES.get(note.credit_note_id)
        if expected is None:
            continue
        credit_total += 1
        match = service.match_credit_note(note)
        got = {
            "matched_invoice_id": match.matched_invoice_id,
            "status": match.status,
            "settles_variance": match.settles_variance,
            "overbilled_amount": match.overbilled_amount,
        }
        if got == expected:
            credit_correct += 1
        else:
            failures.append(
                f"credit-note {note.credit_note_id}: expected {expected}, got {got}"
            )

    # Duplicate-payment recovery: INV-1013 duplicates INV-1001 (the
    # portal resend). Once the original is approved, the recovery
    # pack must draft — evidence computed, guardrails passed.
    recovery_total = 1
    recovery_correct = 0
    try:
        service.approve(
            "INV-1001", approver="eval-harness", reason="eval: approve the original"
        )
        pack = service.draft_duplicate_recovery("INV-1013")
        got = {
            "pack_id": pack.pack_id,
            "original_invoice_id": pack.original_invoice_id,
            "amount": pack.amount,
            "exposure_status": pack.exposure_status,
            "status": pack.status,
            "validation_passed": bool(pack.validation and pack.validation.passed),
            "grounded": bool(pack.verification and pack.verification.grounded),
        }
        expected = {
            "pack_id": "REC-INV-1013",
            "original_invoice_id": "INV-1001",
            "amount": 1180.0,
            "exposure_status": "twin_approved",
            "status": "ready",
            "validation_passed": True,
            "grounded": True,
        }
        if got == expected:
            recovery_correct = 1
        else:
            failures.append(f"recovery pack: expected {expected}, got {got}")
    except Exception as exc:  # a broken recovery path is a failed case
        failures.append(f"recovery pack: {type(exc).__name__}: {exc}")

    total = len(cases)
    return {
        "total": total,
        "credit_note_total": credit_total,
        "credit_note_correct": credit_correct,
        "recovery_total": recovery_total,
        "recovery_correct": recovery_correct,
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
        and scored["credit_note_correct"] == scored["credit_note_total"]
        and scored["recovery_correct"] == scored["recovery_total"]
    )

    _write_results(
        {
            "dataset": "golden.jsonl (synthetic)",
            "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "total": total,
            "outcome_match": f"{scored['outcome_correct']}/{total}",
            "match_status_match": f"{scored['match_correct']}/{total}",
            "credit_note_match": (
                f"{scored['credit_note_correct']}/{scored['credit_note_total']}"
            ),
            "recovery_pack": (
                f"{scored['recovery_correct']}/{scored['recovery_total']}"
            ),
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
    print(
        f"Credit-note matching    : {scored['credit_note_correct']}/"
        f"{scored['credit_note_total']}"
    )
    print(
        f"Recovery pack           : {scored['recovery_correct']}/"
        f"{scored['recovery_total']}"
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
