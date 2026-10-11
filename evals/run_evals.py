"""Golden-dataset evaluation for the Invoice / AP Document Intelligence Agent.

Runs the full pipeline (deterministic rules path, offline fallback
backend) over ``golden.jsonl`` through ONE service instance in file
order — order matters: the duplicate case only duplicates because the
original invoice was analysed earlier in the same run, exactly like a
real AP queue. Reports triage outcome accuracy and match-status
accuracy, then matches the bundled sample credit notes against the
populated store (a credit note only settles because its invoice was
processed earlier in the run), and reconciles the bundled sample
supplier statements against a dedicated store with exact per-statement
expectations — matched / differs / missing / duplicate counts, the
recomputed totals, the statement's own claimed total checked against
them, and the drafted supplier query. The Northbeam statement is then
reconciled a second time from its document form (OCR + vision ports,
parsed in code) and must land on exactly the structured twin's
expectations, its one garbled line flagged as a low-confidence read.
Finally, a dedicated store seeds a payment run — a credit-netted
payment, a payment flagged ACK_REQUIRED by an open recovery claim
(confirmation refused until the acknowledgement is recorded), and a
disputed invoice excluded — held to exact control-report figures,
and the recovery receipt is checked against the supplier roll-up:
it may move only the recovery line, by exactly the received amount.
All cases are synthetic. Run
from the repo root:

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

from ap_agent.samples import (  # noqa: E402
    load_sample_credit_notes,
    load_sample_invoices,
    load_sample_statement_documents,
    load_sample_statements,
)
from ap_agent.schemas import (  # noqa: E402
    CreditNoteInput,
    DocumentInput,
    InvoiceInput,
    SupplierStatement,
)
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
    "CN-3002": {
        "matched_invoice_id": "INV-1003",
        "status": "partial_settlement",
        "settles_variance": False,
        "overbilled_amount": 141.60,
    },
    "CN-3003": {
        "matched_invoice_id": "INV-1002",
        "status": "exceeds_variance",
        "settles_variance": False,
        "overbilled_amount": 94.40,
    },
    "CN-3004": {
        "matched_invoice_id": "",
        "status": "unmatched",
        "settles_variance": False,
        "overbilled_amount": 0.0,
    },
    "CN-3005": {
        "matched_invoice_id": "INV-1001",
        "status": "no_variance",
        "settles_variance": False,
        "overbilled_amount": 0.0,
    },
}

# Statement reconciliation expectations, checked against a dedicated
# store holding INV-1001 + INV-1008 (Northbeam), INV-1002 (Cascade)
# and INV-1007 (Alpen). STMT-NB mixes a match, an amount difference
# and a never-processed line, and its printed total disagrees with
# its own lines; STMT-AW is fully clean; STMT-CI lists one invoice
# twice, which the duplicate screen must flag.
EXPECTED_RECONCILIATIONS = {
    "STMT-NB-2026-10": {
        "matched_count": 1,
        "differs_count": 1,
        "missing_from_ap_count": 1,
        "missing_from_statement_count": 0,
        "duplicate_count": 0,
        "statement_total": 2355.00,
        "ap_total": 2360.00,
        "claimed_total_difference": 45.00,
        "query_draft": True,
    },
    "STMT-AW-2026-10": {
        "matched_count": 1,
        "differs_count": 0,
        "missing_from_ap_count": 0,
        "missing_from_statement_count": 0,
        "duplicate_count": 0,
        "statement_total": 357.00,
        "ap_total": 357.00,
        "claimed_total_difference": 0.00,
        "query_draft": False,
    },
    "STMT-CI-2026-10": {
        "matched_count": 2,
        "differs_count": 0,
        "missing_from_ap_count": 0,
        "missing_from_statement_count": 0,
        "duplicate_count": 1,
        "statement_total": 2548.80,
        "ap_total": 1274.40,
        "claimed_total_difference": 0.00,
        "query_draft": True,
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

    # Statement reconciliation runs against a dedicated store so the
    # expectations stay exact whatever the golden cases did to the
    # main one.
    reconciliation_total = 0
    reconciliation_correct = 0
    recon_service = InvoiceService(store=InMemoryStore(), checkpointer=False)
    samples_by_id = {s["invoice_id"]: s for s in load_sample_invoices()}
    for sample_id in ("INV-1001", "INV-1008", "INV-1002", "INV-1007"):
        recon_service.analyze(
            InvoiceInput.model_validate(samples_by_id[sample_id])
        )
    for payload in load_sample_statements():
        statement = SupplierStatement.model_validate(payload)
        expected = EXPECTED_RECONCILIATIONS.get(statement.statement_id)
        if expected is None:
            continue
        reconciliation_total += 1
        report = recon_service.reconcile_statement(statement)
        got = {
            "matched_count": report.matched_count,
            "differs_count": report.differs_count,
            "missing_from_ap_count": report.missing_from_ap_count,
            "missing_from_statement_count": report.missing_from_statement_count,
            "duplicate_count": report.duplicate_count,
            "statement_total": report.statement_total,
            "ap_total": report.ap_total,
            "claimed_total_difference": report.claimed_total_difference,
            "query_draft": bool(report.supplier_query_draft),
        }
        if got == expected:
            reconciliation_correct += 1
        else:
            failures.append(
                f"reconciliation {statement.statement_id}: "
                f"expected {expected}, got {got}"
            )

    # The same Northbeam statement, arrived as a document: read
    # through the OCR + vision ports and parsed in code, it must
    # reconcile to exactly its structured twin's expectations —
    # and its one garbled line (75.0O) must ride on the report as
    # a flagged low-confidence read, amount recovered.
    for payload in load_sample_statement_documents():
        document = DocumentInput.model_validate(payload)
        if document.document_id != "DOC-STMT-NB-2026-10":
            continue
        reconciliation_total += 1
        report = recon_service.reconcile_statement_document(document)
        expected = EXPECTED_RECONCILIATIONS["STMT-NB-2026-10"]
        got = {
            "matched_count": report.matched_count,
            "differs_count": report.differs_count,
            "missing_from_ap_count": report.missing_from_ap_count,
            "missing_from_statement_count": report.missing_from_statement_count,
            "duplicate_count": report.duplicate_count,
            "statement_total": report.statement_total,
            "ap_total": report.ap_total,
            "claimed_total_difference": report.claimed_total_difference,
            "query_draft": bool(report.supplier_query_draft),
        }
        flagged = {
            line.invoice_number: line
            for line in report.lines
            if line.low_confidence
        }
        if (
            got == expected
            and report.source == "document"
            and not report.refused
            and set(flagged) == {"NB-9999"}
            and flagged["NB-9999"].statement_amount == 75.00
        ):
            reconciliation_correct += 1
        else:
            failures.append(
                f"reconciliation document {document.document_id}: "
                f"expected {expected} with NB-9999 flagged at 75.00, "
                f"got {got}, source={report.source}, "
                f"flagged={sorted(flagged)}"
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

    # Payment-run control report, on a dedicated store: a seeded
    # run holding one clean payment (Cascade's INV-1002, its
    # recorded credit netted), one payment to a supplier with an
    # open recovery claim (Northbeam's INV-1001 — ACK_REQUIRED,
    # and confirmation refuses until the acknowledgement lands),
    # and one disputed invoice excluded (INV-1008, whose amount
    # Northbeam's statement disagrees with). Then the receipt:
    # the supplier roll-up must move by exactly the received
    # amount in its recovery line and nowhere else.
    control_total = 1
    control_correct = 0
    try:
        cs = InvoiceService(store=InMemoryStore(), checkpointer=False)
        by_id = {s["invoice_id"]: s for s in load_sample_invoices()}
        for sample_id in ("INV-1001", "INV-1008", "INV-1002", "INV-1013"):
            cs.analyze(InvoiceInput.model_validate(by_id[sample_id]))
        for invoice_id in ("INV-1001", "INV-1002", "INV-1008"):
            cs.approve(
                invoice_id,
                approver="eval-harness",
                reason="eval: seed the control run",
            )
        cs.draft_duplicate_recovery("INV-1013")
        note_payload = next(
            p
            for p in load_sample_credit_notes()
            if p["credit_note_id"] == "CN-3001"
        )
        cs.apply_credit_note(
            CreditNoteInput.model_validate(note_payload),
            actor="eval-harness",
        )
        statement_payload = next(
            p
            for p in load_sample_statements()
            if p["statement_id"] == "STMT-NB-2026-10"
        )
        cs.reconcile_statement(
            SupplierStatement.model_validate(statement_payload)
        )

        def northbeam_row():
            roll = cs.supplier_positions(as_of="2026-10-05")
            return next(
                r.model_dump()
                for r in roll.suppliers
                if r.supplier_name == "Northbeam Office Supplies"
            )

        roll_before = northbeam_row()
        run = next(
            r
            for r in cs.propose_payment_runs(as_of="2026-10-05")
            if r.currency == "USD"
        )
        report = cs.payment_run_control_report(run.run_id)
        lines = {l.invoice_id: l for l in report.lines}
        gate_refused = False
        try:
            cs.confirm_payment_run(run.run_id, actor="eval-harness")
        except ValueError:
            gate_refused = True
        cs.acknowledge_run_recovery(
            run.run_id, "INV-1001", actor="eval-harness"
        )
        confirmed = (
            cs.confirm_payment_run(run.run_id, actor="eval-harness").status
            == "confirmed"
        )
        cs.transition_recovery_case(
            "INV-1013", to_status="supplier_contacted", actor="eval-harness"
        )
        cs.transition_recovery_case(
            "INV-1013", to_status="credit_received", actor="eval-harness"
        )
        roll_after = northbeam_row()
        got = {
            "inv1001": {
                "match_status": lines["INV-1001"].match_status,
                "gross": lines["INV-1001"].gross_amount,
                "discount": lines["INV-1001"].discount_amount,
                "credit": lines["INV-1001"].credit_amount,
                "net": lines["INV-1001"].net_amount,
                "ack_required": lines["INV-1001"].ack_required,
                "excluded": lines["INV-1001"].excluded,
                "claimed": lines["INV-1001"].recovery_claimed_amount,
            },
            "inv1002": {
                "gross": lines["INV-1002"].gross_amount,
                "credit": lines["INV-1002"].credit_amount,
                "net": lines["INV-1002"].net_amount,
                "ack_required": lines["INV-1002"].ack_required,
                "excluded": lines["INV-1002"].excluded,
            },
            "inv1008_excluded": lines["INV-1008"].excluded,
            "inv1008_net": lines["INV-1008"].net_amount,
            "inv1008_reason_names_statement": (
                "STMT-NB-2026-10" in lines["INV-1008"].exclusion_reason
            ),
            "payable_count": report.payable_count,
            "excluded_count": report.excluded_count,
            "ack_required_count": report.ack_required_count,
            "total_gross": report.total_gross,
            "total_discount": report.total_discount,
            "total_credit": report.total_credit,
            "total_net": report.total_net,
            "excluded_net": report.excluded_net,
            "lines_net_total": report.lines_net_total,
            "totals_agree": report.totals_agree,
            "gate_refused": gate_refused,
            "confirmed_after_ack": confirmed,
            "roll_before_recovery": {
                "open": roll_before["recovery_open_amount"],
                "received": roll_before["recovery_received_amount"],
            },
            "roll_after_recovery": {
                "open": roll_after["recovery_open_amount"],
                "received": roll_after["recovery_received_amount"],
            },
            "roll_payable_unchanged": {
                k: roll_before[k]
                for k in (
                    "gross_invoiced",
                    "credits_applied",
                    "open_amount",
                    "disputed_amount",
                )
            }
            == {
                k: roll_after[k]
                for k in (
                    "gross_invoiced",
                    "credits_applied",
                    "open_amount",
                    "disputed_amount",
                )
            },
        }
        expected = {
            "inv1001": {
                "match_status": "full_match",
                "gross": 1180.00,
                "discount": 23.60,
                "credit": 0.00,
                "net": 1156.40,
                "ack_required": True,
                "excluded": False,
                "claimed": 1180.00,
            },
            "inv1002": {
                "gross": 1274.40,
                "credit": 94.40,
                "net": 1180.00,
                "ack_required": False,
                "excluded": False,
            },
            "inv1008_excluded": True,
            "inv1008_net": 1180.00,
            "inv1008_reason_names_statement": True,
            "payable_count": 2,
            "excluded_count": 1,
            "ack_required_count": 1,
            "total_gross": 2454.40,
            "total_discount": 23.60,
            "total_credit": 94.40,
            "total_net": 2336.40,
            "excluded_net": 1180.00,
            "lines_net_total": 3516.40,
            "totals_agree": True,
            "gate_refused": True,
            "confirmed_after_ack": True,
            "roll_before_recovery": {"open": 1180.00, "received": 0.00},
            "roll_after_recovery": {"open": 0.00, "received": 1180.00},
            "roll_payable_unchanged": True,
        }
        if got == expected:
            control_correct = 1
        else:
            failures.append(
                f"payment-run control: expected {expected}, got {got}"
            )
    except Exception as exc:  # a broken control path is a failed case
        failures.append(f"payment-run control: {type(exc).__name__}: {exc}")

    total = len(cases)
    return {
        "total": total,
        "credit_note_total": credit_total,
        "credit_note_correct": credit_correct,
        "reconciliation_total": reconciliation_total,
        "reconciliation_correct": reconciliation_correct,
        "recovery_total": recovery_total,
        "recovery_correct": recovery_correct,
        "control_total": control_total,
        "control_correct": control_correct,
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
        and scored["reconciliation_correct"] == scored["reconciliation_total"]
        and scored["recovery_correct"] == scored["recovery_total"]
        and scored["control_correct"] == scored["control_total"]
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
            "statement_reconciliation": (
                f"{scored['reconciliation_correct']}/"
                f"{scored['reconciliation_total']}"
            ),
            "recovery_pack": (
                f"{scored['recovery_correct']}/{scored['recovery_total']}"
            ),
            "payment_run_control": (
                f"{scored['control_correct']}/{scored['control_total']}"
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
        f"Statement reconciliation: {scored['reconciliation_correct']}/"
        f"{scored['reconciliation_total']}"
    )
    print(
        f"Recovery pack           : {scored['recovery_correct']}/"
        f"{scored['recovery_total']}"
    )
    print(
        f"Payment-run control     : {scored['control_correct']}/"
        f"{scored['control_total']}"
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
