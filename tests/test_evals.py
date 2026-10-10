"""The golden eval set is a regression gate, not a fixture."""

from __future__ import annotations

import json
import sys
from pathlib import Path

EVALS_DIR = Path(__file__).resolve().parents[1] / "evals"
sys.path.insert(0, str(EVALS_DIR))

import run_evals  # noqa: E402

OUTCOMES = {
    "clean_match", "price_variance", "quantity_variance", "missing_po",
    "missing_receipt", "duplicate_suspected", "unreadable", "not_an_invoice",
}


def _cases() -> list[dict]:
    return [
        json.loads(line)
        for line in (EVALS_DIR / "golden.jsonl").read_text().splitlines()
        if line.strip()
    ]


def test_golden_set_shape():
    cases = _cases()
    assert len(cases) >= 28
    for case in cases:
        assert case["expected_outcome"] in OUTCOMES
        assert case["expected_match_status"] in {
            "full_match", "partial_match", "no_match", "no_po", "no_receipt",
        }
        assert ("sample_id" in case) ^ ("invoice" in case)


def test_golden_set_covers_every_outcome():
    covered = {c["expected_outcome"] for c in _cases()}
    assert covered == OUTCOMES


def test_evals_pass_the_gate():
    scored = run_evals.evaluate(_cases())
    assert scored["outcome_accuracy"] >= 0.95, scored["failures"]
    assert scored["match_accuracy"] >= 0.95, scored["failures"]
