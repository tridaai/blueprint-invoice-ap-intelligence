"""The duplicate screen cannot be bypassed by run mode.

The 2026-10-10 live judged slice (Nemotron-3-Super, 6 cases)
scored case-inv-1004 — the INV-1001 resend — as ``clean_match``
with resolution ``rules_only``, where the golden set expects
``duplicate_suspected``. The deterministic pack had never missed
it. Root cause: the AP-DUP-01 screen was not part of the
pipeline. ``InvoiceService`` computed ``find_duplicate`` over the
store's priors and passed the verdict into ``run_invoice`` —
so the screen fired only for callers who went through the
service AND whose store had accumulated the original. The LLM
eval pack called ``run_invoice`` directly, per case, with no
store and no verdict: the screen never ran, no duplicate
evidence existed, and triage fell through the (genuinely full)
three-way match to a clean outcome.

The fix is structural, in two places:

* ``run_invoice`` computes the verdict itself from whatever
  priors it is handed — a caller-supplied verdict is discarded
  when priors ride along, so no run mode that supplies the
  evidence can skip the screen; and
* the LLM eval pack runs its cases through one shared
  ``InvoiceService`` in file order, the production path, so the
  evidence exists to be screened against.

These tests reproduce the miss in the offline harness (bare
``run_invoice`` over the sample pair) and pin both halves.
"""

from __future__ import annotations

from ap_agent.graph import run_invoice
from ap_agent.samples import sample_invoice_models
from ap_agent.service import InvoiceService
from ap_agent.store import InMemoryStore

SAMPLES = {s.invoice_id: s for s in sample_invoice_models()}

# The store's prior-invoice evidence for INV-1001, in the shape
# ``Store.prior_invoices`` returns it (the stored invoice payload).
PRIORS = [SAMPLES["INV-1001"].model_dump(mode="json")]


def test_run_invoice_screens_a_duplicate_from_priors_alone():
    """The exact pre-fix miss shape, one level down: priors are
    handed to the pipeline but NO caller verdict is — the old
    code trusted the empty default and called the resend clean."""
    result = run_invoice(SAMPLES["INV-1004"], priors=PRIORS)
    assert result.facts.duplicate_of == "INV-1001"
    assert result.triage.outcome == "duplicate_suspected"


def test_a_caller_verdict_cannot_suppress_the_screen():
    """An empty — or outright wrong — caller verdict is discarded
    when priors are present: the verdict is computed from the
    evidence, never inherited."""
    empty = run_invoice(SAMPLES["INV-1004"], priors=PRIORS, duplicate_of="")
    assert empty.facts.duplicate_of == "INV-1001"
    assert empty.triage.outcome == "duplicate_suspected"

    wrong = run_invoice(SAMPLES["INV-1004"], priors=PRIORS, duplicate_of="INV-9999")
    assert wrong.facts.duplicate_of == "INV-1001"
    assert wrong.triage.outcome == "duplicate_suspected"


def test_no_priors_is_no_evidence_not_a_false_positive():
    """With nothing to screen against, the resend is what its own
    facts say — a clean full match. The screen's absence here is
    an absence of evidence; the harness fix (below) is what makes
    sure production-shaped runs always carry the evidence."""
    result = run_invoice(SAMPLES["INV-1004"])
    assert result.facts.duplicate_of == ""
    assert result.triage.outcome == "clean_match"


def test_unrelated_priors_do_not_trip_the_screen():
    """Priors for a different supplier/invoice change nothing:
    the computed verdict stays empty and the match stands."""
    priors = [SAMPLES["INV-1002"].model_dump(mode="json")]
    result = run_invoice(SAMPLES["INV-1004"], priors=priors)
    # INV-1002 shares neither supplier+number nor total with the
    # INV-1004 resend of INV-1001 — but INV-1001 is not among the
    # priors either, so no duplicate can be found.
    assert result.facts.duplicate_of == ""
    assert result.triage.outcome == "clean_match"


def test_the_eval_pack_shape_accumulates_through_one_service():
    """The harness half of the fix, exercised the way the LLM
    pack now runs: one service, one shared store, file order —
    the original lands first, and its resend is screened."""
    service = InvoiceService(store=InMemoryStore(), checkpointer=False)
    first = service.analyze(SAMPLES["INV-1001"])
    assert first.triage.outcome == "clean_match"
    resend = service.analyze(SAMPLES["INV-1004"])
    assert resend.facts.duplicate_of == "INV-1001"
    assert resend.triage.outcome == "duplicate_suspected"
