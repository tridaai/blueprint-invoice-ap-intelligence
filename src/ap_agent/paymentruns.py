"""Payment-run proposals — computed in code over the approved records.

This is how AP actually pays: not invoice by invoice, but in a run —
the approved invoices grouped into a pay list a finance lead
confirms. The agent's part stops exactly where its authority stops:

- Eligibility is a fact from the store: the invoice's human decision
  is ``approved``. Nothing awaiting, rejected, or already inside a
  confirmed run is proposed again.
- Grouping is arithmetic: by currency, then by due window —
  consecutive ``window_days``-long windows anchored on ``as_of``,
  with overdue invoices swept into the first window.
- Totals are summed in code; the discount column reuses each run's
  stored discount evaluation, and counts only while its deadline is
  still reachable inside the window.
- A run is a draft object. Confirming it (service layer) records a
  human's acceptance of the proposal; it pays nothing, moves no
  money, and changes no invoice's state — the customer's AP system
  of record executes the run, as with everything this agent drafts.
- Large runs carry a four-eyes rule: a net total above the
  configured threshold stamps the run ``four_eyes_required``, and
  the hand-off export (service layer) refuses it until a second,
  different confirmer has also confirmed the run.
- Discounts are re-verified at export: a window can close between
  confirmation and hand-off, so :func:`discount_drift` re-prices
  every discounted member as of the export date and itemises any
  drift on the run record — the exported file never promises a
  discount the bank run can no longer take.
- At confirmation, that same leak is projected forward
  (:func:`drift_forecast`): members whose discount deadlines fall
  before the run's expected export date — its proposal date plus
  the measured mean proposal→export lag — are flagged on the run
  as a forecast, so the warning arrives while the confirmer can
  still act on it. A forecast is labelled as one, in the data and
  in the note; the export-time drift check stays the authority.
- And the forecast itself is scored (:func:`forecast_accuracy`):
  every exported run's flags are set against its measured drift,
  so the projection the confirmer trusted earns or loses its keep
  in numbers — precision, recall, and the value on each side of
  the ledger, aggregated into the same metrics section.
"""

from __future__ import annotations

from datetime import date, datetime, timedelta, timezone

from .discounts import evaluate_discount
from .matching import _parse_date
from .schemas import (
    InvoiceInput,
    PaymentRun,
    PaymentRunDrift,
    PaymentRunForecast,
    PaymentRunLine,
    PaymentRunWaitLapse,
)
from .store import ApprovalRecord

DEFAULT_WINDOW_DAYS = 7


def _window_index(due: date, as_of: date, window_days: int) -> int:
    """Which due window a date falls in. Overdue and in-window dates
    share window 0; later dates step forward a window at a time."""
    delta = (due - as_of).days
    if delta < window_days:
        return 0
    return delta // window_days


def _line_for(
    record: ApprovalRecord, as_of: date, credit_amount: float = 0.0
) -> PaymentRunLine:
    result = record.result
    invoice = record.invoice or {}
    gross = float(invoice.get("total_amount") or 0.0)
    discount_amount = 0.0
    discount_deadline = ""
    evaluation = result.discount
    if evaluation is not None and evaluation.worthwhile:
        deadline = _parse_date(evaluation.discount_deadline)
        if deadline is not None and deadline >= as_of:
            discount_amount = round(float(evaluation.gross_discount), 2)
            discount_deadline = evaluation.discount_deadline
    return PaymentRunLine(
        invoice_id=result.invoice_id,
        supplier_name=result.supplier_name or str(invoice.get("supplier_name", "")),
        invoice_number=str(invoice.get("invoice_number", "")),
        due_date=str(invoice.get("due_date", "")),
        gross_amount=gross,
        discount_amount=discount_amount,
        discount_deadline=discount_deadline,
        credit_amount=round(float(credit_amount), 2),
        net_amount=round(
            max(0.0, gross - discount_amount - float(credit_amount)), 2
        ),
    )


def propose_payment_runs(
    records: list[ApprovalRecord],
    *,
    window_days: int = DEFAULT_WINDOW_DAYS,
    as_of: date | None = None,
    exclude_invoice_ids: set[str] | None = None,
    four_eyes_threshold: float | None = None,
    four_eyes_thresholds: dict[str, float] | None = None,
    credits_by_invoice: dict[str, float] | None = None,
) -> list[PaymentRun]:
    """Group the approved, not-yet-run records into proposed runs.

    Pure over the records it is given — the service layer supplies
    the store's records and the membership of confirmed runs. When
    a four-eyes threshold applies, a run whose net total is *above*
    it is stamped ``four_eyes_required``: the hand-off then waits
    for a second confirmer, different from the first (the service
    enforces it). A total exactly at the threshold is not above it.

    The threshold is per-currency: ``four_eyes_thresholds`` maps a
    currency code to its threshold, and ``four_eyes_threshold`` is
    the fallback for currencies the map does not name — 25,000 USD
    and 25,000 JPY are not the same risk, so one global number
    cannot gate both honestly. Whichever figure applied is stamped
    on the run (``four_eyes_threshold``) and named in its note.
    """
    as_of = as_of or date.today()
    window_days = max(1, int(window_days))
    excluded = exclude_invoice_ids or set()
    groups: dict[tuple[str, int], list[ApprovalRecord]] = {}
    for record in records:
        result = record.result
        if result.status != "approved":
            continue
        if result.invoice_id in excluded:
            continue
        invoice = record.invoice or {}
        currency = str(invoice.get("currency") or result.facts.currency or "USD")
        due = _parse_date(str(invoice.get("due_date", ""))) or as_of
        index = _window_index(due, as_of, window_days)
        groups.setdefault((currency, index), []).append(record)

    runs: list[PaymentRun] = []
    credits = credits_by_invoice or {}
    for (currency, index), members in sorted(groups.items()):
        lines = [
            _line_for(record, as_of, credits.get(record.result.invoice_id, 0.0))
            for record in members
        ]
        lines.sort(key=lambda line: (line.due_date or "9999", line.invoice_id))
        dues = [
            parsed
            for line in lines
            if (parsed := _parse_date(line.due_date)) is not None
        ]
        window_end = as_of + timedelta(days=(index + 1) * window_days - 1)
        window_start = min(dues) if dues and index == 0 else as_of + timedelta(
            days=index * window_days
        )
        run = PaymentRun(
            run_id=f"RUN-{currency}-{window_start.isoformat()}-{window_end.isoformat()}",
            currency=currency,
            window_start=window_start.isoformat(),
            window_end=window_end.isoformat(),
            window_days=window_days,
            lines=lines,
            invoice_count=len(lines),
            total_gross=round(sum(line.gross_amount for line in lines), 2),
            total_discount=round(sum(line.discount_amount for line in lines), 2),
            total_credit=round(sum(line.credit_amount for line in lines), 2),
            total_net=round(sum(line.net_amount for line in lines), 2),
        )
        threshold = four_eyes_threshold
        if four_eyes_thresholds:
            threshold = four_eyes_thresholds.get(currency, four_eyes_threshold)
        if threshold is not None:
            run.four_eyes_threshold = threshold
            run.four_eyes_required = run.total_net > threshold
        run.note = (
            f"{run.invoice_count} approved invoice(s), {currency} "
            f"{run.total_net:.2f} net payable in the window ending "
            f"{run.window_end}"
            + (
                f" — includes {run.total_discount:.2f} of early-payment "
                "discounts still capturable"
                if run.total_discount
                else ""
            )
            + ". A proposal only: a finance lead confirms the run; "
            "payment itself happens in the AP system of record."
            + (
                f" The four-eyes threshold for {currency} runs is "
                f"{currency} {threshold:,.2f}."
                if threshold is not None
                else ""
            )
            + (
                " This run is above that threshold: a second "
                "confirmer, different from the first, must also confirm "
                "it before the hand-off file can be generated."
                if run.four_eyes_required
                else ""
            )
        )
        runs.append(run)
    return runs


def run_member_ids(run: PaymentRun) -> set[str]:
    return {line.invoice_id for line in run.lines}


def drift_forecast(
    run: PaymentRun,
    *,
    as_of: date,
    lag_stats: dict | None = None,
    lag_segments: dict | None = None,
    lag_grades: dict | None = None,
    lag_drift_watch: dict | None = None,
    profile_id: str = "",
    excluded_segments: set[str] | frozenset[str] | None = None,
) -> tuple[list[PaymentRunForecast], str]:
    """Forecast, at confirmation, which of a run's discounted
    members will lapse before the run is likely to be exported.

    The metrics have always shown the leak after the fact: runs
    sit between proposal and export, and a member whose discount
    deadline falls inside that sit loses its discount (the export
    drift check then re-prices it at full gross). This projects
    the same numbers forward while the confirmer can still act:
    the expected export date is the run's proposal date plus a
    measured proposal→export lag, and a discounted member is
    flagged when its deadline falls before that date.

    Which lag statistic projects is itself a decision, and it is
    no longer a static one. Given ``lag_segments`` (see
    :func:`lag_statistics`), the statistic is *selected* for this
    run — its governing profile's segment, its currency's, or the
    global pool — by :func:`_select_lag_statistic`, which grades
    the candidates against the accuracy ledger (the recorded
    export outcomes, :func:`grade_lag_statistics`) and prefers
    the statistic with the best recorded recall. The drift watch
    (``lag_drift_watch``, :func:`lag_grade_drift_watch`) can
    demote a candidate whose recent recall has decayed from its
    record; the basis names the demotion when one shapes the
    choice. The basis states the statistic used and why it was
    chosen. The legacy flat ``lag_stats`` argument (the global
    statistic, pre-selection) remains for direct callers.

    Returns (forecast entries, basis). The basis always says what
    the projection rests on — including the honest empty cases:
    no exported runs yet means no measured lag, and no forecast is
    dressed up without one. Pure: the run is not mutated; the
    service stamps the result on the run record at confirmation.
    """
    label = ""
    why = ""
    if lag_segments is not None:
        chosen, label, why = _select_lag_statistic(
            run,
            lag_segments,
            lag_grades or {},
            profile_id,
            lag_drift_watch,
            excluded_segments,
        )
        if chosen is None:
            return [], (
                why
                or "no forecast: no exported runs yet, so there is no "
                "measured proposal→export lag to project from"
            )
        lag_stats = chosen
    if not lag_stats or lag_stats.get("mean_days") is None:
        return [], (
            "no forecast: no exported runs yet, so there is no "
            "measured proposal→export lag to project from"
        )
    proposed = _parse_date(run.proposed_at)
    if proposed is None:
        return [], "no forecast: the run carries no proposal date"
    lag_days = float(lag_stats["mean_days"])
    count = int(lag_stats.get("count") or 0)
    expected_export = proposed + timedelta(days=lag_days)
    if label:
        basis = (
            f"forecast from the {label} proposal→export lag: mean "
            f"{lag_days:.2f} days over {count} exported run(s) — "
            f"{why}; this run proposed {proposed.isoformat()}, so "
            f"its expected export date is {expected_export.isoformat()}"
        )
    else:
        basis = (
            f"forecast from the measured proposal→export lag: mean "
            f"{lag_days:.2f} days over {count} exported run(s); this "
            f"run proposed {proposed.isoformat()}, so its expected "
            f"export date is {expected_export.isoformat()}"
        )
    entries: list[PaymentRunForecast] = []
    for line in run.lines:
        if line.discount_amount <= 0:
            continue
        deadline = _parse_date(line.discount_deadline)
        if deadline is None or deadline >= expected_export:
            continue
        headroom = (deadline - as_of).days
        entries.append(
            PaymentRunForecast(
                invoice_id=line.invoice_id,
                supplier_name=line.supplier_name,
                discount_amount=line.discount_amount,
                discount_deadline=line.discount_deadline,
                headroom_days=headroom,
                expected_export_date=expected_export.isoformat(),
                message=(
                    f"discount of {run.currency} "
                    f"{line.discount_amount:.2f} expires "
                    f"{line.discount_deadline}, before this run's "
                    f"expected export date "
                    f"{expected_export.isoformat()} — export by "
                    f"{line.discount_deadline} or the discount is "
                    "projected to lapse (forecast from the measured "
                    "proposal→export lag, not a promise)"
                ),
            )
        )
    return entries, basis


def _lag_days(start: str | None, end: str | None) -> float | None:
    """Whole-and-part days between two ISO timestamps, or None
    when either is missing or unparseable (an unstamped legacy
    run is excluded from the lag stats, never counted as zero)."""
    if not start or not end:
        return None
    try:
        delta = datetime.fromisoformat(end) - datetime.fromisoformat(start)
    except ValueError:
        return None
    return round(delta.total_seconds() / 86400.0, 2)


def _lag_stats(lags: list[float]) -> dict | None:
    if not lags:
        return None
    return {
        "count": len(lags),
        "mean_days": round(sum(lags) / len(lags), 2),
        "max_days": max(lags),
    }


def lag_statistics(runs: list[PaymentRun]) -> dict:
    """Proposal→export lag statistics, segmented — the raw
    material the forecast's statistic selection chooses from.

    One pool is the global statistic the forecast always used.
    The others exist because the pool mixes unlike work: a
    currency whose runs clear through a slow bank rail, or a
    hand-off profile whose lead days stretch every export, has a
    lag of its own, and the global mean understates exactly the
    runs most likely to drift. Segments: ``by_currency`` over the
    run's currency, ``by_profile`` over the governing profile id
    the export stamped on the run (ungoverned exports join no
    profile segment). Each value is a :func:`_lag_stats` dict or
    None; computed over exported runs only. Pure.
    """
    global_lags: list[float] = []
    by_currency: dict[str, list[float]] = {}
    by_profile: dict[str, list[float]] = {}
    for run in runs:
        if not run.handoff_generated_at:
            continue
        lag = _lag_days(run.proposed_at, run.handoff_generated_at)
        if lag is None:
            continue
        global_lags.append(lag)
        by_currency.setdefault(run.currency, []).append(lag)
        if run.handoff_profile_id:
            by_profile.setdefault(run.handoff_profile_id, []).append(lag)
    return {
        "global": _lag_stats(global_lags),
        "by_currency": {
            key: _lag_stats(values)
            for key, values in sorted(by_currency.items())
        },
        "by_profile": {
            key: _lag_stats(values)
            for key, values in sorted(by_profile.items())
        },
    }


def _lag_grade_members(runs: list[PaymentRun]):
    """The grading replay, per member: for every exported run
    carrying discounted members, each applicable statistic
    (global, the run's currency, the run's governing profile) is
    recomputed WITHOUT that run (leave-one-out), its flag rule
    replayed, and each member scored against the drift its
    export actually measured. Yields one dict per graded member
    per statistic — the shared evidence base the grades
    (:func:`grade_lag_statistics`) aggregate and the drift watch
    (:func:`lag_grade_drift_watch`) windows over time."""
    exported: list[tuple[PaymentRun, float]] = []
    for run in runs:
        if not run.handoff_generated_at:
            continue
        lag = _lag_days(run.proposed_at, run.handoff_generated_at)
        if lag is not None:
            exported.append((run, lag))
    for run, _own_lag in exported:
        discounted = [
            line for line in run.lines if line.discount_amount > 0
        ]
        if not discounted:
            continue
        proposed = _parse_date(run.proposed_at)
        if proposed is None:
            continue
        drifted = {
            entry.invoice_id: entry
            for entry in run.handoff_drift
            if entry.discount_lost > 0.005
        }
        candidates: list[tuple[str, list[float]]] = [
            (
                "global",
                [lag for other, lag in exported if other is not run],
            ),
            (
                f"currency:{run.currency}",
                [
                    lag
                    for other, lag in exported
                    if other is not run and other.currency == run.currency
                ],
            ),
        ]
        if run.handoff_profile_id:
            candidates.append(
                (
                    f"profile:{run.handoff_profile_id}",
                    [
                        lag
                        for other, lag in exported
                        if other is not run
                        and other.handoff_profile_id
                        == run.handoff_profile_id
                    ],
                )
            )
        members = (
            set(drifted)
            | {line.invoice_id for line in discounted}
        )
        amounts = {line.invoice_id: line.discount_amount for line in discounted}
        for key, segment_lags in candidates:
            if not segment_lags:
                continue
            mean_lag = sum(segment_lags) / len(segment_lags)
            expected = proposed + timedelta(days=mean_lag)
            flagged = {
                line.invoice_id
                for line in discounted
                if (deadline := _parse_date(line.discount_deadline))
                is not None
                and deadline < expected
            }
            for invoice_id in members:
                is_flagged = invoice_id in flagged
                is_drifted = invoice_id in drifted
                if is_flagged and is_drifted:
                    bucket = "flagged_drifted"
                elif is_flagged:
                    bucket = "flagged_saved"
                elif is_drifted:
                    bucket = "unflagged_drifted"
                else:
                    bucket = "unflagged_saved"
                yield {
                    "statistic": key,
                    "run": run,
                    "invoice_id": invoice_id,
                    "bucket": bucket,
                    "discount_amount": amounts.get(invoice_id, 0.0),
                    "discount_lost": (
                        drifted[invoice_id].discount_lost if is_drifted else 0.0
                    ),
                }


def grade_lag_statistics(runs: list[PaymentRun]) -> dict:
    """Grade each candidate lag statistic against the accuracy
    ledger — what the exports actually recorded.

    The forecast's report card (:func:`forecast_accuracy`) scores
    the forecasts that were made. This scores the *statistics*
    that could have made them: the per-member replay
    (:func:`_lag_grade_members`) aggregated into the same
    two-by-two cells. Returns ``{statistic_key: {cells…,
    precision, recall, graded_members}}`` — the selection's
    evidence base. A statistic with no graded members is absent,
    never dressed up with a rate over zero cases. Pure.
    """
    cells: dict[str, dict] = {}
    for member in _lag_grade_members(runs):
        cell = cells.setdefault(member["statistic"], _new_accuracy_cell())
        cell[member["bucket"]] += 1
        if member["bucket"] in ("flagged_drifted", "flagged_saved"):
            cell["value_flagged"] = round(
                cell["value_flagged"] + member["discount_amount"], 2
            )
        if member["bucket"] == "flagged_drifted":
            cell["value_lost_flagged"] = round(
                cell["value_lost_flagged"] + member["discount_lost"], 2
            )
        elif member["bucket"] == "unflagged_drifted":
            cell["value_lost_unflagged"] = round(
                cell["value_lost_unflagged"] + member["discount_lost"], 2
            )
    graded: dict[str, dict] = {}
    for key, cell in sorted(cells.items()):
        precision, recall = _accuracy_rates(cell)
        graded[key] = {
            **{
                name: cell[name]
                for name in (
                    "flagged_drifted",
                    "flagged_saved",
                    "unflagged_drifted",
                    "unflagged_saved",
                    "value_flagged",
                    "value_lost_flagged",
                    "value_lost_unflagged",
                )
            },
            "precision": precision,
            "recall": recall,
            "graded_members": sum(
                cell[name]
                for name in (
                    "flagged_drifted",
                    "flagged_saved",
                    "unflagged_drifted",
                    "unflagged_saved",
                )
            ),
        }
    return graded


# The drift watch's windows: the recent sample is the last few
# exported runs (in export order) that carried graded members,
# and a window with fewer graded members than this is an
# anecdote, not a trend — the watch assesses nothing on it.
_DRIFT_WATCH_RECENT_RUNS = 5
_DRIFT_WATCH_MIN_WINDOW_MEMBERS = 3
# A recall fall of this size, earlier record to recent window,
# is the statistic losing the drift it used to catch — not noise.
_DRIFT_WATCH_RECALL_DROP = 0.25


def lag_grade_drift_watch(
    runs: list[PaymentRun],
    *,
    recent_runs: int = _DRIFT_WATCH_RECENT_RUNS,
    min_window_members: int = _DRIFT_WATCH_MIN_WINDOW_MEMBERS,
    recall_drop: float = _DRIFT_WATCH_RECALL_DROP,
) -> dict:
    """Watch the grades for decay: is a statistic still catching
    what its record says it catches?

    The selection trusts a statistic's grade — one number over
    the whole ledger. A bank that slows down does not announce
    itself: the profile segment that used to flag every lapse
    starts missing them, and the pooled grade dilutes the decay
    for months. This watch windows the same per-member replay
    (:func:`_lag_grade_members`) by export order: the members
    graded from the most recent ``recent_runs`` exported runs
    form the recent sample, the rest the earlier record, and a
    statistic whose recent recall has fallen ``recall_drop`` or
    more below its earlier recall is **degrading** — named in
    ``/metrics`` and demoted by the forecast's selection until
    its recent record recovers. A statistic is assessed only
    with at least ``min_window_members`` graded members in BOTH
    windows and a recall on both sides; anything thinner is
    reported unassessed rather than dressed up as stable. Pure.
    """
    members = list(_lag_grade_members(runs))
    graded_runs = sorted(
        {member["run"].run_id: member["run"] for member in members}.values(),
        key=lambda run: (run.handoff_generated_at or "", run.run_id),
    )
    recent_ids = {run.run_id for run in graded_runs[-recent_runs:]} if recent_runs else set()
    windows: dict[str, dict[str, dict]] = {}
    for member in members:
        per_key = windows.setdefault(
            member["statistic"],
            {"overall": _new_accuracy_cell(), "earlier": _new_accuracy_cell(), "recent": _new_accuracy_cell()},
        )
        per_key["overall"][member["bucket"]] += 1
        side = "recent" if member["run"].run_id in recent_ids else "earlier"
        per_key[side][member["bucket"]] += 1

    def _members(cell: dict) -> int:
        return sum(
            cell[name]
            for name in (
                "flagged_drifted",
                "flagged_saved",
                "unflagged_drifted",
                "unflagged_saved",
            )
        )

    watch: dict[str, dict] = {}
    for key in sorted(windows):
        per_key = windows[key]
        overall_recall = _accuracy_rates(per_key["overall"])[1]
        earlier_recall = _accuracy_rates(per_key["earlier"])[1]
        recent_recall = _accuracy_rates(per_key["recent"])[1]
        earlier_members = _members(per_key["earlier"])
        recent_members = _members(per_key["recent"])
        assessed = (
            earlier_members >= min_window_members
            and recent_members >= min_window_members
            and earlier_recall is not None
            and recent_recall is not None
        )
        degrading = bool(
            assessed and earlier_recall - recent_recall >= recall_drop
        )
        watch[key] = {
            "overall_recall": overall_recall,
            "earlier_recall": earlier_recall,
            "recent_recall": recent_recall,
            "earlier_graded_members": earlier_members,
            "recent_graded_members": recent_members,
            "assessed": assessed,
            "degrading": degrading,
        }
    return watch


# A segment mean over one or two exports is an anecdote, not a
# statistic; a grade over three members is a coin toss. The
# selection requires both bars before it trusts either.
_MIN_SEGMENT_RUNS = 3
_MIN_GRADED_MEMBERS = 4


def _select_lag_statistic(
    run: PaymentRun,
    segments: dict,
    grades: dict,
    profile_id: str,
    drift_watch: dict | None = None,
    excluded_segments: set[str] | frozenset[str] | None = None,
) -> tuple[dict | None, str, str]:
    """Choose the lag statistic that projects this run's export,
    and say why — (stats, label, why); (None, "", "") when no
    segment carries a measured lag at all.

    Candidates, most specific first: the governing profile's
    segment, the currency's, the global pool. Among candidates
    with enough measured runs, a candidate the ledger has graded
    over enough members wins on recorded recall (the miss —
    ``unflagged_drifted`` — is the expensive cell; precision,
    then specificity, break ties). With no graded candidate, the
    most specific adequately-measured segment governs. With no
    adequately-measured segment at all, the global pool — the
    pre-selection behaviour — is the fallback, so a young ledger
    forecasts exactly as it did before segments existed.

    The drift watch (``drift_watch``, :func:`lag_grade_drift_watch`)
    can demote a graded candidate: a statistic whose RECENT
    recall has fallen away from its record is not allowed to win
    on the strength of the old record — the selection falls to
    the best standing graded candidate, or, when every graded
    candidate is demoted, to the most specific adequately-
    measured segment — and the why names the demotion either way.

    A governed exclusion (``excluded_segments`` — segment keys an
    approved lag-segment proposal retired, derived from the
    proposal ledger by the service) is stronger than a demotion:
    the candidate is not in the set at all, however good its
    record, and the why names the exclusion. When EVERY
    candidate stands excluded there is nothing the selection
    may crown — it returns no statistic and says why, rather
    than quietly re-admitting a statistic a named human retired.
    """
    candidates: list[tuple[int, str, str, dict]] = []
    profile_stats = (
        segments.get("by_profile", {}).get(profile_id)
        if profile_id
        else None
    )
    if profile_stats:
        candidates.append(
            (2, f"profile:{profile_id}", f"per-profile ({profile_id})", profile_stats)
        )
    currency_stats = segments.get("by_currency", {}).get(run.currency)
    if currency_stats:
        candidates.append(
            (1, f"currency:{run.currency}", f"per-currency ({run.currency})", currency_stats)
        )
    global_stats = segments.get("global")
    if global_stats:
        candidates.append((0, "global", "global", global_stats))
    if not candidates:
        return None, "", ""
    excluded = set(excluded_segments or ())
    excluded_here = [c for c in candidates if c[1] in excluded]
    if excluded_here:
        candidates = [c for c in candidates if c[1] not in excluded]
        if not candidates:
            names = "; ".join(
                f"the {c[2]} statistic" for c in excluded_here
            )
            return (
                None,
                "",
                "no forecast: every candidate lag statistic is "
                f"under a governed exclusion ({names} — retired "
                "by approved lag-segment proposals) and no "
                "statistic may project this run until a re-grade "
                "re-admits one",
            )
    exclusion_clause = ""
    if excluded_here:
        exclusion_clause = "; " + "; ".join(
            f"the {c[2]} statistic stands excluded under a "
            "governed lag-segment decision and was not a candidate"
            for c in excluded_here
        )
    eligible = [
        c for c in candidates if (c[3].get("count") or 0) >= _MIN_SEGMENT_RUNS
    ]
    if not eligible:
        if "global" in excluded:
            # The pre-selection fallback IS the global pool; an
            # excluded pool is no fallback. Nothing may project.
            return (
                None,
                "",
                "no forecast: no segment holds an adequately-"
                "measured lag, and the global pool — the fallback "
                "of last resort — stands excluded under a governed "
                "lag-segment decision",
            )
        stats, label = global_stats, "global"
        return (
            stats,
            label,
            "the only segment with a measured lag so far — segment "
            "statistics govern once one holds "
            f"{_MIN_SEGMENT_RUNS} or more exported runs"
            + exclusion_clause,
        )
    graded = [
        c
        for c in eligible
        if (grades.get(c[1]) or {}).get("graded_members", 0)
        >= _MIN_GRADED_MEMBERS
        and (grades.get(c[1]) or {}).get("recall") is not None
    ]
    if graded:
        def rank(candidate: tuple[int, str, str, dict]) -> tuple:
            grade = grades[candidate[1]]
            return (
                grade["recall"],
                grade["precision"] or 0.0,
                candidate[0],
            )

        watch = drift_watch or {}
        demoted = [
            c for c in graded if (watch.get(c[1]) or {}).get("degrading")
        ]
        demoted_keys = {c[1] for c in demoted}
        standing = [c for c in graded if c[1] not in demoted_keys]

        def demotion_clause() -> str:
            parts = []
            for candidate in demoted:
                entry = watch[candidate[1]]
                parts.append(
                    f"the {candidate[2]} statistic was demoted by "
                    f"the lag-grade drift watch (recent recall "
                    f"{entry['recent_recall']:.2f} over its latest "
                    f"graded members, down from "
                    f"{entry['earlier_recall']:.2f} on the earlier "
                    "record)"
                )
            return "; ".join(parts)

        if standing:
            best = max(standing, key=rank)
            grade = grades[best[1]]
            precision_clause = (
                f" (precision {grade['precision']:.2f})"
                if grade["precision"] is not None
                else ""
            )
            why = (
                "selected on the accuracy ledger: recall "
                f"{grade['recall']:.2f} over {grade['graded_members']} "
                f"graded members{precision_clause}, the best recorded "
                f"of {len(graded)} graded candidate(s)"
            )
            if demoted:
                why = f"{why}; {demotion_clause()}"
            return best[3], best[2], why + exclusion_clause
        # Every graded candidate is demoted: the forecast falls
        # back to the most specific adequately-measured segment,
        # with the demotions named — never to a statistic the
        # watch just caught decaying.
        best = max(eligible, key=lambda c: c[0])
        return (
            best[3],
            best[2],
            f"{demotion_clause()}; no graded candidate stands, so "
            "the most specific adequately-measured segment governs"
            + exclusion_clause,
        )
    best = max(eligible, key=lambda c: c[0])
    return (
        best[3],
        best[2],
        "no candidate statistic has graded history over "
        f"{_MIN_GRADED_MEMBERS} or more members yet, so the most "
        "specific adequately-measured segment governs"
        + exclusion_clause,
    )


def _new_accuracy_cell() -> dict:
    return {
        "flagged_drifted": 0,
        "flagged_saved": 0,
        "unflagged_drifted": 0,
        "unflagged_saved": 0,
        "value_flagged": 0.0,
        "value_lost_flagged": 0.0,
        "value_lost_unflagged": 0.0,
    }


def _accuracy_rates(cell: dict) -> tuple[float | None, float | None]:
    flagged = cell["flagged_drifted"] + cell["flagged_saved"]
    drifted = cell["flagged_drifted"] + cell["unflagged_drifted"]
    precision = round(cell["flagged_drifted"] / flagged, 4) if flagged else None
    recall = round(cell["flagged_drifted"] / drifted, 4) if drifted else None
    return precision, recall


def forecast_accuracy(runs: list[PaymentRun]) -> dict:
    """Score the confirmation-time drift forecasts against what
    the exports actually measured — the ``forecast_accuracy``
    entry of :func:`handoff_metrics`.

    A forecast is a projection; this is its report card. For every
    *exported* run whose confirmation carried a forecast computed
    from a measured lag (its ``drift_forecast_basis`` says so),
    each discounted member lands in one cell of the two-by-two:

    - **flagged & drifted** — the warning fired and the discount
      still lapsed: the forecast was right; the process was slow.
    - **flagged & saved** — the warning fired and the member
      exported in time: the outcome the warning exists to buy
      (the forecast cannot distinguish "saved by the warning" from
      "would have been fine" — it claims neither).
    - **unflagged & drifted** — the miss: a discount lapsed that
      no forecast named. This is the cell that argues for a
      better lag statistic — or for retiring the forecast.
    - **unflagged & saved** — the quiet majority, counted so the
      rates have honest denominators.

    A member counts as drifted when its export-time drift entry
    lost value (``discount_lost`` > 0); a partial re-price is
    drift too. Precision = flagged∩drifted ÷ flagged; recall =
    flagged∩drifted ÷ drifted; ``None`` when the denominator is
    empty — no rate is dressed up over zero cases.

    Runs exported before forecasts existed, and runs whose
    confirmation honestly made no forecast (no measured lag yet),
    are *unscored*: their drift is reported separately
    (``unscored_drift``) rather than counted against a forecast
    nobody made. Pure over the runs given.
    """
    overall = _new_accuracy_cell()
    by_currency: dict[str, dict] = {}
    scored_runs = 0
    unscored_runs = 0
    unscored_members = 0
    unscored_value = 0.0
    for run in runs:
        if not run.handoff_generated_at:
            continue
        drifted = {
            entry.invoice_id: entry
            for entry in run.handoff_drift
            if entry.discount_lost > 0.005
        }
        if not run.drift_forecast_basis.startswith("forecast from"):
            unscored_runs += 1
            unscored_members += len(drifted)
            unscored_value += sum(
                entry.discount_lost for entry in drifted.values()
            )
            continue
        scored_runs += 1
        cell = by_currency.setdefault(run.currency, _new_accuracy_cell())
        flagged = {entry.invoice_id: entry for entry in run.drift_forecast}
        members = (
            set(flagged)
            | set(drifted)
            | {
                line.invoice_id
                for line in run.lines
                if line.discount_amount > 0
            }
        )
        for target in (overall, cell):
            target["value_flagged"] = round(
                target["value_flagged"]
                + sum(entry.discount_amount for entry in flagged.values()),
                2,
            )
        for invoice_id in members:
            is_flagged = invoice_id in flagged
            is_drifted = invoice_id in drifted
            if is_flagged and is_drifted:
                bucket = "flagged_drifted"
            elif is_flagged:
                bucket = "flagged_saved"
            elif is_drifted:
                bucket = "unflagged_drifted"
            else:
                bucket = "unflagged_saved"
            for target in (overall, cell):
                target[bucket] += 1
                if is_drifted:
                    key = (
                        "value_lost_flagged"
                        if is_flagged
                        else "value_lost_unflagged"
                    )
                    target[key] = round(
                        target[key] + drifted[invoice_id].discount_lost, 2
                    )
    precision, recall = _accuracy_rates(overall)
    return {
        "scored_runs": scored_runs,
        "unscored_runs": unscored_runs,
        "members": {
            key: overall[key]
            for key in (
                "flagged_drifted",
                "flagged_saved",
                "unflagged_drifted",
                "unflagged_saved",
            )
        },
        "precision": precision,
        "recall": recall,
        "value_flagged": overall["value_flagged"],
        "value_lost_flagged": overall["value_lost_flagged"],
        "value_lost_unflagged": overall["value_lost_unflagged"],
        "by_currency": {
            currency: {
                **{
                    key: cell[key]
                    for key in (
                        "flagged_drifted",
                        "flagged_saved",
                        "unflagged_drifted",
                        "unflagged_saved",
                        "value_flagged",
                        "value_lost_flagged",
                        "value_lost_unflagged",
                    )
                },
                "precision": _accuracy_rates(cell)[0],
                "recall": _accuracy_rates(cell)[1],
            }
            for currency, cell in sorted(by_currency.items())
        },
        "unscored_drift": {
            "members": unscored_members,
            "value_lost": round(unscored_value, 2),
        },
    }


def _new_completion_cell() -> dict:
    return {
        "exports": 0,
        "members": 0,
        "members_completed": 0,
        "members_marked": 0,
        "value_total": 0.0,
        "value_completed": 0.0,
        "value_marked": 0.0,
    }


def _completion_shares(cell: dict) -> dict:
    cell["member_completion_share"] = (
        round(cell["members_completed"] / cell["members"], 4)
        if cell["members"]
        else None
    )
    cell["value_completion_share"] = (
        round(cell["value_completed"] / cell["value_total"], 4)
        if cell["value_total"]
        else None
    )
    return cell


def creditor_completion_metrics(runs: list[PaymentRun]) -> dict:
    """Creditor-completion coverage — the ``creditor_completion``
    entry of :func:`handoff_metrics`.

    Master coverage (in ``/metrics``' ``supplier_master`` section)
    counts *records*: how much of the processed flow has a
    governed master entry. This counts *exports*: of the members
    actually handed to the bank, how many — and how much value —
    carried creditor accounts completed from the master, and how
    many still carried the ``SUPPLIER-MASTER-AT-IMPORT`` marker
    for the importing system to fill. The marker is the residual
    import risk; this is its price tag, per currency and per
    hand-off format, read off each run's completion receipt
    (``handoff_creditor_entries``, a per-member record of the
    latest export).

    A member's value is its export-time net amount (the figure
    the bank was told to pay). Runs exported before completion
    receipts existed export-count but receipt-less: they are
    reported as ``exports_without_receipt`` so the coverage
    denominator is never quietly smaller than the export count.

    ``by_execution_group`` splits the same coverage by the
    PmtInf grouping the member was exported under: each entry's
    stamped execution date — its batch-booking group's date, or
    the single-PmtInf file's date — so the marker risk is priced
    per value date, not just per file. Entries from before the
    stamp (and CSV exports, which carry no execution date) are
    absent from the split, never silently grouped.
    Pure over the runs given.
    """
    overall = _new_completion_cell()
    groups: dict[tuple[str, str], dict] = {}
    execution_groups: dict[str, dict] = {}
    execution_group_runs: dict[str, set] = {}
    exports_total = 0
    exports_without_receipt = 0

    def accumulate(cell: dict, *, completed: bool, value: float) -> None:
        cell["members"] += 1
        cell["value_total"] = round(cell["value_total"] + value, 2)
        if completed:
            cell["members_completed"] += 1
            cell["value_completed"] = round(cell["value_completed"] + value, 2)
        else:
            cell["members_marked"] += 1
            cell["value_marked"] = round(cell["value_marked"] + value, 2)

    for run in runs:
        if not run.handoff_generated_at:
            continue
        exports_total += 1
        if not run.handoff_creditor_entries:
            exports_without_receipt += 1
            continue
        nets = {line.invoice_id: line.net_amount for line in run.lines}
        cell = groups.setdefault(
            (run.currency, run.handoff_format or "unknown"),
            _new_completion_cell(),
        )
        overall["exports"] += 1
        cell["exports"] += 1
        for entry in run.handoff_creditor_entries:
            value = round(nets.get(entry.invoice_id, 0.0), 2)
            completed = entry.source == "supplier_master"
            for target in (overall, cell):
                accumulate(target, completed=completed, value=value)
            if entry.execution_date:
                # The PmtInf grouping split: under batch booking a
                # member's group is its own execution date; under a
                # single-PmtInf file it is the file's date. Either
                # way the marker risk is priced per value date.
                ecell = execution_groups.setdefault(
                    entry.execution_date, _new_completion_cell()
                )
                execution_group_runs.setdefault(entry.execution_date, set()).add(
                    run.run_id
                )
                accumulate(ecell, completed=completed, value=value)
    for date_key, cell in execution_groups.items():
        cell["exports"] = len(execution_group_runs[date_key])
    return {
        "exports_total": exports_total,
        "exports_with_receipt": overall["exports"],
        "exports_without_receipt": exports_without_receipt,
        "overall": _completion_shares(overall),
        "by_currency_format": [
            {
                "currency": currency,
                "format": format_name,
                **_completion_shares(cell),
            }
            for (currency, format_name), cell in sorted(groups.items())
        ],
        "by_execution_group": [
            {
                "execution_date": date_key,
                "runs": len(execution_group_runs[date_key]),
                **_completion_shares(cell),
            }
            for date_key, cell in sorted(execution_groups.items())
        ],
    }


def price_four_eyes_wait(
    run: PaymentRun,
) -> tuple[float, list[PaymentRunWaitLapse]]:
    """Price a four-eyes run's second-confirmation wait: the
    member discounts that were capturable when the first
    confirmer signed and had lapsed by the time the second did.

    The wait was previously a feeling — the run sat, the queue
    aged it, the digest flagged the stall — but nobody could say
    what the stall *cost*. This is the number: for each line
    carrying a capturable discount, the discount is charged to
    the wait exactly when its deadline falls inside the wait —
    on or after the first confirmation's date (so it was still
    takeable when the run entered the wait) and before the
    second confirmation's date (so it was gone when the run
    left it), under the same capturable-on-the-deadline-day rule
    the proposal and the drift check use. A discount that died
    before the first confirmation is the calendar's fault, not
    the wait's, and is never charged here (the drift machinery
    prices the whole confirm-to-export span separately); a
    discount still alive at the second confirmation cost the
    wait nothing.

    Pure over the run record: the service stamps the result onto
    the run when the second confirmation lands. Runs without
    both confirmations — including every run that never needed
    a second confirmer — price at zero."""
    first = _parse_date(run.confirmed_at)
    second = _parse_date(run.second_confirmed_at)
    if first is None or second is None:
        return 0.0, []
    lapses: list[PaymentRunWaitLapse] = []
    for line in run.lines:
        if line.discount_amount <= 0:
            continue
        deadline = _parse_date(line.discount_deadline)
        if deadline is None:
            continue
        if first <= deadline < second:
            lapses.append(
                PaymentRunWaitLapse(
                    invoice_id=line.invoice_id,
                    supplier_name=line.supplier_name,
                    discount_amount=line.discount_amount,
                    discount_deadline=line.discount_deadline,
                )
            )
    return round(sum(lapse.discount_amount for lapse in lapses), 2), lapses


def wait_cost_budget_status(
    runs: list[PaymentRun], budgets: dict[str, float], *, month: str
) -> dict:
    """The four-eyes wait's cost against its monthly budgets,
    for one calendar month (``YYYY-MM``).

    Round 11 priced the wait per run; a price nobody owns is
    still just a number. This read model gives it an owner: for
    each currency with wait cost in the month *or* a configured
    budget, the month's destroyed-discount total (summed from
    the runs' stamps, charged to the month the second
    confirmation landed — the moment the wait ended and its
    cost became final), the budget in force, and whether the
    month is over it, by how much. A currency with no budget
    configured reports ``budget: None`` and can never be over:
    unbudgeted is a fact to state, not a zero ceiling to breach
    against. Pure over the runs given."""
    cells: dict[str, dict] = {}
    for run in runs:
        if not run.second_confirmed_at:
            continue
        second = _parse_date(run.second_confirmed_at)
        if second is None or second.strftime("%Y-%m") != month:
            continue
        cell = cells.setdefault(
            run.currency,
            {"currency": run.currency, "wait_cost": 0.0, "runs_waited": 0},
        )
        cell["runs_waited"] += 1
        cell["wait_cost"] = round(
            cell["wait_cost"] + run.four_eyes_wait_cost, 2
        )
    rows: list[dict] = []
    for currency in sorted(set(cells) | set(budgets)):
        cell = cells.get(
            currency,
            {"currency": currency, "wait_cost": 0.0, "runs_waited": 0},
        )
        budget = budgets.get(currency)
        over = budget is not None and cell["wait_cost"] > budget
        rows.append(
            {
                **cell,
                "budget": budget,
                "over_budget": over,
                "excess": (
                    round(cell["wait_cost"] - budget, 2) if over else 0.0
                ),
            }
        )
    return {
        "month": month,
        "budgets_configured": bool(budgets),
        "currencies": rows,
        "breach_currencies": [r["currency"] for r in rows if r["over_budget"]],
    }


def group_answer_metrics(runs: list[PaymentRun]) -> dict:
    """The bank's per-group answers, aggregated: over the
    batch-booked exports, how many groups stand accepted,
    rejected, or unanswered — and how many bounced groups were
    re-presented alone. The feedback loop's coverage: a high
    unanswered count is bank silence, and it is the queue's
    group waits (aged) that chase it. Pure; the group reads
    come from ``handoff`` (the receipts are the record, never
    recomputed)."""
    from .handoff import batch_group_members, current_group_receipts

    groups = accepted = rejected = unanswered = 0
    reexported_groups = 0
    reexports = 0
    for run in runs:
        if not run.handoff_batch_booking or not run.handoff_generated_at:
            continue
        members_by_group = batch_group_members(run)
        if not members_by_group:
            continue
        standing = current_group_receipts(run)
        groups += len(members_by_group)
        for execution_date in members_by_group:
            receipt = standing.get(execution_date)
            if receipt is None:
                unanswered += 1
            elif receipt.status == "accepted":
                accepted += 1
            elif receipt.status == "rejected":
                rejected += 1
        reexported = {
            entry.execution_date for entry in run.handoff_group_reexports
        }
        reexported_groups += len(reexported)
        reexports += len(run.handoff_group_reexports)
    return {
        "groups_total": groups,
        "accepted": accepted,
        "rejected": rejected,
        "unanswered": unanswered,
        "reexported_groups": reexported_groups,
        "reexports": reexports,
    }


def _answer_day_stats(days: list[int]) -> dict | None:
    """count / median / mean / max over day counts — the
    follow-through gradings' shared shape (the information-wait
    grading's days-to-answer block). None when nothing was
    measured, never a dressed-up zero block."""
    if not days:
        return None
    ordered = sorted(days)
    count = len(ordered)
    median = (
        ordered[count // 2]
        if count % 2 == 1
        else round((ordered[count // 2 - 1] + ordered[count // 2]) / 2, 2)
    )
    return {
        "count": count,
        "median": median,
        "mean": round(sum(ordered) / count, 2),
        "max": ordered[-1],
    }


def _stamp_datetime(value: str) -> datetime | None:
    """A stored stamp as a datetime, for ordering events that
    share a calendar day (a rejection recorded in the morning
    and the re-export it caused in the afternoon are one
    episode's end and the next one's start, not a tie). Falls
    back to the date at midnight when only a date parses."""
    if not value:
        return None
    try:
        stamp = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        parsed = _parse_date(value)
        if parsed is None:
            return None
        stamp = datetime(parsed.year, parsed.month, parsed.day)
    if stamp.tzinfo is None:
        stamp = stamp.replace(tzinfo=timezone.utc)
    return stamp


def group_answer_follow_through(
    runs: list[PaymentRun], *, as_of: date, aging_days: int
) -> dict:
    """Grade the aged bank silences against what happened next:
    did the chase window's passing move the bank?

    The ageing read (:func:`ap_agent.queue.unanswered_group_items`)
    names the groups the bank has not answered past
    ``aging_days`` — a snapshot. This is the same loop graded
    over time, the information-wait escalation grading's shape
    applied to the bank loop. Each (group, file) pair is an
    *episode*: the whole-file export opens one, each solo
    re-export opens the next. An episode that stayed unanswered
    past the aging threshold is graded by how it ended:

    - ``answered_after_ageing`` — a receipt was recorded for
      the episode, after the crossing (accepted and rejected
      counted apart). The bank moved, late.
    - ``reexported_after_ageing`` — the episode's late answer
      was a rejection and the group was re-presented alone: the
      silence ended with the file itself being redone, which is
      a different lesson than a late booking.
    - ``still_silent`` — the episode is the group's current one
      and no answer has come: the silence the queue ages today.

    Episodes that closed inside the threshold never aged and
    are not graded — a bank being a bank is not a finding.
    Days-to-answer runs from the episode's file going out to
    its answer landing, over the answered and re-exported
    episodes. Pure over the runs given; the receipts and
    re-exports on the run are the record, never recomputed."""
    from .handoff import batch_group_members

    graded = answered = answered_accepted = answered_rejected = 0
    reexported = still_silent = 0
    answer_days: list[int] = []
    silent_groups: list[dict] = []
    for run in runs:
        if not run.handoff_batch_booking or not run.handoff_generated_at:
            continue
        members_by_group = batch_group_members(run)
        if not members_by_group:
            continue
        export_stamp = _stamp_datetime(run.handoff_generated_at)
        if export_stamp is None:
            continue
        reexports_by_group: dict[str, list] = {}
        for entry in run.handoff_group_reexports:
            reexports_by_group.setdefault(entry.execution_date, []).append(
                entry
            )
        receipts_by_group: dict[str, list] = {}
        for receipt in run.handoff_group_receipts:
            receipts_by_group.setdefault(receipt.execution_date, []).append(
                receipt
            )
        for execution_date in members_by_group:
            sends: list[datetime] = [export_stamp]
            for entry in sorted(
                reexports_by_group.get(execution_date, []),
                key=lambda e: e.reexported_at,
            ):
                sent = _stamp_datetime(entry.reexported_at)
                if sent is not None and sent > sends[-1]:
                    sends.append(sent)
            receipts = sorted(
                receipts_by_group.get(execution_date, []),
                key=lambda r: r.recorded_at,
            )
            for index, sent in enumerate(sends):
                next_send = (
                    sends[index + 1] if index + 1 < len(sends) else None
                )
                episode_receipt = None
                episode_recorded: datetime | None = None
                for receipt in receipts:
                    recorded = _stamp_datetime(receipt.recorded_at)
                    if recorded is None or recorded < sent:
                        continue  # an earlier file's answer
                    if next_send is not None and recorded >= next_send:
                        continue  # the next file's answer, not this one's
                    episode_receipt = receipt
                    episode_recorded = recorded
                    break
                if episode_recorded is not None:
                    end = episode_recorded.date()
                elif next_send is not None:
                    end = next_send.date()
                else:
                    end = as_of
                if (end - sent.date()).days <= aging_days:
                    continue  # closed (or stands) inside the threshold
                graded += 1
                if episode_receipt is None:
                    if next_send is None:
                        still_silent += 1
                        silent_groups.append(
                            {
                                "run_id": run.run_id,
                                "execution_date": execution_date,
                            }
                        )
                    # No answer and yet a later file exists cannot
                    # happen (a re-export needs the rejection), so
                    # there is nothing else to grade here.
                    continue
                answer_days.append((end - sent.date()).days)
                if (
                    next_send is not None
                    and episode_receipt.status == "rejected"
                ):
                    reexported += 1
                    continue
                answered += 1
                if episode_receipt.status == "accepted":
                    answered_accepted += 1
                elif episode_receipt.status == "rejected":
                    answered_rejected += 1
    return {
        "aging_threshold_days": aging_days,
        "episodes_graded": graded,
        "answered_after_ageing": answered,
        "answered_accepted": answered_accepted,
        "answered_rejected": answered_rejected,
        "reexported_after_ageing": reexported,
        "still_silent": still_silent,
        "silence_ended_rate": (
            round((answered + reexported) / graded, 4) if graded else None
        ),
        "days_to_answer": _answer_day_stats(answer_days),
        "still_silent_groups": sorted(
            silent_groups,
            key=lambda g: (g["run_id"], g["execution_date"]),
        ),
    }


def group_reexport_analytics(runs: list[PaymentRun]) -> dict:
    """The re-exports, read back as a population: did the
    corrected files fare better than the ones that bounced?

    Each re-exported group is graded by its standing receipt —
    the bank's eventual word on the solo file: accepted (the
    correction worked), rejected again (the group bounced
    twice, a pattern with a supplier and a reason attached),
    or still awaiting its answer. Around that spine, the
    rejection population itself is read: the bank's reasons as
    patterns (the same words, counted — a bank whose groups
    keep bouncing for one reason is a profile conversation,
    and this is the read that starts it), the suppliers whose
    groups were rejected more than once across the whole
    feedback loop, and the drift the re-exports re-priced —
    the discount value the second files had to give back.
    Pure over the runs; the receipts and re-export records are
    the evidence, never recomputed."""
    from .handoff import batch_group_members, current_group_receipts

    reexports_total = 0
    accepted = rejected_again = awaiting_answer = 0
    groups_bounced_twice = 0
    reasons: dict[str, int] = {}
    supplier_rejections: dict[str, int] = {}
    drift_by_currency: dict[str, float] = {}
    drift_members = 0
    for run in runs:
        if not run.handoff_batch_booking or not run.handoff_generated_at:
            continue
        members_by_group = batch_group_members(run)
        if not members_by_group:
            continue
        standing = current_group_receipts(run)
        suppliers_by_invoice = {
            line.invoice_id: line.supplier_name for line in run.lines
        }
        rejected_by_group: dict[str, int] = {}
        for receipt in run.handoff_group_receipts:
            if receipt.status != "rejected":
                continue
            rejected_by_group[receipt.execution_date] = (
                rejected_by_group.get(receipt.execution_date, 0) + 1
            )
            if receipt.detail:
                reasons[receipt.detail] = reasons.get(receipt.detail, 0) + 1
            members = receipt.member_invoice_ids or members_by_group.get(
                receipt.execution_date, []
            )
            for supplier in {
                suppliers_by_invoice.get(m, "") for m in members
            } - {""}:
                supplier_rejections[supplier] = (
                    supplier_rejections.get(supplier, 0) + 1
                )
        groups_bounced_twice += sum(
            1 for count in rejected_by_group.values() if count >= 2
        )
        reexports_total += len(run.handoff_group_reexports)
        for entry in run.handoff_group_reexports:
            if entry.drift_total:
                drift_by_currency[run.currency] = round(
                    drift_by_currency.get(run.currency, 0.0)
                    + entry.drift_total,
                    2,
                )
            drift_members += len(entry.drift)
        for execution_date in {
            entry.execution_date for entry in run.handoff_group_reexports
        }:
            receipt = standing.get(execution_date)
            if receipt is None:
                awaiting_answer += 1
            elif receipt.status == "accepted":
                accepted += 1
            elif receipt.status == "rejected":
                rejected_again += 1
    decided = accepted + rejected_again
    return {
        "reexports_total": reexports_total,
        "groups_reexported": accepted + rejected_again + awaiting_answer,
        "accepted_after_reexport": accepted,
        "rejected_again": rejected_again,
        "awaiting_answer": awaiting_answer,
        "acceptance_rate": (
            round(accepted / decided, 4) if decided else None
        ),
        "groups_bounced_twice": groups_bounced_twice,
        "rejection_reasons": [
            {"reason": reason, "count": count}
            for reason, count in sorted(
                reasons.items(), key=lambda kv: (-kv[1], kv[0])
            )
        ],
        "repeat_rejection_suppliers": [
            {"supplier_name": supplier, "rejections": count}
            for supplier, count in sorted(
                supplier_rejections.items(), key=lambda kv: (-kv[1], kv[0])
            )
            if count >= 2
        ],
        "drift_repriced_by_currency": drift_by_currency,
        "drift_members_repriced": drift_members,
    }


def handoff_metrics(
    runs: list[PaymentRun],
    *,
    as_of: date | None = None,
    group_aging_days: int = 5,
) -> dict:
    """The hand-off story, aggregated over the stored runs — the
    ``payment_runs`` section of ``/metrics``.

    Round 6 made drift visible per run; this makes the *discipline*
    measurable: how much discount value drift destroyed, in which
    currencies, and how long runs sat between proposal,
    confirmation and export — the confirm→export lag is the
    process leak that causes drift, so the two are reported side
    by side. The confirmation-time forecast is scored in the same
    section (:func:`forecast_accuracy`): a projection that is
    never graded is just a decoration. The four-eyes wait is
    priced in the same section too (``four_eyes_wait``): the
    discounts the second-confirmation wait itself destroyed,
    aggregated from the stamps on the runs — the control's cost,
    next to the leaks it exists to prevent. Pure over the runs
    given; runs exported before the receipt fields existed simply
    contribute what they carry.
    """
    by_status: dict[str, int] = {}
    by_format: dict[str, int] = {}
    exported = 0
    runs_with_drift = 0
    drift_members = 0
    drift_value: dict[str, float] = {}
    confirm_lags: list[float] = []
    proposal_lags: list[float] = []
    # The priced four-eyes wait, aggregated from the stamps the
    # second confirmations left on the runs: how many runs have
    # been through the wait, what the waits cost per currency,
    # and how many member discounts died inside them. A run is
    # counted from its second confirmation onward, exported or
    # not; runs that never waited carry no stamp.
    second_confirmed_runs = 0
    wait_cost_value: dict[str, float] = {}
    wait_lapsed_members = 0
    for run in runs:
        by_status[run.status] = by_status.get(run.status, 0) + 1
        if run.second_confirmed_at:
            second_confirmed_runs += 1
            wait_lapsed_members += len(run.four_eyes_wait_lapses)
            if run.four_eyes_wait_cost:
                wait_cost_value[run.currency] = round(
                    wait_cost_value.get(run.currency, 0.0)
                    + run.four_eyes_wait_cost,
                    2,
                )
        if run.handoff_drift:
            runs_with_drift += 1
            drift_members += len(run.handoff_drift)
            if run.handoff_drift_total:
                drift_value[run.currency] = round(
                    drift_value.get(run.currency, 0.0) + run.handoff_drift_total, 2
                )
        if not run.handoff_generated_at:
            continue
        exported += 1
        if run.handoff_format:
            by_format[run.handoff_format] = by_format.get(run.handoff_format, 0) + 1
        lag = _lag_days(run.confirmed_at, run.handoff_generated_at)
        if lag is not None:
            confirm_lags.append(lag)
        lag = _lag_days(run.proposed_at, run.handoff_generated_at)
        if lag is not None:
            proposal_lags.append(lag)
    return {
        "runs_total": len(runs),
        "by_status": by_status,
        "exported_runs": exported,
        "exports_by_format": by_format,
        "drift": {
            "runs_with_drift": runs_with_drift,
            "members_drifted": drift_members,
            "value_lost_by_currency": drift_value,
        },
        "export_lag": {
            "confirm_to_export": _lag_stats(confirm_lags),
            "proposal_to_export": _lag_stats(proposal_lags),
            "proposal_to_export_by_currency": lag_statistics(runs)[
                "by_currency"
            ],
            "proposal_to_export_by_profile": lag_statistics(runs)[
                "by_profile"
            ],
        },
        "forecast_accuracy": forecast_accuracy(runs),
        "lag_statistic_grades": grade_lag_statistics(runs),
        "lag_grade_drift_watch": lag_grade_drift_watch(runs),
        "creditor_completion": creditor_completion_metrics(runs),
        "group_answers": group_answer_metrics(runs),
        # The bank-silence loop, graded: the aged group waits
        # read against what happened next — a late answer, a
        # re-presentation, or silence still standing — so the
        # ageing's chase has a measured effect beside its
        # coverage counts.
        "group_answer_follow_through": group_answer_follow_through(
            runs,
            as_of=as_of or datetime.now(timezone.utc).date(),
            aging_days=group_aging_days,
        ),
        # The re-exports as a population: the corrected groups'
        # eventual standing, the bank's rejection reasons as
        # patterns, the repeat-rejection suppliers, and the
        # drift the second files re-priced.
        "group_reexports": group_reexport_analytics(runs),
        "four_eyes_wait": {
            "runs_second_confirmed": second_confirmed_runs,
            "cost_by_currency": wait_cost_value,
            "discounts_lapsed": wait_lapsed_members,
        },
    }


def discount_drift(
    run: PaymentRun,
    records: dict[str, ApprovalRecord],
    *,
    as_of: date,
) -> tuple[PaymentRun, list[PaymentRunDrift]]:
    """Re-verify a confirmed run's discounts at export time.

    A run is confirmed against the discounts capturable on proposal
    day; the hand-off can be generated days later, after a member's
    discount window has closed — and a file that still nets off a
    lapsed discount short-pays the supplier (the full total is what
    is owed once the window closes) while telling the bank a figure
    nobody can take. So export re-prices every member that carries
    a discount, from its stored invoice, as of the export date:

    - Still capturable at the confirmed figure → the line stands.
    - No longer capturable (or capturable at a different figure) →
      the line is adjusted to the export-time figure — full gross
      when the window has closed — and the drift is itemised in
      the returned entries: which invoice, which deadline, how
      much discount was lost, and why.

    The export proceeds with the adjusted figures; the drift is
    never silent and never blocks the clean members. Pure: the
    input run is not mutated — the service layer applies the
    adjusted copy to the stored run alongside the export receipt.
    """
    adjusted = run.model_copy(deep=True)
    drift: list[PaymentRunDrift] = []
    for line in adjusted.lines:
        if line.discount_amount <= 0:
            continue
        record = records.get(line.invoice_id)
        capturable = 0.0
        if record is None:
            reason = (
                "the invoice record is no longer in the store, so "
                "its discount cannot be re-verified"
            )
        else:
            invoice = InvoiceInput.model_validate(record.invoice or {})
            evaluation = evaluate_discount(invoice, as_of=as_of)
            if evaluation is not None and evaluation.worthwhile:
                capturable = round(float(evaluation.gross_discount), 2)
            if capturable > 0:
                reason = (
                    "the discount re-priced at export differs from "
                    "the figure the run was confirmed with"
                )
            else:
                reason = (
                    f"the discount window closed on "
                    f"{line.discount_deadline} — the discount is no "
                    "longer capturable, so the member is exported at "
                    "its full gross amount"
                )
        if abs(capturable - line.discount_amount) <= 0.005:
            continue  # still capturable at the confirmed figure
        drift.append(
            PaymentRunDrift(
                invoice_id=line.invoice_id,
                supplier_name=line.supplier_name,
                discount_deadline=line.discount_deadline,
                discount_at_proposal=line.discount_amount,
                discount_at_export=capturable,
                discount_lost=round(line.discount_amount - capturable, 2),
                reason=reason,
            )
        )
        line.discount_amount = capturable
        line.net_amount = round(
            max(0.0, line.gross_amount - capturable - line.credit_amount), 2
        )
    if drift:
        adjusted.total_discount = round(
            sum(line.discount_amount for line in adjusted.lines), 2
        )
        adjusted.total_net = round(
            sum(line.net_amount for line in adjusted.lines), 2
        )
    return adjusted, drift
