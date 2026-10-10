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

from datetime import date, datetime, timedelta

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


def _line_for(record: ApprovalRecord, as_of: date) -> PaymentRunLine:
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
        net_amount=round(gross - discount_amount, 2),
    )


def propose_payment_runs(
    records: list[ApprovalRecord],
    *,
    window_days: int = DEFAULT_WINDOW_DAYS,
    as_of: date | None = None,
    exclude_invoice_ids: set[str] | None = None,
    four_eyes_threshold: float | None = None,
    four_eyes_thresholds: dict[str, float] | None = None,
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
    for (currency, index), members in sorted(groups.items()):
        lines = [_line_for(record, as_of) for record in members]
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
    profile_id: str = "",
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
    the statistic with the best recorded recall. The basis states
    the statistic used and why it was chosen. The legacy flat
    ``lag_stats`` argument (the global statistic, pre-selection)
    remains for direct callers.

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
            run, lag_segments, lag_grades or {}, profile_id
        )
        if chosen is None:
            return [], (
                "no forecast: no exported runs yet, so there is no "
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


def grade_lag_statistics(runs: list[PaymentRun]) -> dict:
    """Grade each candidate lag statistic against the accuracy
    ledger — what the exports actually recorded.

    The forecast's report card (:func:`forecast_accuracy`) scores
    the forecasts that were made. This scores the *statistics*
    that could have made them: for every exported run carrying
    discounted members, each applicable statistic (global, the
    run's currency, the run's governing profile) is recomputed
    WITHOUT that run (leave-one-out — a statistic never grades
    itself on its own evidence), the flag rule is replayed with
    it, and the replayed flags are scored against the drift the
    export actually measured, in the same two-by-two cells.
    Returns ``{statistic_key: {cells…, precision, recall,
    graded_members}}`` — the selection's evidence base. A
    statistic with no graded members is absent, never dressed up
    with a rate over zero cases. Pure.
    """
    exported: list[tuple[PaymentRun, float]] = []
    for run in runs:
        if not run.handoff_generated_at:
            continue
        lag = _lag_days(run.proposed_at, run.handoff_generated_at)
        if lag is not None:
            exported.append((run, lag))
    cells: dict[str, dict] = {}
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
            cell = cells.setdefault(key, _new_accuracy_cell())
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
                cell[bucket] += 1
                if is_flagged:
                    cell["value_flagged"] = round(
                        cell["value_flagged"]
                        + next(
                            line.discount_amount
                            for line in discounted
                            if line.invoice_id == invoice_id
                        ),
                        2,
                    )
                if is_drifted:
                    value_key = (
                        "value_lost_flagged"
                        if is_flagged
                        else "value_lost_unflagged"
                    )
                    cell[value_key] = round(
                        cell[value_key]
                        + drifted[invoice_id].discount_lost,
                        2,
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
    eligible = [
        c for c in candidates if (c[3].get("count") or 0) >= _MIN_SEGMENT_RUNS
    ]
    if not eligible:
        stats, label = global_stats, "global"
        return (
            stats,
            label,
            "the only segment with a measured lag so far — segment "
            "statistics govern once one holds "
            f"{_MIN_SEGMENT_RUNS} or more exported runs",
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

        best = max(graded, key=rank)
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
        return best[3], best[2], why
    best = max(eligible, key=lambda c: c[0])
    return (
        best[3],
        best[2],
        "no candidate statistic has graded history over "
        f"{_MIN_GRADED_MEMBERS} or more members yet, so the most "
        "specific adequately-measured segment governs",
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


def handoff_metrics(runs: list[PaymentRun]) -> dict:
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
        "creditor_completion": creditor_completion_metrics(runs),
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
        line.net_amount = round(line.gross_amount - capturable, 2)
    if drift:
        adjusted.total_discount = round(
            sum(line.discount_amount for line in adjusted.lines), 2
        )
        adjusted.total_net = round(
            sum(line.net_amount for line in adjusted.lines), 2
        )
    return adjusted, drift
