"""Approved-not-posted reconciliation — the ERP loop's open items.

The posting-receipt loop (``service.record_posting_receipt``)
closes only when a receipt arrives. An approved invoice whose
receipt never comes — or whose only receipt says the ERP refused
the posting — is a silent break: AP believes the invoice is
handled, the ledger disagrees. This read model names those
invoices:

- Only human-approved invoices appear (only approvals post).
- An invoice leaves the report the moment a ``posted`` receipt is
  recorded; a ``failed`` receipt keeps it on the report, carrying
  the ERP's reason — a failure nobody retried is still open.
- Age runs from the gate decision (``decided_at``, stamped by the
  service), falling back to the submission and then the invoice
  date for records decided before the stamp existed; the item
  says which anchor its age uses.
- Only invoices older than the grace window are reported
  (``POSTING_RECEIPT_GRACE_DAYS``, default 7): a receipt that is
  merely in flight is not an open item yet.

It writes nothing, posts nothing, and chases nobody — like the
aging queue, it answers a daily question in work order, oldest
first.
"""

from __future__ import annotations

from datetime import date, datetime, timezone

from .matching import _parse_date
from .schemas import ApprovedNotPostedReport, OpenPostingItem
from .store import ApprovalRecord

DEFAULT_GRACE_DAYS = 7


def _stamp_datetime(value: str) -> datetime | None:
    """A stored stamp as a datetime (naive stamps read as
    UTC), for ordering a firing against the receipt that may
    have answered it. Falls back to the date at midnight."""
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


def _days_stats(days: list[int]) -> dict | None:
    """count / median / mean / max over day counts — the
    follow-through gradings' shared shape. None when nothing
    was measured."""
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


def _anchor(record: ApprovalRecord) -> tuple[date | None, str]:
    """The date an approval's age runs from, and its basis."""
    result = record.result
    invoice = record.invoice or {}
    for value, basis in (
        (result.decided_at, "decision"),
        (invoice.get("submitted_at"), "submission"),
        (invoice.get("invoice_date"), "invoice_date"),
    ):
        parsed = _parse_date(str(value)) if value else None
        if parsed is not None:
            return parsed, basis
    return None, ""


def approved_not_posted(
    records: list[ApprovalRecord],
    *,
    as_of: date,
    grace_days: int = DEFAULT_GRACE_DAYS,
) -> ApprovedNotPostedReport:
    """The approved invoices with no posted receipt older than the
    grace window, as of a date. Pure over the records given."""
    items: list[OpenPostingItem] = []
    totals: dict[str, float] = {}
    for record in records:
        result = record.result
        if result.status != "approved":
            continue
        receipt = result.posting_receipt
        if receipt is not None and receipt.status == "posted":
            continue  # the loop closed for this invoice
        anchor, basis = _anchor(record)
        age_days = (as_of - anchor).days if anchor is not None else None
        if age_days is None or age_days <= grace_days:
            continue  # within the grace window — in flight, not open
        invoice = record.invoice or {}
        currency = str(invoice.get("currency") or result.facts.currency or "USD")
        total = invoice.get("total_amount")
        if total is not None:
            totals[currency] = round(totals.get(currency, 0.0) + float(total), 2)
        items.append(
            OpenPostingItem(
                invoice_id=result.invoice_id,
                supplier_name=result.supplier_name
                or str(invoice.get("supplier_name", "")),
                invoice_number=str(invoice.get("invoice_number", "")),
                total_amount=float(total) if total is not None else None,
                currency=currency,
                decided_by=result.decided_by or "",
                decided_at=result.decided_at or "",
                age_days=age_days,
                age_basis=basis,
                receipt_state="failed" if receipt is not None else "none",
                receipt_detail=receipt.detail if receipt is not None else "",
            )
        )
    items.sort(key=lambda item: (-(item.age_days or 0), item.invoice_id))
    return ApprovedNotPostedReport(
        as_of=as_of.isoformat(),
        grace_days=grace_days,
        count=len(items),
        totals_by_currency=totals,
        items=items,
    )


def posting_ladder_analytics(records: list[ApprovalRecord]) -> dict:
    """Grade the posting-escalation ladder's recorded rungs
    against what happened next: which rung, if any, the posting
    followed.

    The ladder's ledger (``result.posting_escalations``)
    records that a crossing was pushed — rung 1 past the
    threshold, rung 2 past twice it. This reads each *episode*
    (one receipt state, from its start date — the ledger's own
    dedupe key) against the record as it now stands:

    - ``resolved_after_rung_1`` / ``resolved_after_rung_2`` —
      a posted receipt landed after that rung fired and before
      any higher rung did: the posting followed that shout.
      (A posted receipt after rung 2 cannot credit rung 1 —
      rung 1 already had its window and the posting outlasted
      it, which is exactly why rung 2 exists.)
    - ``closed_otherwise`` — the episode ended without the
      ladder resolving it: the posting closed on a receipt that
      did not follow the push, or a failed receipt moved the
      case into a new episode (the failure is information, but
      it is not the ladder's resolution, and the grade does
      not let the ladder claim it).
    - ``still_open`` — no posted receipt, no newer episode:
      the case stands exactly as it stood when the last rung
      fired. Neglect, counted.

    Days-to-post runs from the resolving rung's dispatch to
    the posted receipt. Read beside the information waits'
    own follow-through (``/metrics`` carries both), the two
    ladders price the same question for the two loops that
    wait on someone else: the supplier, and the ERP. Pure
    over the records."""
    episodes_graded = 0
    firings = 0
    firings_by_rung = {"1": 0, "2": 0}
    resolved = {"1": 0, "2": 0}
    closed_otherwise = 0
    still_open = 0
    post_days: list[int] = []
    still_open_ids: list[str] = []
    for record in records:
        result = record.result
        ledger = result.posting_escalations
        if not ledger:
            continue
        firings += len(ledger)
        for entry in ledger:
            firings_by_rung["2" if entry.level >= 2 else "1"] += 1
        by_episode: dict[tuple[str, str], dict[int, datetime]] = {}
        for entry in ledger:
            stamp = _stamp_datetime(entry.dispatched_at)
            if stamp is None:
                continue
            rungs = by_episode.setdefault(
                (entry.receipt_state, entry.episode_since), {}
            )
            rung = 2 if entry.level >= 2 else 1
            if rung not in rungs or stamp < rungs[rung]:
                rungs[rung] = stamp
        if not by_episode:
            continue
        ordered = sorted(
            by_episode.values(), key=lambda rungs: min(rungs.values())
        )
        receipt = result.posting_receipt
        posted = (
            _stamp_datetime(receipt.recorded_at)
            if receipt is not None and receipt.status == "posted"
            else None
        )
        for index, rungs in enumerate(ordered):
            episodes_graded += 1
            rung1 = rungs.get(1)
            rung2 = rungs.get(2)
            resolved_rung: int | None = None
            if posted is not None and rung2 is not None and posted >= rung2:
                resolved_rung = 2
            elif posted is not None and rung1 is not None and posted >= rung1:
                resolved_rung = 1
            if resolved_rung is not None:
                resolved[str(resolved_rung)] += 1
                post_days.append(
                    (posted.date() - rungs[resolved_rung].date()).days
                )
            elif posted is not None or index + 1 < len(ordered):
                # Posted without following this episode's push,
                # or a newer episode exists (a failed receipt
                # moved the case on): closed, but not by the
                # ladder.
                closed_otherwise += 1
            else:
                still_open += 1
                still_open_ids.append(result.invoice_id)
    return {
        "episodes_graded": episodes_graded,
        "firings": firings,
        "firings_by_rung": firings_by_rung,
        "resolved_after_rung_1": resolved["1"],
        "resolved_after_rung_2": resolved["2"],
        "closed_otherwise": closed_otherwise,
        "still_open": still_open,
        "resolution_rate": (
            round((resolved["1"] + resolved["2"]) / episodes_graded, 4)
            if episodes_graded
            else None
        ),
        "days_to_post": _days_stats(post_days),
        "still_open_invoice_ids": sorted(set(still_open_ids)),
    }
