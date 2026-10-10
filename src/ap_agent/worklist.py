"""The marker-risk worklist — completion coverage, turned into
supplier-onboarding work.

Creditor-completion coverage (``paymentruns.creditor_completion_
metrics``) prices the residual import risk: how much exported
value still carried the ``SUPPLIER-MASTER-AT-IMPORT`` marker
because the governed master held no approved bank details for the
supplier. A price is not a plan. This read model turns the same
export receipts into the plan — the suppliers to onboard into
the master, ranked by the marked value they keep sending to the
bank:

- One row per (supplier, currency): the total marked (uncompleted)
  export value, how many exported invoices carried the marker,
  which invoices they were, and when the supplier last exported
  marked — the fraud screen's blind spot as a prioritised queue
  with an owner-shaped order: biggest exposure first.
- A supplier who has since been onboarded — the master now holds
  approved bank details for them — drops off the worklist: their
  marked exports are history, not work. They are counted
  (``resolved_supplier_count`` and value), so the queue's
  shrinkage is visible rather than silent.
- Values are the export-time net amounts the bank was told to
  pay — the same figures the coverage metric prices — read off
  each run's completion receipt (``handoff_creditor_entries``).
  Runs exported before receipts existed contribute nothing:
  there is no receipt to rank from, and the worklist invents
  none.
- The listed rows also know where the onboarding *flow* stands,
  joined from the governed change pipeline: a supplier whose
  latest master change is still live — pending inside its
  decision window, or approved and awaiting its effective date —
  is ``in_flight``: the change's id, proposer, and age ride the
  row, so the queue does not chase work already being worked
  (the decision is owed by a second operator, and the row says
  who proposed, so it also says who may *not* decide it). A
  supplier whose latest change was rejected returns to the list
  wearing the rejection — decider, date, and the recorded note —
  so re-proposing is deliberate. An expired pending change is
  neither: it decided nothing, and the row reads open again.
- A pending change also ages against its own decision window:
  the row carries the days of window left, and inside the
  warning band (``MASTER_CHANGE_EXPIRY_WARNING_DAYS``, default
  7) the change is flagged expiring soon — the early-warning
  surface that lets the owed decision rescue the onboarding
  before the window closes silently. The digest carries the
  same warning (``digest.build_digest``) for the suppliers
  whose windows are closing.

Pure over the inputs given: the service supplies the store's
runs, the set of suppliers the master now covers, and the
store's master-change requests. It writes nothing, onboards
nobody — onboarding is the governed master-change path, one
click away in the console, and it stays a human's governed act.
"""

from __future__ import annotations

from datetime import date
from typing import Iterable

from .master_changes import change_age_days, is_change_expired
from .matching import _parse_date
from .schemas import (
    MarkerWorklist,
    MarkerWorklistItem,
    MasterChangeRequest,
    PaymentRun,
)


def _flow_state(
    change: MasterChangeRequest | None,
    *,
    as_of: date,
    expiry_days: int,
    expiry_warning_days: int = 7,
) -> dict:
    """One supplier's onboarding flow state, from its latest
    governed master change. Live means the change can still
    land: pending inside its decision window, or approved and
    not yet applied (its effective date is still ahead — once
    applied with bank details the supplier resolves off the
    worklist entirely, so a listed approved change is by
    construction an unapplied one).

    A *pending* change also ages against its window: the row
    carries how many days of window remain and flags the
    change ``in_flight_expiring_soon`` once the remainder is
    inside the warning band — the early warning that lets a
    decision rescue the change before day ``expiry_days``
    silently returns the supplier to open. An approved change
    never expires, so it carries neither figure."""
    if change is None:
        return {"onboarding_state": "open"}
    if change.status == "pending" and not is_change_expired(
        change, as_of, expiry_days
    ):
        age = change_age_days(change, as_of)
        days_left = expiry_days - age if age is not None else None
        return {
            "onboarding_state": "in_flight",
            "in_flight_change_id": change.request_id,
            "in_flight_proposed_by": change.proposed_by,
            "in_flight_age_days": age,
            "in_flight_days_to_expiry": days_left,
            "in_flight_expiring_soon": (
                days_left is not None and days_left <= expiry_warning_days
            ),
        }
    if change.status == "approved" and not change.applied_at:
        return {
            "onboarding_state": "in_flight",
            "in_flight_change_id": change.request_id,
            "in_flight_proposed_by": change.proposed_by,
            "in_flight_age_days": change_age_days(change, as_of),
        }
    if change.status == "rejected":
        return {
            "onboarding_state": "rejected",
            "rejected_change_id": change.request_id,
            "rejected_by": change.decided_by or "",
            "rejected_at": change.decided_at or "",
            "rejection_note": change.decision_note or "",
        }
    # Expired (or a decided change that never applied bank
    # details the master now shows): open work again.
    return {"onboarding_state": "open"}


def marker_worklist(
    runs: Iterable[PaymentRun],
    *,
    onboarded_suppliers: Iterable[str] = (),
    changes: Iterable[MasterChangeRequest] = (),
    as_of: date | None = None,
    expiry_days: int = 30,
    expiry_warning_days: int = 7,
) -> MarkerWorklist:
    """Rank suppliers by marked export value, from the runs'
    completion receipts. ``onboarded_suppliers`` names suppliers
    the master now covers with approved bank details (the service
    checks the master with the same has-details rule the export
    completion uses); they are resolved, not work. ``changes``
    are the governed master-change requests (store order — the
    last one per supplier is its latest), joined onto the listed
    rows as the onboarding flow state; ``as_of`` anchors change
    ages and the pending-expiry check, against the configured
    ``expiry_days`` decision window, with
    ``expiry_warning_days`` the band inside which a pending
    change is flagged as expiring soon."""
    onboarded = set(onboarded_suppliers)
    as_of = as_of or date.today()
    latest_change: dict[str, MasterChangeRequest] = {}
    for change in changes:
        if change.supplier_name:
            latest_change[change.supplier_name] = change
    cells: dict[tuple[str, str], dict] = {}
    resolved_cells: dict[tuple[str, str], dict] = {}
    resolved_suppliers: set[str] = set()
    for run in runs:
        if not run.handoff_generated_at or not run.handoff_creditor_entries:
            continue  # no export, or a receipt-less legacy export
        export_date = _parse_date(run.handoff_generated_at)
        nets = {line.invoice_id: line.net_amount for line in run.lines}
        for entry in run.handoff_creditor_entries:
            if entry.source == "supplier_master":
                continue  # completed from the master — no marker, no risk
            supplier = entry.supplier_name or ""
            if not supplier:
                continue
            if supplier in onboarded:
                resolved_suppliers.add(supplier)
                target = resolved_cells
            else:
                target = cells
            cell = target.setdefault(
                (supplier, run.currency),
                {
                    "marked_value": 0.0,
                    "invoice_count": 0,
                    "invoice_ids": set(),
                    "runs": set(),
                    "last_export_date": None,
                },
            )
            cell["marked_value"] = round(
                cell["marked_value"] + nets.get(entry.invoice_id, 0.0), 2
            )
            cell["invoice_count"] += 1
            cell["invoice_ids"].add(entry.invoice_id)
            cell["runs"].add(run.run_id)
            if export_date is not None and (
                cell["last_export_date"] is None
                or export_date > cell["last_export_date"]
            ):
                cell["last_export_date"] = export_date

    items = [
        MarkerWorklistItem(
            supplier_name=supplier,
            currency=currency,
            marked_value=cell["marked_value"],
            invoice_count=cell["invoice_count"],
            invoice_ids=sorted(cell["invoice_ids"]),
            export_count=len(cell["runs"]),
            last_export_date=(
                cell["last_export_date"].isoformat()
                if cell["last_export_date"] is not None
                else ""
            ),
            **_flow_state(
                latest_change.get(supplier),
                as_of=as_of,
                expiry_days=expiry_days,
                expiry_warning_days=expiry_warning_days,
            ),
        )
        for (supplier, currency), cell in cells.items()
    ]
    # Ranked by exposure: currency groups in order (values are
    # never compared across currencies), marked value descending
    # inside the group, supplier name as the stable tiebreak.
    items.sort(
        key=lambda item: (item.currency, -item.marked_value, item.supplier_name)
    )
    totals: dict[str, float] = {}
    for item in items:
        totals[item.currency] = round(
            totals.get(item.currency, 0.0) + item.marked_value, 2
        )
    resolved_value: dict[str, float] = {}
    for (_supplier, currency), cell in resolved_cells.items():
        resolved_value[currency] = round(
            resolved_value.get(currency, 0.0) + cell["marked_value"], 2
        )
    return MarkerWorklist(
        items=items,
        supplier_count=len({item.supplier_name for item in items}),
        totals_by_currency=totals,
        resolved_supplier_count=len(resolved_suppliers),
        resolved_value_by_currency=resolved_value,
        in_flight_count=len(
            {
                item.supplier_name
                for item in items
                if item.onboarding_state == "in_flight"
            }
        ),
        rejected_count=len(
            {
                item.supplier_name
                for item in items
                if item.onboarding_state == "rejected"
            }
        ),
        expiring_soon_count=len(
            {
                item.supplier_name
                for item in items
                if item.in_flight_expiring_soon
            }
        ),
    )
