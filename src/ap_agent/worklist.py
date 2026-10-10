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

Pure over the runs given: the service supplies the store's runs
and the set of suppliers the master now covers. It writes
nothing, onboards nobody — onboarding is the governed
master-change path, one click away in the console, and it stays
a human's governed act.
"""

from __future__ import annotations

from typing import Iterable

from .matching import _parse_date
from .schemas import MarkerWorklist, MarkerWorklistItem, PaymentRun


def marker_worklist(
    runs: Iterable[PaymentRun],
    *,
    onboarded_suppliers: Iterable[str] = (),
) -> MarkerWorklist:
    """Rank suppliers by marked export value, from the runs'
    completion receipts. ``onboarded_suppliers`` names suppliers
    the master now covers with approved bank details (the service
    checks the master with the same has-details rule the export
    completion uses); they are resolved, not work."""
    onboarded = set(onboarded_suppliers)
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
    )
