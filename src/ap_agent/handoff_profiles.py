"""Hand-off routing profiles — the payee side's say in the export.

A payment run leaves as one file in one format, but the right
format is not a property of the run: it belongs to the payee
side. One customer's ERP ingests the CSV; another's bank wants
the pain.001 document; a third needs lead days between the
file's arrival and the requested execution date; and some
banks reject a file whose creditor account is still the
``SUPPLIER-MASTER-AT-IMPORT`` marker instead of real details.
Until now the export call chose the format alone, every time,
for everybody. The profile is where an operator records the
routing knowledge once — scoped to a single supplier or to a
whole currency — and the export resolves it deterministically:

1. A format named explicitly on the export call wins. The
   human at the keyboard outranks stored configuration.
2. Else the run's **currency profile** — the ERP target the
   whole currency pays through.
3. Else a **supplier profile** — but a run is one file, so a
   supplier profile governs only when every profiled member
   of the run points at the *same* profile values: one voice,
   not a vote. Split profiles leave the run ungoverned.
4. Else the historical default: the CSV.

The governing profile contributes more than the format. Its
``lead_days`` sets a pain.001 export's requested execution
date — the later of the run's window end and (export date +
lead days): a bank that needs three days to process gets
three days, and no profile ever pulls a payment ahead of the
window's last due date. And its ``markers_rejected`` flag is
read at *proposal* time: a member whose supplier's bank
refuses markers, while the governed master still holds no
approved bank details for that supplier, is flagged on the
run when it is proposed — ``handoff_profile_flags`` — so
the import failure is prevented at the desk, not discovered
at the bank.

Profiles are governed configuration: setting one names its
actor and appends to the profile's audit trail (the service
owns that write; this module is the pure read side — ids,
resolution, flags). Nothing here writes, exports, or pays.
"""

from __future__ import annotations

import re
from datetime import date, timedelta
from typing import Iterable

from .handoff import HANDOFF_FORMAT_CSV
from .matching import _parse_date
from .schemas import HandoffProfile, PaymentRun

_SLUG = re.compile(r"[^A-Z0-9]+")


def profile_id_for(scope: str, *, supplier_name: str = "", currency: str = "") -> str:
    """The deterministic profile id for a scope key: one
    profile per supplier, one per currency, so setting a
    profile twice updates the same record (and its audit
    trail grows) instead of forking the configuration."""
    if scope == "supplier":
        slug = _SLUG.sub("-", (supplier_name or "").upper()).strip("-")
        return f"HP-SUP-{slug}"
    return f"HP-CUR-{(currency or '').upper()}"


def profile_detail(profile: HandoffProfile) -> str:
    """The profile's effective values in one line — the audit
    event's wording, and the console's summary of a profile."""
    parts = [f"format {profile.format}", f"lead {profile.lead_days}d"]
    if profile.markers_rejected:
        parts.append("markers rejected by the payee's bank")
    if profile.batch_booking:
        parts.append("batch booking (one PmtInf per execution date)")
    if profile.note:
        parts.append(profile.note)
    return "; ".join(parts)


def governing_profile(
    run: PaymentRun, profiles: Iterable[HandoffProfile]
) -> HandoffProfile | None:
    """The one profile that governs a run, or None.

    The currency profile for the run's currency wins over
    supplier profiles (the ERP target is the stronger fact —
    it is where the file is going). Below that, supplier
    profiles govern only in unison: the distinct profiles
    among the run's members must be exactly one, compared on
    their routing values (format, lead days, marker stance) —
    two suppliers sharing identical values are one voice even
    under two profile records.
    """
    profiles = list(profiles)
    for profile in profiles:
        if profile.scope == "currency" and profile.currency == run.currency:
            return profile
    by_supplier = {
        profile.supplier_name: profile
        for profile in profiles
        if profile.scope == "supplier"
    }
    member_profiles = [
        by_supplier[line.supplier_name]
        for line in run.lines
        if line.supplier_name in by_supplier
    ]
    if not member_profiles:
        return None
    voices = {
        (p.format, p.lead_days, p.markers_rejected, p.batch_booking)
        for p in member_profiles
    }
    if len(voices) != 1:
        return None  # split voices: no profile governs this run
    return member_profiles[0]


def resolve_export_format(
    run: PaymentRun,
    profiles: Iterable[HandoffProfile],
    requested_format: str = "",
) -> tuple[str, HandoffProfile | None, str]:
    """Resolve an export's format: (format, governing profile,
    basis in words). An explicitly requested format wins and
    the governing profile (if any) still rides along — its
    lead days are the payee's processing fact, whoever picked
    the format. With no request, the governing profile's
    format applies; with neither, the CSV default."""
    profile = governing_profile(run, profiles)
    if requested_format:
        return requested_format, profile, "requested on the export call"
    if profile is not None:
        return profile.format, profile, f"profile {profile.profile_id}"
    return HANDOFF_FORMAT_CSV, None, "default (no profile governs the run)"


def requested_execution_date(
    run: PaymentRun, profile: HandoffProfile | None, *, export_date: date
) -> str:
    """The ReqExctnDt a pain.001 export should carry: the
    run's window end, pushed out to (export date + the
    governing profile's lead days) when that is later. Lead
    days buy the bank processing time; they never pull a
    payment ahead of the window's last due date, and with no
    governing profile — or no lead — the window end stands,
    exactly as before profiles existed."""
    window_end = run.window_end or export_date.isoformat()
    if profile is None or profile.lead_days <= 0:
        return window_end
    lead_date = export_date + timedelta(days=profile.lead_days)
    window_date = _parse_date(window_end)
    if window_date is not None and lead_date <= window_date:
        return window_end
    return lead_date.isoformat()


def marker_flags(
    run: PaymentRun,
    profiles: Iterable[HandoffProfile],
    *,
    master_covered: Iterable[str] = (),
) -> list[str]:
    """The proposal-time flags for one run: one line per
    member whose supplier profile says the payee's bank
    rejects at-import markers while the supplier master holds
    no approved bank details for them (``master_covered`` —
    the service checks the master with the completion's own
    has-details rule). Such a member would export carrying
    the marker its bank refuses: flag it now, at proposal,
    while onboarding the supplier is still cheap. Pure over
    the inputs; the wording names the supplier, the profile,
    and the remedy."""
    by_supplier = {
        profile.supplier_name: profile
        for profile in profiles
        if profile.scope == "supplier"
    }
    covered = set(master_covered)
    flags: list[str] = []
    seen: set[str] = set()
    for line in run.lines:
        profile = by_supplier.get(line.supplier_name)
        if profile is None or not profile.markers_rejected:
            continue
        if line.supplier_name in covered or line.supplier_name in seen:
            continue
        seen.add(line.supplier_name)
        flags.append(
            f"{line.supplier_name}: profile {profile.profile_id} records "
            "that this supplier's bank rejects at-import creditor markers, "
            "and the supplier master holds no approved bank details for "
            "them — onboard the supplier through the governed "
            "master-change path before export, or the bank will refuse "
            "the file at import"
        )
    return flags
