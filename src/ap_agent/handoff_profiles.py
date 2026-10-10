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
resolution, flags). ``profile_effectiveness`` is the grading
read over the exports the profiles governed: routed exports,
marker incidents against the ungoverned population, flag
follow-through, and lead promises kept. Nothing here writes,
exports, or pays.
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


def profile_effectiveness(
    profiles: Iterable[HandoffProfile], runs: Iterable[PaymentRun]
) -> list[dict]:
    """Grade the profiles against the exports they governed.

    Round 12 made profiles govern exports and stamped the
    governing profile on every exported run; this is the read
    back. Per profile, over the stored runs:

    - **exports routed** — the exports stamped with this
      profile, and their format mix. A profile that governs
      nothing is configuration theatre, and the grade says so.
    - **marker incidents** — of the members those exports
      carried, how many (and how much value) still exported
      with the at-import creditor marker, beside the same rate
      over the *ungoverned* exports: a profile whose marker
      rate matches the ungoverned population is not routing
      anything safer.
    - **flag follow-through** — the suppliers this profile's
      marker-rejection stance flagged at proposal time: did
      they stop exporting marked? A supplier flagged and still
      marked on a later export is the flag failing in public.
    - **lead promises kept** — for pain.001 exports under a
      profile with lead days, the written execution date(s)
      against the promise (export date + lead days, read from
      the export's own stamp). A written date inside the lead
      is a regression in the resolution path, counted here as
      a breach — the alarm the export itself never raises.

    Pure over the runs given; profiles keep the input order.
    Rates are member shares (marked / members), ``None`` when
    the population is empty — never a dressed-up zero.
    """
    runs = list(runs)
    exported = [run for run in runs if run.handoff_generated_at]

    def marker_stats(population: list[PaymentRun]) -> dict:
        members = 0
        marked = 0
        value_total = 0.0
        value_marked = 0.0
        for run in population:
            nets = {line.invoice_id: line.net_amount for line in run.lines}
            for entry in run.handoff_creditor_entries:
                members += 1
                value = round(nets.get(entry.invoice_id, 0.0), 2)
                value_total = round(value_total + value, 2)
                if entry.source != "supplier_master":
                    marked += 1
                    value_marked = round(value_marked + value, 2)
        return {
            "members": members,
            "members_marked": marked,
            "value_total": value_total,
            "value_marked": value_marked,
            "marker_rate": round(marked / members, 4) if members else None,
        }

    ungoverned = marker_stats(
        [run for run in exported if not run.handoff_profile_id]
    )

    grades: list[dict] = []
    for profile in profiles:
        governed = [
            run for run in exported if run.handoff_profile_id == profile.profile_id
        ]
        stats = marker_stats(governed)
        format_mix: dict[str, int] = {}
        for run in governed:
            format_mix[run.handoff_format or "unknown"] = (
                format_mix.get(run.handoff_format or "unknown", 0) + 1
            )

        # Flag follow-through: the suppliers this profile's
        # stance flagged at proposal time (the flag text names
        # the profile — marker_flags writes it), and whether a
        # later export still carried them marked.
        flagged: dict[str, dict] = {}
        marker = f": profile {profile.profile_id} records"
        for run in runs:
            for flag in run.handoff_profile_flags:
                if marker not in flag:
                    continue
                supplier = flag.split(marker, 1)[0]
                slot = flagged.setdefault(
                    supplier,
                    {
                        "supplier": supplier,
                        "first_flagged_at": run.proposed_at,
                        "later_exports": 0,
                        "still_marked_exports": 0,
                        "still_marked_value": 0.0,
                    },
                )
                if run.proposed_at and (
                    not slot["first_flagged_at"]
                    or run.proposed_at < slot["first_flagged_at"]
                ):
                    slot["first_flagged_at"] = run.proposed_at
        for slot in flagged.values():
            for run in exported:
                if slot["first_flagged_at"] and (
                    run.handoff_generated_at or ""
                ) <= slot["first_flagged_at"]:
                    continue
                nets = {line.invoice_id: line.net_amount for line in run.lines}
                for entry in run.handoff_creditor_entries:
                    if entry.supplier_name != slot["supplier"]:
                        continue
                    slot["later_exports"] += 1
                    if entry.source != "supplier_master":
                        slot["still_marked_exports"] += 1
                        slot["still_marked_value"] = round(
                            slot["still_marked_value"]
                            + nets.get(entry.invoice_id, 0.0),
                            2,
                        )
            slot["stopped_exporting_marked"] = (
                (slot["still_marked_exports"] == 0)
                if slot["later_exports"]
                else None
            )

        # Lead promises: the written execution date(s) against
        # export date + the profile's lead, per governed
        # pain.001 export. The profile's CURRENT lead days are
        # the promise read here — a profile edited since the
        # export grades against its present values, stated in
        # the module docstring's spirit: this is a read of the
        # configuration in force, not an audit of its history.
        lead_checked = lead_kept = 0
        if profile.lead_days > 0:
            for run in governed:
                if run.handoff_format != "pain001" or not run.handoff_execution_date:
                    continue
                generated = _parse_date(run.handoff_generated_at or "")
                if generated is None:
                    continue
                floor = generated + timedelta(days=profile.lead_days)
                written = [
                    _parse_date(part.strip())
                    for part in run.handoff_execution_date.split(",")
                ]
                if any(day is None for day in written):
                    continue
                lead_checked += 1
                if all(day >= floor for day in written):
                    lead_kept += 1

        notes: list[str] = []
        if not governed:
            notes.append(
                "governs no exports yet — configuration on paper only"
            )
        elif (
            stats["marker_rate"] is not None
            and ungoverned["marker_rate"] is not None
            and stats["marker_rate"] >= ungoverned["marker_rate"]
            and stats["members_marked"] > 0
        ):
            notes.append(
                "its exports carry the at-import marker at the same "
                "rate as ungoverned exports "
                f"({stats['marker_rate']:.0%} vs "
                f"{ungoverned['marker_rate']:.0%}) — the routing is "
                "not landing safer files"
            )
        for slot in flagged.values():
            if slot["still_marked_exports"]:
                notes.append(
                    f"{slot['supplier']} was flagged at proposal time "
                    "and still exported marked on "
                    f"{slot['still_marked_exports']} later export(s) "
                    f"({slot['still_marked_value']:.2f} of value) — "
                    "the flag did not stop the marker"
                )
        if lead_checked and lead_kept < lead_checked:
            notes.append(
                f"the lead promise was breached on "
                f"{lead_checked - lead_kept} of {lead_checked} checked "
                f"export(s): a written execution date gave the bank "
                f"less than the profile's {profile.lead_days}d lead"
            )

        grades.append(
            {
                "profile_id": profile.profile_id,
                "scope": profile.scope,
                "supplier_name": profile.supplier_name,
                "currency": profile.currency,
                "exports_governed": len(governed),
                "format_mix": format_mix,
                **stats,
                "ungoverned_marker_rate": ungoverned["marker_rate"],
                "flagged_suppliers": [
                    flagged[name] for name in sorted(flagged)
                ],
                "lead_days": profile.lead_days,
                "lead_checked": lead_checked,
                "lead_kept": lead_kept,
                "lead_breached": lead_checked - lead_kept,
                "notes": notes,
            }
        )
    return grades


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
