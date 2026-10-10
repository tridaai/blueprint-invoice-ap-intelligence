"""The duplicate-twin graph: cross-shape screening, read as clusters.

Duplicate screening (``store.find_duplicate``, AP-DUP-01) judges
identity on a canonical key, so the same invoice arriving as a
portal single, a bundle segment, and a statement-era resend is
flagged whichever shape arrived second — but the flag lives on
the later record alone (``facts.duplicate_of``), and an auditor
who wants the whole picture has to open every record and walk
the links by hand. When a third copy arrives the links form a
chain, and "which records are the same invoice?" stops being a
one-record question.

This module is the graph read over the stored records: records
are nodes, and two records are linked when the screen linked
them (a ``duplicate_of`` edge) or when they share the screen's
own evidence — the same canonical key (supplier + invoice
number, normalised by ``store.duplicate_key``) at the same
total. Connected components of two or more are **twin
clusters**: each carries its members with their arrival shapes,
the explicit edges, and the key evidence the cluster rests on,
so a reviewer sees the whole cluster — original, copies, and
the shapes they arrived in — without opening each record.

Pure over the records given; statement lines are not records
and stay where they are screened (the reconciliation report).
"""

from __future__ import annotations

from .store import ApprovalRecord, duplicate_key

_TOTAL_TOLERANCE = 0.005  # the screen's own tolerance (find_duplicate)


def record_arrival_shape(record: ApprovalRecord) -> str:
    """The shape a record arrived in ("single" unless the intake
    stamped otherwise — records saved before the stamp existed
    read as singles)."""
    return getattr(record, "arrival_shape", "") or "single"


def _node(record: ApprovalRecord) -> dict:
    result = record.result
    invoice = record.invoice or {}
    return {
        "invoice_id": result.invoice_id,
        "supplier_name": result.supplier_name or invoice.get("supplier_name", ""),
        "invoice_number": invoice.get("invoice_number", ""),
        "total_amount": invoice.get("total_amount"),
        "currency": invoice.get("currency", ""),
        "arrival_shape": record_arrival_shape(record),
        "outcome": result.triage.outcome,
        "status": result.status,
        "duplicate_of": result.facts.duplicate_of or "",
    }


def _same_key_and_total(a: dict, b: dict) -> bool:
    """The screen's evidence, pairwise: same canonical key and
    totals within the screen's tolerance. An empty invoice-number
    key never links — the screen refuses it too."""
    key_a = duplicate_key(a["supplier_name"], a["invoice_number"])
    if not key_a[1]:
        return False
    if key_a != duplicate_key(b["supplier_name"], b["invoice_number"]):
        return False
    if a["total_amount"] is None or b["total_amount"] is None:
        return False
    return abs(float(a["total_amount"]) - float(b["total_amount"])) < _TOTAL_TOLERANCE


def twin_clusters(records: list[ApprovalRecord]) -> list[dict]:
    """The twin clusters over the stored records, deterministically
    ordered (by each cluster's first member id; members by invoice
    id). Singletons — records no other record twins with — are
    not clusters and are not returned."""
    nodes = [_node(record) for record in records]
    by_id = {node["invoice_id"]: node for node in nodes}
    parent = {node["invoice_id"]: node["invoice_id"] for node in nodes}

    def find(x: str) -> str:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: str, b: str) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)

    edges: list[dict] = []
    for node in nodes:
        target = node["duplicate_of"]
        if target and target in by_id:
            union(node["invoice_id"], target)
            edges.append(
                {
                    "duplicate": node["invoice_id"],
                    "original": target,
                    "basis": "screened duplicate_of (AP-DUP-01)",
                }
            )
    for index, node in enumerate(nodes):
        for other in nodes[index + 1:]:
            if _same_key_and_total(node, other):
                union(node["invoice_id"], other["invoice_id"])

    groups: dict[str, list[dict]] = {}
    for node in nodes:
        groups.setdefault(find(node["invoice_id"]), []).append(node)

    clusters: list[dict] = []
    for members in groups.values():
        if len(members) < 2:
            continue
        members = sorted(members, key=lambda n: n["invoice_id"])
        member_ids = {m["invoice_id"] for m in members}
        keyed = next(
            (m for m in members if duplicate_key(m["supplier_name"], m["invoice_number"])[1]),
            members[0],
        )
        canonical_supplier, canonical_number = duplicate_key(
            keyed["supplier_name"], keyed["invoice_number"]
        )
        clusters.append(
            {
                "cluster_id": f"TWIN-{members[0]['invoice_id']}",
                "size": len(members),
                "canonical_supplier": canonical_supplier,
                "canonical_invoice_number": canonical_number,
                "totals": sorted(
                    {
                        round(float(m["total_amount"]), 2)
                        for m in members
                        if m["total_amount"] is not None
                    }
                ),
                "arrival_shapes": sorted({m["arrival_shape"] for m in members}),
                "members": members,
                "edges": [
                    edge
                    for edge in edges
                    if edge["duplicate"] in member_ids and edge["original"] in member_ids
                ],
            }
        )
    clusters.sort(key=lambda c: c["members"][0]["invoice_id"])
    return clusters


def cluster_membership(clusters: list[dict]) -> dict[str, dict]:
    """invoice_id -> its cluster's id + size, for row-level joins
    (the audit export's twin columns)."""
    membership: dict[str, dict] = {}
    for cluster in clusters:
        for member in cluster["members"]:
            membership[member["invoice_id"]] = {
                "cluster_id": cluster["cluster_id"],
                "size": cluster["size"],
            }
    return membership
