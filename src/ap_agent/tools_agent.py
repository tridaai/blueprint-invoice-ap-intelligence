"""Agent tools for the diagnosis step (provider mode).

In provider mode the diagnosis is *agentic*: instead of one composed
call over pre-assembled facts, the model runs a bounded tool loop —
it may pull more policy context, the computed three-way match facts,
the supplier's history from prior invoices, or the purchase order
itself before it composes the exception summary. The loop itself
lives in the backends (``model_backends.py``, per-provider tool
formats); this module owns what the tools ARE: their schemas (one
neutral definition both provider formats render from) and their
dispatch.

Every tool here is a read over data the pipeline already holds — the
policy corpus behind the retriever interface, the store's history
entries, and the computed match. No tool sends, files, mutates, or
decides anything; the graph's human-gate rule is untouched.

The bound: ``DIAGNOSIS_MAX_TOOL_CALLS`` (default 4, hard cap 6) caps
how many tool calls one diagnosis may make. When the budget runs out
the model is told to compose with what it has.
"""

from __future__ import annotations

from .config import env_int, load_dotenv
from .matching import format_match_summary
from .schemas import InvoiceInput, MatchFacts
from .store import format_outcome_counts, supplier_summary

DEFAULT_MAX_TOOL_CALLS = 4
HARD_MAX_TOOL_CALLS = 6

# Neutral tool definitions: name, description, JSON-schema parameters.
# The Anthropic backend renders these as {name, description,
# input_schema}; the OpenAI backend as {type: "function", function:
# {name, description, parameters}}. One definition, two wire formats.
TOOL_SPECS: list[dict] = [
    {
        "name": "search_policies",
        "description": (
            "Search the accounts-payable policy corpus with your own query "
            "wording and get the top matching policies with their IDs. "
            "Use when the policies already provided look incomplete."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search query in AP wording."}
            },
            "required": ["query"],
        },
    },
    {
        "name": "match_facts",
        "description": (
            "The computed three-way match for THIS invoice: match status, "
            "line-by-line results against the purchase order and goods "
            "receipt, totals and tax variances, duplicate check. Always "
            "prefer these over assumptions."
        ),
        "parameters": {"type": "object", "properties": {}},
    },
    {
        "name": "supplier_history",
        "description": (
            "Prior invoices from a supplier from the stored history: "
            "total count and outcome counts — a supplier whose invoices "
            "routinely arrive with variances is itself a signal."
        ),
        "parameters": {
            "type": "object",
            "properties": {"supplier_name": {"type": "string"}},
            "required": ["supplier_name"],
        },
    },
    {
        "name": "po_lookup",
        "description": (
            "The purchase order attached to this case: its number, "
            "currency, and ordered lines with quantities and unit "
            "prices — the order side of the three-way match."
        ),
        "parameters": {"type": "object", "properties": {}},
    },
]


def diagnosis_max_tool_calls() -> int:
    """The tool-call budget for one diagnosis (env, read at run time)."""
    load_dotenv()
    return min(max(env_int("DIAGNOSIS_MAX_TOOL_CALLS", DEFAULT_MAX_TOOL_CALLS), 0), HARD_MAX_TOOL_CALLS)


class DiagnosisToolBox:
    """Dispatches the diagnosis tools over the data one run already holds.

    ``dispatch(name, args)`` returns ``(result_text, summary)``: the
    text fed back to the model, and a one-line summary for the trace.
    Unknown tools and bad arguments raise — the backend loop converts a
    raise into an error tool-result and the model composes without it,
    so a broken tool degrades the diagnosis, never the run.
    """

    def __init__(
        self,
        *,
        invoice: InvoiceInput,
        triage: dict,
        facts: MatchFacts,
        discrepancies: list[str] | None = None,
        retriever=None,
        priors: list[dict] | None = None,
    ) -> None:
        self._invoice = invoice
        self._triage = triage
        self._facts = facts
        self._discrepancies = discrepancies or []
        self._retriever = retriever
        self._priors = priors or []

    def dispatch(self, name: str, args: dict) -> tuple[str, str]:
        handler = {
            "search_policies": self._search_policies,
            "match_facts": self._match_facts,
            "supplier_history": self._supplier_history,
            "po_lookup": self._po_lookup,
        }.get(name)
        if handler is None:
            raise ValueError(f"unknown tool: {name}")
        return handler(args or {})

    # -- the tools ------------------------------------------------------

    def _search_policies(self, args: dict) -> tuple[str, str]:
        query = str(args.get("query") or "").strip()
        if not query:
            raise ValueError("search_policies needs a non-empty 'query'")
        if self._retriever is None:
            raise ValueError("no retriever is configured for this run")
        results = self._retriever.retrieve(query, top_k=3)
        if not results:
            return "No policies matched that query.", f"search_policies({query!r}) -> 0 policies"
        text = "\n".join(f"[{r.policy_id}] {r.title}: {r.text}" for r in results)
        ids = ", ".join(r.policy_id for r in results)
        return text, f"search_policies({query!r}) -> {len(results)} policies: {ids}"

    def _match_facts(self, args: dict) -> tuple[str, str]:
        f = self._facts
        lines = [
            f"triage: {self._triage.get('outcome')} (severity {self._triage.get('severity')}, confidence {self._triage.get('confidence')})",
            f"three-way match: {format_match_summary(f)}",
            f"match status: {f.match_status} · score {f.match_score}",
        ]
        for lm in f.line_matches:
            lines.append(
                f"line {lm.description}: {lm.status} — invoiced {lm.invoice_qty} @ "
                f"{lm.invoice_price}; ordered {lm.po_qty} @ {lm.po_price}; "
                f"received {lm.received_qty} ({lm.detail})"
            )
        lines += [f"discrepancy: {d}" for d in f.discrepancies]
        lines += [f"extraction discrepancy: {d}" for d in self._discrepancies]
        summary = f"match_facts -> {f.match_status}, score={f.match_score}"
        return "\n".join(lines), summary

    def _supplier_history(self, args: dict) -> tuple[str, str]:
        supplier = str(args.get("supplier_name") or self._invoice.supplier_name)
        data = supplier_summary(self._priors, supplier)
        total = data["supplier_count"]
        if total == 0:
            return (
                f"No prior invoices from supplier {supplier!r} in the stored history.",
                f"supplier_history({supplier!r}) -> no prior invoices",
            )
        breakdown = format_outcome_counts(data["supplier_outcome_counts"])
        return (
            f"{total} prior invoice(s) from supplier {supplier!r} in the stored "
            f"history; outcomes: {breakdown}.",
            f"supplier_history({supplier!r}) -> {total} prior invoice(s), outcomes: {breakdown}",
        )

    def _po_lookup(self, args: dict) -> tuple[str, str]:
        po = self._invoice.purchase_order
        if po is None:
            return (
                "No purchase order is attached to this case.",
                "po_lookup -> no purchase order on file",
            )
        lines = [
            f"purchase order {po.po_number} (currency {po.currency or self._invoice.currency or '—'})"
        ]
        for item in po.lines:
            lines.append(
                f"ordered: {item.description} x{item.quantity} @ {item.unit_price}"
            )
        return "\n".join(lines), f"po_lookup -> PO {po.po_number}, {len(po.lines)} line(s)"
