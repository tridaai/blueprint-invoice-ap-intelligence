"""Node resilience policy: timeouts and retries for provider calls.

One central policy table for the pipeline's model work, applied by
wrapping the backend (the same wrapper pattern as the token budget's
``_BudgetGuard`` — the graph wraps budget first, resilience outside
it, so every retry attempt re-checks the budget):

- **Timeout.** Every provider call runs with a per-node timeout
  (``NODE_TIMEOUT_SECONDS``, default 180 — generous, because the SDK
  already has its own ``LLM_TIMEOUT_SECONDS``; this is the ceiling
  that keeps a wedged endpoint from wedging the run). A timed-out
  call raises :class:`~ap_agent.errors.ProviderError` in the
  product's clean style. Deterministic node work is pure in-process
  code and needs no timeout — the provider call is the only part of
  a node that can hang, so it is the part the policy governs.
- **Retry.** Only the idempotent language steps retry, and only on
  ``ProviderError``: extraction, the classification cross-check,
  diagnosis, options, verification, review. Max 2 attempts (the
  initial try + one retry) with small backoff. Retries are recorded
  on the wrapper's ``retry_log`` and land in the trace
  ("attempt 2 after provider error"). Drafting never retries — a
  redraft is a guardrail decision, not a transport retry — and
  nothing at or past the gate retries.

A timed-out call's worker thread cannot be killed (Python threads);
it is abandoned and finishes on its own, bounded by the SDK timeout.
"""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FuturesTimeoutError
from dataclasses import dataclass

from .config import env_int, load_dotenv
from .errors import ProviderError

# Provider capability method -> the pipeline node it serves (mirrors
# graph._PROVIDER_STEP_NODES; the two tables answer different
# questions — budget degradation vs resilience — about the same map).
_METHOD_NODES = {
    "extract_document_fields": "extract",
    "classify_with_llm": "classify",
    "diagnose": "diagnose",
    "diagnose_with_tools": "diagnose",
    "propose_options": "options",
    "draft_documents": "draft",
    "verify_draft": "verify",
    "review_draft": "review",
    "compose_information_request": "gate",
}

# The idempotent language steps: re-asking costs tokens, changes
# nothing else. Drafting and the gate are deliberately absent.
_RETRYABLE_NODES = {"extract", "classify", "diagnose", "options", "verify", "review"}

DEFAULT_NODE_TIMEOUT_SECONDS = 180
MAX_ATTEMPTS = 2  # initial try + one retry
BACKOFF_SECONDS = 0.25


@dataclass(frozen=True)
class NodePolicy:
    max_attempts: int
    timeout_seconds: float


def node_timeout_seconds() -> float:
    """``NODE_TIMEOUT_SECONDS`` (env, read at call time)."""
    load_dotenv()
    value = env_int("NODE_TIMEOUT_SECONDS", DEFAULT_NODE_TIMEOUT_SECONDS)
    return float(value) if value > 0 else float(DEFAULT_NODE_TIMEOUT_SECONDS)


def policy_for(method: str, timeout_seconds: float) -> NodePolicy:
    retryable = _METHOD_NODES[method] in _RETRYABLE_NODES
    return NodePolicy(
        max_attempts=MAX_ATTEMPTS if retryable else 1,
        timeout_seconds=timeout_seconds,
    )


class ResilientBackend:
    """Backend wrapper enforcing the resilience policy per provider call.

    Non-provider attributes (``name``, ``usage_totals``, …) pass
    through untouched, as do methods outside the policy table.
    ``retry_log`` accumulates one entry per retry —
    ``{"node", "method", "attempt", "error"}`` — for the trace.
    """

    def __init__(self, backend, timeout_seconds: float | None = None) -> None:
        object.__setattr__(self, "_backend", backend)
        object.__setattr__(
            self,
            "_timeout_seconds",
            timeout_seconds if timeout_seconds is not None else node_timeout_seconds(),
        )
        object.__setattr__(self, "retry_log", [])

    def __getattr__(self, name: str):
        backend = object.__getattribute__(self, "_backend")
        attr = getattr(backend, name)  # AttributeError propagates — getattr defaults work
        if name not in _METHOD_NODES or not callable(attr):
            return attr
        policy = policy_for(name, object.__getattribute__(self, "_timeout_seconds"))

        def guarded(*args, **kwargs):
            return self._call_with_policy(name, policy, attr, args, kwargs)

        return guarded

    def _call_with_policy(self, method, policy: NodePolicy, fn, args, kwargs):
        attempt = 1
        while True:
            try:
                return self._call_timed(method, policy, fn, args, kwargs)
            except ProviderError as exc:
                if attempt >= policy.max_attempts:
                    raise
                attempt += 1
                self.retry_log.append(
                    {
                        "node": _METHOD_NODES[method],
                        "method": method,
                        "attempt": attempt,
                        "error": str(exc),
                    }
                )
                time.sleep(BACKOFF_SECONDS * (attempt - 1))

    def _call_timed(self, method, policy: NodePolicy, fn, args, kwargs):
        executor = ThreadPoolExecutor(max_workers=1)
        try:
            future = executor.submit(fn, *args, **kwargs)
            return future.result(timeout=policy.timeout_seconds)
        except FuturesTimeoutError:
            backend_name = getattr(
                object.__getattribute__(self, "_backend"), "name", "provider"
            )
            raise ProviderError(
                f"{backend_name} provider call failed (timeout): the {method} "
                f"call did not answer within NODE_TIMEOUT_SECONDS="
                f"{policy.timeout_seconds:g}s. Likely fix: raise "
                "NODE_TIMEOUT_SECONDS, or check the endpoint's health."
            ) from None
        finally:
            executor.shutdown(wait=False)


# ---------------------------------------------------------------------------
# Transient-error retry for the document ports (OCR, vision)
# ---------------------------------------------------------------------------
#
# The language steps above retry through ResilientBackend; the OCR and
# vision ports call their endpoints directly and had no retry at all,
# so one "503 worker limit reached" from a hosted catalog sent the
# vision verdict to the metadata fallback. Only transient failures
# retry — rate limits, 5xx, timeouts, dropped connections — never an
# auth or request error, which fails on the first attempt as before.

_TRANSIENT_STATUS = {408, 409, 429, 500, 502, 503, 504}
_TRANSIENT_NAMES = {"APITimeoutError", "APIConnectionError", "Timeout", "ConnectError", "ReadTimeout"}
DEFAULT_PORT_MAX_RETRIES = 2
DEFAULT_PORT_RETRY_BACKOFF_SECONDS = 2.0


def is_transient(exc: BaseException) -> bool:
    """A provider failure worth retrying: rate limit, 5xx, timeout,
    dropped connection."""
    status = getattr(exc, "status_code", None)
    if isinstance(status, int):
        return status in _TRANSIENT_STATUS
    return type(exc).__name__ in _TRANSIENT_NAMES


def call_with_transient_retry(fn, *, sleep=time.sleep):
    """``fn()`` with up to ``NIM_PORT_MAX_RETRIES`` retries (default 2)
    on transient failures, backing off exponentially from
    ``NIM_PORT_RETRY_BACKOFF_SECONDS`` (default 2.0). Anything else —
    and the last transient failure — propagates unchanged."""
    load_dotenv()
    retries = max(0, env_int("NIM_PORT_MAX_RETRIES", DEFAULT_PORT_MAX_RETRIES))
    from .config import env_float

    backoff = max(0.0, env_float("NIM_PORT_RETRY_BACKOFF_SECONDS", DEFAULT_PORT_RETRY_BACKOFF_SECONDS))
    attempt = 0
    while True:
        try:
            return fn()
        except Exception as exc:
            if attempt >= retries or not is_transient(exc):
                raise
            sleep(backoff * (2 ** attempt))
            attempt += 1

