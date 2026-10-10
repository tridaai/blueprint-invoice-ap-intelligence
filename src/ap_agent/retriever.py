"""Retrieval behind a small interface — the policy-retrieval layer.

Three implementations ship in the repo, selected with the ``RETRIEVER``
environment variable (see :func:`get_retriever`):

- ``keyword`` (default) — transparent token overlap, deterministic, offline.
- ``semantic`` — embeds the policy corpus and the query with an embeddings
  API and ranks by cosine similarity **in pgvector**: the vectors live in
  the same PostgreSQL that holds the records (``policy_embeddings``,
  migrations/0002), so there is one database to run, back up, and
  secure — no second vector service to operate. Embeddings come from
  the local Ollama server when ``MODEL_BACKEND=ollama``, otherwise from
  an OpenAI-compatible endpoint — the NIM profile points
  ``OPENAI_BASE_URL`` at NVIDIA's hosted endpoint with an NV-Embed
  model (``OPENAI_EMBEDDING_MODEL``, e.g. ``nvidia/nemotron-3-embed-1b``);
  Anthropic has no embeddings API, so ``MODEL_BACKEND=anthropic``
  still needs the OpenAI-compatible key for this retriever — the
  failure message says exactly that when the key is missing.
- ``hybrid`` — runs both, merges the candidate pools, and applies an
  explicit rerank step: reciprocal-rank score fusion in code (see
  :func:`rerank_fused`). This is score-fusion reranking, not a
  cross-encoder — the repo ships no cross-encoder and claims none.
  (A deployment that wants a cross-encoder can put the NVIDIA
  reranker NIM behind this same port; the fusion step is where its
  scores would plug in.)

The agent only depends on the ``Retriever`` protocol, so swapping the
implementation touches this file, not the graph.
"""

from __future__ import annotations

import math
import re

from .config import env_float, env_str, load_dotenv
from .errors import translate_construction_error, translate_provider_error
from .model_backends import DEFAULT_TIMEOUT_SECONDS, _missing_sdk_error
from .policies_data import POLICIES
from .schemas import RetrievedPolicy

_STOPWORDS = {
    "the", "a", "an", "and", "or", "of", "to", "in", "on", "for", "is", "was",
    "by", "with", "when", "if", "it", "its", "are", "be", "at", "as",
}


def _tokens(text: str) -> list[str]:
    return [t for t in re.findall(r"[a-z0-9]+", text.lower()) if t not in _STOPWORDS]


# The Retriever protocol is declared in ports.py (the seam registry);
# it is re-exported here so existing imports keep working.
from .ports import Retriever  # noqa: E402,F401


class KeywordRetriever:
    """Transparent keyword-overlap retriever over the policy corpus.

    Deliberately simple and deterministic so evals and tests are stable
    offline. Score = shared tokens normalised by policy length.
    """

    name = "keyword"

    def __init__(self, policies: list[dict[str, str]] | None = None) -> None:
        self._policies = policies if policies is not None else POLICIES
        self._index = [
            (p, set(_tokens(f"{p['title']} {p['text']}"))) for p in self._policies
        ]

    def retrieve(self, query: str, top_k: int = 3) -> list[RetrievedPolicy]:
        query_tokens = set(_tokens(query))
        if not query_tokens:
            return []
        scored: list[RetrievedPolicy] = []
        for policy, tokens in self._index:
            overlap = len(query_tokens & tokens)
            if overlap == 0:
                continue
            score = round(overlap / math.sqrt(len(tokens)), 4)
            scored.append(
                RetrievedPolicy(
                    policy_id=policy["id"],
                    title=policy["title"],
                    text=policy["text"],
                    score=score,
                    sources=["keyword"],
                )
            )
        scored.sort(key=lambda r: r.score, reverse=True)
        return scored[:top_k]


def _build_embeddings_client(mode_label: str):
    """Resolve the embeddings client, failing loudly when unusable.

    Embeddings follow the configured stack:

    - ``MODEL_BACKEND=ollama`` → embeddings from the local Ollama
      server (``OLLAMA_EMBEDDING_MODEL``, default ``nomic-embed-text``)
      — fully local, no cloud key.
    - Otherwise → an OpenAI-compatible endpoint (``OPENAI_BASE_URL``):
      plain OpenAI, or NVIDIA NIM in the lead profile
      (``OPENAI_EMBEDDING_MODEL=nvidia/nemotron-3-embed-1b`` against
      ``https://integrate.api.nvidia.com/v1``). Anthropic offers no
      embeddings API, so with ``MODEL_BACKEND=anthropic`` (or ``mock``)
      an ``OPENAI_API_KEY`` is still required, and the error says so.

    Returns (client, model, provider_name, base_url).
    """
    load_dotenv()
    backend = (env_str("MODEL_BACKEND") or "mock").lower()
    timeout = env_float("LLM_TIMEOUT_SECONDS", DEFAULT_TIMEOUT_SECONDS)
    if backend == "ollama":
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise _missing_sdk_error("ollama (embeddings)", "OpenAI") from exc
        base_url = env_str("OLLAMA_BASE_URL", "http://localhost:11434/v1")
        try:
            client = OpenAI(
                api_key="ollama",  # placeholder — Ollama ignores it
                base_url=base_url,
                timeout=timeout,
                max_retries=0,
            )
        except Exception as exc:  # construction failures are translated too
            raise translate_construction_error(
                exc,
                backend=f"ollama embeddings (RETRIEVER={mode_label})",
                base_url=base_url,
            ) from exc
        return client, env_str("OLLAMA_EMBEDDING_MODEL", "nomic-embed-text"), "ollama", base_url
    if not env_str("OPENAI_API_KEY"):
        hint = (
            f" MODEL_BACKEND={backend} has no embeddings API to use instead —"
            " drafting and embeddings are separate providers here."
            if backend != "openai"
            else ""
        )
        raise RuntimeError(
            f"RETRIEVER={mode_label} needs embeddings, which run through an "
            "OpenAI-compatible endpoint, but OPENAI_API_KEY is not set." + hint
            + " Add OPENAI_API_KEY=<your key> to the .env file in the repo root "
            "(copy .env.example to .env) or export it, set MODEL_BACKEND=ollama "
            "to embed locally with Ollama, or set RETRIEVER=keyword."
        )
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise _missing_sdk_error("openai (embeddings)", "OpenAI") from exc
    client_kwargs: dict = {
        "api_key": env_str("OPENAI_API_KEY"),
        "timeout": timeout,
        "max_retries": 0,
    }
    base_url = env_str("OPENAI_BASE_URL")
    if base_url:
        client_kwargs["base_url"] = base_url
    model = env_str("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
    try:
        client = OpenAI(**client_kwargs)
    except Exception as exc:  # construction failures are translated too
        raise translate_construction_error(
            exc,
            backend=f"openai-compatible embeddings (RETRIEVER={mode_label})",
            base_url=base_url or "https://api.openai.com/v1",
        ) from exc
    return (client, model, "openai", base_url or "https://api.openai.com/v1")


def _vector_literal(vector: list[float]) -> str:
    return "[" + ",".join(f"{x:.8f}" for x in vector) + "]"


class PgVectorRetriever:
    """Cosine-similarity retriever over the policy corpus, in pgvector.

    The corpus is embedded once per embedding model (lazily, on first
    use) into the ``policy_embeddings`` table of the same PostgreSQL
    that holds the approval records; each query is embedded per call
    and ranked by pgvector's cosine operator. Same return shape as the
    keyword retriever: ``RetrievedPolicy`` snippets with similarity
    scores. Requires ``DATABASE_URL`` — semantic retrieval is a
    production feature here, not a local-file one.
    """

    name = "semantic"
    vector_store = "pgvector"

    def __init__(
        self,
        policies: list[dict[str, str]] | None = None,
        *,
        mode_label: str = "semantic",
        database_url: str | None = None,
    ) -> None:
        self._policies = policies if policies is not None else POLICIES
        # The RETRIEVER value this instance serves ("semantic", or
        # "hybrid" when the hybrid retriever owns it) — error messages
        # name the value the operator actually set, never a sibling mode.
        self._mode_label = mode_label
        load_dotenv()
        self._database_url = database_url or env_str("DATABASE_URL")
        if not self._database_url:
            raise RuntimeError(
                f"RETRIEVER={mode_label} stores policy vectors in pgvector, "
                "but DATABASE_URL is not set. Semantic retrieval runs on the "
                "same PostgreSQL as the records — set DATABASE_URL "
                "(see .env.example), or set RETRIEVER=keyword."
            )
        try:
            import psycopg
        except ImportError as exc:
            raise RuntimeError(
                "RETRIEVER=semantic needs the psycopg driver, which is not "
                "installed. Install the postgres extra: uv sync --extra postgres "
                "(pip fallback: pip install -e \".[postgres]\"), then re-run."
            ) from exc
        self._psycopg = psycopg
        (
            self._client,
            self._model,
            self._embeddings_backend,
            self._embeddings_base_url,
        ) = _build_embeddings_client(mode_label)
        self._indexed = False

    def _embed(self, texts: list[str]) -> list[list[float]]:
        try:
            response = self._client.embeddings.create(model=self._model, input=texts)
        except Exception as exc:
            raise translate_provider_error(
                exc,
                backend=f"{self._embeddings_backend} embeddings (RETRIEVER={self._mode_label})",
                base_url=self._embeddings_base_url,
            ) from exc
        ordered = sorted(response.data, key=lambda d: getattr(d, "index", 0))
        return [list(d.embedding) for d in ordered]

    def _connect(self):
        return self._psycopg.connect(self._database_url)

    def _ensure_index(self) -> None:
        """Embed any corpus policies missing for the current model."""
        if self._indexed:
            return
        from .migrations import run_migrations

        with self._connect() as conn:
            run_migrations(conn)
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT policy_id FROM policy_embeddings WHERE model = %s",
                    (self._model,),
                )
                present = {row[0] for row in cur.fetchall()}
                missing = [p for p in self._policies if p["id"] not in present]
                if missing:
                    vectors = self._embed(
                        [f"{p['title']} {p['text']}" for p in missing]
                    )
                    for policy, vector in zip(missing, vectors):
                        cur.execute(
                            "INSERT INTO policy_embeddings (policy_id, model, embedding) "
                            "VALUES (%s, %s, %s::vector) "
                            "ON CONFLICT (policy_id, model) DO UPDATE "
                            "SET embedding = EXCLUDED.embedding",
                            (policy["id"], self._model, _vector_literal(vector)),
                        )
            conn.commit()
        self._indexed = True

    def retrieve(self, query: str, top_k: int = 3) -> list[RetrievedPolicy]:
        self._ensure_index()
        query_vector = _vector_literal(self._embed([query])[0])
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT policy_id, 1 - (embedding <=> %s::vector) AS score "
                    "FROM policy_embeddings WHERE model = %s "
                    "ORDER BY embedding <=> %s::vector LIMIT %s",
                    (query_vector, self._model, query_vector, top_k),
                )
                rows = cur.fetchall()
        by_id = {p["id"]: p for p in self._policies}
        retrieved: list[RetrievedPolicy] = []
        for policy_id, score in rows:
            policy = by_id.get(policy_id)
            if policy is None:  # foreign id in a shared database — skip it
                continue
            retrieved.append(
                RetrievedPolicy(
                    policy_id=policy_id,
                    title=policy["title"],
                    text=policy["text"],
                    score=round(float(score), 4),
                    sources=["semantic"],
                )
            )
        return retrieved


def rerank_fused(
    keyword_results: list[RetrievedPolicy],
    semantic_results: list[RetrievedPolicy],
    top_k: int = 3,
    *,
    rrf_k: int = 60,
) -> list[RetrievedPolicy]:
    """The hybrid rerank step: reciprocal-rank score fusion, in code.

    Keyword scores (token overlap) and semantic scores (cosine) live on
    incomparable scales, so the merge never compares them directly.
    Each list contributes ``1 / (rrf_k + rank)`` per policy; the fused
    score is the sum across lists, and policies found by both retrievers
    rank above single-list finds at similar ranks. The returned score
    IS the fused score — labelled as such via ``sources`` ("keyword",
    "semantic", or both). This is score-fusion reranking: transparent
    and deterministic, not a cross-encoder model.
    """
    fused: dict[str, dict] = {}
    for source, results in (("keyword", keyword_results), ("semantic", semantic_results)):
        for rank, result in enumerate(results):
            entry = fused.setdefault(
                result.policy_id, {"policy": result, "score": 0.0, "sources": []}
            )
            entry["score"] += 1.0 / (rrf_k + rank + 1)
            entry["sources"].append(source)
            if result.score > entry["policy"].score:
                entry["policy"] = result
    merged = sorted(
        fused.values(), key=lambda e: (-e["score"], e["policy"].policy_id)
    )
    reranked: list[RetrievedPolicy] = []
    for entry in merged[:top_k]:
        sources = [s for s in ("keyword", "semantic") if s in entry["sources"]]
        reranked.append(
            entry["policy"].model_copy(
                update={"score": round(entry["score"], 4), "sources": sources}
            )
        )
    return reranked


class HybridRetriever:
    """Keyword + pgvector semantic retrieval with an explicit rerank step.

    Pipeline: both retrievers produce a candidate pool → the pools are
    merged by policy ID → :func:`rerank_fused` reranks by reciprocal-rank
    fusion → the top-k are returned and cited exactly like the other
    retrievers' results. ``last_stats`` records the pool sizes of the
    most recent call so the trace can show merge → rerank honestly.
    """

    name = "hybrid"

    def __init__(self, policies: list[dict[str, str]] | None = None) -> None:
        self._keyword = KeywordRetriever(policies)
        self._semantic = PgVectorRetriever(policies, mode_label="hybrid")
        self.last_stats: dict[str, int] = {}

    @property
    def vector_store(self) -> str:
        """The store serving the semantic half (pgvector)."""
        return self._semantic.vector_store

    def retrieve(self, query: str, top_k: int = 3) -> list[RetrievedPolicy]:
        pool = top_k * 2
        keyword_results = self._keyword.retrieve(query, top_k=pool)
        semantic_results = self._semantic.retrieve(query, top_k=pool)
        self.last_stats = {
            "keyword_pool": len(keyword_results),
            "semantic_pool": len(semantic_results),
        }
        return rerank_fused(keyword_results, semantic_results, top_k=top_k)


def get_retriever(name: str | None = None) -> Retriever:
    """Select the retriever by name or the RETRIEVER env var (default: keyword).

    Loads the repo-root ``.env`` first (real environment variables win), so
    every surface — API, CLI, traced demo — selects the same way.
    """
    load_dotenv()
    selected = (name or env_str("RETRIEVER") or "keyword").lower()
    if selected == "keyword":
        return KeywordRetriever()
    if selected == "semantic":
        return PgVectorRetriever()
    if selected == "hybrid":
        return HybridRetriever()
    raise ValueError(
        f"Unknown RETRIEVER: {selected!r} (expected keyword | semantic | hybrid)"
    )
