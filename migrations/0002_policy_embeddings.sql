-- 0002_policy_embeddings: pgvector store for policy-corpus embeddings.
--
-- Semantic / hybrid retrieval (retriever.py, PgVectorRetriever)
-- embeds the bundled AP policy corpus once per embedding model and
-- keeps the vectors here, in the same Postgres that holds the
-- records — one database to run, back up, and secure. The corpus is
-- keyed by (policy_id, model): a model change re-embeds under its own
-- key instead of silently mixing vector spaces. The vector column is
-- dimension-less on purpose: embedding models differ (768, 1024,
-- 1536, ...), and pgvector's cosine operator works across them.

CREATE TABLE IF NOT EXISTS policy_embeddings (
    policy_id TEXT NOT NULL,
    model     TEXT NOT NULL,
    embedding vector NOT NULL,
    PRIMARY KEY (policy_id, model)
);
