-- 0003_checkpoints: LangGraph checkpoint tables for the approval gate.
--
-- The Postgres checkpoint saver (checkpoints.py) persists graph state
-- here: a run pauses at the human gate with its state under
-- thread_id = invoice id, and the service's approve/reject resumes
-- the same thread. Layout mirrors the reference LangGraph savers:
-- checkpoints hold the serialised checkpoint + metadata, channel
-- values live in checkpoint_blobs, pending writes in
-- checkpoint_writes. Blobs are LangGraph-serde bytes (BYTEA).

CREATE TABLE IF NOT EXISTS checkpoints (
    thread_id            TEXT NOT NULL,
    checkpoint_ns        TEXT NOT NULL,
    checkpoint_id        TEXT NOT NULL,
    parent_checkpoint_id TEXT,
    type                 TEXT,
    checkpoint           BYTEA,
    metadata_type        TEXT,
    metadata             BYTEA,
    PRIMARY KEY (thread_id, checkpoint_ns, checkpoint_id)
);

CREATE TABLE IF NOT EXISTS checkpoint_blobs (
    thread_id     TEXT NOT NULL,
    checkpoint_ns TEXT NOT NULL,
    channel       TEXT NOT NULL,
    version       TEXT NOT NULL,
    type          TEXT,
    value         BYTEA,
    PRIMARY KEY (thread_id, checkpoint_ns, channel, version)
);

CREATE TABLE IF NOT EXISTS checkpoint_writes (
    thread_id     TEXT NOT NULL,
    checkpoint_ns TEXT NOT NULL,
    checkpoint_id TEXT NOT NULL,
    task_id       TEXT NOT NULL,
    idx           INTEGER NOT NULL,
    channel       TEXT NOT NULL,
    type          TEXT,
    value         BYTEA,
    task_path     TEXT NOT NULL DEFAULT '',
    PRIMARY KEY (thread_id, checkpoint_ns, checkpoint_id, task_id, idx)
);
