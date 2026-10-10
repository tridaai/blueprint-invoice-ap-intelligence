"""Checkpointed approval gate: LangGraph state persistence over PostgreSQL.

The approval gate is a real pause in the graph, not a status flag:
with a checkpointer attached, a run executes up to the gate and its
graph state persists (thread_id = the invoice record id); the
service's approve/reject then *resumes* the thread with the decision,
and the graph completes. A process restart loses nothing — a new
service instance over the same database sees the store record AND
the paused thread.

The split of responsibilities, kept deliberately:

- the **store** (``store.py``) is the record of decisions — analyses,
  approvals, reasons, dispatch outcomes; every surface reads it;
- the **checkpointer** (this module) holds graph state — where the
  run paused and what it carried. It never decides anything.

Implementation: the pinned LangGraph ships the checkpoint base +
in-memory saver only (the Postgres saver is a separate package), so
:class:`PostgresCheckpointSaver` implements that base over psycopg 3
— one connection per call, opened and closed per call (thread-safe
by construction, like ``PostgresStore``), checkpoint blobs serialised
with LangGraph's own serde, channel values in side tables exactly as
the reference savers lay them out. The tables are created by
``migrations/0003_checkpoints.sql``; the saver runs migrations on
construction, so it works against a fresh database too.

When ``DATABASE_URL`` is not set, the checkpointer falls back to
LangGraph's in-process ``InMemorySaver`` — right for tests and
ephemeral runs, gone on restart; the Postgres saver is the production
story. ``CHECKPOINTS=off`` disables the whole thing: no checkpointer
is attached and the gate flow is the service-state flow alone.
"""

from __future__ import annotations

import asyncio
from typing import Any, AsyncIterator, Iterator, Sequence

from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import (
    WRITES_IDX_MAP,
    BaseCheckpointSaver,
    ChannelVersions,
    Checkpoint,
    CheckpointMetadata,
    CheckpointTuple,
    get_checkpoint_id,
    get_checkpoint_metadata,
)

from .config import env_str, load_dotenv

_SELECT = (
    "SELECT thread_id, checkpoint_ns, checkpoint_id, parent_checkpoint_id, "
    "type, checkpoint, metadata_type, metadata FROM checkpoints"
)


def _bytes(value) -> bytes:
    """BYTEA comes back as memoryview — the serde wants bytes."""
    return bytes(value) if value is not None else b""


class PostgresCheckpointSaver(BaseCheckpointSaver):
    """A minimal LangGraph checkpoint saver over PostgreSQL (psycopg 3)."""

    def __init__(self, database_url: str) -> None:
        super().__init__()
        try:
            import psycopg
            from psycopg.rows import dict_row
        except ImportError as exc:
            raise RuntimeError(
                "DATABASE_URL is set but the psycopg driver is not installed. "
                "Install the postgres extra: uv sync --extra postgres "
                "(pip fallback: pip install -e \".[postgres]\"), then re-run."
            ) from exc
        self._url = database_url
        self._psycopg = psycopg
        self._row_factory = dict_row
        from .migrations import run_migrations

        with self._connect() as conn:
            run_migrations(conn)

    def _connect(self):
        return self._psycopg.connect(self._url, row_factory=self._row_factory)

    # -- reads -----------------------------------------------------------

    def _load_channel_values(
        self, cur, thread_id: str, ns: str, versions: ChannelVersions
    ) -> dict[str, Any]:
        values: dict[str, Any] = {}
        for channel, version in versions.items():
            cur.execute(
                "SELECT type, value FROM checkpoint_blobs WHERE thread_id = %s "
                "AND checkpoint_ns = %s AND channel = %s AND version = %s",
                (thread_id, ns, channel, str(version)),
            )
            row = cur.fetchone()
            if row is not None and row["type"] != "empty":
                values[channel] = self.serde.loads_typed((row["type"], _bytes(row["value"])))
        return values

    def _tuple_from_row(self, cur, row: dict, config: RunnableConfig) -> CheckpointTuple:
        thread_id, ns = row["thread_id"], row["checkpoint_ns"]
        checkpoint: Checkpoint = self.serde.loads_typed(
            (row["type"], _bytes(row["checkpoint"]))
        )
        checkpoint = {
            **checkpoint,
            "channel_values": self._load_channel_values(
                cur, thread_id, ns, checkpoint["channel_versions"]
            ),
        }
        cur.execute(
            "SELECT task_id, channel, type, value FROM checkpoint_writes "
            "WHERE thread_id = %s AND checkpoint_ns = %s AND checkpoint_id = %s ORDER BY idx",
            (thread_id, ns, row["checkpoint_id"]),
        )
        write_rows = cur.fetchall()
        parent_id = row["parent_checkpoint_id"]
        return CheckpointTuple(
            config=config,
            checkpoint=checkpoint,
            metadata=self.serde.loads_typed((row["metadata_type"], _bytes(row["metadata"]))),
            parent_config=(
                {
                    "configurable": {
                        "thread_id": thread_id,
                        "checkpoint_ns": ns,
                        "checkpoint_id": parent_id,
                    }
                }
                if parent_id
                else None
            ),
            pending_writes=[
                (r["task_id"], r["channel"], self.serde.loads_typed((r["type"], _bytes(r["value"]))))
                for r in write_rows
            ],
        )

    def get_tuple(self, config: RunnableConfig) -> CheckpointTuple | None:
        thread_id = config["configurable"]["thread_id"]
        ns = config["configurable"].get("checkpoint_ns", "")
        checkpoint_id = get_checkpoint_id(config)
        with self._connect() as conn:
            with conn.cursor() as cur:
                if checkpoint_id:
                    cur.execute(
                        _SELECT
                        + " WHERE thread_id = %s AND checkpoint_ns = %s AND checkpoint_id = %s",
                        (thread_id, ns, checkpoint_id),
                    )
                else:
                    cur.execute(
                        _SELECT
                        + " WHERE thread_id = %s AND checkpoint_ns = %s "
                        "ORDER BY checkpoint_id DESC LIMIT 1",
                        (thread_id, ns),
                    )
                row = cur.fetchone()
                if row is None:
                    return None
                effective_config = config
                if not checkpoint_id:
                    effective_config = {
                        "configurable": {
                            "thread_id": thread_id,
                            "checkpoint_ns": ns,
                            "checkpoint_id": row["checkpoint_id"],
                        }
                    }
                return self._tuple_from_row(cur, row, effective_config)

    def list(
        self,
        config: RunnableConfig | None,
        *,
        filter: dict[str, Any] | None = None,
        before: RunnableConfig | None = None,
        limit: int | None = None,
    ) -> Iterator[CheckpointTuple]:
        thread_id = (config or {}).get("configurable", {}).get("thread_id")
        ns = (config or {}).get("configurable", {}).get("checkpoint_ns")
        before_id = get_checkpoint_id(before) if before else None
        with self._connect() as conn:
            with conn.cursor() as cur:
                query = _SELECT
                clauses, params = [], []
                if thread_id is not None:
                    clauses.append("thread_id = %s")
                    params.append(thread_id)
                if ns is not None:
                    clauses.append("checkpoint_ns = %s")
                    params.append(ns)
                if before_id:
                    clauses.append("checkpoint_id < %s")
                    params.append(before_id)
                if clauses:
                    query += " WHERE " + " AND ".join(clauses)
                query += " ORDER BY checkpoint_id DESC"
                cur.execute(query, params)
                rows = cur.fetchall()
                tuples = [
                    self._tuple_from_row(
                        cur,
                        row,
                        {
                            "configurable": {
                                "thread_id": row["thread_id"],
                                "checkpoint_ns": row["checkpoint_ns"],
                                "checkpoint_id": row["checkpoint_id"],
                            }
                        },
                    )
                    for row in rows
                ]
        count = 0
        for tup in tuples:
            if limit is not None and count >= limit:
                break
            if filter and any(
                tup.metadata.get(key) != value for key, value in filter.items()
            ):
                continue
            count += 1
            yield tup

    # -- writes ----------------------------------------------------------

    def put(
        self,
        config: RunnableConfig,
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
        new_versions: ChannelVersions,
    ) -> RunnableConfig:
        thread_id = config["configurable"]["thread_id"]
        ns = config["configurable"]["checkpoint_ns"]
        stored = checkpoint.copy()
        values: dict[str, Any] = stored.pop("channel_values")  # type: ignore[misc]
        checkpoint_type, checkpoint_blob = self.serde.dumps_typed(stored)
        metadata_type, metadata_blob = self.serde.dumps_typed(
            get_checkpoint_metadata(config, metadata)
        )
        with self._connect() as conn:
            with conn.cursor() as cur:
                for channel, version in new_versions.items():
                    if channel in values:
                        blob_type, blob = self.serde.dumps_typed(values[channel])
                    else:
                        blob_type, blob = "empty", b""
                    cur.execute(
                        "INSERT INTO checkpoint_blobs "
                        "(thread_id, checkpoint_ns, channel, version, type, value) "
                        "VALUES (%s, %s, %s, %s, %s, %s) "
                        "ON CONFLICT (thread_id, checkpoint_ns, channel, version) "
                        "DO UPDATE SET type = EXCLUDED.type, value = EXCLUDED.value",
                        (thread_id, ns, channel, str(version), blob_type, blob),
                    )
                cur.execute(
                    "INSERT INTO checkpoints "
                    "(thread_id, checkpoint_ns, checkpoint_id, parent_checkpoint_id, "
                    "type, checkpoint, metadata_type, metadata) "
                    "VALUES (%s, %s, %s, %s, %s, %s, %s, %s) "
                    "ON CONFLICT (thread_id, checkpoint_ns, checkpoint_id) "
                    "DO UPDATE SET parent_checkpoint_id = EXCLUDED.parent_checkpoint_id, "
                    "type = EXCLUDED.type, checkpoint = EXCLUDED.checkpoint, "
                    "metadata_type = EXCLUDED.metadata_type, metadata = EXCLUDED.metadata",
                    (
                        thread_id,
                        ns,
                        checkpoint["id"],
                        config["configurable"].get("checkpoint_id"),
                        checkpoint_type,
                        checkpoint_blob,
                        metadata_type,
                        metadata_blob,
                    ),
                )
            conn.commit()
        return {
            "configurable": {
                "thread_id": thread_id,
                "checkpoint_ns": ns,
                "checkpoint_id": checkpoint["id"],
            }
        }

    def put_writes(
        self,
        config: RunnableConfig,
        writes: Sequence[tuple[str, Any]],
        task_id: str,
        task_path: str = "",
    ) -> None:
        thread_id = config["configurable"]["thread_id"]
        ns = config["configurable"].get("checkpoint_ns", "")
        checkpoint_id = config["configurable"]["checkpoint_id"]
        with self._connect() as conn:
            with conn.cursor() as cur:
                for idx, (channel, value) in enumerate(writes):
                    mapped = WRITES_IDX_MAP.get(channel, idx)
                    blob_type, blob = self.serde.dumps_typed(value)
                    # Special channels (negative idx: errors, interrupts,
                    # resumes) overwrite; ordinary writes are first-wins,
                    # mirroring the reference savers.
                    conflict = (
                        "DO UPDATE SET channel = EXCLUDED.channel, type = EXCLUDED.type, "
                        "value = EXCLUDED.value, task_path = EXCLUDED.task_path"
                        if mapped < 0
                        else "DO NOTHING"
                    )
                    cur.execute(
                        "INSERT INTO checkpoint_writes "
                        "(thread_id, checkpoint_ns, checkpoint_id, task_id, idx, "
                        "channel, type, value, task_path) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s) "
                        f"ON CONFLICT (thread_id, checkpoint_ns, checkpoint_id, task_id, idx) {conflict}",
                        (thread_id, ns, checkpoint_id, task_id, mapped, channel, blob_type, blob, task_path),
                    )
            conn.commit()

    def delete_thread(self, thread_id: str) -> None:
        with self._connect() as conn:
            with conn.cursor() as cur:
                for table in ("checkpoints", "checkpoint_blobs", "checkpoint_writes"):
                    cur.execute(f"DELETE FROM {table} WHERE thread_id = %s", (thread_id,))
            conn.commit()

    # -- async delegates (the pipeline runs sync; these keep the saver
    #    usable from async LangGraph callers) -----------------------------

    async def aget_tuple(self, config: RunnableConfig) -> CheckpointTuple | None:
        return await asyncio.to_thread(self.get_tuple, config)

    async def alist(
        self,
        config: RunnableConfig | None,
        *,
        filter: dict[str, Any] | None = None,
        before: RunnableConfig | None = None,
        limit: int | None = None,
    ) -> AsyncIterator[CheckpointTuple]:
        for tup in await asyncio.to_thread(
            lambda: list(self.list(config, filter=filter, before=before, limit=limit))
        ):
            yield tup

    async def aput(
        self,
        config: RunnableConfig,
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
        new_versions: ChannelVersions,
    ) -> RunnableConfig:
        return await asyncio.to_thread(self.put, config, checkpoint, metadata, new_versions)

    async def aput_writes(
        self,
        config: RunnableConfig,
        writes: Sequence[tuple[str, Any]],
        task_id: str,
        task_path: str = "",
    ) -> None:
        await asyncio.to_thread(self.put_writes, config, writes, task_id, task_path)

    async def adelete_thread(self, thread_id: str) -> None:
        await asyncio.to_thread(self.delete_thread, thread_id)


# ---------------------------------------------------------------------------
# Selection from configuration
# ---------------------------------------------------------------------------

def checkpoints_enabled() -> bool:
    """``CHECKPOINTS`` env (read at call time): on unless explicitly off."""
    load_dotenv()
    raw = (env_str("CHECKPOINTS") or "on").strip().lower()
    return raw not in {"off", "0", "false", "no"}


def get_checkpointer():
    """The configured checkpointer, or None when CHECKPOINTS=off.

    PostgreSQL when ``DATABASE_URL`` is set (production — the paused
    gate survives restarts and is shared across processes); otherwise
    LangGraph's in-process in-memory saver (tests, ephemeral runs).
    """
    if not checkpoints_enabled():
        return None
    load_dotenv()
    url = env_str("DATABASE_URL")
    if url:
        return PostgresCheckpointSaver(url)
    from langgraph.checkpoint.memory import InMemorySaver

    return InMemorySaver()
