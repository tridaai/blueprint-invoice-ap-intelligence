"""Schema migrations: numbered SQL files + a tiny runner.

Production systems version their schema. The SQL lives in
``migrations/`` — visible at the repo root and packaged inside
``ap_agent/migrations/`` (identical files; a test asserts they stay
in sync) — and this runner applies any pending files, in order, each
in its own transaction, recording them in ``schema_migrations``.

The runner is invoked at startup: the Postgres store, the Postgres
checkpoint saver, and the pgvector retriever each run it on
construction (it is idempotent and cheap once applied), so a fresh
deployment against an empty database needs no separate step. It can
also be run explicitly:

    DATABASE_URL=postgresql://... python -m ap_agent.migrations

No ORM, no migration framework — twelve lines of bookkeeping over
psycopg, because the schema is three small files and the runner
should be auditable at a glance.
"""

from __future__ import annotations

from pathlib import Path

_MIGRATIONS_DIR = Path(__file__).resolve().parent / "migrations"

_TRACKING_DDL = """
CREATE TABLE IF NOT EXISTS schema_migrations (
    version    TEXT PRIMARY KEY,
    applied_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
"""


def migration_files() -> list[tuple[str, Path]]:
    """The packaged migrations, in application order: [(version, path)]."""
    files = []
    for path in sorted(_MIGRATIONS_DIR.glob("*.sql")):
        version = path.name.split("_", 1)[0]
        files.append((version, path))
    return files


def run_migrations(conn) -> list[str]:
    """Apply pending migrations on an open psycopg connection.

    Returns the versions applied by this call ([] when up to date).
    """
    applied: list[str] = []
    with conn.cursor() as cur:
        cur.execute(_TRACKING_DDL)
        cur.execute("SELECT version FROM schema_migrations")
        done = {row[0] for row in cur.fetchall()}
    conn.commit()
    for version, path in migration_files():
        if version in done:
            continue
        sql = path.read_text(encoding="utf-8")
        with conn.cursor() as cur:
            cur.execute(sql)
            cur.execute(
                "INSERT INTO schema_migrations (version) VALUES (%s)", (version,)
            )
        conn.commit()
        applied.append(version)
    return applied


def migrate_database(database_url: str) -> list[str]:
    """Connect to ``database_url`` and apply pending migrations."""
    import psycopg

    with psycopg.connect(database_url) as conn:
        return run_migrations(conn)


def main() -> None:
    import os

    url = os.environ.get("DATABASE_URL")
    if not url:
        raise SystemExit("DATABASE_URL is not set — nothing to migrate.")
    applied = migrate_database(url)
    if applied:
        print(f"applied migrations: {', '.join(applied)}")
    else:
        print("schema is up to date (no pending migrations)")


if __name__ == "__main__":
    main()
