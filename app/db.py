"""Postgres + pgvector access.

All DDL lives here rather than in the Supabase SQL editor, so the dev database
and the throwaway container CI uses in Phase 4 can never drift apart.
"""

import logging
from contextlib import contextmanager

from pgvector.psycopg import register_vector
from psycopg_pool import ConnectionPool

from app.config import settings

logger = logging.getLogger(__name__)

SCHEMA_SQL = """
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS rag_chunks (
    id           BIGSERIAL PRIMARY KEY,
    source       TEXT      NOT NULL,
    chunk_index  INT       NOT NULL,
    content      TEXT      NOT NULL,
    embedding    vector({dim}) NOT NULL,
    created_at   TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (source, chunk_index)
);

CREATE INDEX IF NOT EXISTS rag_chunks_embedding_idx
    ON rag_chunks USING hnsw (embedding vector_cosine_ops);
"""

_pool: ConnectionPool | None = None


def _configure(conn) -> None:
    """Run on every new connection: pgvector adapters + autocommit."""
    conn.autocommit = True
    register_vector(conn)


def _validate_url(url: str) -> None:
    """Catch malformed connection strings up front.

    Without this, a stray '@' turns the password into part of the hostname and
    psycopg spends 30 seconds retrying DNS before failing with an error that
    doesn't name the real cause.
    """
    if "[YOUR-PASSWORD]" in url or "[" in url.split("@")[0]:
        raise RuntimeError(
            "DATABASE_URL still contains the [YOUR-PASSWORD] placeholder. "
            "Replace the whole placeholder, brackets included, with your password."
        )

    creds = url.split("://", 1)[-1].rsplit("@", 1)[0]
    if "@" in creds:
        raise RuntimeError(
            "DATABASE_URL has more than one '@'. If your password contains '@', "
            "percent-encode it as '%40' (likewise ':' -> '%3A', '/' -> '%2F', "
            "'#' -> '%23'). Otherwise the host is parsed from the wrong position."
        )


def get_pool() -> ConnectionPool:
    global _pool
    if _pool is None:
        if not settings.database_url:
            raise RuntimeError(
                "DATABASE_URL is not set. Copy .env.example to .env and paste "
                "your Supabase Session pooler connection string."
            )
        _validate_url(settings.database_url)
        _pool = ConnectionPool(
            conninfo=settings.database_url,
            min_size=1,
            max_size=5,
            open=True,
            configure=_configure,
            # Supabase's pooler runs in transaction mode, which cannot hold
            # server-side prepared statements across checkouts.
            kwargs={"prepare_threshold": None},
        )
    return _pool


@contextmanager
def get_conn():
    with get_pool().connection() as conn:
        yield conn


def init_schema() -> None:
    """Idempotent: safe to call on every boot."""
    with get_conn() as conn:
        conn.execute(SCHEMA_SQL.format(dim=settings.embedding_dim))
    logger.info("Schema ready (rag_chunks, dim=%d)", settings.embedding_dim)


def chunk_count() -> int:
    with get_conn() as conn:
        row = conn.execute("SELECT count(*) FROM rag_chunks").fetchone()
    return row[0] if row else 0


def close_pool() -> None:
    global _pool
    if _pool is not None:
        _pool.close()
        _pool = None
