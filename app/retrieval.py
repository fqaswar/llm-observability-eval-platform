"""pgvector similarity search, traced as retriever -> embedding + search."""

import time
from contextlib import nullcontext
from dataclasses import dataclass

from langfuse import get_client

from app.config import settings
from app.db import get_conn
from app.embeddings import embed_query
from app.tracing import tracing_enabled


@dataclass
class RetrievedChunk:
    source: str
    chunk_index: int
    content: str
    score: float  # cosine similarity, 1.0 == identical


@dataclass
class RetrievalResult:
    chunks: list[RetrievedChunk]
    embed_ms: float
    search_ms: float


# `<=>` is cosine DISTANCE (0 = identical), so similarity is 1 - distance.
SEARCH_SQL = """
SELECT source, chunk_index, content, 1 - (embedding <=> %s) AS score
FROM rag_chunks
ORDER BY embedding <=> %s
LIMIT %s
"""


def _observe(**kwargs):
    """Langfuse observation, or a no-op when tracing is off."""
    if not tracing_enabled():
        return nullcontext()
    return get_client().start_as_current_observation(**kwargs)


def retrieve(question: str, k: int | None = None) -> RetrievalResult:
    k = k if k is not None else settings.rag_top_k

    with _observe(
        name="retrieve",
        as_type="retriever",
        input={"question": question},
        metadata={"top_k": k, "embedding_model": settings.embedding_model},
    ) as retriever:
        # --- embed the query ---
        with _observe(
            name="embed-query",
            as_type="embedding",
            input=question,
            model=settings.embedding_model,
            metadata={"dimensions": settings.embedding_dim},
        ):
            t0 = time.perf_counter()
            qvec = embed_query(question)
            embed_ms = (time.perf_counter() - t0) * 1000

        # --- search pgvector ---
        with _observe(
            name="pgvector-search",
            as_type="span",
            input={"top_k": k},
        ) as search:
            t1 = time.perf_counter()
            with get_conn() as conn:
                rows = conn.execute(SEARCH_SQL, (qvec, qvec, k)).fetchall()
            search_ms = (time.perf_counter() - t1) * 1000

            chunks = [
                RetrievedChunk(source=r[0], chunk_index=r[1], content=r[2], score=float(r[3]))
                for r in rows
            ]
            if hasattr(search, "update"):
                search.update(
                    output={"rows": len(chunks)},
                    metadata={"search_ms": round(search_ms, 1)},
                )

        # Log the retrieved sources and scores — this is what you inspect when a
        # faithfulness score drops and you need to know whether retrieval or
        # generation was at fault.
        if hasattr(retriever, "update"):
            retriever.update(
                output=[
                    {
                        "source": c.source,
                        "chunk_index": c.chunk_index,
                        "score": round(c.score, 4),
                        "preview": c.content[:200],
                    }
                    for c in chunks
                ],
                metadata={
                    "chunks_retrieved": len(chunks),
                    "top_score": round(chunks[0].score, 4) if chunks else None,
                    "min_score": round(chunks[-1].score, 4) if chunks else None,
                    "embed_ms": round(embed_ms, 1),
                    "search_ms": round(search_ms, 1),
                },
            )

    return RetrievalResult(chunks=chunks, embed_ms=embed_ms, search_ms=search_ms)
