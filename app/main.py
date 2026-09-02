"""FastAPI surface — a thin wrapper over app.rag.answer_question."""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from app.config import settings
from app.db import chunk_count, close_pool, init_schema
from app.embeddings import get_model
from app.rag import answer_question
from app.tracing import flush as flush_traces
from app.tracing import tracing_enabled

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(_: FastAPI):
    init_schema()
    get_model()  # warm the embedding model so the first request isn't a 5s outlier
    logger.info(
        "Ready — %d chunks indexed, tracing %s",
        chunk_count(),
        "ON" if tracing_enabled() else "OFF",
    )
    yield
    flush_traces()  # buffered traces would be lost on exit otherwise
    close_pool()


app = FastAPI(title="LLM Observability & Eval Platform", lifespan=lifespan)


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    top_k: int | None = Field(default=None, ge=1, le=20)


class SourceOut(BaseModel):
    source: str
    chunk_index: int
    score: float


class AskResponse(BaseModel):
    question: str
    answer: str
    sources: list[SourceOut]
    contexts: list[str]
    usage: dict
    timings_ms: dict
    trace_id: str | None = None


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "chunks_indexed": chunk_count(),
        "model": settings.rag_model,
        "embedding_model": settings.embedding_model,
        "top_k": settings.rag_top_k,
        "tracing": tracing_enabled(),
    }


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest) -> AskResponse:
    if chunk_count() == 0:
        raise HTTPException(
            status_code=503,
            detail="No documents indexed. Run: python -m ingest.fetch_docs && "
            "python -m ingest.ingest",
        )

    result = answer_question(req.question, k=req.top_k)

    return AskResponse(
        question=result.question,
        answer=result.answer,
        sources=[
            SourceOut(source=c.source, chunk_index=c.chunk_index, score=round(c.score, 4))
            for c in result.chunks
        ],
        contexts=result.contexts,
        usage={
            "input_tokens": result.input_tokens,
            "output_tokens": result.output_tokens,
            "model": result.model,
        },
        timings_ms={
            "embed": round(result.embed_ms, 1),
            "search": round(result.search_ms, 1),
            "generate": round(result.generate_ms, 1),
            "total": round(result.total_ms, 1),
        },
        trace_id=result.trace_id,
    )
