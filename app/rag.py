"""The orchestrator — the single seam the whole project hangs off.

`answer_question()` is called in-process by both the FastAPI endpoint
(app/main.py) and the Phase 3 eval runner. Keeping it a plain function means the
eval harness needs no HTTP glue, and Phase 2 can wrap it in one trace decorator.

It returns `contexts` alongside the answer because RAGAS scores faithfulness and
context precision against exactly those strings — an API shape that discarded
them would make the eval phase impossible.
"""

from dataclasses import dataclass, field

from contextlib import nullcontext

from langfuse import get_client, observe, propagate_attributes

from app.config import settings
from app.generation import generate
from app.retrieval import RetrievedChunk, retrieve
from app.tracing import tracing_enabled


@dataclass
class AnswerResult:
    question: str
    answer: str
    contexts: list[str]
    chunks: list[RetrievedChunk] = field(repr=False, default_factory=list)
    input_tokens: int = 0
    output_tokens: int = 0
    embed_ms: float = 0.0
    search_ms: float = 0.0
    generate_ms: float = 0.0
    model: str = ""
    top_k: int = 0
    trace_id: str | None = None  # populated in Phase 2

    @property
    def total_ms(self) -> float:
        return self.embed_ms + self.search_ms + self.generate_ms


@observe(name="rag-query")
def answer_question(
    question: str, k: int | None = None, user_id: str | None = None
) -> AnswerResult:
    """One traced request: retrieve -> generate, nested under a single trace."""
    k = k if k is not None else settings.rag_top_k
    traced = tracing_enabled()

    # propagate_attributes must WRAP the work — nested observations inherit
    # these trace-level attributes as they are created.
    attrs = (
        propagate_attributes(
            user_id=user_id,
            tags=["rag", f"top_k={k}"],
            metadata={"top_k": k, "model": settings.rag_model},
        )
        if traced
        else nullcontext()
    )

    with attrs:
        retrieval = retrieve(question, k=k)
        generation = generate(question, retrieval.chunks)

    result = AnswerResult(
        question=question,
        answer=generation.answer,
        contexts=[c.content for c in retrieval.chunks],
        chunks=retrieval.chunks,
        input_tokens=generation.input_tokens,
        output_tokens=generation.output_tokens,
        embed_ms=retrieval.embed_ms,
        search_ms=retrieval.search_ms,
        generate_ms=generation.latency_ms,
        model=generation.model,
        top_k=k,
    )

    if traced:
        client = get_client()
        # Trace-level input/output so the dashboard list is readable at a glance
        # without having to open each trace.
        client.set_current_trace_io(input=question, output=result.answer)
        client.update_current_span(
            metadata={
                "sources": [c.source for c in result.chunks],
                "top_score": round(result.chunks[0].score, 4) if result.chunks else None,
                "total_ms": round(result.total_ms, 1),
            }
        )
        result.trace_id = client.get_current_trace_id()

    return result
