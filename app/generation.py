"""Claude generation over retrieved context."""

import time
from contextlib import nullcontext
from dataclasses import dataclass

from anthropic import Anthropic
from langfuse import get_client as get_langfuse

from app.config import settings
from app.retrieval import RetrievedChunk
from app.tracing import tracing_enabled

_client: Anthropic | None = None


def get_client() -> Anthropic:
    global _client
    if _client is None:
        if not settings.anthropic_api_key:
            raise RuntimeError(
                "ANTHROPIC_API_KEY is not set. Copy .env.example to .env and add it."
            )
        _client = Anthropic(api_key=settings.anthropic_api_key)
    return _client


@dataclass
class GenerationResult:
    answer: str
    prompt: str
    input_tokens: int
    output_tokens: int
    latency_ms: float
    model: str


def build_prompt(question: str, chunks: list[RetrievedChunk]) -> str:
    """Number the sources so the answer can cite them and a human can audit."""
    if not chunks:
        context = "(no relevant documentation was retrieved)"
    else:
        context = "\n\n".join(
            f"[{i}] source: {c.source}\n{c.content}" for i, c in enumerate(chunks, 1)
        )
    return (
        f"<context>\n{context}\n</context>\n\n"
        f"Question: {question}\n\n"
        f"Answer using only the context above."
    )


def _observe(**kwargs):
    """Langfuse observation, or a no-op when tracing is off."""
    if not tracing_enabled():
        return nullcontext()
    return get_langfuse().start_as_current_observation(**kwargs)


def generate(question: str, chunks: list[RetrievedChunk]) -> GenerationResult:
    prompt = build_prompt(question, chunks)

    with _observe(
        name="generate",
        as_type="generation",
        model=settings.rag_model,
        input=[
            {"role": "system", "content": settings.system_prompt},
            {"role": "user", "content": prompt},
        ],
        model_parameters={"max_tokens": settings.rag_max_tokens},
        metadata={"context_chunks": len(chunks)},
    ) as gen:
        t0 = time.perf_counter()
        resp = get_client().messages.create(
            model=settings.rag_model,
            max_tokens=settings.rag_max_tokens,
            system=settings.system_prompt,
            messages=[{"role": "user", "content": prompt}],
        )
        latency_ms = (time.perf_counter() - t0) * 1000

        answer = "".join(b.text for b in resp.content if b.type == "text").strip()

        if hasattr(gen, "update"):
            # Send Anthropic's own token counts — Langfuse's fallback tokenizer
            # is inaccurate for Claude. Cost in USD is inferred from the model
            # name against Langfuse's built-in price table.
            gen.update(
                output=answer,
                usage_details={
                    "input": resp.usage.input_tokens,
                    "output": resp.usage.output_tokens,
                },
                metadata={
                    "stop_reason": resp.stop_reason,
                    "latency_ms": round(latency_ms, 1),
                },
            )

    return GenerationResult(
        answer=answer,
        prompt=prompt,
        input_tokens=resp.usage.input_tokens,
        output_tokens=resp.usage.output_tokens,
        latency_ms=latency_ms,
        model=settings.rag_model,
    )
