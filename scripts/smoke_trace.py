"""Ask one question and print the resulting Langfuse trace URL.

    .venv310/bin/python -m scripts.smoke_trace ["your question"]

Use this to confirm tracing works end to end without starting the web server.
"""

import sys

from app.rag import answer_question
from app.tracing import flush, tracing_enabled

DEFAULT_Q = "What is prompt caching and when should I use it?"


def main() -> int:
    question = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_Q
    print(f"tracing enabled: {tracing_enabled()}")

    r = answer_question(question)

    print(f"\nQ: {question}")
    print(f"A: {r.answer[:300]}{'…' if len(r.answer) > 300 else ''}\n")
    print("sources:")
    for c in r.chunks:
        print(f"  {c.score:.3f}  {c.source}#{c.chunk_index}")
    print(f"\ntokens : {r.input_tokens} in / {r.output_tokens} out ({r.model})")
    print(
        f"timings: embed {r.embed_ms:.0f}ms  search {r.search_ms:.0f}ms  "
        f"generate {r.generate_ms:.0f}ms  total {r.total_ms:.0f}ms"
    )

    flush()  # short-lived process: without this the trace is never sent

    if r.trace_id:
        from langfuse import get_client

        print(f"\ntrace  : {get_client().get_trace_url(trace_id=r.trace_id)}")
    else:
        print("\nNo trace produced — tracing is disabled.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
