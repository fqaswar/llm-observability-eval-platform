"""Show what retrieval returns for a question, without calling Claude.

    .venv310/bin/python -m scripts.peek "your question" [k]

Useful while writing golden-set questions: confirms a fact is actually
retrievable before you commit a reference answer that depends on it.
"""

import sys

from app.retrieval import retrieve


def main() -> int:
    if len(sys.argv) < 2:
        print(__doc__)
        return 2
    question = sys.argv[1]
    k = int(sys.argv[2]) if len(sys.argv) > 2 else 3

    for c in retrieve(question, k=k).chunks:
        print(f"\n--- {c.score:.3f}  {c.source}#{c.chunk_index} ---")
        print(c.content[:700])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
