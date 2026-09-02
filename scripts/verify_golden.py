"""Check that each golden-set question retrieves plausible context.

Retrieval only — no Claude calls, so this is free. Run it after editing the
golden set: a low top score on a `factual` question usually means the fact is
not in the corpus, which would make the eval measure the wrong thing.

    .venv310/bin/python -m scripts.verify_golden
"""

import json
from pathlib import Path

from app.retrieval import retrieve

GOLDEN = Path("eval/golden_set.json")
LOW_SCORE = 0.70  # below this, retrieval probably missed


def main() -> int:
    data = json.loads(GOLDEN.read_text())
    questions = data["questions"]
    print(f"{len(questions)} questions\n")

    suspicious = []
    for q in questions:
        chunks = retrieve(q["question"], k=3).chunks
        top = chunks[0] if chunks else None
        tag = q["tags"][0]
        score = top.score if top else 0.0
        flag = ""
        # Traps are *expected* to retrieve loosely related material; only
        # non-trap questions with weak retrieval are a problem.
        if tag != "trap" and score < LOW_SCORE:
            flag = "  <-- weak retrieval"
            suspicious.append(q["id"])
        src = f"{top.source[:44]}#{top.chunk_index}" if top else "-"
        print(f"  {q['id']}  {tag:9} {score:.3f}  {src}{flag}")

    print()
    if suspicious:
        print(f"Check these: {', '.join(suspicious)}")
        print("Use scripts/peek.py to see what they actually retrieve.")
    else:
        print("All non-trap questions retrieve plausible context.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
