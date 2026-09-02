"""Chunk -> embed -> upsert into pgvector.

Idempotent: re-running updates existing rows rather than duplicating them, so
you can safely re-ingest after changing the chunker.

    python -m ingest.ingest
    python -m ingest.ingest --reset     # clear the table first
"""

import argparse
from pathlib import Path

from app.config import settings
from app.db import chunk_count, get_conn, init_schema
from app.embeddings import embed_documents
from ingest.chunker import chunk_markdown

RAW_DIR = Path("data/raw")

UPSERT_SQL = """
INSERT INTO rag_chunks (source, chunk_index, content, embedding)
VALUES (%s, %s, %s, %s)
ON CONFLICT (source, chunk_index)
DO UPDATE SET content = EXCLUDED.content,
              embedding = EXCLUDED.embedding,
              created_at = now()
"""


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reset", action="store_true", help="delete existing rows first")
    ap.add_argument("--limit", type=int, help="only ingest N files (for CI/smoke tests)")
    args = ap.parse_args()

    files = sorted(RAW_DIR.glob("*.md"))
    if args.limit:
        files = files[: args.limit]
    if not files:
        print(f"No markdown in {RAW_DIR}/ — run: python -m ingest.fetch_docs")
        return 1

    init_schema()

    if args.reset:
        with get_conn() as conn:
            conn.execute("TRUNCATE rag_chunks RESTART IDENTITY")
        print("Cleared rag_chunks")

    print(f"Chunking {len(files)} files…")
    records: list[tuple[str, int, str]] = []
    for path in files:
        text = path.read_text(encoding="utf-8")
        chunks = chunk_markdown(
            text,
            max_tokens=settings.chunk_size_tokens,
            overlap_tokens=settings.chunk_overlap_tokens,
        )
        records.extend((path.name, i, c) for i, c in enumerate(chunks))

    if not records:
        print("Nothing to ingest.")
        return 1

    print(f"{len(records)} chunks — embedding with {settings.embedding_model}…")
    vectors = embed_documents([r[2] for r in records])

    print("Upserting…")
    with get_conn() as conn:
        with conn.cursor() as cur:
            for (source, idx, content), vec in zip(records, vectors):
                cur.execute(UPSERT_SQL, (source, idx, content, vec))

    total = chunk_count()
    avg = sum(len(r[2].split()) for r in records) / len(records)
    print(f"\nDone. {len(records)} chunks written, {total} rows in rag_chunks.")
    print(f"Average chunk size: {avg:.0f} words")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
