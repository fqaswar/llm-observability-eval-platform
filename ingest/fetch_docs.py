"""Download the Anthropic documentation corpus into data/raw/.

The docs site publishes a machine-readable index at /llms.txt listing every page
as a `.md` URL, so the corpus is reproducible: this script is committed, the
downloaded markdown is gitignored.

    python -m ingest.fetch_docs            # default section subset
    python -m ingest.fetch_docs --all      # every page (~690)
    python -m ingest.fetch_docs --limit 20 # smoke test
"""

import argparse
import re
import sys
import time
from pathlib import Path

import httpx

INDEX_URL = "https://docs.claude.com/llms.txt"
RAW_DIR = Path("data/raw")

# Sections that make good RAG material: prose and reference, with concrete
# facts to ask about. Release notes and API-error tables are excluded — they are
# mostly changelogs and status codes, which make for poor eval questions.
DEFAULT_SECTIONS = (
    "/docs/en/about-claude/",
    "/docs/en/build-with-claude/",
    "/docs/en/agents-and-tools/",
    "/docs/en/api/",
    "/docs/en/test-and-evaluate/",
)

URL_RE = re.compile(r"https://[^\s)\"']+\.md")


def discover_urls(session: httpx.Client) -> list[str]:
    resp = session.get(INDEX_URL, timeout=30)
    resp.raise_for_status()
    return sorted(set(URL_RE.findall(resp.text)))


def slugify(url: str) -> str:
    """https://…/docs/en/build-with-claude/prompt-caching.md
       -> build-with-claude__prompt-caching.md"""
    path = url.split("/docs/en/", 1)[-1].removesuffix(".md")
    return path.replace("/", "__") + ".md"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--all", action="store_true", help="fetch every page")
    ap.add_argument("--limit", type=int, help="cap the number of pages")
    args = ap.parse_args()

    RAW_DIR.mkdir(parents=True, exist_ok=True)

    with httpx.Client(follow_redirects=True, headers={"User-Agent": "rag-eval-demo"}) as s:
        urls = discover_urls(s)
        print(f"Index lists {len(urls)} pages")

        if not args.all:
            urls = [u for u in urls if any(sec in u for sec in DEFAULT_SECTIONS)]
            print(f"{len(urls)} match the default sections")
        if args.limit:
            urls = urls[: args.limit]

        written = skipped = failed = 0
        for i, url in enumerate(urls, 1):
            dest = RAW_DIR / slugify(url)
            if dest.exists():
                skipped += 1
                continue
            try:
                r = s.get(url, timeout=30)
                r.raise_for_status()
                text = r.text.strip()
                if len(text) < 200:  # nav stubs and redirect pages
                    skipped += 1
                    continue
                dest.write_text(text, encoding="utf-8")
                written += 1
            except Exception as exc:  # noqa: BLE001 — one bad page shouldn't abort
                print(f"  ! {url}: {exc}", file=sys.stderr)
                failed += 1
            if i % 25 == 0:
                print(f"  …{i}/{len(urls)}")
            time.sleep(0.05)  # be polite

    print(f"\nWrote {written}, skipped {skipped}, failed {failed} -> {RAW_DIR}/")
    return 0 if written or skipped else 1


if __name__ == "__main__":
    raise SystemExit(main())
