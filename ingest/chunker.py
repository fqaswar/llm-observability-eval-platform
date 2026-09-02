"""Markdown-aware chunking.

Splits on headings first so a chunk rarely straddles two topics, then packs
sections up to a token budget. Oversized sections are split on paragraph
boundaries with overlap, so a fact spanning a boundary still appears whole in
one chunk.

Token counts are approximated as words * 1.3 — good enough for sizing, and it
avoids a tokenizer dependency in the ingest path.
"""

import re
from dataclasses import dataclass

HEADING_RE = re.compile(r"^(#{1,4})\s+(.*)$", re.MULTILINE)
WORDS_TO_TOKENS = 1.3


def estimate_tokens(text: str) -> int:
    return int(len(text.split()) * WORDS_TO_TOKENS)


@dataclass
class Section:
    heading: str
    body: str

    @property
    def text(self) -> str:
        return f"{self.heading}\n\n{self.body}".strip() if self.heading else self.body.strip()


def split_sections(markdown: str) -> list[Section]:
    matches = list(HEADING_RE.finditer(markdown))
    if not matches:
        return [Section(heading="", body=markdown)]

    sections = []
    preamble = markdown[: matches[0].start()].strip()
    if preamble:
        sections.append(Section(heading="", body=preamble))

    for i, m in enumerate(matches):
        end = matches[i + 1].start() if i + 1 < len(matches) else len(markdown)
        body = markdown[m.end() : end].strip()
        if body:
            sections.append(Section(heading=m.group(0).strip(), body=body))
    return sections


def _split_block(block: str, max_tokens: int) -> list[str]:
    """Last-resort split for a block with no blank lines — long markdown tables
    and dense list runs. Splits on newlines, then mid-line if a single line is
    still too long, so no chunk can exceed the budget."""
    if estimate_tokens(block) <= max_tokens:
        return [block]

    pieces: list[str] = []
    current: list[str] = []
    current_tokens = 0

    for line in block.splitlines():
        ltok = estimate_tokens(line)
        if ltok > max_tokens:  # one line longer than the whole budget
            if current:
                pieces.append("\n".join(current))
                current, current_tokens = [], 0
            words = line.split()
            step = max(1, int(max_tokens / WORDS_TO_TOKENS))
            pieces.extend(" ".join(words[i : i + step]) for i in range(0, len(words), step))
            continue
        if current and current_tokens + ltok > max_tokens:
            pieces.append("\n".join(current))
            current, current_tokens = [], 0
        current.append(line)
        current_tokens += ltok

    if current:
        pieces.append("\n".join(current))
    return pieces


def _split_oversized(section: Section, max_tokens: int, overlap_tokens: int) -> list[str]:
    """Break one long section on paragraph boundaries, carrying overlap."""
    # The heading is re-attached to every piece below, so reserve room for it.
    budget = max_tokens - estimate_tokens(section.heading) if section.heading else max_tokens
    budget = max(budget, 50)

    paras = [
        piece
        for p in re.split(r"\n\s*\n", section.body)
        if p.strip()
        for piece in _split_block(p.strip(), budget)
    ]
    chunks: list[str] = []
    current: list[str] = []
    current_tokens = 0

    for para in paras:
        ptok = estimate_tokens(para)
        if current and current_tokens + ptok > budget:
            chunks.append("\n\n".join(current))
            # Carry trailing paragraphs back as overlap.
            carry, carried = [], 0
            for prev in reversed(current):
                t = estimate_tokens(prev)
                if carried + t > overlap_tokens:
                    break
                carry.insert(0, prev)
                carried += t
            current, current_tokens = carry, carried
        current.append(para)
        current_tokens += ptok

    if current:
        chunks.append("\n\n".join(current))

    # Repeat the heading on every piece so each chunk stays self-describing.
    if section.heading:
        chunks = [f"{section.heading}\n\n{c}" for c in chunks]
    return chunks


def chunk_markdown(markdown: str, max_tokens: int = 800, overlap_tokens: int = 100) -> list[str]:
    chunks: list[str] = []
    buffer: list[str] = []
    buffer_tokens = 0

    for section in split_sections(markdown):
        text = section.text
        if not text:
            continue
        stok = estimate_tokens(text)

        if stok > max_tokens:
            if buffer:
                chunks.append("\n\n".join(buffer))
                buffer, buffer_tokens = [], 0
            chunks.extend(_split_oversized(section, max_tokens, overlap_tokens))
            continue

        if buffer and buffer_tokens + stok > max_tokens:
            chunks.append("\n\n".join(buffer))
            buffer, buffer_tokens = [], 0

        buffer.append(text)
        buffer_tokens += stok

    if buffer:
        chunks.append("\n\n".join(buffer))

    # Drop fragments too small to answer anything.
    return [c.strip() for c in chunks if estimate_tokens(c) >= 20]
