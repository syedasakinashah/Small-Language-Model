"""Split documents into overlapping, page-tagged chunks.

Chunks are the unit of both retrieval and citation, so each one must remember
exactly which document and page it came from -- that is what lets the tutor say
"Physics.pdf, page 12" instead of asking the student to trust it.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

TARGET_WORDS = 180      # big enough to hold a full explanation
OVERLAP_WORDS = 45      # so a definition split across a boundary is never lost
MIN_WORDS = 25          # below this a chunk is noise (stray caption, page number)


@dataclass
class Chunk:
    text: str
    doc_name: str
    page: int
    subject: str = ""
    index: int = 0

    @property
    def citation(self) -> str:
        return f"{self.doc_name} · p.{self.page}"


def _split_sentences(text: str) -> list[str]:
    """Sentence split that does not break on decimals or common abbreviations."""
    protected = re.sub(r"(\b(?:e\.g|i\.e|etc|vs|Dr|Mr|Mrs|Fig|Eq|approx|No)\.)",
                       lambda m: m.group(1).replace(".", "<DOT>"), text)
    protected = re.sub(r"(\d)\.(\d)", r"\1<DOT>\2", protected)
    parts = re.split(r"(?<=[.!?])\s+|\n{2,}", protected)
    return [p.replace("<DOT>", ".").strip() for p in parts if p.strip()]


def chunk_document(
    doc,
    subject: str = "",
    target_words: int = TARGET_WORDS,
    overlap_words: int = OVERLAP_WORDS,
) -> list[Chunk]:
    """Turn a :class:`pdf.Document` into overlapping chunks.

    Chunks never span a page boundary, so the page citation is always exact.
    """
    chunks: list[Chunk] = []
    for page in doc.pages:
        sentences = _split_sentences(page.text)
        if not sentences:
            continue

        buffer: list[str] = []
        count = 0
        for sent in sentences:
            words = sent.split()
            buffer.append(sent)
            count += len(words)
            if count >= target_words:
                chunks.append(Chunk("", doc.name, page.number, subject))
                chunks[-1].text = " ".join(buffer)
                # keep a tail of the previous chunk so context carries over
                tail: list[str] = []
                tail_count = 0
                for s in reversed(buffer):
                    tail.insert(0, s)
                    tail_count += len(s.split())
                    if tail_count >= overlap_words:
                        break
                buffer, count = tail, tail_count

        leftover = " ".join(buffer).strip()
        if leftover and len(leftover.split()) >= MIN_WORDS:
            # avoid emitting a chunk that is purely the overlap tail
            if not chunks or leftover != chunks[-1].text:
                chunks.append(Chunk(leftover, doc.name, page.number, subject))

    for i, c in enumerate(chunks):
        c.index = i
    return chunks


def chunk_documents(docs, subject: str = "") -> list[Chunk]:
    out: list[Chunk] = []
    for d in docs:
        out.extend(chunk_document(d, subject=subject))
    for i, c in enumerate(out):
        c.index = i
    return out
    #ffff 
