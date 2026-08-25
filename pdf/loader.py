"""Read a PDF into clean, page-tagged text.

The old app dumped raw ``page.get_text()`` straight into the answer engine, which
is why answers looked like broken column soup: PDF text arrives with hard line
breaks mid-sentence, hyphenated word splits, and a repeated header/footer on
every page. All of that has to go before retrieval, or every search matches the
running header instead of the content.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field

import fitz  # PyMuPDF


# --------------------------------------------------------------------------
# data model
# --------------------------------------------------------------------------
@dataclass
class Page:
    number: int          # 1-based, matches what the student sees in a reader
    text: str


@dataclass
class Document:
    name: str
    pages: list[Page] = field(default_factory=list)
    raw: bytes | None = None

    @property
    def page_count(self) -> int:
        return len(self.pages)

    @property
    def text(self) -> str:
        return "\n\n".join(p.text for p in self.pages)

    @property
    def word_count(self) -> int:
        return sum(len(p.text.split()) for p in self.pages)


# --------------------------------------------------------------------------
# cleaning
# --------------------------------------------------------------------------
_LIGATURES = {
    "ﬀ": "ff", "ﬁ": "fi", "ﬂ": "fl", "ﬃ": "ffi",
    "ﬄ": "ffl", "‘": "'", "’": "'", "“": '"',
    "”": '"', "–": "-", "—": " - ", " ": " ",
}

# A line that is just "14", "Page 14", "14 | Chapter 2" etc.
_PAGE_NUMBER_LINE = re.compile(r"^\s*(page\s*)?\d{1,4}\s*(\||of\s+\d+)?\s*[\w\s]{0,20}$", re.I)


def _normalise(text: str) -> str:
    for bad, good in _LIGATURES.items():
        text = text.replace(bad, good)
    # a word split across a line break: "photo-\nsynthesis" -> "photosynthesis"
    text = re.sub(r"(\w)-\s*\n\s*(\w)", r"\1\2", text)
    return text


def _join_wrapped_lines(text: str) -> str:
    """Undo PDF hard-wrapping, but keep real paragraph and list breaks."""
    lines = [ln.rstrip() for ln in text.split("\n")]
    out: list[str] = []
    for line in lines:
        stripped = line.strip()
        if not stripped:
            out.append("")
            continue
        starts_new_block = bool(
            re.match(r"^\s*([-*•●]|\d+[.)]|[a-z][.)]|\([a-z0-9]+\))\s+", stripped)
            or re.match(r"^(chapter|section|unit|lesson|exercise|example|figure|table|q\d*[.)])\b",
                        stripped, re.I)
            or (stripped == stripped.upper() and len(stripped) > 3)   # ALL-CAPS heading
        )
        prev = out[-1] if out else ""
        # continue the previous line only when it was cut mid-sentence
        if prev and not starts_new_block and not re.search(r"[.!?:;]$", prev):
            out[-1] = prev + " " + stripped
        else:
            out.append(stripped)
    text = "\n".join(out)
    return re.sub(r"\n{3,}", "\n\n", text)


def _find_repeated_lines(page_texts: list[str], min_pages: int = 4) -> set[str]:
    """Detect running headers/footers: short lines repeating across many pages."""
    if len(page_texts) < min_pages:
        return set()
    counts: Counter[str] = Counter()
    for t in page_texts:
        lines = [ln.strip() for ln in t.split("\n") if ln.strip()]
        # headers/footers live in the first and last few lines of a page
        for ln in set(lines[:3] + lines[-3:]):
            if 3 <= len(ln) <= 90:
                counts[ln] += 1
    threshold = max(3, int(len(page_texts) * 0.5))
    return {ln for ln, c in counts.items() if c >= threshold}


def _strip_boilerplate(text: str, boilerplate: set[str]) -> str:
    kept = []
    for ln in text.split("\n"):
        s = ln.strip()
        if s in boilerplate:
            continue
        if _PAGE_NUMBER_LINE.match(s) and len(s) <= 25:
            continue
        kept.append(ln)
    return "\n".join(kept)


# --------------------------------------------------------------------------
# public API
# --------------------------------------------------------------------------
def load_pdf_bytes(data: bytes, name: str = "document.pdf") -> Document:
    """Parse PDF bytes into a cleaned :class:`Document`."""
    with fitz.open(stream=data, filetype="pdf") as doc:
        raw_pages = [page.get_text("text") for page in doc]

    raw_pages = [_normalise(t) for t in raw_pages]
    boilerplate = _find_repeated_lines(raw_pages)

    pages: list[Page] = []
    for i, t in enumerate(raw_pages, start=1):
        t = _strip_boilerplate(t, boilerplate)
        t = _join_wrapped_lines(t)
        t = re.sub(r"[ \t]{2,}", " ", t).strip()
        if t:
            pages.append(Page(number=i, text=t))

    return Document(name=name, pages=pages, raw=data)


def load_pdf(uploaded_file) -> Document:
    """Parse a Streamlit ``UploadedFile`` (or any file-like object)."""
    data = uploaded_file.read()
    name = getattr(uploaded_file, "name", "document.pdf")
    return load_pdf_bytes(data, name)
