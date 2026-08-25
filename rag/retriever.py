"""Retrieval: BM25 keyword scoring, optionally fused with semantic embeddings.

BM25 is the default and carries the app on its own. It is pure numpy: no torch,
no model weights, a few megabytes of RAM. That matters here -- this machine has
under 2 GB free, and loading even a 90 MB embedding model crashes the process
outright, which would take Streamlit down with it.

Semantic embeddings are therefore strictly opt-in via TUTOR_USE_EMBEDDINGS=1.
They improve paraphrased questions ("why do things fall?" never appears verbatim
in a physics book), so turn them on when running somewhere with spare memory.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass

import numpy as np

from .chunker import Chunk

EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def embeddings_enabled() -> bool:
    """Opt-in only. Loading the model on a low-memory box hard-crashes Python."""
    return os.environ.get("TUTOR_USE_EMBEDDINGS", "").strip().lower() in {"1", "true", "yes"}

_STOPWORDS = {
    "the", "a", "an", "and", "or", "but", "if", "of", "to", "in", "on", "for",
    "with", "is", "are", "was", "were", "be", "been", "it", "its", "this",
    "that", "these", "those", "as", "at", "by", "from", "what", "which", "who",
    "how", "why", "when", "where", "does", "do", "did", "can", "could", "would",
    "should", "explain", "tell", "me", "about", "please", "i", "my", "you",
}


@dataclass
class RetrievedChunk:
    chunk: Chunk
    score: float

    @property
    def text(self) -> str:
        return self.chunk.text

    @property
    def citation(self) -> str:
        return self.chunk.citation


# --------------------------------------------------------------------------
# embedding backend (lazy + cached, shared across all subjects)
# --------------------------------------------------------------------------
_embedder = None
_embedder_failed = False


def get_embedder():
    """Load the sentence-transformer once. Returns ``None`` unless opted in."""
    global _embedder, _embedder_failed
    if not embeddings_enabled():
        return None
    if _embedder is not None or _embedder_failed:
        return _embedder
    try:
        from sentence_transformers import SentenceTransformer
        _embedder = SentenceTransformer(EMBED_MODEL_NAME)
    except Exception:
        _embedder_failed = True
        _embedder = None
    return _embedder


def embeddings_available() -> bool:
    return get_embedder() is not None


# --------------------------------------------------------------------------
# keyword scoring (BM25-lite, always available)
# --------------------------------------------------------------------------
def _tokenize(text: str) -> list[str]:
    return [w for w in re.findall(r"[a-z0-9]+", text.lower()) if w not in _STOPWORDS]


class _KeywordIndex:
    def __init__(self, chunks: list[Chunk]):
        self.docs = [_tokenize(c.text) for c in chunks]
        self.doc_len = np.array([len(d) or 1 for d in self.docs], dtype=np.float32)
        self.avg_len = float(self.doc_len.mean()) if len(self.doc_len) else 1.0
        self.df: dict[str, int] = {}
        self.tf: list[dict[str, int]] = []
        for d in self.docs:
            counts: dict[str, int] = {}
            for w in d:
                counts[w] = counts.get(w, 0) + 1
            self.tf.append(counts)
            for w in counts:
                self.df[w] = self.df.get(w, 0) + 1
        self.n = max(1, len(self.docs))

    def score(self, query: str, k1: float = 1.5, b: float = 0.75) -> np.ndarray:
        terms = _tokenize(query)
        scores = np.zeros(self.n, dtype=np.float32)
        if not terms:
            return scores
        for term in set(terms):
            df = self.df.get(term, 0)
            if df == 0:
                continue
            idf = float(np.log(1 + (self.n - df + 0.5) / (df + 0.5)))
            for i in range(self.n):
                f = self.tf[i].get(term, 0)
                if f:
                    denom = f + k1 * (1 - b + b * self.doc_len[i] / self.avg_len)
                    scores[i] += idf * (f * (k1 + 1)) / denom
        return scores


def _minmax(a: np.ndarray) -> np.ndarray:
    if a.size == 0:
        return a
    lo, hi = float(a.min()), float(a.max())
    if hi - lo < 1e-9:
        return np.zeros_like(a)
    return (a - lo) / (hi - lo)


# --------------------------------------------------------------------------
# retriever
# --------------------------------------------------------------------------
class Retriever:
    """Index a set of chunks and search them semantically + lexically."""

    def __init__(self, chunks: list[Chunk], use_embeddings: bool = True):
        self.chunks = chunks
        self.keyword = _KeywordIndex(chunks) if chunks else None
        self.vectors = None
        self.uses_embeddings = False

        if chunks and use_embeddings:
            model = get_embedder()
            if model is not None:
                try:
                    self.vectors = model.encode(
                        [c.text for c in chunks],
                        normalize_embeddings=True,
                        batch_size=32,
                        show_progress_bar=False,
                    )
                    self.uses_embeddings = True
                except Exception:
                    self.vectors = None

    def __len__(self) -> int:
        return len(self.chunks)

    def search(self, query: str, top_k: int = 5, min_score: float = 0.06) -> list[RetrievedChunk]:
        if not self.chunks or self.keyword is None:
            return []

        kw_raw = self.keyword.score(query)
        kw = _minmax(kw_raw)

        if self.vectors is not None:
            model = get_embedder()
            q = model.encode([query], normalize_embeddings=True)
            sem_raw = (self.vectors @ q[0]).astype(np.float32)
            sem = _minmax(sem_raw)
            # semantic leads, keywords keep it honest on exact terms
            combined = 0.65 * sem + 0.35 * kw
            # a genuinely unrelated question should return nothing, not the
            # least-bad chunk -- that was the old app's core failure
            if float(sem_raw.max()) < 0.18:
                return []
        else:
            combined = kw
            # nothing in the question matched any indexed term: say so rather
            # than handing back whichever paragraph scored least badly
            if float(kw_raw.max()) <= 0.0:
                return []

        order = np.argsort(-combined)[: top_k * 3]
        results: list[RetrievedChunk] = []
        seen_pages: set[tuple[str, int]] = set()
        for i in order:
            score = float(combined[i])
            if score < min_score:
                continue
            c = self.chunks[int(i)]
            key = (c.doc_name, c.page)
            # spread citations across pages instead of returning 5 near-duplicates
            if key in seen_pages and len(results) >= 2:
                continue
            seen_pages.add(key)
            results.append(RetrievedChunk(c, score))
            if len(results) >= top_k:
                break
        return results

    def representative_chunks(self, n: int = 12) -> list[Chunk]:
        """A spread of chunks across the material, for notes/quiz generation."""
        if not self.chunks:
            return []
        if len(self.chunks) <= n:
            return list(self.chunks)
        step = len(self.chunks) / n
        return [self.chunks[int(i * step)] for i in range(n)]
