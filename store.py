"""Disk-backed library so a student's subjects survive closing the tab.

Layout under ``data/library/``::

    <subject>/
        subject.json      notes, quizzes, flashcards, chat history
        files/*.pdf       the original uploads

The PDFs are the source of truth: text and chunks are re-derived on load rather
than cached, so improving the parser or the chunker upgrades every existing
library instead of leaving stale extractions behind.
"""

from __future__ import annotations

import json
import re
import shutil
from pathlib import Path

from pdf import load_pdf_bytes
from rag import chunk_document

ROOT = Path(__file__).resolve().parent
LIBRARY = ROOT / "data" / "library"


def _safe_name(name: str) -> str:
    """Filesystem-safe folder name; keeps the subject readable on disk."""
    cleaned = re.sub(r"[^\w\s-]", "", name).strip()
    return re.sub(r"\s+", "_", cleaned) or "subject"


def _subject_dir(subject: str) -> Path:
    return LIBRARY / _safe_name(subject)


def blank_subject() -> dict:
    return {"docs": [], "chunks": [], "notes": [], "quizzes": [], "cards": [], "messages": []}


# --------------------------------------------------------------------------
# writing
# --------------------------------------------------------------------------
class StorageError(RuntimeError):
    """Saving failed - usually a full disk. The app keeps working in memory."""


def save_pdf(subject: str, filename: str, data: bytes) -> Path:
    """Store an uploaded PDF. Raises :class:`StorageError` if it can't be written."""
    files = _subject_dir(subject) / "files"
    try:
        files.mkdir(parents=True, exist_ok=True)

        target = files / Path(filename).name
        stem, suffix = target.stem, target.suffix or ".pdf"
        counter = 2
        while target.exists():                  # never silently overwrite
            target = files / f"{stem} ({counter}){suffix}"
            counter += 1

        target.write_bytes(data)
        return target
    except OSError as exc:
        raise StorageError(str(exc)) from exc


def save_subject(subject: str, data: dict) -> bool:
    """Persist a subject's study material. Returns False if it couldn't be saved.

    A failure here (almost always a full disk) must not interrupt a lesson, so
    the caller carries on with what's in memory.
    """
    directory = _subject_dir(subject)
    payload = {
        "name": subject,
        "notes": data.get("notes", []),
        "quizzes": data.get("quizzes", []),
        "cards": data.get("cards", []),
        "messages": data.get("messages", []),
    }
    try:
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "subject.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        return True
    except OSError:
        return False


def save_chat(messages: list) -> bool:
    """Persist the front-page conversation. Returns False if it couldn't be saved."""
    try:
        LIBRARY.mkdir(parents=True, exist_ok=True)
        (LIBRARY / "chat.json").write_text(
            json.dumps(messages[-200:], ensure_ascii=False, indent=2), encoding="utf-8"
        )
        return True
    except OSError:
        return False


def load_chat() -> list:
    path = LIBRARY / "chat.json"
    if not path.exists():
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, list) else []
    except (json.JSONDecodeError, OSError):
        return []


def delete_subject(subject: str) -> None:
    shutil.rmtree(_subject_dir(subject), ignore_errors=True)


def delete_pdf(subject: str, filename: str) -> None:
    path = _subject_dir(subject) / "files" / Path(filename).name
    path.unlink(missing_ok=True)


# --------------------------------------------------------------------------
# reading
# --------------------------------------------------------------------------
def load_library() -> dict[str, dict]:
    """Rebuild every saved subject from disk. Corrupt entries are skipped, not fatal."""
    library: dict[str, dict] = {}
    if not LIBRARY.exists():
        return library

    for directory in sorted(p for p in LIBRARY.iterdir() if p.is_dir()):
        meta_path = directory / "subject.json"
        meta = {}
        if meta_path.exists():
            try:
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                meta = {}

        name = meta.get("name") or directory.name.replace("_", " ")
        entry = blank_subject()
        entry["notes"] = meta.get("notes", [])
        entry["quizzes"] = meta.get("quizzes", [])
        entry["cards"] = meta.get("cards", [])
        entry["messages"] = meta.get("messages", [])

        for pdf_path in sorted((directory / "files").glob("*.pdf")):
            try:
                doc = load_pdf_bytes(pdf_path.read_bytes(), pdf_path.name)
            except Exception:
                continue        # a damaged file shouldn't hide the rest of the subject
            entry["docs"].append(doc)
            entry["chunks"].extend(chunk_document(doc, subject=name))

        if entry["docs"] or entry["notes"] or entry["messages"]:
            library[name] = entry

    return library


def library_exists() -> bool:
    return LIBRARY.exists() and any(LIBRARY.iterdir())
