"""Saving and reloading the library, plus subject auto-detection.

The point of these checks: a student closes the tab, comes back tomorrow, and
their subjects, PDFs, notes and chat history are still there.
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import fitz                                      # noqa: E402

import store                                     # noqa: E402
from pdf import load_pdf_bytes                   # noqa: E402
from rag import chunk_document                   # noqa: E402
from rag.subject import detect_from_filename, detect_subject   # noqa: E402
from tests.test_pipeline import make_pdf         # noqa: E402


def check(label: str, condition: bool, detail: str = "") -> bool:
    print(f"{'PASS' if condition else 'FAIL'}  {label}" + (f"  [{detail}]" if detail else ""))
    return condition


def pdf_of(text: str, pages: int = 2) -> bytes:
    doc = fitz.open()
    for _ in range(pages):
        page = doc.new_page()
        page.insert_textbox(fitz.Rect(50, 50, 545, 780), text, fontsize=10)
    data = doc.tobytes()
    doc.close()
    return data


BIOLOGY = ("The cell is the basic unit of life. Photosynthesis occurs in the "
           "chloroplast, where chlorophyll absorbs light. Enzymes catalyse "
           "reactions. Mitochondria carry out respiration. Chromosomes carry "
           "genetic information inside the nucleus of every organism.")
PHYSICS = ("Velocity is the rate of change of displacement. Acceleration relates "
           "to force through Newton's second law. Momentum is conserved in "
           "collisions. Friction opposes motion. A circuit carries current "
           "measured in ampere and driven by voltage.")


def main() -> int:
    results = []

    # Run against a scratch library so a real one is never touched.
    original = store.LIBRARY
    scratch = ROOT / "data" / "_test_library"
    shutil.rmtree(scratch, ignore_errors=True)
    store.LIBRARY = scratch

    try:
        # --- detection ----------------------------------------------------
        bio_doc = load_pdf_bytes(pdf_of(BIOLOGY), "chapter_4_final.pdf")
        subject, confidence = detect_subject(bio_doc)
        results.append(check("detects Biology from content", subject == "Biology",
                             f"{subject} @ {confidence}"))

        phy_doc = load_pdf_bytes(pdf_of(PHYSICS), "notes.pdf")
        results.append(check("detects Physics from content",
                             detect_subject(phy_doc)[0] == "Physics"))

        noise = load_pdf_bytes(
            pdf_of("Lorem ipsum filler text with no academic vocabulary at all."), "x.pdf")
        results.append(check("declines to guess on unrelated text",
                             detect_subject(noise)[0] is None))
        results.append(check("falls back to the filename",
                             detect_from_filename("physics_ch4.pdf") == "Physics"))

        # --- save ----------------------------------------------------------
        raw = make_pdf()
        doc = load_pdf_bytes(raw, "Biology.pdf")
        store.save_pdf("Biology", "Biology.pdf", raw)
        store.save_subject("Biology", {
            "docs": [doc],
            "chunks": chunk_document(doc, subject="Biology"),
            "notes": [{"title": "Notes 1", "body": "Photosynthesis happens in the chloroplast."}],
            "quizzes": [],
            "cards": [{"front": "What is a chloroplast?", "back": "An organelle.", "citation": "Biology.pdf p.1"}],
            "messages": [{"role": "user", "content": "where does photosynthesis happen?"}],
        })
        results.append(check("PDF written to disk",
                             (scratch / "Biology" / "files" / "Biology.pdf").exists()))

        # --- reload (the "next day" path) -----------------------------------
        library = store.load_library()
        results.append(check("subject restored", "Biology" in library, str(list(library))))

        entry = library.get("Biology", {})
        results.append(check("PDF restored", len(entry.get("docs", [])) == 1))
        results.append(check("chunks rebuilt from the PDF", len(entry.get("chunks", [])) > 0,
                             f"{len(entry.get('chunks', []))} chunks"))
        results.append(check("chunks keep their subject tag",
                             all(c.subject == "Biology" for c in entry.get("chunks", []))))
        results.append(check("notes restored", len(entry.get("notes", [])) == 1))
        results.append(check("flashcards restored", len(entry.get("cards", [])) == 1))
        results.append(check("chat history restored", len(entry.get("messages", [])) == 1))

        # --- duplicate filenames must not overwrite --------------------------
        store.save_pdf("Biology", "Biology.pdf", raw)
        saved = list((scratch / "Biology" / "files").glob("*.pdf"))
        results.append(check("same filename saved alongside, not over",
                             len(saved) == 2, str([p.name for p in saved])))

        # --- delete -----------------------------------------------------------
        store.delete_subject("Biology")
        results.append(check("subject deleted from disk",
                             "Biology" not in store.load_library()))

    finally:
        shutil.rmtree(scratch, ignore_errors=True)
        store.LIBRARY = original

    passed, total = sum(results), len(results)
    print(f"\n{passed}/{total} checks passed")
    return 0 if passed == total else 1


def test_store():
    assert main() == 0


if __name__ == "__main__":
    sys.exit(main())
