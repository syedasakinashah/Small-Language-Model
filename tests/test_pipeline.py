"""End-to-end checks for the ingest -> chunk -> retrieve -> answer pipeline.

Run:  python -m pytest tests -q        (or: python tests/test_pipeline.py)
"""

from __future__ import annotations

import sys
from pathlib import Path

import fitz

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from llm import Tutor, get_backend           # noqa: E402
from pdf import load_pdf_bytes               # noqa: E402
from rag import Retriever, chunk_document    # noqa: E402

PAGES = [
    ("Photosynthesis is the process by which green plants convert light energy "
     "into chemical energy. It takes place in the chloroplast, an organelle that "
     "contains the green pigment chlorophyll. Chlorophyll absorbs light most "
     "strongly in the blue and red parts of the spectrum. The overall reaction "
     "consumes carbon dioxide and water and releases oxygen as a by-product."),
    ("Respiration is the process by which cells release energy from glucose. "
     "Aerobic respiration requires oxygen and produces carbon dioxide and water. "
     "The mitochondrion is the organelle where aerobic respiration occurs. "
     "Anaerobic respiration releases far less energy per molecule of glucose."),
    ("Newton's second law states that the acceleration of an object is directly "
     "proportional to the net force acting on it and inversely proportional to "
     "its mass. This is written as F = ma. A larger mass therefore accelerates "
     "less for the same applied force."),
]


def make_pdf() -> bytes:
    doc = fitz.open()
    for body in PAGES:
        page = doc.new_page()
        page.insert_textbox(fitz.Rect(60, 60, 540, 760), body, fontsize=11)
    data = doc.tobytes()
    doc.close()
    return data


def check(label: str, condition: bool, detail: str = "") -> bool:
    print(f"{'PASS' if condition else 'FAIL'}  {label}" + (f"  [{detail}]" if detail else ""))
    return condition


def main() -> int:
    results = []
    doc = load_pdf_bytes(make_pdf(), "Biology.pdf")

    results.append(check("PDF parsed into pages", doc.page_count == 3, f"{doc.page_count} pages"))
    results.append(check("text extracted", doc.word_count > 100, f"{doc.word_count} words"))
    results.append(check(
        "line wrapping repaired",
        "chloroplast" in doc.text and "chloro plast" not in doc.text,
    ))

    chunks = chunk_document(doc, subject="Biology")
    results.append(check("chunks built", len(chunks) > 0, f"{len(chunks)} chunks"))
    results.append(check(
        "every chunk carries a page citation",
        all(c.page >= 1 and c.doc_name == "Biology.pdf" for c in chunks),
    ))

    retriever = Retriever(chunks)

    hits = retriever.search("where does photosynthesis happen?")
    results.append(check("finds the right passage", bool(hits) and "chloroplast" in hits[0].text.lower(),
                         hits[0].citation if hits else "no hits"))

    hits2 = retriever.search("what is F = ma")
    results.append(check("matches exact formula terms",
                         bool(hits2) and "newton" in hits2[0].text.lower(),
                         hits2[0].citation if hits2 else "no hits"))

    # The old app always returned *something*; returning nothing here is the fix.
    off_topic = retriever.search("zzzqqq unrelated gibberish topic")
    results.append(check("refuses to match unrelated questions", off_topic == []))

    tutor = Tutor(get_backend(force="extractive"))

    answer = tutor.answer("where does photosynthesis happen?", hits)
    results.append(check("answer produced", len(answer) > 50))
    results.append(check("answer cites a source", "p." in answer))

    empty = tutor.answer("unrelated", [])
    results.append(check("says so when it doesn't know", "couldn't find" in empty.lower()))

    context = [type(hits[0])(c, 1.0) for c in retriever.representative_chunks(8)]

    notes = tutor.notes(context)
    results.append(check("notes generated", len(notes) > 80))

    quiz = tutor.quiz(context, n=4)
    results.append(check("quiz generated", len(quiz) > 0, f"{len(quiz)} questions"))
    results.append(check(
        "quiz answers are valid indices",
        all(0 <= q["answer_index"] < len(q["options"]) for q in quiz),
    ))
    results.append(check(
        "quiz options are distinct",
        all(len(set(q["options"])) == len(q["options"]) for q in quiz),
    ))

    cards = tutor.flashcards(context, n=6)
    results.append(check("flashcards generated", len(cards) > 0, f"{len(cards)} cards"))
    results.append(check(
        "flashcard fronts are real questions",
        all("What does this describe" not in c["front"] for c in cards),
    ))

    passed, total = sum(results), len(results)
    print(f"\n{passed}/{total} checks passed")
    return 0 if passed == total else 1


# pytest entry point
def test_pipeline():
    assert main() == 0


if __name__ == "__main__":
    sys.exit(main())
