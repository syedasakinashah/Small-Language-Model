"""Drive the real Streamlit app headlessly and assert it renders without errors.

Covers the layout the app actually ships: chat on the front page, everything
else behind the profile button. Also pins the regression that mattered most --
uploading a PDF used to re-add the same file on every rerun and spin forever.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from streamlit.testing.v1 import AppTest    # noqa: E402

from pdf import load_pdf_bytes              # noqa: E402
from rag import chunk_document              # noqa: E402
from tests.test_pipeline import make_pdf    # noqa: E402

APP = str(ROOT / "app" / "app.py")


def check(label: str, condition: bool, detail: str = "") -> bool:
    print(f"{'PASS' if condition else 'FAIL'}  {label}" + (f"  [{detail}]" if detail else ""))
    return condition


def err(at) -> str:
    return str(at.exception[0].message) if at.exception else ""


def seed(at):
    """Put one subject in the session, the way the upload handler would."""
    doc = load_pdf_bytes(make_pdf(), "Biology.pdf")
    at.session_state["subjects"] = {
        "Biology": {
            "docs": [doc],
            "chunks": chunk_document(doc, subject="Biology"),
            "notes": [], "quizzes": [], "cards": [], "messages": [],
        }
    }


def main() -> int:
    results = []

    # --- front page is the chatbot ---------------------------------------
    at = AppTest.from_file(APP, default_timeout=90).run()
    results.append(check("app starts with no exception", not at.exception, err(at)))
    results.append(check("front page is the chat view", at.session_state["view"] == "chat"))
    results.append(check("chat input present", len(at.chat_input) == 1))
    results.append(check("hero shown before any conversation",
                         any("Miss RUBI" in m.value for m in at.markdown)))
    results.append(check("no subject grid on the front page",
                         not any("My subjects" in m.value for m in at.markdown)))
    results.append(check("prompts for a first upload",
                         any("upload" in i.value.lower() for i in at.info)))

    # --- profile opens the subject list -----------------------------------
    seed(at)
    at.session_state["view"] = "profile"
    at.run()
    results.append(check("profile view renders", not at.exception, err(at)))
    results.append(check("profile lists subjects",
                         any("My subjects" in m.value for m in at.markdown)))
    results.append(check("subject card shown",
                         any("Biology" in m.value for m in at.markdown)))
    open_button = next((b for b in at.button if b.label == "Open"), None)
    results.append(check("subject has an Open button", open_button is not None))

    # --- opening a subject ------------------------------------------------
    if open_button is not None:
        open_button.click().run()
        results.append(check("opening a subject works", not at.exception, err(at)))
        results.append(check("lands on the subject view",
                             at.session_state["view"] == "subject"
                             and at.session_state["current"] == "Biology"))

    # --- every subject section renders ------------------------------------
    for tab in ["PDF", "Quiz", "Flashcards", "Notes", "Check"]:
        at.session_state["view"] = "subject"
        at.session_state["current"] = "Biology"
        at.session_state["tab"] = tab
        at.run()
        results.append(check(f"section '{tab}' renders", not at.exception, err(at)))

    # --- the three sections the product is built around actually work ------
    at.session_state["tab"] = "Quiz"
    at.run()
    quiz_button = next((b for b in at.button if "Generate a new quiz" in b.label), None)
    if quiz_button is not None:
        quiz_button.click().run()
        results.append(check("quiz generates", not at.exception, err(at)))
        results.append(check("quiz stored",
                             len(at.session_state["subjects"]["Biology"]["quizzes"]) == 1))
    else:
        results.append(check("quiz button present", False))

    at.session_state["tab"] = "Flashcards"
    at.run()
    cards_button = next((b for b in at.button if "Generate flashcards" in b.label), None)
    if cards_button is not None:
        cards_button.click().run()
        results.append(check("flashcards generate", not at.exception, err(at)))
        results.append(check("flashcards stored",
                             len(at.session_state["subjects"]["Biology"]["cards"]) > 0))
    else:
        results.append(check("flashcards button present", False))

    # --- back to chat, and the chat can reach the uploaded material --------
    at.session_state["view"] = "chat"
    at.run()
    results.append(check("returns to chat cleanly", not at.exception, err(at)))
    results.append(check("scope selector offers the subject",
                         any("Biology" in str(s.options) for s in at.selectbox)))

    # --- the regression that mattered: reruns must not re-add the PDF -----
    before = len(at.session_state["subjects"]["Biology"]["docs"])
    at.run()
    at.run()
    after = len(at.session_state["subjects"]["Biology"]["docs"])
    results.append(check("reruns do not duplicate PDFs", before == after, f"{before} -> {after}"))

    passed, total = sum(results), len(results)
    print(f"\n{passed}/{total} checks passed")
    return 0 if passed == total else 1


def test_app_ui():
    assert main() == 0


if __name__ == "__main__":
    sys.exit(main())
