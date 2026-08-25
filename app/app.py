# --------------------------------------------------------------------------
# Miss RUBI - Personal AI Study Tutor
#
# Run:  streamlit run app/app.py     (from the project root)
#
# Layout: the front page is a chatbot and nothing else. Subjects, PDFs,
# quizzes and flashcards live behind the profile button, top right.
# --------------------------------------------------------------------------

from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

import streamlit as st

# Streamlit puts the script's own folder on sys.path, not the project root,
# so the sibling packages (pdf/, rag/, llm/) need it added explicitly.
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import store                                                 # noqa: E402
from llm import Tutor, get_backend                           # noqa: E402
from llm.tutor import classify_intent, small_talk            # noqa: E402
from pdf import load_pdf_bytes                               # noqa: E402
from rag import Retriever, chunk_document                    # noqa: E402
from rag.retriever import RetrievedChunk                     # noqa: E402
from rag.subject import detect_from_filename, detect_subject  # noqa: E402

st.set_page_config(
    page_title="Miss RUBI · AI Study Tutor",
    page_icon="💎",
    layout="centered",
    initial_sidebar_state="collapsed",
)

SUBJECT_ICONS = {
    "physics": "📘", "mathematics": "📗", "math": "📗", "maths": "📗",
    "biology": "📙", "computer science": "💻", "computer": "💻",
    "chemistry": "🧪", "history": "📜", "english": "📖", "urdu": "📕",
    "economics": "📈", "islamiat": "🕌", "geography": "🌍", "statistics": "📊",
}

ALL_SUBJECTS = "All my subjects"


def subject_icon(name: str) -> str:
    return SUBJECT_ICONS.get(name.strip().lower(), "📚")


# ==========================================================================
# STATE
# ==========================================================================
@st.cache_resource(show_spinner="Loading your library...")
def load_saved_library() -> dict:
    return store.load_library()


def init_state():
    defaults = {
        "view": "chat",          # chat | profile | subject
        "current": None,
        "tab": "PDF",
        "scope": ALL_SUBJECTS,
        "chat": [],
        "api_key": "",
        "urdu": False,
        "pending": None,         # question queued from a suggestion chip
        "quiz_answers": {},
        "quiz_submitted": {},
        "flipped": {},
        "upload_token": 0,       # bumped to reset the uploader after a successful add
    }
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)

    if "subjects" not in st.session_state:
        st.session_state.subjects = load_saved_library()
        st.session_state.chat = store.load_chat()


init_state()


def ensure_subject(name: str) -> str:
    name = name.strip()
    for existing in st.session_state.subjects:           # case-insensitive match
        if existing.lower() == name.lower():
            return existing
    st.session_state.subjects[name] = store.blank_subject()
    return name


def persist(subject: str) -> None:
    data = st.session_state.subjects.get(subject)
    if data is not None:
        store.save_subject(subject, data)


def go(view: str, subject: str | None = None, tab: str | None = None):
    st.session_state.view = view
    if subject is not None:
        st.session_state.current = subject
    if tab is not None:
        st.session_state.tab = tab
    st.rerun()


# ==========================================================================
# RETRIEVAL
# ==========================================================================
@st.cache_resource(show_spinner=False)
def build_retriever(key: str, fingerprint: str, _chunks: list) -> Retriever:
    """Index a set of chunks once; `fingerprint` busts the cache when they change."""
    return Retriever(_chunks)


def _fingerprint(chunks: list) -> str:
    return f"{len(chunks)}:{sum(len(c.text) for c in chunks)}"


def retriever_for(subject: str) -> Retriever:
    chunks = st.session_state.subjects[subject]["chunks"]
    return build_retriever(subject, _fingerprint(chunks), chunks)


def library_chunks() -> list:
    out = []
    for data in st.session_state.subjects.values():
        out.extend(data["chunks"])
    return out


def search_scope(query: str, top_k: int = 5):
    scope = st.session_state.scope
    if scope != ALL_SUBJECTS and scope in st.session_state.subjects:
        return retriever_for(scope).search(query, top_k=top_k)
    chunks = library_chunks()
    if not chunks:
        return []
    return build_retriever("__all__", _fingerprint(chunks), chunks).search(query, top_k=top_k)


def scope_context(n: int = 10):
    """A spread across whatever the chat is currently scoped to."""
    scope = st.session_state.scope
    chunks = (st.session_state.subjects[scope]["chunks"]
              if scope != ALL_SUBJECTS and scope in st.session_state.subjects
              else library_chunks())
    if not chunks:
        return []
    retriever = build_retriever(f"ctx_{scope}", _fingerprint(chunks), chunks)
    return [RetrievedChunk(c, 1.0) for c in retriever.representative_chunks(n)]


def broad_context(subject: str, n: int = 10):
    return [RetrievedChunk(c, 1.0) for c in retriever_for(subject).representative_chunks(n)]


def tutor() -> Tutor:
    return Tutor(get_backend(api_key=st.session_state.api_key or None))


# ==========================================================================
# DESIGN SYSTEM
# ==========================================================================
def inject_css():
    st.markdown("""
    <style>
      :root {
        --rubi:        #B23A5E;
        --rubi-deep:   #8E1E43;
        --rubi-soft:   rgba(178,58,94,.10);
        --ink:         #241722;
        --muted:       rgba(36,23,34,.58);
        --line:        rgba(36,23,34,.12);
        --surface:     #FFFFFF;
      }
      @media (prefers-color-scheme: dark) {
        :root {
          --rubi:      #F08AAC;
          --rubi-deep: #F5B8CD;
          --rubi-soft: rgba(240,138,172,.14);
          --ink:       #F7EFF3;
          --muted:     rgba(247,239,243,.60);
          --line:      rgba(247,239,243,.16);
          --surface:   #221A20;
        }
      }

      .block-container { padding-top: 1rem; padding-bottom: 6rem; max-width: 780px; }
      [data-testid="stSidebar"], #MainMenu, footer, header { display: none; }

      /* ---------- top bar ---------- */
      .rubi-topbar {
        display: flex; align-items: center; gap: 10px;
        padding: 6px 2px 14px 2px; border-bottom: 1px solid var(--line);
        margin-bottom: 18px;
      }
      .rubi-brand { display: flex; align-items: center; gap: 9px; font-weight: 700;
                    font-size: 16px; color: var(--ink); letter-spacing: -.2px; }
      .rubi-brand .dot {
        width: 28px; height: 28px; border-radius: 9px; display: grid;
        place-items: center; font-size: 15px;
        background: linear-gradient(135deg, var(--rubi), var(--rubi-deep));
      }

      /* ---------- hero ---------- */
      .rubi-hero { text-align: center; padding: 44px 0 26px 0; }
      .rubi-hero .gem { font-size: 46px; line-height: 1; }
      .rubi-hero h1 {
        font-size: 30px; font-weight: 800; letter-spacing: -.7px;
        margin: 14px 0 8px 0; color: var(--ink);
      }
      .rubi-hero p { color: var(--muted); font-size: 15px; margin: 0 auto; max-width: 430px;
                     line-height: 1.55; }

      /* ---------- cards ---------- */
      .rubi-card {
        background: var(--surface); border: 1px solid var(--line);
        border-radius: 16px; padding: 16px 18px; margin-bottom: 10px;
      }
      .rubi-card .title { font-weight: 700; font-size: 15px; color: var(--ink); }
      .rubi-card .meta  { color: var(--muted); font-size: 12.5px; margin-top: 3px; }

      .subject-card {
        background: var(--surface); border: 1px solid var(--line);
        border-radius: 18px; padding: 18px; position: relative; overflow: hidden;
      }
      .subject-card::before {
        content: ""; position: absolute; inset: 0 auto 0 0; width: 4px;
        background: linear-gradient(180deg, var(--rubi), var(--rubi-deep));
      }
      .subject-card .icon { font-size: 26px; }
      .subject-card .name { font-weight: 700; font-size: 16px; margin-top: 6px;
                            color: var(--ink); }
      .subject-card .meta { color: var(--muted); font-size: 12.5px; margin-top: 3px; }

      /* ---------- citations ---------- */
      .cite {
        display: inline-block; font-size: 11.5px; font-weight: 600;
        padding: 3px 10px; border-radius: 999px; margin: 3px 6px 0 0;
        background: var(--rubi-soft); color: var(--rubi);
      }

      /* ---------- flashcards ---------- */
      .flash {
        border: 1px solid var(--line); border-radius: 16px; padding: 22px 16px;
        min-height: 132px; display: flex; align-items: center; justify-content: center;
        text-align: center; font-size: 14.5px; line-height: 1.5; color: var(--ink);
        background: var(--surface); font-weight: 650;
      }
      .flash.back {
        background: var(--rubi-soft); border-color: transparent; font-weight: 450;
      }

      /* ---------- section heading ---------- */
      .rubi-h { font-size: 19px; font-weight: 750; color: var(--ink);
                margin: 4px 0 2px 0; letter-spacing: -.3px; }
      .rubi-sub { color: var(--muted); font-size: 13.5px; margin-bottom: 14px; }

      /* ---------- controls ---------- */
      div.stButton > button {
        border-radius: 11px; border: 1px solid var(--line); font-weight: 600;
        transition: border-color .15s ease, color .15s ease;
      }
      div.stButton > button:hover { border-color: var(--rubi); color: var(--rubi); }
      div.stButton > button[kind="primary"] {
        background: var(--rubi); border-color: var(--rubi); color: #fff;
      }
      [data-testid="stChatInput"] textarea { font-size: 15px; }
      [data-testid="stChatMessage"] { background: transparent; padding: 6px 0; }
      .stRadio [role="radiogroup"] { gap: 6px; }
    </style>
    """, unsafe_allow_html=True)


def citations_html(results) -> str:
    seen, out = set(), []
    for r in results:
        if r.citation in seen:
            continue
        seen.add(r.citation)
        out.append(f'<span class="cite">📄 {r.citation}</span>')
    return "".join(out)


# ==========================================================================
# TOP BAR
# ==========================================================================
def render_topbar():
    st.markdown(
        '<div class="rubi-topbar"><div class="rubi-brand">'
        '<span class="dot">💎</span> Miss RUBI</div></div>',
        unsafe_allow_html=True,
    )
    left, right = st.columns([5, 2])

    with left:
        if st.session_state.view != "chat":
            if st.button("← Back to chat", key="back_to_chat"):
                go("chat")

    with right:
        subjects = st.session_state.subjects
        with st.popover("👤  Profile", use_container_width=True):
            st.markdown(f'<div class="rubi-card"><div class="title">My study profile</div>'
                        f'<div class="meta">{len(subjects)} subject(s) · '
                        f'{sum(len(d["docs"]) for d in subjects.values())} PDF(s)</div></div>',
                        unsafe_allow_html=True)

            if st.button("📚 My subjects", use_container_width=True, key="p_subjects"):
                go("profile")
            if st.button("📤 Upload a PDF", use_container_width=True, key="p_upload"):
                go("profile")

            if subjects:
                st.caption("Open a subject")
                for name in subjects:
                    if st.button(f"{subject_icon(name)} {name}", key=f"p_{name}",
                                 use_container_width=True):
                        go("subject", subject=name, tab="PDF")

            st.divider()
            backend = get_backend(api_key=st.session_state.api_key or None)
            st.caption(f"AI engine · **{backend.info.label}**")
            st.text_input("Anthropic API key", type="password", key="api_key",
                          placeholder="sk-ant-...",
                          help="Unlocks explanations in the tutor's own words, Urdu, "
                               "and answer-checking. Kept in this session only.")
            st.toggle("🌐 Also explain in Urdu", key="urdu")

            if st.session_state.chat:
                if st.button("🧹 Clear chat", use_container_width=True, key="p_clear"):
                    st.session_state.chat = []
                    store.save_chat([])
                    go("chat")


# ==========================================================================
# UPLOAD  (opens your own file browser)
# ==========================================================================
AUTO_SUBJECT = "🔎 Detect from the PDF"


def render_upload(default_subject: str | None = None, compact: bool = False):
    names = list(st.session_state.subjects)
    token = st.session_state.upload_token

    with st.form(f"upload_{token}", clear_on_submit=True, border=not compact):
        files = st.file_uploader(
            "Browse your computer for PDFs", type=["pdf"], accept_multiple_files=True,
            key=f"files_{token}",
        )
        if default_subject:
            picked, typed = default_subject, ""
            st.caption(f"Filing under **{default_subject}**")
        else:
            col1, col2 = st.columns([2, 3])
            picked = col1.selectbox("Subject", [AUTO_SUBJECT, "➕ New subject..."] + names)
            typed = col2.text_input(
                "Subject name", placeholder="e.g. Physics",
                disabled=picked != "➕ New subject...",
                help="Leave it on 'Detect' and I'll read the PDF and work it out.",
            )
        submitted = st.form_submit_button("📤 Add to my library", use_container_width=True,
                                          type="primary")

    # Handled outside the form: a form only reports its values on submit, which
    # is what stops the old app's infinite re-upload loop.
    if not submitted:
        return
    if not files:
        st.warning("Choose at least one PDF first.")
        return
    if picked == "➕ New subject..." and not typed.strip():
        st.warning("Give the new subject a name.")
        return

    added, skipped = 0, 0
    for f in files:
        raw = f.getvalue()
        try:
            doc = load_pdf_bytes(raw, f.name)
        except Exception as exc:
            st.error(f"Couldn't read **{f.name}** — is it a valid PDF? ({exc})")
            skipped += 1
            continue
        if doc.word_count < 30:
            st.error(f"**{f.name}** has almost no selectable text. It's probably a "
                     "scanned image, which this app can't read yet.")
            skipped += 1
            continue

        # Each file is filed on its own, so a mixed upload lands in the right places.
        if picked == AUTO_SUBJECT:
            guess, confidence = detect_subject(doc)
            guess = guess or detect_from_filename(f.name)
            if guess:
                subject = ensure_subject(guess)
                st.info(f"📄 {f.name} → **{guess}**"
                        + (f" ({int(confidence * 100)}% confident)" if confidence else ""))
            else:
                subject = ensure_subject("General")
                st.info(f"📄 {f.name}: couldn't tell the subject — filed under **General**.")
        else:
            subject = ensure_subject(typed if picked == "➕ New subject..." else picked)

        data = st.session_state.subjects[subject]
        data["docs"].append(doc)
        data["chunks"].extend(chunk_document(doc, subject=subject))
        added += 1

        # A full disk must not cost the student their lesson: the PDF is already
        # indexed in memory and fully usable, it just won't survive a restart.
        try:
            store.save_pdf(subject, f.name, raw)
            persist(subject)
        except store.StorageError:
            st.warning(f"**{f.name}** is loaded and ready to use, but couldn't be "
                       "saved — your disk is full, so it won't be here after a "
                       "restart. Free up some space and upload it again.")

    if added:
        st.session_state.upload_token += 1
        st.success(f"Added {added} PDF(s) — saved for next time.")
        st.rerun()
    elif skipped:
        st.warning("Nothing was added.")


# ==========================================================================
# 1. FRONT PAGE - the chatbot
# ==========================================================================
def suggestion_chips() -> str | None:
    """Give a new student something to click instead of a blank prompt."""
    subjects = list(st.session_state.subjects)
    if not subjects:
        return None
    first = subjects[0]
    ideas = [
        f"Summarise my {first} notes",
        "What are the key terms I should know?",
        "Explain the main idea simply",
        "What can you do?",
    ]
    cols = st.columns(2)
    for i, idea in enumerate(ideas):
        if cols[i % 2].button(idea, key=f"chip_{i}", use_container_width=True):
            return idea
    return None


def answer_question(question: str):
    """Shared by the chat input and the suggestion chips."""
    st.session_state.chat.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    intent = classify_intent(question)

    with st.chat_message("assistant", avatar="💎"):
        if intent in {"greeting", "thanks", "capability"}:
            answer = small_talk(intent, list(st.session_state.subjects))
            st.markdown(answer)
            cites = ""
        else:
            results = (scope_context(10) if intent == "summarize"
                       else search_scope(question))
            answer = st.write_stream(
                tutor().answer_stream(question, results,
                                      also_urdu=st.session_state.urdu, intent=intent)
            )
            cites = citations_html(results)
            if cites:
                st.markdown(cites, unsafe_allow_html=True)

    st.session_state.chat.append(
        {"role": "assistant", "content": answer, "citations": cites}
    )
    store.save_chat(st.session_state.chat)
    st.rerun()


def render_chat_page():
    has_material = bool(library_chunks())

    if not st.session_state.chat:
        st.markdown("""
        <div class="rubi-hero">
          <div class="gem">💎</div>
          <h1>Hey, I'm Miss RUBI</h1>
          <p>Upload your notes and ask me anything from them.
             Every answer comes with the page it came from.</p>
        </div>
        """, unsafe_allow_html=True)

        if not has_material:
            st.info("📎 Start by uploading a PDF — the **Upload** button is just below.")
        else:
            chip = suggestion_chips()
            if chip:
                answer_question(chip)

    for msg in st.session_state.chat:
        with st.chat_message(msg["role"], avatar="💎" if msg["role"] == "assistant" else None):
            st.markdown(msg["content"])
            if msg.get("citations"):
                st.markdown(msg["citations"], unsafe_allow_html=True)

    # attach + scope sit just above the input, so the conversation stays the focus
    col1, col2 = st.columns([1, 2])
    with col1:
        with st.popover("📎 Upload", use_container_width=True):
            render_upload(compact=True)
    with col2:
        options = [ALL_SUBJECTS] + list(st.session_state.subjects)
        if st.session_state.scope not in options:
            st.session_state.scope = ALL_SUBJECTS
        st.selectbox("Answer from", options, key="scope", label_visibility="collapsed")

    question = st.chat_input("Ask Miss RUBI anything from your notes...")
    if question:
        answer_question(question)


# ==========================================================================
# 2. PROFILE - the subject list
# ==========================================================================
def render_profile_page():
    st.markdown('<div class="rubi-h">📚 My subjects</div>'
                '<div class="rubi-sub">Everything you\'ve uploaded, grouped by subject. '
                'Open one for its PDFs, quizzes and flashcards.</div>',
                unsafe_allow_html=True)

    subjects = st.session_state.subjects
    if not subjects:
        st.info("No subjects yet — upload your first PDF below.")
    else:
        cols = st.columns(2)
        for i, (name, data) in enumerate(subjects.items()):
            with cols[i % 2]:
                st.markdown(
                    f'<div class="subject-card">'
                    f'<div class="icon">{subject_icon(name)}</div>'
                    f'<div class="name">{name}</div>'
                    f'<div class="meta">{len(data["docs"])} PDF(s) · '
                    f'{len(data["quizzes"])} quiz(zes) · {len(data["cards"])} cards</div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )
                if st.button("Open", key=f"open_{name}", use_container_width=True):
                    go("subject", subject=name, tab="PDF")
                st.write("")

    st.divider()
    st.markdown('<div class="rubi-h">📤 Upload a PDF</div>'
                '<div class="rubi-sub">Click below to browse your computer and pick a file.'
                '</div>', unsafe_allow_html=True)
    render_upload()


# ==========================================================================
# 3. SUBJECT - PDF / Quiz / Flashcards / Notes / Check answer
# ==========================================================================
def render_pdf_tab(subject: str):
    data = st.session_state.subjects[subject]
    if not data["docs"]:
        st.info("No PDFs here yet.")
    for doc in data["docs"]:
        st.markdown(
            f'<div class="rubi-card"><div class="title">📄 {doc.name}</div>'
            f'<div class="meta">{doc.page_count} pages · {doc.word_count:,} words · '
            f'saved to disk</div></div>',
            unsafe_allow_html=True,
        )
        with st.expander(f"Preview text · {doc.name}"):
            st.text(doc.text[:2000] + ("..." if len(doc.text) > 2000 else ""))

    st.divider()
    st.caption("Add another PDF to this subject")
    render_upload(subject, compact=True)

    with st.expander("🗑️ Remove this subject"):
        st.caption(f"Permanently deletes **{subject}** — PDFs, notes, quizzes, "
                   "flashcards and history — from disk. This cannot be undone.")
        confirm = st.text_input(f"Type “{subject}” to confirm", key=f"del_{subject}")
        if st.button("Delete permanently", key=f"delbtn_{subject}", type="primary"):
            if confirm.strip().lower() != subject.lower():
                st.warning("The name didn't match — nothing was deleted.")
            else:
                store.delete_subject(subject)
                st.session_state.subjects.pop(subject, None)
                load_saved_library.clear()
                go("profile", subject=None)


def render_quiz_tab(subject: str):
    data = st.session_state.subjects[subject]
    col1, col2 = st.columns([3, 1])
    count = col2.number_input("Questions", 3, 10, 5)
    if col1.button("✨ Generate a new quiz", use_container_width=True, type="primary"):
        with st.spinner("Writing questions..."):
            questions = tutor().quiz(broad_context(subject, 10), n=int(count))
        if not questions:
            st.warning("I couldn't build questions from this material yet.")
        else:
            data["quizzes"].insert(0, {
                "title": f"Quiz · {datetime.now():%d %b, %I:%M %p}",
                "questions": questions,
            })
            persist(subject)
            st.rerun()

    if not data["quizzes"]:
        st.info("No quizzes yet. Generate one to test yourself.")
        return

    for qi, quiz in enumerate(data["quizzes"]):
        key = f"{subject}_{qi}"
        with st.expander(f"{quiz['title']} · {len(quiz['questions'])} questions",
                         expanded=qi == 0):
            answers = st.session_state.quiz_answers.setdefault(key, {})
            for i, q in enumerate(quiz["questions"]):
                st.markdown(f"**Q{i + 1}. {q['question']}**")
                choice = st.radio("Choose one", q["options"], key=f"{key}_{i}",
                                  index=None, label_visibility="collapsed")
                if choice is not None:
                    answers[i] = q["options"].index(choice)

            if st.button("✅ Check my answers", key=f"submit_{key}", type="primary"):
                st.session_state.quiz_submitted[key] = True

            if st.session_state.quiz_submitted.get(key):
                score = 0
                for i, q in enumerate(quiz["questions"]):
                    picked, correct = answers.get(i), q["answer_index"]
                    if picked == correct:
                        score += 1
                        st.success(f"Q{i + 1}: correct — {q['options'][correct]}")
                    else:
                        chosen = q["options"][picked] if picked is not None else "no answer"
                        st.error(f"Q{i + 1}: you said *{chosen}* · "
                                 f"correct answer is **{q['options'][correct]}**")
                    if q.get("explanation"):
                        st.caption(q["explanation"])
                    if q.get("citation"):
                        st.markdown(f'<span class="cite">📄 {q["citation"]}</span>',
                                    unsafe_allow_html=True)
                st.markdown(f"### Score: {score}/{len(quiz['questions'])}")


def render_flashcards_tab(subject: str):
    data = st.session_state.subjects[subject]
    if st.button("✨ Generate flashcards", use_container_width=True, type="primary"):
        with st.spinner("Making cards..."):
            cards = tutor().flashcards(broad_context(subject, 12), n=10)
        if not cards:
            st.warning("I couldn't build flashcards from this material yet.")
        else:
            data["cards"] = cards
            st.session_state.flipped = {}
            persist(subject)
            st.rerun()

    if not data["cards"]:
        st.info("No flashcards yet.")
        return

    cols = st.columns(2)
    for i, card in enumerate(data["cards"]):
        key = f"{subject}_{i}"
        flipped = st.session_state.flipped.get(key, False)
        with cols[i % 2]:
            body = card["back"] if flipped else card["front"]
            st.markdown(f'<div class="flash {"back" if flipped else ""}">{body}</div>',
                        unsafe_allow_html=True)
            if st.button("🔄 Flip", key=f"flip_{key}", use_container_width=True):
                st.session_state.flipped[key] = not flipped
                st.rerun()
            if flipped and card.get("citation"):
                st.markdown(f'<span class="cite">📄 {card["citation"]}</span>',
                            unsafe_allow_html=True)
            st.write("")


def render_notes_tab(subject: str):
    data = st.session_state.subjects[subject]
    if st.button("✨ Generate revision notes", use_container_width=True, type="primary"):
        with st.spinner("Reading your material..."):
            text = tutor().notes(broad_context(subject, 12), also_urdu=st.session_state.urdu)
        data["notes"].insert(0, {
            "title": f"Notes · {datetime.now():%d %b, %I:%M %p}",
            "body": text,
        })
        persist(subject)
        st.rerun()

    if not data["notes"]:
        st.info("No notes yet. Generate a set from your PDFs.")
    for note in data["notes"]:
        with st.expander(note["title"], expanded=note is data["notes"][0]):
            st.markdown(note["body"])


def render_check_tab(subject: str):
    st.caption("Write an answer in your own words and I'll tell you what's missing "
               "and which misconception caused it.")
    with st.form(f"check_{subject}"):
        question = st.text_input("The question you're answering")
        answer = st.text_area("Your answer", height=150)
        submitted = st.form_submit_button("🔍 Check my answer", use_container_width=True,
                                          type="primary")

    if not submitted:
        return
    if not question.strip() or not answer.strip():
        st.warning("Fill in both the question and your answer.")
        return

    results = retriever_for(subject).search(f"{question} {answer}")
    with st.spinner("Marking..."):
        feedback = tutor().analyse_answer(question, answer, results,
                                          also_urdu=st.session_state.urdu)
    st.markdown(feedback)
    cites = citations_html(results)
    if cites:
        st.markdown(cites, unsafe_allow_html=True)


SUBJECT_TABS = {
    "PDF": ("📄 PDF", render_pdf_tab),
    "Quiz": ("🧠 Quiz", render_quiz_tab),
    "Flashcards": ("🃏 Cards", render_flashcards_tab),
    "Notes": ("📝 Notes", render_notes_tab),
    "Check": ("✅ Check", render_check_tab),
}


def render_subject_page(subject: str):
    data = st.session_state.subjects[subject]
    st.markdown(f'<div class="rubi-h">{subject_icon(subject)} {subject}</div>'
                f'<div class="rubi-sub">{len(data["docs"])} PDF(s) · '
                f'{len(data["chunks"])} passages indexed</div>',
                unsafe_allow_html=True)

    keys = list(SUBJECT_TABS)
    labels = [SUBJECT_TABS[k][0] for k in keys]
    if st.session_state.tab not in keys:
        st.session_state.tab = "PDF"
    chosen = st.radio("Section", labels, horizontal=True, label_visibility="collapsed",
                      index=keys.index(st.session_state.tab))
    st.session_state.tab = keys[labels.index(chosen)]
    st.divider()

    if not data["chunks"] and st.session_state.tab != "PDF":
        st.warning("Upload a PDF for this subject first.")
        return
    SUBJECT_TABS[st.session_state.tab][1](subject)


# ==========================================================================
# MAIN
# ==========================================================================
def main():
    inject_css()
    render_topbar()

    view = st.session_state.view
    if view == "subject":
        subject = st.session_state.current
        if subject in st.session_state.subjects:
            render_subject_page(subject)
        else:
            st.session_state.view = "profile"
            render_profile_page()
    elif view == "profile":
        render_profile_page()
    else:
        render_chat_page()


if __name__ == "__main__":
    main()
