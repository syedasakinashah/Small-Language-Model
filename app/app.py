# --------------------------------------------------------------------------
# Miss RUBI — Personal AI Study Tutor (offline prototype)
#
# Run with:   streamlit run app.py
# Requires:   pip install streamlit pymupdf
# --------------------------------------------------------------------------

import streamlit as st
import fitz  # PyMuPDF
import re
import random
import base64
from datetime import datetime

# ==========================================================================
# PAGE CONFIG
# ==========================================================================
st.set_page_config(
    page_title="Miss RUBI — Your AI Tutor",
    page_icon="💎",
    layout="wide",
    initial_sidebar_state="collapsed",
)

SUBJECT_ICONS = {
    "physics": "📘",
    "mathematics": "📗",
    "math": "📗",
    "biology": "📙",
    "computer science": "💻",
    "chemistry": "🧪",
    "history": "📜",
    "english": "📖",
    "economics": "📈",
    "management": "📊",
}
DEFAULT_ICON = "📚"


def get_icon(name: str) -> str:
    return SUBJECT_ICONS.get(name.strip().lower(), DEFAULT_ICON)


# ==========================================================================
# SESSION STATE
# ==========================================================================
def init_state():
    defaults = {
        "theme": "light",
        "subjects": {},           # name -> {pdfs, notes, quizzes, flashcards}
        "general_messages": [
            {"role": "assistant", "content": (
                "Hey there! I'm **Miss RUBI**, your personal study tutor. 👋\n\n"
                "Upload a PDF and tell me which subject it belongs to (e.g. "
                "*\"Here is my Physics PDF, create a Physics section for it\"*) "
                "and I'll organize everything for you."
            )}
        ],
        "subject_messages": {},   # subject -> [ {role, content} ]
        "selected_subject": None,
        "subject_tab": {},        # subject -> "PDF" | "Notes" | "Quizzes" | "Flashcards"
        "show_add_subject": False,
        "pending_upload_key": 0,  # forces file_uploader reset
        "flip_state": {},         # card key -> bool
        "quiz_open": {},          # quiz key -> bool
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


init_state()


def ensure_subject(name: str):
    if name not in st.session_state.subjects:
        st.session_state.subjects[name] = {
            "pdfs": [], "notes": [], "quizzes": [], "flashcards": []
        }
    if name not in st.session_state.subject_messages:
        st.session_state.subject_messages[name] = []
    if name not in st.session_state.subject_tab:
        st.session_state.subject_tab[name] = "PDF"


# ==========================================================================
# PDF / TEXT HELPERS
# ==========================================================================
def extract_pdf(file) -> tuple[str, int]:
    raw = file.read()
    doc = fitz.open(stream=raw, filetype="pdf")
    text = ""
    for page in doc:
        text += page.get_text() + "\n"
    return text, doc.page_count, raw


def all_text_for(subject: str) -> str:
    return "\n".join(p["text"] for p in st.session_state.subjects[subject]["pdfs"])


def sentences_from(text: str, min_len=40):
    parts = re.split(r"(?<=[.!?])\s+", text.replace("\n", " "))
    return [s.strip() for s in parts if len(s.strip()) >= min_len]


def keyword_score_sentences(text: str, top_n=8):
    words = re.findall(r"\b[a-zA-Z]{4,}\b", text.lower())
    freq = {}
    for w in words:
        freq[w] = freq.get(w, 0) + 1
    sents = sentences_from(text)
    scored = []
    for s in sents:
        sw = re.findall(r"\b[a-zA-Z]{4,}\b", s.lower())
        score = sum(freq.get(w, 0) for w in sw) / (len(sw) + 1)
        scored.append((score, s))
    scored.sort(key=lambda x: -x[0])
    return [s for _, s in scored[:top_n]]


def generate_notes(subject: str):
    text = all_text_for(subject)
    if not text.strip():
        return None
    bullets = keyword_score_sentences(text, top_n=8)
    note = {
        "title": f"Notes — {datetime.now().strftime('%b %d, %I:%M %p')}",
        "bullets": bullets if bullets else ["Not enough content to summarize yet."],
    }
    st.session_state.subjects[subject]["notes"].insert(0, note)
    return note


def generate_quiz(subject: str, n=5):
    text = all_text_for(subject)
    if not text.strip():
        return None
    sents = sentences_from(text, min_len=50)
    if not sents:
        return None
    chosen = random.sample(sents, min(n, len(sents)))
    questions = []
    for s in chosen:
        words = s.split()
        if len(words) > 6:
            blank_idx = random.randint(2, len(words) - 2)
            answer = words[blank_idx].strip(".,;:")
            words[blank_idx] = "_____"
            q_text = " ".join(words)
        else:
            answer = s
            q_text = f"Explain: {s}"
        questions.append({"question": q_text, "answer": answer})
    difficulty = random.choice(["Easy", "Medium", "Hard"])
    quiz = {
        "title": f"Quiz — {datetime.now().strftime('%b %d, %I:%M %p')}",
        "questions": questions,
        "difficulty": difficulty,
    }
    st.session_state.subjects[subject]["quizzes"].insert(0, quiz)
    return quiz


def generate_flashcards(subject: str, n=10):
    text = all_text_for(subject)
    if not text.strip():
        return None
    sents = sentences_from(text, min_len=35)
    cards = []
    for s in sents:
        m = re.match(r"(.{3,60}?)\bis\b(.+)", s, re.IGNORECASE)
        if m:
            front = m.group(1).strip().rstrip(",") + "?"
            back = (m.group(1).strip() + " is" + m.group(2)).strip()
        else:
            front = "What does this describe?"
            back = s
        cards.append({"front": front, "back": back})
        if len(cards) >= n:
            break
    if not cards:
        return None
    st.session_state.subjects[subject]["flashcards"] = cards
    return cards


# ==========================================================================
# CHAT INTENT PARSING (subject creation from natural language)
# ==========================================================================
SECTION_PATTERNS = [
    r"(?:create|make|start|add)\s+(?:a|an)?\s*([a-zA-Z][a-zA-Z &]{1,30}?)\s+section",
    r"section\s+for\s+([a-zA-Z][a-zA-Z &]{1,30})",
]


def detect_subject_name(message: str, known_subjects) -> str | None:
    low = message.lower()
    for name in known_subjects:
        if name.lower() in low:
            return name
    for pat in SECTION_PATTERNS:
        m = re.search(pat, low)
        if m:
            return m.group(1).strip().title()
    return None


# ==========================================================================
# THEME / CSS
# ==========================================================================
def theme_colors():
    if st.session_state.theme == "light":
        return dict(
            bg="#FFF9FB", card="#FFFFFF", soft="#FDEFF4", text="#241522",
            subtext="#7A6470", accent="#8E1E43", accent2="#B23A5E",
            border="rgba(142,30,67,0.15)", chatbubble="#FDEFF4",
        )
    return dict(
        bg="#161018", card="#231A26", soft="#2C1E29", text="#F6EEF2",
        subtext="#C9B8C2", accent="#E85D8A", accent2="#F2A6C0",
        border="rgba(255,255,255,0.08)", chatbubble="#2C1E29",
    )


def inject_css():
    c = theme_colors()
    st.markdown(f"""
    <style>
    .stApp {{ background: {c['bg']}; color: {c['text']}; }}
    h1,h2,h3,h4,p,label,span,.stMarkdown {{ color: {c['text']}; }}
    .block-container {{ padding-top: 1.5rem; max-width: 1100px; }}

    .rubi-logo {{ text-align:center; font-size: 46px; margin-bottom: -6px; }}
    .rubi-title {{
        text-align:center; font-size: 42px; font-weight: 800;
        color:{c['accent']}; font-family: Georgia, 'Times New Roman', serif;
        margin-bottom: 4px;
    }}
    .rubi-subtitle {{
        max-width: 640px; margin: 0 auto 28px auto; text-align:center;
        background:{c['soft']}; color:{c['text']}; padding:18px 26px;
        border-radius: 20px; font-size:15.5px; line-height:1.6;
        border:1px solid {c['border']};
    }}

    .chat-user {{
        background:{c['accent']}; color:#fff; padding:12px 18px;
        border-radius:18px 18px 4px 18px; margin:6px 0; margin-left:30%;
        text-align:left; font-size:14.5px;
    }}
    .chat-ai {{
        background:{c['chatbubble']}; color:{c['text']}; padding:12px 18px;
        border-radius:18px 18px 18px 4px; margin:6px 30% 6px 0;
        border:1px solid {c['border']}; font-size:14.5px; line-height:1.55;
    }}

    .section-label {{
        font-weight:700; font-size:18px; margin: 22px 0 10px 0; color:{c['text']};
    }}

    .subject-card {{
        background:{c['card']}; border:1px solid {c['border']};
        border-radius:18px; padding:20px; text-align:left;
        transition: transform .15s ease, box-shadow .15s ease;
        height: 100%;
    }}
    .subject-card:hover {{ transform: translateY(-3px); box-shadow: 0 8px 24px rgba(0,0,0,0.08); }}
    .subject-card .icon {{ font-size:30px; }}
    .subject-card .name {{ font-weight:700; font-size:17px; margin-top:6px; }}
    .subject-card .count {{ color:{c['subtext']}; font-size:13px; margin-bottom:10px; }}

    .add-card {{
        background:{c['soft']}; border: 2px dashed {c['accent2']};
        border-radius:18px; display:flex; align-items:center; justify-content:center;
        height: 100%; min-height: 130px; font-size:34px; color:{c['accent']};
    }}

    .big-tile {{
        background:{c['card']}; border:1px solid {c['border']}; border-radius:22px;
        padding: 30px 18px; text-align:center; transition: transform .15s ease;
    }}
    .big-tile:hover {{ transform: translateY(-4px); }}
    .big-tile .emoji {{ font-size:38px; }}
    .big-tile .label {{ font-weight:700; font-size:16px; margin-top:8px; }}

    .note-card, .quiz-card, .flash-front, .flash-back {{
        background:{c['card']}; border:1px solid {c['border']}; border-radius:16px;
        padding:18px 20px; margin-bottom:14px;
    }}
    .flash-front, .flash-back {{
        min-height: 120px; display:flex; align-items:center; justify-content:center;
        text-align:center; font-weight:600; font-size:16px;
        background: linear-gradient(135deg, {c['soft']}, {c['card']});
    }}

    div.stButton > button {{
        border-radius: 999px !important; border:1px solid {c['border']} !important;
        background: {c['card']}; color:{c['text']};
    }}
    div.stButton > button:hover {{ border-color:{c['accent']} !important; color:{c['accent']} !important; }}
    </style>
    """, unsafe_allow_html=True)


# ==========================================================================
# HEADER (logo + theme toggle + nav)
# ==========================================================================
def render_header():
    left, mid, right = st.columns([1, 6, 1])
    with right:
        icon = "☀️" if st.session_state.theme == "dark" else "🌙"
        if st.button(icon, key="theme_toggle"):
            st.session_state.theme = "dark" if st.session_state.theme == "light" else "light"
            st.rerun()
    with left:
        if st.session_state.selected_subject is not None:
            if st.button("← Back", key="back_home"):
                st.session_state.selected_subject = None
                st.rerun()


# ==========================================================================
# HOME SCREEN
# ==========================================================================
def render_hero():
    st.markdown('<div class="rubi-logo">💎</div>', unsafe_allow_html=True)
    st.markdown('<div class="rubi-title">Miss RUBI</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="rubi-subtitle">Hey! I\'m your personal AI tutor 👋<br>'
        'Upload PDFs from your different subjects and I\'ll organize everything for you.<br>'
        'You can ask me to generate quizzes, flashcards, and notes from your study material.</div>',
        unsafe_allow_html=True,
    )


def render_home_chat():
    for m in st.session_state.general_messages:
        css = "chat-user" if m["role"] == "user" else "chat-ai"
        st.markdown(f'<div class="{css}">{m["content"]}</div>', unsafe_allow_html=True)

    with st.expander("📎 Attach a PDF to this message", expanded=False):
        uploaded = st.file_uploader(
            "Choose a PDF", type=["pdf"],
            key=f"home_pdf_{st.session_state.pending_upload_key}",
        )

    prompt = st.chat_input("Ask Miss RUBI anything...")

    if prompt:
        st.session_state.general_messages.append({"role": "user", "content": prompt})

        detected = detect_subject_name(prompt, st.session_state.subjects.keys())

        if uploaded is not None:
            text, pages, _raw = extract_pdf(uploaded)
            if detected:
                ensure_subject(detected)
                st.session_state.subjects[detected]["pdfs"].append(
                    {"name": uploaded.name, "text": text, "pages": pages}
                )
                reply = f"Done! I created a **{detected}** section for you and added *{uploaded.name}*. 📚"
            else:
                reply = (
                    "I've received your PDF! Which subject should I file it under? "
                    "Try something like *\"Create a Physics section for this.\"*"
                )
            st.session_state.pending_upload_key += 1
        else:
            if detected and detected not in st.session_state.subjects:
                ensure_subject(detected)
                reply = f"Done! I created a **{detected}** section for you. 📚 Upload a PDF whenever you're ready."
            else:
                # general Q&A across all subjects
                q_words = set(re.findall(r"\b[a-zA-Z]{3,}\b", prompt.lower()))
                best = None
                for subj, data in st.session_state.subjects.items():
                    for pdf in data["pdfs"]:
                        for para in re.split(r"\n\s*\n", pdf["text"]):
                            pw = set(re.findall(r"\b[a-zA-Z]{3,}\b", para.lower()))
                            score = len(q_words & pw)
                            if score > 0 and (best is None or score > best[0]):
                                best = (score, para.strip(), subj)
                if best:
                    reply = f"**From {best[2]}:**\n\n{best[1][:500]}"
                else:
                    reply = (
                        "I couldn't find that in your uploaded material yet. "
                        "Upload a PDF and tell me the subject, and I'll take it from there!"
                    )

        st.session_state.general_messages.append({"role": "assistant", "content": reply})
        st.rerun()


def render_subjects_grid():
    st.markdown('<div class="section-label">📚 My Subjects</div>', unsafe_allow_html=True)
    names = list(st.session_state.subjects.keys())
    n_cols = 4
    cells = names + ["__add__"]
    rows = [cells[i:i + n_cols] for i in range(0, len(cells), n_cols)]

    for row in rows:
        cols = st.columns(n_cols)
        for i, item in enumerate(row):
            with cols[i]:
                if item == "__add__":
                    st.markdown('<div class="add-card">+</div>', unsafe_allow_html=True)
                    if st.button("Add Subject", key="add_subject_btn", use_container_width=True):
                        st.session_state.show_add_subject = True
                        st.rerun()
                else:
                    data = st.session_state.subjects[item]
                    st.markdown(
                        f'<div class="subject-card">'
                        f'<div class="icon">{get_icon(item)}</div>'
                        f'<div class="name">{item}</div>'
                        f'<div class="count">{len(data["pdfs"])} PDF(s)</div>'
                        f'</div>',
                        unsafe_allow_html=True,
                    )
                    if st.button(f"Open {item}", key=f"open_{item}", use_container_width=True):
                        ensure_subject(item)
                        st.session_state.selected_subject = item
                        st.rerun()


@st.dialog("Add a new subject")
def add_subject_dialog():
    st.write("How would you like to add it?")
    tab1, tab2 = st.tabs(["✏️ Create Manually", "💬 Ask Miss RUBI"])
    with tab1:
        name = st.text_input("Subject name", placeholder="e.g. Chemistry")
        if st.button("Create", key="create_manual_subject"):
            if name.strip():
                ensure_subject(name.strip())
                st.session_state.show_add_subject = False
                st.rerun()
            else:
                st.warning("Please enter a subject name.")
    with tab2:
        st.info(
            "Close this dialog and, in the chat box, upload a PDF and say something like:\n\n"
            "*\"Here is my Biology PDF, create a Biology section for it.\"*\n\n"
            "Miss RUBI will detect the subject and set everything up for you."
        )
    if st.button("Cancel", key="cancel_add_subject"):
        st.session_state.show_add_subject = False
        st.rerun()


# ==========================================================================
# SUBJECT PAGE
# ==========================================================================
def render_subject_tiles(subject):
    tiles = [("PDF", "📄"), ("Notes", "📝"), ("Quizzes", "🧠"), ("Flashcards", "🃏")]
    cols = st.columns(4)
    for (label, emoji), col in zip(tiles, cols):
        with col:
            st.markdown(
                f'<div class="big-tile"><div class="emoji">{emoji}</div>'
                f'<div class="label">{label}</div></div>',
                unsafe_allow_html=True,
            )
            if st.button(f"Open {label}", key=f"tile_{subject}_{label}", use_container_width=True):
                st.session_state.subject_tab[subject] = label
                st.rerun()


def render_pdf_section(subject):
    st.markdown('<div class="section-label">📄 PDFs</div>', unsafe_allow_html=True)
    uploaded = st.file_uploader(
        f"Upload another PDF for {subject}", type=["pdf"], key=f"pdf_upload_{subject}"
    )
    if uploaded:
        text, pages, raw = extract_pdf(uploaded)
        st.session_state.subjects[subject]["pdfs"].append(
            {"name": uploaded.name, "text": text, "pages": pages, "raw": raw}
        )
        st.success(f"✅ {uploaded.name} added to {subject}!")
        st.rerun()

    pdfs = st.session_state.subjects[subject]["pdfs"]
    if not pdfs:
        st.info("No PDFs uploaded yet for this subject.")
        return

    for i, pdf in enumerate(pdfs):
        with st.container():
            st.markdown(
                f'<div class="note-card"><b>📄 {pdf["name"]}</b> '
                f'&nbsp;·&nbsp; {pdf.get("pages", "?")} page(s)</div>',
                unsafe_allow_html=True,
            )
            if pdf.get("raw"):
                b64 = base64.b64encode(pdf["raw"]).decode()
                with st.expander("Preview"):
                    st.markdown(
                        f'<iframe src="data:application/pdf;base64,{b64}" '
                        f'width="100%" height="450"></iframe>',
                        unsafe_allow_html=True,
                    )
            else:
                with st.expander("Preview (text)"):
                    st.text(pdf["text"][:1500] + ("..." if len(pdf["text"]) > 1500 else ""))


def render_notes_section(subject):
    st.markdown('<div class="section-label">📝 Notes</div>', unsafe_allow_html=True)
    if st.button("✨ Generate Notes with Miss RUBI", key=f"gen_notes_{subject}"):
        result = generate_notes(subject)
        if result is None:
            st.warning("Upload a PDF first so I have something to summarize!")
        st.rerun()

    notes = st.session_state.subjects[subject]["notes"]
    if not notes:
        st.info("No notes generated yet.")
        return
    for note in notes:
        bullets_html = "".join(f"<li>{b}</li>" for b in note["bullets"])
        st.markdown(
            f'<div class="note-card"><b>{note["title"]}</b><ul>{bullets_html}</ul></div>',
            unsafe_allow_html=True,
        )


def render_quiz_section(subject):
    st.markdown('<div class="section-label">🧠 Quizzes</div>', unsafe_allow_html=True)
    if st.button("✨ Generate New Quiz", key=f"gen_quiz_{subject}"):
        result = generate_quiz(subject)
        if result is None:
            st.warning("Upload a PDF first so I can build questions from it!")
        st.rerun()

    quizzes = st.session_state.subjects[subject]["quizzes"]
    if not quizzes:
        st.info("No quizzes generated yet.")
        return

    for qi, quiz in enumerate(quizzes):
        key = f"{subject}_{qi}"
        st.markdown(
            f'<div class="quiz-card"><b>{quiz["title"]}</b><br>'
            f'{len(quiz["questions"])} questions &nbsp;·&nbsp; Difficulty: {quiz["difficulty"]}</div>',
            unsafe_allow_html=True,
        )
        open_now = st.session_state.quiz_open.get(key, False)
        label = "Close Quiz" if open_now else "Start Quiz"
        if st.button(label, key=f"start_{key}"):
            st.session_state.quiz_open[key] = not open_now
            st.rerun()
        if st.session_state.quiz_open.get(key, False):
            for i, q in enumerate(quiz["questions"]):
                st.write(f"**Q{i + 1}.** {q['question']}")
                st.text_input("Your answer", key=f"ans_{key}_{i}")


def render_flashcards_section(subject):
    st.markdown('<div class="section-label">🃏 Flashcards</div>', unsafe_allow_html=True)
    if st.button("✨ Generate Flashcards", key=f"gen_flash_{subject}"):
        result = generate_flashcards(subject)
        if result is None:
            st.warning("Upload a PDF first so I have material to build flashcards from!")
        st.rerun()

    cards = st.session_state.subjects[subject]["flashcards"]
    if not cards:
        st.info("No flashcards yet.")
        return

    cols = st.columns(3)
    for i, card in enumerate(cards):
        key = f"{subject}_{i}"
        flipped = st.session_state.flip_state.get(key, False)
        with cols[i % 3]:
            content = card["back"] if flipped else card["front"]
            css_class = "flash-back" if flipped else "flash-front"
            st.markdown(f'<div class="{css_class}">{content}</div>', unsafe_allow_html=True)
            if st.button("🔄 Flip", key=f"flip_btn_{key}", use_container_width=True):
                st.session_state.flip_state[key] = not flipped
                st.rerun()


def render_subject_chat(subject):
    st.markdown('<div class="section-label">💬 Ask about ' + subject + '</div>', unsafe_allow_html=True)
    for m in st.session_state.subject_messages[subject]:
        css = "chat-user" if m["role"] == "user" else "chat-ai"
        st.markdown(f'<div class="{css}">{m["content"]}</div>', unsafe_allow_html=True)

    q = st.chat_input(f"Ask something about {subject}...", key=f"chat_{subject}")
    if q:
        st.session_state.subject_messages[subject].append({"role": "user", "content": q})
        text = all_text_for(subject)
        words = set(re.findall(r"\b[a-zA-Z]{3,}\b", q.lower()))
        best = None
        for para in re.split(r"\n\s*\n", text):
            pw = set(re.findall(r"\b[a-zA-Z]{3,}\b", para.lower()))
            score = len(words & pw)
            if score > 0 and (best is None or score > best[0]):
                best = (score, para.strip())
        answer = best[1][:500] if best else f"I couldn't find this in your {subject} PDFs yet."
        st.session_state.subject_messages[subject].append({"role": "assistant", "content": answer})
        st.rerun()


def render_subject_page(subject):
    ensure_subject(subject)
    st.markdown(
        f'<div class="rubi-title" style="font-size:32px;">{get_icon(subject)} {subject}</div>'
        f'<div style="text-align:center;color:var(--subtext);margin-bottom:20px;">'
        f'Your {subject} study space</div>',
        unsafe_allow_html=True,
    )
    render_subject_tiles(subject)
    st.write("")
    tab = st.session_state.subject_tab.get(subject, "PDF")
    if tab == "PDF":
        render_pdf_section(subject)
    elif tab == "Notes":
        render_notes_section(subject)
    elif tab == "Quizzes":
        render_quiz_section(subject)
    elif tab == "Flashcards":
        render_flashcards_section(subject)
    st.divider()
    render_subject_chat(subject)


# ==========================================================================
# MAIN
# ==========================================================================
def main():
    inject_css()
    render_header()

    if st.session_state.show_add_subject:
        add_subject_dialog()

    if st.session_state.selected_subject:
        render_subject_page(st.session_state.selected_subject)
    else:
        render_hero()
        render_home_chat()
        st.divider()
        render_subjects_grid()


if __name__ == "__main__":
    main()
    