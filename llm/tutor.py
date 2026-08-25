"""Miss RUBI's brain: prompts and study-task logic.

Everything the tutor says is grounded in retrieved passages and carries a page
citation. That is the whole point of the product -- a student has to be able to
check the answer against the book, and a tutor that invents facts is worse than
no tutor at all.
"""

from __future__ import annotations

import json
import random
import re
from typing import Iterator

from rag import RetrievedChunk

from .backends import Backend, ExtractiveBackend

# --------------------------------------------------------------------------
# prompts
# --------------------------------------------------------------------------
PERSONA = """You are Miss RUBI, a warm and patient study tutor for school and \
college students in Pakistan. You explain things clearly, at the level of a \
student who is meeting the topic for the first time."""

ANSWER_SYSTEM = PERSONA + """

You answer ONLY from the SOURCE passages the student's own textbook gives you.

Rules:
- If the sources do not contain the answer, say so plainly and suggest what to \
upload or ask instead. Never invent facts, numbers, or definitions.
- Cite the source for each claim using the exact tag shown, e.g. [Physics.pdf p.12].
- Lead with a direct answer in 1-2 sentences, then explain.
- Use simple language and a concrete example where it helps.
- Keep it focused: a short answer for a short question.
- If the student writes in Urdu or Roman Urdu, reply in the same language."""

URDU_SUFFIX = """

After your English explanation, add a section headed "**اردو میں وضاحت**" with \
the same explanation in simple Urdu."""

NOTES_SYSTEM = PERSONA + """

Write revision notes from the SOURCE passages only.

Format:
- A one-line summary of what the material covers.
- 5-9 bullet points, each a complete, self-contained fact a student could revise from.
- Each bullet ends with its citation tag.
- Then "**Key terms**": 3-6 term - short definition pairs found in the sources.
Do not add anything the sources do not state."""

QUIZ_SYSTEM = PERSONA + """

Write quiz questions from the SOURCE passages only.

Return ONLY a JSON array, no prose around it. Each element:
{"question": "...", "options": ["A", "B", "C", "D"], "answer_index": 0,
 "explanation": "why that option is right", "citation": "File.pdf p.3"}

Rules:
- Questions must be answerable from the sources alone.
- The three wrong options must be plausible and related to the topic, never silly.
- Vary the position of the correct answer.
- Test understanding, not word-for-word recall."""

FLASHCARD_SYSTEM = PERSONA + """

Write flashcards from the SOURCE passages only.

Return ONLY a JSON array, no prose around it. Each element:
{"front": "a question or term", "back": "the answer, 1-3 sentences",
 "citation": "File.pdf p.3"}

Rules:
- The front is a real question or a term to define, never "What does this describe?".
- The back must be complete enough to learn from on its own."""

ANALYSIS_SYSTEM = PERSONA + """

The student has answered a question. Judge their answer against the SOURCE \
passages and help them improve.

Structure your reply exactly like this:
**Verdict:** Correct / Partly correct / Incorrect
**What you got right:** ... (skip if nothing)
**What's missing or wrong:** ... (skip if nothing)
**The misconception:** name the underlying misunderstanding, if there is one.
**Model answer:** a short correct answer, with citations.

Be encouraging but honest. Never mark a wrong answer as correct."""


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------
_URDU_RANGE = re.compile(r"[؀-ۿ]")
_ROMAN_URDU_HINTS = {
    "kya", "kyun", "kyu", "hai", "hain", "samjhao", "batao", "mujhe", "mera",
    "kaise", "kaisay", "matlab", "tafseel", "urdu", "smjhao", "bata", "kar",
}


def wants_urdu(text: str) -> bool:
    if _URDU_RANGE.search(text):
        return True
    words = set(re.findall(r"[a-z]+", text.lower()))
    return len(words & _ROMAN_URDU_HINTS) >= 2


def build_sources_block(results: list[RetrievedChunk]) -> str:
    parts = []
    for r in results:
        parts.append(f"[{r.citation}]\n{r.text}")
    return "\n\n---\n\n".join(parts)


# --------------------------------------------------------------------------
# intent -- so "hello" isn't answered with "I couldn't find that in your PDFs"
# --------------------------------------------------------------------------
_GREETINGS = {
    "hi", "hello", "hey", "salam", "salaam", "assalam", "assalamualaikum",
    "aoa", "hy", "yo", "good morning", "good evening", "good afternoon",
}
_THANKS = {"thanks", "thank you", "thankyou", "shukriya", "jazakallah", "ty", "thx"}
_CAPABILITY_HINTS = {
    "what can you do", "who are you", "what are you", "how do you work",
    "help", "what do you do", "kya kar sakti", "tum kon ho", "how to use",
}
_SUMMARY_HINTS = {
    "summarise", "summarize", "summary", "overview", "what is this about",
    "what's this about", "khulasa", "main points", "key points", "what does this cover",
}


def classify_intent(text: str) -> str:
    """Route a message: chit-chat, a question about the tutor, or real study work."""
    cleaned = re.sub(r"[^\w\s']", " ", text.lower()).strip()
    cleaned = re.sub(r"\s+", " ", cleaned)
    if not cleaned:
        return "greeting"

    if cleaned in _GREETINGS or (len(cleaned.split()) <= 3
                                 and any(cleaned.startswith(g) for g in _GREETINGS)):
        return "greeting"
    if cleaned in _THANKS or (len(cleaned.split()) <= 3
                              and any(t in cleaned for t in _THANKS)):
        return "thanks"
    if any(h in cleaned for h in _CAPABILITY_HINTS):
        return "capability"
    if any(h in cleaned for h in _SUMMARY_HINTS):
        return "summarize"
    return "study"


def small_talk(intent: str, subjects: list[str]) -> str:
    """Replies that need no retrieval at all."""
    have = ", ".join(f"**{s}**" for s in subjects[:6]) if subjects else ""

    if intent == "greeting":
        if subjects:
            return (f"Hello! 👋 I'm **Miss RUBI**, your study tutor.\n\n"
                    f"I've read your notes on {have}. Ask me anything from them — "
                    f"I'll explain it and show you exactly which page it came from.")
        return ("Hello! 👋 I'm **Miss RUBI**, your study tutor.\n\n"
                "Upload a PDF of your notes or textbook using the **📎 Upload** button "
                "below, and I'll answer questions from it, make quizzes, and build "
                "flashcards for you.")

    if intent == "thanks":
        return "You're very welcome. Ask me anything else whenever you're ready. 😊"

    lines = [
        "I'm **Miss RUBI**. I only teach from *your* uploaded material, and I show "
        "you the page for everything I say — so you can always check me.\n",
        "**What I can do**",
        "- 💬 Answer questions from your notes, in English or Urdu",
        "- 📝 Write revision notes from a chapter",
        "- 🧠 Build quizzes and mark them, explaining every answer",
        "- 🃏 Make flashcards from the real definitions in your book",
        "- ✅ Check an answer you've written and name the misconception behind it",
    ]
    if subjects:
        lines.append(f"\nRight now I've read your {have}. What would you like to start with?")
    else:
        lines.append("\nUpload a PDF with the **📎 Upload** button to get started.")
    return "\n".join(lines)


def _first_json_array(text: str):
    """Pull the first JSON array out of a model reply, tolerating stray prose."""
    match = re.search(r"\[.*\]", text, re.S)
    if not match:
        return None
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        # models sometimes leave a trailing comma
        cleaned = re.sub(r",\s*([\]}])", r"\1", match.group(0))
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            return None


# --------------------------------------------------------------------------
# extractive fallbacks -- used when no generative model is available
# --------------------------------------------------------------------------
_STOP = {
    "the", "a", "an", "and", "or", "of", "to", "in", "on", "for", "with", "is",
    "are", "was", "were", "be", "it", "its", "this", "that", "as", "at", "by",
    "from", "what", "which", "who", "how", "why", "when", "where", "does", "do",
    "did", "can", "explain", "tell", "me", "about", "please", "i", "my", "you",
}


def _keywords(text: str) -> set[str]:
    """Content words. Keeps 2-character tokens so 'ma' in 'F = ma' survives."""
    return {w for w in re.findall(r"[a-z0-9]{2,}", text.lower()) if w not in _STOP}


# What a question is *asking for* decides which sentence answers it. "Where does
# photosynthesis happen?" wants the sentence naming the chloroplast, not the one
# defining photosynthesis -- matching on keywords alone gets this backwards.
_QUESTION_CUES = [
    (re.compile(r"^\s*where\b|\bwhere (?:does|do|is|are)\b", re.I),
     re.compile(r"\b(in|inside|within|takes? place|occurs?|located|found|happens?)\b", re.I)),
    (re.compile(r"^\s*why\b|\bwhy (?:does|do|is|are)\b", re.I),
     re.compile(r"\b(because|since|due to|so that|as a result|therefore|reason)\b", re.I)),
    (re.compile(r"^\s*how\b|\bhow (?:does|do|is|are|can)\b", re.I),
     re.compile(r"\b(by|through|process|method|steps?|first|then|using)\b", re.I)),
    (re.compile(r"\b(what is|what are|define|definition|meaning of)\b", re.I),
     re.compile(r"\b(is|are|means|refers to|defined as|called|known as)\b", re.I)),
]


def _rank_sentences(question: str, results: list[RetrievedChunk], limit: int = 4):
    """Pick the individual sentences that answer the question, not whole blocks.

    Quoting a 700-character block was the old behaviour and it read like a data
    dump. Ranking sentence by sentence is what makes the reply feel answered.
    """
    from rag.chunker import _split_sentences

    q_words = _keywords(question)
    if not q_words:
        # Symbol-only questions ("F = ma") leave nothing after filtering.
        q_words = set(re.findall(r"[a-z0-9]+", question.lower()))

    cue = next((c for pattern, c in _QUESTION_CUES if pattern.search(question)), None)

    scored: list[tuple[float, str, str]] = []
    seen: set[str] = set()

    for rank, r in enumerate(results):
        for sent in _split_sentences(r.text):
            words = sent.split()
            if not (5 <= len(words) <= 60):
                continue
            key = sent.lower()[:70]
            if key in seen:
                continue
            seen.add(key)

            s_words = _keywords(sent)
            overlap = len(q_words & s_words)
            # a bare formula may not tokenise, so check the raw string too
            literal = sum(1 for w in q_words if len(w) <= 3 and w in sent.lower())
            answers_cue = cue is not None and bool(cue.search(sent))

            # "It takes place in the chloroplast" answers "where does
            # photosynthesis happen?" while sharing no words with it. Inside the
            # best-matching passage a pronoun almost always refers back to the
            # thing being asked about, so cue-matching sentences stay in.
            if not overlap and not literal and not (answers_cue and rank == 0):
                continue

            score = (overlap + 0.5 * literal) / (len(q_words) ** 0.5 + 1)
            score += 0.25 * (len(results) - rank) / len(results)      # trust retrieval order
            if answers_cue:
                score += 0.6                                          # answers what was asked
            elif cue is None and re.search(r"\b(is|are|means|refers to)\b", sent, re.I):
                score += 0.2
            scored.append((score, sent.strip(), r.citation))

    scored.sort(key=lambda x: -x[0])
    return scored[:limit]


def _extractive_answer(question: str, results: list[RetrievedChunk]) -> str:
    if not results:
        return (
            "I couldn't find that in your uploaded material.\n\n"
            "Two things that usually help:\n"
            "- Use a word that appears in the book (e.g. *photosynthesis* rather than *food making*)\n"
            "- Upload the PDF for this chapter with the **📎 Upload** button"
        )

    picked = _rank_sentences(question, results)
    if not picked:
        # Retrieval found the right area but no sentence matched the wording.
        best = results[0]
        snippet = best.text.strip()
        if len(snippet) > 600:
            snippet = snippet[:600].rsplit(" ", 1)[0] + "..."
        return (f"The closest thing in your notes is this, from **{best.citation}**:\n\n"
                f"> {snippet}\n\n"
                f"Try asking about a specific term from that passage.")

    lead = picked[0][1]
    parts = [f"**{lead}**", ""]

    if len(picked) > 1:
        parts.append("More from your notes:")
        for _, sent, citation in picked[1:]:
            parts.append(f"- {sent} `{citation}`")
        parts.append("")

    pages = []
    for _, _, citation in picked:
        if citation not in pages:
            pages.append(citation)
    parts.append("📖 Read it in full at " + ", ".join(f"**{p}**" for p in pages) + ".")
    return "\n".join(parts)


def _extractive_summary(results: list[RetrievedChunk]) -> str:
    if not results:
        return ("There's nothing uploaded yet. Add a PDF with the **📎 Upload** button "
                "and I'll summarise it for you.")
    body = _extractive_notes(results)
    return body + "\n\n*Ask me about any of these points and I'll go deeper.*"


def _extractive_notes(results: list[RetrievedChunk]) -> str:
    if not results:
        return "Upload a PDF first so I have something to summarise."
    from rag.chunker import _split_sentences

    bullets, seen = [], set()
    for r in results:
        for sent in _split_sentences(r.text):
            words = sent.split()
            if not (8 <= len(words) <= 45):
                continue
            key = sent.lower()[:60]
            if key in seen:
                continue
            seen.add(key)
            bullets.append(f"- {sent}  `{r.citation}`")
            break                      # one strong line per passage keeps it varied
    body = "\n".join(bullets[:9])
    return f"**Revision notes from your material**\n\n{body}"


def _extractive_quiz(results: list[RetrievedChunk], n: int) -> list[dict]:
    """Cloze questions with distractors drawn from other real terms in the text."""
    from rag.chunker import _split_sentences

    pool: list[tuple[str, str]] = []
    for r in results:
        for sent in _split_sentences(r.text):
            if 10 <= len(sent.split()) <= 35:
                pool.append((sent, r.citation))

    vocabulary = sorted({
        w for sent, _ in pool
        for w in re.findall(r"\b[A-Za-z]{5,}\b", sent)
    })
    if len(pool) < 1 or len(vocabulary) < 4:
        return []

    questions = []
    for sent, citation in random.sample(pool, min(n, len(pool))):
        candidates = [w for w in re.findall(r"\b[A-Za-z]{5,}\b", sent)
                      if w.lower() not in {"which", "there", "these", "those", "where"}]
        if not candidates:
            continue
        answer = random.choice(candidates)
        distractors = random.sample(
            [w for w in vocabulary if w.lower() != answer.lower()], 3
        )
        options = distractors + [answer]
        random.shuffle(options)
        questions.append({
            "question": sent.replace(answer, "______", 1),
            "options": options,
            "answer_index": options.index(answer),
            "explanation": f"The passage reads: “{sent}”",
            "citation": citation,
        })
    return questions


def _extractive_flashcards(results: list[RetrievedChunk], n: int) -> list[dict]:
    from rag.chunker import _split_sentences

    cards = []
    definition = re.compile(
        r"^(?:The\s+|A\s+|An\s+)?([A-Z][\w\s\-]{2,50}?)\s+(?:is|are|refers to|means)\s+(.{20,})$"
    )
    for r in results:
        for sent in _split_sentences(r.text):
            m = definition.match(sent.strip())
            if not m:
                continue
            term = m.group(1).strip()
            cards.append({
                "front": f"What is {term}?",
                "back": sent.strip(),
                "citation": r.citation,
            })
            if len(cards) >= n:
                return cards
    return cards


# --------------------------------------------------------------------------
# the tutor
# --------------------------------------------------------------------------
class Tutor:
    def __init__(self, backend: Backend):
        self.backend = backend

    @property
    def is_extractive(self) -> bool:
        return isinstance(self.backend, ExtractiveBackend)

    # -- question answering -------------------------------------------------
    def answer_stream(
        self,
        question: str,
        results: list[RetrievedChunk],
        also_urdu: bool = False,
        intent: str = "study",
    ) -> Iterator[str]:
        if intent == "summarize":
            if self.is_extractive:
                yield _extractive_summary(results)
                return
            if results:
                yield from self.backend.stream(
                    NOTES_SYSTEM + (URDU_SUFFIX if also_urdu else ""),
                    f"SOURCE PASSAGES:\n\n{build_sources_block(results)}",
                    max_tokens=1600,
                )
                return

        if self.is_extractive:
            yield _extractive_answer(question, results)
            return

        if not results:
            yield ("I couldn't find that in your uploaded material.\n\n"
                   "Try a word that appears in the book, or upload the PDF for this "
                   "chapter with the **📎 Upload** button.")
            return

        system = ANSWER_SYSTEM + (URDU_SUFFIX if (also_urdu or wants_urdu(question)) else "")
        user = (
            f"SOURCE PASSAGES:\n\n{build_sources_block(results)}\n\n"
            f"STUDENT'S QUESTION: {question}"
        )
        yield from self.backend.stream(system, user, max_tokens=1400)

    def answer(self, question: str, results: list[RetrievedChunk], also_urdu: bool = False) -> str:
        return "".join(self.answer_stream(question, results, also_urdu))

    # -- notes --------------------------------------------------------------
    def notes(self, results: list[RetrievedChunk], also_urdu: bool = False) -> str:
        if not results:
            return "Upload a PDF first so I have something to summarise."
        if self.is_extractive:
            return _extractive_notes(results)
        system = NOTES_SYSTEM + (URDU_SUFFIX if also_urdu else "")
        return self.backend.generate(
            system, f"SOURCE PASSAGES:\n\n{build_sources_block(results)}", max_tokens=1600
        )

    # -- quiz ---------------------------------------------------------------
    def quiz(self, results: list[RetrievedChunk], n: int = 5) -> list[dict]:
        if not results:
            return []
        if self.is_extractive:
            return _extractive_quiz(results, n)

        raw = self.backend.generate(
            QUIZ_SYSTEM,
            f"SOURCE PASSAGES:\n\n{build_sources_block(results)}\n\n"
            f"Write exactly {n} questions.",
            max_tokens=2000,
        )
        data = _first_json_array(raw)
        if not data:
            return _extractive_quiz(results, n)      # never leave the student empty-handed

        cleaned = []
        for item in data:
            try:
                options = [str(o) for o in item["options"]][:4]
                idx = int(item["answer_index"])
                if not options or not (0 <= idx < len(options)):
                    continue
                cleaned.append({
                    "question": str(item["question"]),
                    "options": options,
                    "answer_index": idx,
                    "explanation": str(item.get("explanation", "")),
                    "citation": str(item.get("citation", "")),
                })
            except (KeyError, TypeError, ValueError):
                continue
        return cleaned or _extractive_quiz(results, n)

    # -- flashcards ---------------------------------------------------------
    def flashcards(self, results: list[RetrievedChunk], n: int = 10) -> list[dict]:
        if not results:
            return []
        if self.is_extractive:
            return _extractive_flashcards(results, n)

        raw = self.backend.generate(
            FLASHCARD_SYSTEM,
            f"SOURCE PASSAGES:\n\n{build_sources_block(results)}\n\n"
            f"Write exactly {n} flashcards.",
            max_tokens=1800,
        )
        data = _first_json_array(raw)
        if not data:
            return _extractive_flashcards(results, n)

        cleaned = []
        for item in data:
            try:
                cleaned.append({
                    "front": str(item["front"]),
                    "back": str(item["back"]),
                    "citation": str(item.get("citation", "")),
                })
            except (KeyError, TypeError):
                continue
        return cleaned or _extractive_flashcards(results, n)

    # -- answer analysis ----------------------------------------------------
    def analyse_answer(
        self,
        question: str,
        student_answer: str,
        results: list[RetrievedChunk],
        also_urdu: bool = False,
    ) -> str:
        if self.is_extractive:
            if not results:
                return "I need the source material to check this answer."
            return (
                "**I can't grade answers without an AI engine turned on.**\n\n"
                "Here's the relevant passage so you can compare it yourself:\n\n"
                + _extractive_answer(question, results)
            )
        system = ANALYSIS_SYSTEM + (URDU_SUFFIX if also_urdu else "")
        user = (
            f"SOURCE PASSAGES:\n\n{build_sources_block(results)}\n\n"
            f"QUESTION: {question}\n\nSTUDENT'S ANSWER: {student_answer}"
        )
        return self.backend.generate(system, user, max_tokens=1200)
