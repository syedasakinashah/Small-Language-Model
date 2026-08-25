# 💎 Miss RUBI — Personal AI Study Tutor

Upload your textbooks as PDFs. Ask questions in English or Urdu. Every answer
quotes the page it came from, so you can always check it against the book.

- **Ask** — questions answered from *your* material, with page citations
- **Notes** — revision notes generated from the uploaded chapters
- **Quiz** — multiple-choice questions with explanations, marked instantly
- **Flashcards** — flip cards built from the real definitions in your text
- **Check my answer** — write an answer, get told what's missing and *why*

---

## Quick start

Double-click **`run.bat`**, or from a terminal in this folder:

```bat
.venv\Scripts\python.exe -m streamlit run app/app.py
```

Then open <http://localhost:8501>.

First time on a new machine:

```bat
python -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
```

---

## Turning on real AI answers

The app runs out of the box with a **built-in reader** that finds and quotes the
right passage. It never invents anything, but it quotes rather than explains.

For genuine explanations, Urdu translation, and answer-checking, connect Claude:

1. Get a key at <https://console.anthropic.com>
2. Open the app → sidebar → **⚙️ AI engine** → paste the key

The key lives in that browser session only — it is never written to disk. To set
it permanently instead:

```bat
setx ANTHROPIC_API_KEY "sk-ant-..."
```

### Engines, in the order the app picks them

| Engine | Needs | Gives you |
|---|---|---|
| **Claude API** | API key + internet | Full explanations, Urdu, answer analysis |
| **Offline AI model** | ~2.5 GB disk, ~3 GB free RAM | Same features, no internet, slower |
| **Built-in reader** | nothing | Finds and quotes the right passage |

The sidebar always shows which engine is live, so a demo never silently degrades.

> **On this machine:** the offline model is disabled. There is 7.8 GB of RAM with
> only ~1.4 GB free, and loading the model weights aborts the Python process —
> which would kill the whole app mid-demo. Use the Claude API here. On a machine
> with more free memory, run `python scripts/setup_models.py`, then set
> `TUTOR_USE_EMBEDDINGS=1` and `TUTOR_USE_LOCAL_LLM=1`.

---

## How it works

```
PDF  →  clean text      pdf/loader.py    de-hyphenate, rejoin wrapped lines,
                                         strip repeated headers and footers
     →  chunks          rag/chunker.py   ~180-word overlapping passages,
                                         each tagged with its page number
     →  retrieval       rag/retriever.py BM25 keyword scoring (+ optional
                                         semantic embeddings), returns nothing
                                         when the question isn't covered
     →  answer          llm/tutor.py     grounded prompts; the model may only
                                         use the retrieved passages
```

Two design rules do most of the work:

**Never answer from nothing.** If retrieval finds no relevant passage, the tutor
says so instead of returning the least-bad paragraph. A confident wrong answer is
worse for a student than "I don't know".

**Always cite.** Every claim carries a `File.pdf · p.12` tag, so a student can
verify it in seconds.

---

## Project layout

```
app/app.py            Streamlit UI
pdf/loader.py         PDF → clean, page-tagged text
rag/chunker.py        text → overlapping passages with citations
rag/retriever.py      BM25 + optional embeddings
llm/backends.py       Claude / local model / built-in reader
llm/tutor.py          prompts and study-task logic
scripts/setup_models.py   one-time offline-model download
tests/                pipeline and headless UI tests
```

## Tests

```bat
.venv\Scripts\python.exe tests/test_pipeline.py    # ingest → retrieve → answer
.venv\Scripts\python.exe tests/test_app_ui.py      # drives the real UI headlessly
```

## Limitations

- **Scanned PDFs won't work.** The app reads embedded text, not images. A photo
  of a page needs OCR, which isn't wired up — the app tells you when this happens
  instead of failing silently.
- Uploads live in the browser session; closing the tab clears the library.
