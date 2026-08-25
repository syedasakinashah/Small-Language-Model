"""Answer-generation backends, best-available-first.

Three engines, tried in this order:

1. ``ClaudeBackend``  - Claude API. Best explanations, fluent Urdu, real answer
   analysis. Needs ``ANTHROPIC_API_KEY`` and internet.
2. ``LocalBackend``   - a small instruct model via transformers. Fully offline
   once the weights are cached; slower on CPU but genuinely generative.
3. ``ExtractiveBackend`` - no model at all. Builds a structured answer out of the
   retrieved passages. Always available, so the app never dies with an error.

The app shows which one is live, so a demo never silently degrades.
"""

from __future__ import annotations

import os
import textwrap
from dataclasses import dataclass
from typing import Iterator

CLAUDE_MODEL = "claude-opus-5"
LOCAL_MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"


@dataclass
class BackendInfo:
    key: str
    label: str
    detail: str
    offline: bool


def free_ram_gb() -> float:
    """Available physical memory, or +inf when we can't tell (never block on a guess)."""
    try:
        import ctypes

        class _MemoryStatus(ctypes.Structure):
            _fields_ = [
                ("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
            ]

        status = _MemoryStatus()
        status.dwLength = ctypes.sizeof(_MemoryStatus)
        ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status))
        return status.ullAvailPhys / (1024 ** 3)
    except Exception:
        return float("inf")


# --------------------------------------------------------------------------
# base
# --------------------------------------------------------------------------
class Backend:
    info: BackendInfo

    def generate(self, system: str, user: str, max_tokens: int = 1200) -> str:
        raise NotImplementedError

    def stream(self, system: str, user: str, max_tokens: int = 1200) -> Iterator[str]:
        """Default: no real streaming, just hand back the whole answer once."""
        yield self.generate(system, user, max_tokens)


# --------------------------------------------------------------------------
# 1. Claude
# --------------------------------------------------------------------------
class ClaudeBackend(Backend):
    info = BackendInfo(
        key="claude",
        label="Claude API",
        detail=f"{CLAUDE_MODEL} - full explanations, Urdu, answer analysis",
        offline=False,
    )

    def __init__(self, api_key: str | None = None):
        import anthropic
        self._anthropic = anthropic
        self.client = anthropic.Anthropic(api_key=api_key) if api_key else anthropic.Anthropic()

    @staticmethod
    def sdk_installed() -> bool:
        try:
            import anthropic  # noqa: F401
        except ImportError:
            return False
        return True

    @staticmethod
    def is_configured(api_key: str | None = None) -> bool:
        return bool(api_key or os.environ.get("ANTHROPIC_API_KEY")) and ClaudeBackend.sdk_installed()

    def _request(self, system: str, user: str, max_tokens: int):
        # Streaming keeps long answers from tripping the SDK's HTTP timeout.
        # `fallbacks` re-runs the request on another model if a safety
        # classifier declines it, so a student never sees a dead end.
        return self.client.beta.messages.stream(
            model=CLAUDE_MODEL,
            max_tokens=max_tokens,
            system=system,
            thinking={"type": "adaptive"},
            output_config={"effort": "medium"},
            betas=["server-side-fallback-2026-07-01"],
            fallbacks="default",
            messages=[{"role": "user", "content": user}],
        )

    def stream(self, system: str, user: str, max_tokens: int = 1200) -> Iterator[str]:
        try:
            with self._request(system, user, max_tokens) as stream:
                for text in stream.text_stream:
                    yield text
                final = stream.get_final_message()
            if final.stop_reason == "refusal":
                yield ("\n\n_I can't answer that one. Try rephrasing it, "
                       "or ask about something in your uploaded material._")
        except self._anthropic.AuthenticationError:
            yield "**Claude API key rejected.** Check `ANTHROPIC_API_KEY` and restart the app."
        except self._anthropic.RateLimitError:
            yield "**Rate limited by the Claude API.** Wait a few seconds and ask again."
        except self._anthropic.APIConnectionError:
            yield "**No internet connection to the Claude API.** The offline engine can still answer."
        except self._anthropic.APIStatusError as e:
            yield f"**Claude API error ({e.status_code}).** {e.message}"

    def generate(self, system: str, user: str, max_tokens: int = 1200) -> str:
        return "".join(self.stream(system, user, max_tokens))


# --------------------------------------------------------------------------
# 2. Local instruct model (offline)
# --------------------------------------------------------------------------
class LocalBackend(Backend):
    info = BackendInfo(
        key="local",
        label="Offline AI model",
        detail=f"{LOCAL_MODEL_NAME} running on this machine - no internet needed",
        offline=True,
    )

    _tokenizer = None
    _model = None

    def __init__(self):
        LocalBackend._load()

    @classmethod
    def _load(cls):
        if cls._model is not None:
            return
        from transformers import AutoModelForCausalLM, AutoTokenizer
        cls._tokenizer = AutoTokenizer.from_pretrained(LOCAL_MODEL_NAME)
        cls._model = AutoModelForCausalLM.from_pretrained(LOCAL_MODEL_NAME)
        cls._model.eval()

    @staticmethod
    def is_available() -> bool:
        """Opt-in, weights cached, and enough free RAM to actually load them.

        The memory check is not paranoia: torch aborts the whole process when an
        allocation fails, so an over-optimistic attempt here takes Streamlit down
        with it and the student just sees the app vanish.
        """
        if os.environ.get("TUTOR_USE_LOCAL_LLM", "").strip().lower() not in {"1", "true", "yes"}:
            return False
        if free_ram_gb() < 3.0:
            return False
        try:
            from huggingface_hub import snapshot_download
            snapshot_download(LOCAL_MODEL_NAME, local_files_only=True)
            return True
        except Exception:
            return False

    def generate(self, system: str, user: str, max_tokens: int = 1200) -> str:
        import torch
        tok, model = LocalBackend._tokenizer, LocalBackend._model
        prompt = tok.apply_chat_template(
            [{"role": "system", "content": system},
             {"role": "user", "content": user}],
            tokenize=False, add_generation_prompt=True,
        )
        inputs = tok(prompt, return_tensors="pt", truncation=True, max_length=3072)
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=min(max_tokens, 700),   # CPU generation is slow
                do_sample=True,
                temperature=0.6,
                top_p=0.9,
                pad_token_id=tok.eos_token_id,
            )
        text = tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        return text.strip()


# --------------------------------------------------------------------------
# 3. Extractive (always works, zero dependencies beyond the retriever)
# --------------------------------------------------------------------------
class ExtractiveBackend(Backend):
    info = BackendInfo(
        key="extractive",
        label="Built-in reader",
        detail="Answers straight from your PDFs - works with no model installed",
        offline=True,
    )

    def generate(self, system: str, user: str, max_tokens: int = 1200) -> str:
        # `tutor.py` hands this backend a pre-built answer in the user payload,
        # after the ANSWER marker. Nothing to generate, just surface it.
        marker = "<<<EXTRACTIVE_ANSWER>>>"
        if marker in user:
            return user.split(marker, 1)[1].strip()
        return textwrap.dedent("""\
            I can only answer from the study material you've uploaded.
            Upload a PDF for this subject and ask again.""")


# --------------------------------------------------------------------------
# selection
# --------------------------------------------------------------------------
_local_singleton: Backend | None = None


def get_backend(api_key: str | None = None, force: str | None = None) -> Backend:
    """Return the best backend available.

    ``api_key`` lets the UI supply a Claude key without touching the environment.
    ``force`` pins a specific backend by key.
    """
    global _local_singleton

    order = [force] if force else ["claude", "local", "extractive"]
    for key in order:
        try:
            if key == "claude" and ClaudeBackend.is_configured(api_key):
                return ClaudeBackend(api_key or os.environ.get("ANTHROPIC_API_KEY"))
            if key == "local" and LocalBackend.is_available():
                if _local_singleton is None:      # weights load once per process
                    _local_singleton = LocalBackend()
                return _local_singleton
            if key == "extractive":
                return ExtractiveBackend()
        except Exception:
            continue

    return ExtractiveBackend()


def backend_status(api_key: str | None = None) -> list[tuple[BackendInfo, bool, str]]:
    """(info, ready, why-not) for each backend - drives the UI status panel."""
    rows = []

    if ClaudeBackend.is_configured(api_key):
        rows.append((ClaudeBackend.info, True, ""))
    elif not ClaudeBackend.sdk_installed():
        rows.append((ClaudeBackend.info, False, "run: pip install anthropic"))
    else:
        rows.append((ClaudeBackend.info, False, "paste an API key in the sidebar"))

    if LocalBackend.is_available():
        rows.append((LocalBackend.info, True, ""))
    else:
        free = free_ram_gb()
        if free < 3.0:
            why = f"needs ~3 GB free RAM (this machine has {free:.1f} GB free)"
        else:
            why = "set TUTOR_USE_LOCAL_LLM=1 and run scripts/setup_models.py once"
        rows.append((LocalBackend.info, False, why))

    rows.append((ExtractiveBackend.info, True, ""))
    return rows
