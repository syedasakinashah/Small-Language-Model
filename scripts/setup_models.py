"""One-time download of the offline models. Needs internet once, then never again.

Only worth running if you want the tutor to work with no internet at all.
Requires roughly 2.5 GB of disk and ~3 GB of free RAM at runtime -- check the
report this script prints before committing to it.
"""

import sys

EMBED = "sentence-transformers/all-MiniLM-L6-v2"
GENERATOR = "Qwen/Qwen2.5-0.5B-Instruct"


def free_ram_gb() -> float:
    try:
        import ctypes

        class MemoryStatus(ctypes.Structure):
            _fields_ = [
                ("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
            ]

        status = MemoryStatus()
        status.dwLength = ctypes.sizeof(MemoryStatus)
        ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status))
        return status.ullAvailPhys / (1024 ** 3)
    except Exception:
        return float("inf")


def main() -> int:
    free = free_ram_gb()
    print(f"Free RAM: {free:.1f} GB")
    if free < 3.0:
        print(
            "\nNot enough free memory to load these models. Loading them anyway\n"
            "would abort the process (and take the app down with it).\n"
            "Close some programs and retry, or use the Claude API engine instead."
        )
        return 1

    try:
        from sentence_transformers import SentenceTransformer
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError:
        print("\nMissing packages. Install them first:\n"
              "  pip install torch transformers sentence-transformers")
        return 1

    print(f"\n[1/2] {EMBED} (~90 MB)")
    model = SentenceTransformer(EMBED)
    print("      OK, embedding size:", model.encode(["test"]).shape[-1])

    print(f"\n[2/2] {GENERATOR} (~1 GB)")
    AutoTokenizer.from_pretrained(GENERATOR)
    AutoModelForCausalLM.from_pretrained(GENERATOR)
    print("      OK")

    print("\nDone. Now enable them:\n"
          "  set TUTOR_USE_EMBEDDINGS=1\n"
          "  set TUTOR_USE_LOCAL_LLM=1\n"
          "then restart the app.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
