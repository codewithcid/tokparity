"""Get a model's tokenizer files (never the weights) and, for llama.cpp, a vocab-only GGUF."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from tokparity.paths import cache_dir

TOKENIZER_FILES = [
    "config.json",
    "tokenizer*",
    "*.model",
    "*.tiktoken",
    "special_tokens_map.json",
    "vocab*",
    "merges.txt",
    "generation_config.json",
    "added_tokens.json",
    "*.py",
    "chat_template*",
    "tekken.json",
    "original/tokenizer.model",
]


def model_dir(model: str) -> Path:
    """A local directory as is, or a Hugging Face repo id downloaded (tokenizer files only) to the cache."""
    if Path(model).is_dir():
        return Path(model)
    from huggingface_hub import snapshot_download

    target = cache_dir() / "models" / model.replace("/", "__")
    snapshot_download(model, local_dir=target, allow_patterns=TOKENIZER_FILES)
    return target


def vocab_gguf(model: str, llama_cpp_src: str | Path) -> Path:
    """Convert the tokenizer to a vocab-only GGUF with llama.cpp's own converter (needs its requirements)."""
    src = model_dir(model)
    out = cache_dir() / "gguf" / f"{src.name}.gguf"
    if out.exists():
        return out
    out.parent.mkdir(parents=True, exist_ok=True)
    script = Path(llama_cpp_src) / "convert_hf_to_gguf.py"
    r = subprocess.run(
        [sys.executable, str(script), str(src), "--vocab-only", "--outfile", str(out)],
        capture_output=True,
        encoding="utf-8",
        errors="replace",
    )
    if r.returncode != 0 or not out.exists():
        tail = "\n".join((r.stderr or r.stdout).strip().splitlines()[-8:])
        raise RuntimeError(f"convert_hf_to_gguf.py failed for {model}:\n{tail}")
    return out
