"""Find and load a model's original tokenizer, the one its tokenizer.json was converted from.

Supported: SentencePiece models, tiktoken rank files (Meta's `original/tokenizer.model` for Llama 3 and 4),
and Mistral's `tekken.json`.
"""

from __future__ import annotations

import json
from pathlib import Path

SPM_NAMES = ("tokenizer.model", "spiece.model", "sentencepiece.bpe.model", "sentencepiece.model")
ORIGINAL_NAMES = ("tekken.json", "original/tokenizer.model", *SPM_NAMES)


def original_file(files) -> str | None:
    """The original tokenizer file among a repo's file names, by priority."""
    files = set(files)
    return next((name for name in ORIGINAL_NAMES if name in files), None)


def is_tiktoken_file(path: Path) -> bool:
    """tiktoken rank files are text lines "<base64 token> <rank>"; SentencePiece models are protobuf."""
    first = path.read_bytes()[:256].split(b"\n")[0].strip()
    parts = first.split(b" ")
    return len(parts) == 2 and parts[1].isdigit()


def load_original(model_dir: str | Path, original: str | None = None):
    """Reference backend for the original tokenizer in `model_dir` (detected when `original` is None)."""
    model_dir = Path(model_dir)
    if original is None:
        found = [p.relative_to(model_dir).as_posix() for p in model_dir.rglob("*") if p.is_file()]
        original = original_file(found)
        if original is None:
            raise FileNotFoundError(f"no original tokenizer ({', '.join(ORIGINAL_NAMES)}) in {model_dir}")
    path = model_dir / original
    if original == "tekken.json":
        from tokparity.backends.mistral import Tekken

        return Tekken(path)
    if is_tiktoken_file(path):
        from tokparity.backends.tiktoken_backend import LLAMA3_PATTERN, LLAMA4_PATTERN, TikToken

        config = model_dir / "config.json"
        is_llama4 = config.exists() and "llama4" in json.dumps(json.loads(config.read_text(encoding="utf-8")))
        return TikToken(rank_file=path, pattern=LLAMA4_PATTERN if is_llama4 else LLAMA3_PATTERN)
    from tokparity.backends.sentencepiece import SentencePiece

    vocab = model_dir / "tokenizer.json"
    return SentencePiece(path, vocab_from=vocab if vocab.exists() else None)
