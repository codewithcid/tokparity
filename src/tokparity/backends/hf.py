"""Reference backend: Hugging Face `tokenizers` on the model's own tokenizer.json."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from tokenizers import Tokenizer


class HFTokenizer:
    def __init__(self, model_dir: str | Path):
        path = Path(model_dir) / "tokenizer.json"
        if not path.exists():
            raise FileNotFoundError(f"{path} not found (tokparity compares against tokenizer.json)")
        self.name = "hf-tokenizers"
        self._tok = Tokenizer.from_file(str(path))
        # Some tokenizer.json files enable padding or truncation, which would change ids in a batch.
        self._tok.no_padding()
        self._tok.no_truncation()

    def encode_batch(self, texts: Sequence[str]) -> list[list[int]]:
        return [e.ids for e in self._tok.encode_batch(list(texts), add_special_tokens=False)]

    def piece(self, token_id: int) -> str:
        return self._tok.id_to_token(token_id) or f"<{token_id}>"
