"""Mistral's own tokenizer library (`mistral_common`) on a model's `tekken.json`."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path


class Tekken:
    def __init__(self, tekken_file: str | Path):
        from mistral_common.tokens.tokenizers.tekken import Tekkenizer

        self.name = "mistral_common (tekken)"
        self._tok = Tekkenizer.from_file(str(tekken_file))

    def encode_batch(self, texts: Sequence[str]) -> list[list[int]]:
        return [self._tok.encode(t, bos=False, eos=False) for t in texts]

    def piece(self, token_id: int) -> str:
        return self._tok.id_to_piece(token_id)
