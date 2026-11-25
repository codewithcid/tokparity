"""Tokenizer backends. Each one turns a batch of strings into token ids, without special tokens."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol


class Backend(Protocol):
    name: str

    def encode_batch(self, texts: Sequence[str]) -> list[list[int]]: ...

    def piece(self, token_id: int) -> str: ...
