"""Original tiktoken tokenizers: a named encoding (`o200k_base`, ...) or a `.tiktoken`-format rank file with
its split pattern (Meta's Llama 3 and 4 `original/tokenizer.model`).
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

# The split pattern in Meta's llama-models repository (llama3 tokenizer), also used by Llama 4.
LLAMA3_PATTERN = (
    r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+"
    r"|\s+(?!\S)|\s+"
)
LLAMA4_PATTERN = (
    r"[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*[\p{Ll}\p{Lm}\p{Lo}\p{M}]+(?i:'s|'t|'re|'ve|'m|'ll|'d)?"
    r"|[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]+[\p{Ll}\p{Lm}\p{Lo}\p{M}]*(?i:'s|'t|'re|'ve|'m|'ll|'d)?"
    r"|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n/]*|\s*[\r\n]+|\s+(?!\S)|\s+"
)


class TikToken:
    def __init__(
        self, encoding: str | None = None, rank_file: str | Path | None = None, pattern: str | None = None
    ):
        import tiktoken

        if encoding:
            self._enc = tiktoken.get_encoding(encoding)
            self.name = f"tiktoken ({encoding})"
        else:
            from tiktoken.load import load_tiktoken_bpe

            if rank_file is None or pattern is None:
                raise ValueError("give `encoding`, or `rank_file` and `pattern`")
            ranks = load_tiktoken_bpe(str(rank_file))
            self._enc = tiktoken.Encoding(
                name=Path(rank_file).name, pat_str=pattern, mergeable_ranks=ranks, special_tokens={}
            )
            self.name = f"tiktoken ({Path(rank_file).parent.name}/{Path(rank_file).name})"

    def encode_batch(self, texts: Sequence[str]) -> list[list[int]]:
        return self._enc.encode_ordinary_batch(list(texts))

    def piece(self, token_id: int) -> str:
        return self._enc.decode_single_token_bytes(token_id).decode("utf-8", "replace")
