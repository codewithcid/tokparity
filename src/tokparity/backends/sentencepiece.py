"""Original SentencePiece model (`tokenizer.model`, `spiece.model`, `sentencepiece.bpe.model`).

Ids are taken from the pieces through the vocabulary of the model's tokenizer.json, so that models whose
Hugging Face ids are shifted from SentencePiece's (fairseq-style XLM-R, for example) compare correctly. A
piece that is missing from that vocabulary becomes -1 and always counts as a difference.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path


class SentencePiece:
    def __init__(self, model_file: str | Path, vocab_from: str | Path | None = None):
        import sentencepiece as spm

        self.model_file = Path(model_file)
        self.name = f"sentencepiece ({self.model_file.name})"
        self._sp = spm.SentencePieceProcessor(model_file=str(self.model_file))
        self._vocab = None
        if vocab_from is not None:
            from tokenizers import Tokenizer

            tok = Tokenizer.from_file(str(Path(vocab_from)))
            self._vocab = tok.get_vocab(with_added_tokens=True)
            self._inverse = {i: p for p, i in self._vocab.items()}

    def encode_batch(self, texts: Sequence[str]) -> list[list[int]]:
        if self._vocab is None:
            return [list(ids) for ids in self._sp.encode(list(texts))]
        # Go through ids, not `out_type=str`: for an unknown character SentencePiece returns the character
        # itself as the piece, while its id (and `id_to_piece`) is the unknown token.
        to_piece = self._sp.id_to_piece
        return [[self._vocab.get(to_piece(i), -1) for i in ids] for ids in self._sp.encode(list(texts))]

    def piece(self, token_id: int) -> str:
        if self._vocab is not None:
            return self._inverse.get(token_id, f"<{token_id}>")
        return self._sp.id_to_piece(token_id)

    def describe_normalizer(self) -> dict:
        """The model's normalization settings, which conversions to tokenizer.json sometimes drop."""
        from sentencepiece import sentencepiece_model_pb2 as pb

        m = pb.ModelProto()
        m.ParseFromString(self.model_file.read_bytes())
        n = m.normalizer_spec
        return json.loads(
            json.dumps(
                {
                    "name": n.name,
                    "add_dummy_prefix": n.add_dummy_prefix,
                    "remove_extra_whitespaces": n.remove_extra_whitespaces,
                    "escape_whitespaces": n.escape_whitespaces,
                    "has_precompiled_charsmap": len(n.precompiled_charsmap) > 0,
                    "model_type": pb.TrainerSpec.ModelType.Name(m.trainer_spec.model_type),
                    "byte_fallback": m.trainer_spec.byte_fallback,
                }
            )
        )
