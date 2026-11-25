import base64

import pytest

from tokparity.originals import is_tiktoken_file, load_original, original_file


def test_original_file_priority():
    assert original_file({"tokenizer.json", "tekken.json", "tokenizer.model"}) == "tekken.json"
    assert original_file({"original/tokenizer.model", "tokenizer.model"}) == "original/tokenizer.model"
    assert original_file({"spiece.model"}) == "spiece.model"
    assert original_file({"tokenizer.json"}) is None


def _byte_rank_file(path):
    # A tiktoken rank file with the 256 single bytes and one merge ("ab").
    lines = [f"{base64.b64encode(bytes([b])).decode()} {b}" for b in range(256)]
    lines.append(f"{base64.b64encode(b'ab').decode()} 256")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_tiktoken_rank_file_is_detected_and_loaded(tmp_path):
    pytest.importorskip("tiktoken")
    original = tmp_path / "original"
    original.mkdir()
    _byte_rank_file(original / "tokenizer.model")
    assert is_tiktoken_file(original / "tokenizer.model")
    tok = load_original(tmp_path)
    assert tok.encode_batch(["ab c"]) == [[256, 32, 99]]
    assert tok.piece(256) == "ab"


def test_sentencepiece_model_is_detected_and_loaded(tmp_path):
    spm = pytest.importorskip("sentencepiece")
    (tmp_path / "train.txt").write_text("hello world\nhello there\nworld peace\n" * 50, encoding="utf-8")
    spm.SentencePieceTrainer.train(
        input=str(tmp_path / "train.txt"),
        model_prefix=str(tmp_path / "tokenizer"),
        vocab_size=30,
        model_type="bpe",
        minloglevel=2,
    )
    assert not is_tiktoken_file(tmp_path / "tokenizer.model")
    tok = load_original(tmp_path, "tokenizer.model")
    ids = tok.encode_batch(["hello world"])[0]
    assert "".join(tok.piece(i) for i in ids).replace("▁", " ").strip() == "hello world"
    assert tok.describe_normalizer()["model_type"] == "BPE"


def test_missing_original_is_a_clear_error(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_original(tmp_path)


def test_unknown_characters_map_to_the_unknown_token(tmp_path):
    spm = pytest.importorskip("sentencepiece")
    from tokenizers import Tokenizer, models

    (tmp_path / "train.txt").write_text("hello world\nhello there\nworld peace\n" * 50, encoding="utf-8")
    spm.SentencePieceTrainer.train(
        input=str(tmp_path / "train.txt"),
        model_prefix=str(tmp_path / "spiece"),
        vocab_size=18,
        model_type="unigram",
        minloglevel=2,
    )
    sp = spm.SentencePieceProcessor(model_file=str(tmp_path / "spiece.model"))
    vocab = {sp.id_to_piece(i): i for i in range(sp.get_piece_size())}
    Tokenizer(models.WordLevel(vocab, unk_token="<unk>")).save(str(tmp_path / "tokenizer.json"))
    tok = load_original(tmp_path, "spiece.model")
    ids = tok.encode_batch(["hello " + chr(0x4E16)])[0]
    assert -1 not in ids
    assert vocab["<unk>"] in ids
