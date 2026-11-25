import json
import unicodedata

import pytest

from tokparity import compare, minimize
from tokparity.classify import tags
from tokparity.cli import main
from tokparity.corpora import from_file, stress
from tokparity.report import _esc, single

COMBINING_ACUTE = chr(0x301)
ZWNJ = chr(0x200C)


class CharTokenizer:
    """Fake backend: one token per code point."""

    name = "chars"

    def encode_batch(self, texts):
        return [[ord(c) for c in t] for t in texts]

    def piece(self, token_id):
        return chr(token_id)


class NFCTokenizer(CharTokenizer):
    """Fake backend that NFC-normalizes first, like Qwen's tokenizer.json."""

    name = "nfc-chars"

    def encode_batch(self, texts):
        return super().encode_batch([unicodedata.normalize("NFC", t) for t in texts])


def test_minimize_finds_the_single_offending_pair():
    text = "a long sentence with one cafe" + COMBINING_ACUTE + " in the middle"

    def differs(cands):
        return [unicodedata.normalize("NFC", c) != c for c in cands]

    assert minimize(text, differs) == "e" + COMBINING_ACUTE


def test_minimize_rejects_inputs_that_agree():
    with pytest.raises(ValueError):
        minimize("same", lambda cands: [False] * len(cands))


def test_minimize_result_still_differs_and_is_one_minimal():
    text = "xx ab yy ab zz"

    def differs(cands):
        return ["ab" in c for c in cands]

    small = minimize(text, differs)
    assert small == "ab"
    assert not any(differs([small[:i] + small[i + 1 :] for i in range(len(small))]))


def test_compare_reports_and_minimizes_mismatches():
    texts = ["plain ascii", "cafe" + COMBINING_ACUTE + " noir", "caf" + chr(0xE9)]
    result = compare(NFCTokenizer(), CharTokenizer(), texts, labels=["a", "b", "c"])
    assert result.n_mismatches == 1
    m = result.mismatches[0]
    assert m.label == "b"
    assert m.minimal == "e" + COMBINING_ACUTE
    assert "not-nfc" in m.tags and "combining-mark" in m.tags
    assert result.by_label == {"a": [1, 0], "b": [1, 1], "c": [1, 0]}


def test_compare_does_not_reminimize_known_repros():
    texts = ["e" + COMBINING_ACUTE + " one", "two e" + COMBINING_ACUTE]
    result = compare(NFCTokenizer(), CharTokenizer(), texts)
    assert [m.minimal for m in result.mismatches] == ["e" + COMBINING_ACUTE, None]


@pytest.mark.parametrize(
    "text, expected",
    [
        ("e" + COMBINING_ACUTE, {"not-nfc", "combining-mark"}),
        ("a" + ZWNJ + "b", {"zero-width-joiner"}),
        ("  #", {"punctuation", "whitespace-run", "leading-space", "ascii-only"}),
        (chr(0x967) + chr(0x968), {"non-ascii-digit"}),
        ("KYC " + chr(0x906), {"mixed-script"}),
    ],
)
def test_tags(text, expected):
    assert expected <= set(tags(text))


def test_stress_suite_is_deterministic_and_covers_known_bug_classes():
    a, b = stress(), stress()
    assert a == b
    labels = {label.split(":")[0] for _, label in a}
    assert {"decomposed", "nfc-changes", "combining-mark", "zwnj", "zwj", "digits"} <= labels
    texts = [t for t, _ in a]
    assert all(texts) and len(set(texts)) == len(texts)
    assert any(unicodedata.normalize("NFC", t) != t for t in texts)
    assert "a  #" in texts  # whitespace before punctuation (llama.cpp falcon-h1)


def test_from_file_formats(tmp_path):
    (tmp_path / "x.txt").write_text("one\n\ntwo\n", encoding="utf-8")
    (tmp_path / "y.json").write_text(json.dumps(["a", {"text": "b", "label": "L"}]), encoding="utf-8")
    (tmp_path / "z.jsonl").write_text('{"text": "c"}\n', encoding="utf-8")
    assert from_file(tmp_path / "x.txt") == [("one", "x"), ("two", "x")]
    assert from_file(tmp_path / "y.json") == [("a", "y"), ("b", "L")]
    assert from_file(tmp_path / "z.jsonl") == [("c", "z")]


def test_report_escapes_invisible_characters():
    assert _esc("e" + COMBINING_ACUTE) == "e\\u0301"
    assert _esc("a" + ZWNJ + "b") == "a\\u200cb"
    assert _esc("a|b") == "a\\u007cb"


def test_report_and_cli_summary(tmp_path, capsys):
    result = compare(NFCTokenizer(), CharTokenizer(), ["cafe" + COMBINING_ACUTE]).to_dict()
    result["model"] = "demo/model"
    md = single(result, "demo/model")
    assert "1 of 1 texts" in md and "e\\u0301" in md
    path = tmp_path / "r.json"
    path.write_text(json.dumps(result), encoding="utf-8")
    assert main(["report", str(path)]) == 0
    assert "| demo/model | chars | 1 | 1 | 100.00 |" in capsys.readouterr().out


def test_reference_ignores_padding_and_truncation_in_tokenizer_json(tmp_path):
    from tokenizers import Tokenizer, models, pre_tokenizers

    from tokparity.backends.hf import HFTokenizer

    tok = Tokenizer(models.WordLevel({"[PAD]": 0, "a": 1, "b": 2, "[UNK]": 3}, unk_token="[UNK]"))
    tok.pre_tokenizer = pre_tokenizers.Whitespace()
    tok.enable_padding(pad_id=0, pad_token="[PAD]")
    tok.enable_truncation(max_length=2)
    tok.save(str(tmp_path / "tokenizer.json"))
    assert HFTokenizer(tmp_path).encode_batch(["a", "a b a b"]) == [[1], [1, 2, 1, 2]]


class FlakyTokenizer(CharTokenizer):
    """Differs from CharTokenizer only when it sees more than one text at a time."""

    name = "flaky"

    def encode_batch(self, texts):
        ids = super().encode_batch(texts)
        return [i + [0] for i in ids] if len(texts) > 1 else ids


def test_batch_dependent_mismatch_is_reported_not_raised():
    result = compare(CharTokenizer(), FlakyTokenizer(), ["x", "y"])
    assert result.n_mismatches == 2
    assert result.mismatches[0].minimal is None
    assert result.mismatches[0].tags == ["batch-dependent"]
