"""Test corpora: a generated Unicode stress suite, SIB-200 (FLORES-200 sentences), or your own file.

Every corpus is a list of (text, label) pairs; labels group results in the report.
"""

from __future__ import annotations

import csv
import json
import sys
import unicodedata
from pathlib import Path

from tokparity.classify import script

ZWNJ, ZWJ = chr(0x200C), chr(0x200D)
NBSP, NNBSP, IDEO_SPACE = chr(0xA0), chr(0x202F), chr(0x3000)


def _letters_before(cp: int, limit: int = 160) -> str | None:
    """Nearest letter at or before `cp` (same block in practice), used as a base for a combining mark."""
    for c in range(cp, max(cp - limit, 0), -1):
        ch = chr(c)
        if unicodedata.category(ch) in ("Lo", "Ll", "Lu"):
            return ch
    return None


def stress() -> list[tuple[str, str]]:
    """Deterministic edge cases that tokenizer ports tend to get wrong."""
    out: list[tuple[str, str]] = []
    bmp = [chr(c) for c in range(0x20, 0x10000) if not 0xD800 <= c <= 0xDFFF]

    # 1. Canonically decomposed text, and characters that NFC itself changes (composition exclusions).
    # Hangul syllables and CJK compatibility ideographs are regular, so a sample of each is enough.
    for ch in bmp:
        cp = ord(ch)
        if (0xAC00 <= cp <= 0xD7A3 and cp % 211) or (0xF900 <= cp <= 0xFAFF and cp % 23):
            continue
        nfd = unicodedata.normalize("NFD", ch)
        if len(nfd) > 1 and unicodedata.category(ch).startswith("L"):
            out.append((f"{nfd}{nfd} {ch}{ch} {nfd}", f"decomposed:{script(ch)}"))
        if unicodedata.normalize("NFC", ch) != ch and unicodedata.category(ch).startswith("L"):
            out.append((f"{ch}{ch} {ch}", f"nfc-changes:{script(ch)}"))

    # 2. Every combining mark after a letter of its own block, inside and at the end of words.
    for ch in bmp:
        if unicodedata.category(ch) in ("Mn", "Mc") and unicodedata.combining(ch) == 0:
            base = _letters_before(ord(ch))
            if base:
                out.append((f"{base}{ch}{base} {base}{ch}", f"combining-mark:{script(ch)}"))

    # 3. Zero-width joiners inside words (Persian, Indic) and in emoji sequences.
    for word_a, word_b, lab in [
        ("\u0645\u06cc", "\u062e\u0648\u0627\u0647\u0645", "Persian"),
        ("\u0d05\u0d35\u0d28\u0d4d", " \u0d35\u0d28\u0d4d\u0d28\u0d41", "Malayalam"),
        ("\u0915\u094d", "\u0937", "Devanagari"),
        ("\u09b0\u094d", "\u09af", "Bengali"),
    ]:
        for joiner, name in ((ZWNJ, "zwnj"), (ZWJ, "zwj")):
            out.append((f"{word_a}{joiner}{word_b}", f"{name}:{lab}"))
    man, woman, girl = chr(0x1F468), chr(0x1F469), chr(0x1F467)
    out.append((f"{man}{ZWJ}{woman}{ZWJ}{girl} family", "zwj:emoji"))

    # 4. Decimal digits of every script, alone, after a space, and inside words/numbers.
    for ch in bmp + [chr(c) for c in range(0x10000, 0x1FFFF)]:
        if unicodedata.category(ch) == "Nd" and unicodedata.decimal(ch) == 0:
            d = "".join(chr(ord(ch) + i) for i in range(10))
            out.append((f"{d} {d[1:4]}a{d[5]} x{d[:2]}", f"digits:{script(ch)}"))

    # 5. Whitespace and punctuation around words.
    ws = [
        "a  #",
        "    }",
        "x  .",
        ")]  # c",
        "a\t\tb",
        "a \n b",
        "  leading",
        "trailing  ",
        "\n\nnewlines",
        f"a{NBSP}b",
        f"a{NNBSP}b",
        f"a{IDEO_SPACE}b",
        " " * 40 + "deep indent",
        "a" + " " * 600 + "b",
        "end.\n\n\nNext",
        "\r\nwindows\r\n",
        "word...  !!",
        "\u0985\u09ae\u09bf \u0964",
        "\u0641\u064a \u060c \u0647\u0630\u0627",
    ]
    out += [(t, "whitespace/punctuation") for t in ws]

    # 6. Case changes and contractions (regexes with case-insensitive or case-split groups).
    cases = [
        "HelloWorld",
        "JSONParser",
        "iPhone 15",
        "McDonald's",
        "don't",
        "I'M SURE",
        "THEY'RE",
        "we'Ve",
        "camelCaseVariableName",
        "\u00c9COLE\u00e9cole",
        "\u0130stanbul \u0131",
    ]
    out += [(t, "case/contractions") for t in cases]

    # 7. Code-mixed text, compatibility characters and symbols.
    mixed = [
        "\u0906\u092a\u0915\u093e KYC \u0906\u091c \u092c\u0902\u0926",
        "\u0d28\u0d3f\u0d19\u0d4d\u0d19\u0d33\u0d41\u0d1f\u0d46 OTP 4321",
        "\uff21\uff22\uff23\uff11\uff12\uff13",
        "\ufb01nance \ufb02ow",
        "x\u00b2 + y\u00b3",
        "\u2460\u2461\u2462",
        "\u00bd \u2153",
        "\U0001f600\U0001f3fd\U0001f44d\U0001f3ff",
        "\u65e5\u672c\u8a9e\u306e\u30c6\u30ad\u30b9\u30c8 mixed",
        "\uac00\u1100\u1161 \ud55c\uad6d\uc5b4",
    ]
    out += [(t, "mixed/compat") for t in mixed]
    return out


def sib200(per_language: int = 60, cache: Path | None = None) -> list[tuple[str, str]]:
    """First `per_language` test sentences of each of the 205 SIB-200 languages (CC BY-SA 4.0)."""
    from huggingface_hub import HfApi, hf_hub_download

    files = [
        f for f in HfApi().list_repo_files("Davlan/sib200", repo_type="dataset") if f.endswith("/test.tsv")
    ]
    out = []
    for f in sorted(files):
        path = hf_hub_download("Davlan/sib200", f, repo_type="dataset", cache_dir=cache)
        with open(path, encoding="utf-8", newline="") as fh:
            rows = list(csv.DictReader(fh, delimiter="\t", quoting=csv.QUOTE_NONE))
        lang = f.split("/")[-2]
        out += [(r["text"], lang) for r in rows[:per_language] if r.get("text")]
    return out


def from_file(path: str | Path) -> list[tuple[str, str]]:
    """.txt (one text per line), .json (list of strings or of {"text", "label"}), or .jsonl."""
    p = Path(path)
    label = p.stem
    if p.suffix == ".txt":
        return [(line, label) for line in p.read_text(encoding="utf-8").splitlines() if line]
    rows = (
        json.loads(p.read_text(encoding="utf-8"))
        if p.suffix == ".json"
        else [json.loads(line) for line in p.read_text(encoding="utf-8").splitlines() if line.strip()]
    )
    return [(r, label) if isinstance(r, str) else (r["text"], r.get("label", label)) for r in rows]


def load(spec: str) -> list[tuple[str, str]]:
    if spec == "stress":
        return stress()
    if spec.startswith("sib200"):
        n = int(spec.split(":")[1]) if ":" in spec else 60
        return sib200(n)
    if Path(spec).exists():
        return from_file(spec)
    sys.exit(f"unknown corpus {spec!r}: use 'stress', 'sib200[:N]' or a file path")
