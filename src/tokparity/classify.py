"""Tag a (minimized) mismatch with the Unicode features it contains, to group bugs by likely cause."""

from __future__ import annotations

import re
import unicodedata

ZWNJ, ZWJ = chr(0x200C), chr(0x200D)
_WS_RUN = re.compile(r"\s{2,}")


def tags(text: str) -> list[str]:
    out = []
    if unicodedata.normalize("NFC", text) != text:
        out.append("not-nfc")
    cats = [unicodedata.category(c) for c in text]
    if any(c.startswith("M") for c in cats):
        out.append("combining-mark")
    if ZWNJ in text or ZWJ in text:
        out.append("zero-width-joiner")
    if any(c == "Nd" and not ch.isascii() for c, ch in zip(cats, text, strict=True)):
        out.append("non-ascii-digit")
    if any(c.startswith("P") for c in cats):
        out.append("punctuation")
    if _WS_RUN.search(text):
        out.append("whitespace-run")
    if text[:1].isspace():
        out.append("leading-space")
    if text[-1:].isspace():
        out.append("trailing-space")
    if any(c == "So" for c in cats):
        out.append("symbol/emoji")
    if any(c.startswith("L") and ch.isascii() for c, ch in zip(cats, text, strict=True)) and any(
        c.startswith("L") and not ch.isascii() for c, ch in zip(cats, text, strict=True)
    ):
        out.append("mixed-script")
    if text.isascii():
        out.append("ascii-only")
    return out


def script(ch: str) -> str:
    """Rough script name of a character from its Unicode name (e.g. 'DEVANAGARI', 'LATIN')."""
    try:
        return unicodedata.name(ch).split()[0]
    except ValueError:
        return "UNKNOWN"
