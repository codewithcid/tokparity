"""Shrink a mismatching text to a minimal one that still mismatches (delta debugging).

`differs` takes a batch of candidate strings and returns, for each, whether the two tokenizers still
disagree on it. Candidates are evaluated a whole round at a time, so backends that run in a separate
process (Ollama, llama-tokenize) pay the startup cost once per round instead of once per string.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

Oracle = Callable[[Sequence[str]], Sequence[bool]]


def _trim(text: str, differs: Oracle) -> str:
    """Cut the longest prefix and suffix that can go, halving the step each time."""
    step = max(1, len(text) // 2)
    while step >= 1 and len(text) > 1:
        cands = [c for c in (text[step:], text[:-step]) if c]
        ok = [c for c, d in zip(cands, differs(cands), strict=True) if d]
        if ok:
            text = min(ok, key=len)
            step = min(step, max(1, len(text) // 2))
        else:
            step //= 2
    return text


def _ddmin(text: str, differs: Oracle, budget: int) -> str:
    chars = list(text)
    n = 2
    while len(chars) >= 2 and budget > 0:
        size = -(-len(chars) // n)
        cands = ["".join(chars[:i] + chars[i + size :]) for i in range(0, len(chars), size)]
        cands = [c for c in cands if c]
        budget -= len(cands)
        hit = next((c for c, d in zip(cands, differs(cands), strict=True) if d), None)
        if hit is not None:
            chars = list(hit)
            n = max(n - 1, 2)
        elif n >= len(chars):
            break
        else:
            n = min(len(chars), n * 2)
    return "".join(chars)


def minimize(text: str, differs: Oracle, budget: int = 4000) -> str:
    """Return a short string on which the tokenizers still disagree (1-minimal when the budget allows)."""
    if not differs([text])[0]:
        raise ValueError("the tokenizers agree on the input; nothing to minimize")
    return _ddmin(_trim(text, differs), differs, budget)
