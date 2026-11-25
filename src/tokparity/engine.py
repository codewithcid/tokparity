"""Run a reference and a candidate tokenizer over texts and collect every disagreement."""

from __future__ import annotations

import time
from collections import Counter
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field

from tokparity.classify import tags
from tokparity.minimize import minimize


@dataclass
class Mismatch:
    text: str
    label: str | None
    first_diff: int
    reference: list[int]
    candidate: list[int]
    minimal: str | None = None
    minimal_reference: list[int] | None = None
    minimal_candidate: list[int] | None = None
    tags: list[str] = field(default_factory=list)


@dataclass
class Result:
    reference: str
    candidate: str
    n_texts: int
    seconds: float
    mismatches: list[Mismatch]
    by_label: dict[str, list[int]]  # label -> [n_texts, n_mismatches]

    @property
    def n_mismatches(self) -> int:
        return len(self.mismatches)

    def tag_counts(self) -> Counter:
        return Counter(t for m in self.mismatches if m.minimal is not None for t in m.tags)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["n_mismatches"] = self.n_mismatches
        return d


def _first_diff(a: Sequence[int], b: Sequence[int]) -> int:
    return next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), min(len(a), len(b)))


def compare(
    reference,
    candidate,
    texts: Sequence[str],
    labels: Sequence[str] | None = None,
    minimize_n: int = 20,
    batch_size: int = 4096,
) -> Result:
    """Encode `texts` with both backends. The first `minimize_n` mismatches are shrunk to minimal repros
    (distinct ones: a mismatch whose minimal repro was already seen is not shrunk again)."""
    labels = list(labels) if labels is not None else [None] * len(texts)
    t0 = time.time()
    mismatches: list[Mismatch] = []
    by_label: dict[str, list[int]] = {}
    for start in range(0, len(texts), batch_size):
        chunk = list(texts[start : start + batch_size])
        ref_ids = reference.encode_batch(chunk)
        cand_ids = candidate.encode_batch(chunk)
        for text, label, a, b in zip(chunk, labels[start : start + batch_size], ref_ids, cand_ids):
            stats = by_label.setdefault(label or "-", [0, 0])
            stats[0] += 1
            if a != b:
                stats[1] += 1
                mismatches.append(Mismatch(text, label, _first_diff(a, b), list(a), list(b)))

    def differs(cands: Sequence[str]) -> list[bool]:
        return [x != y for x, y in zip(reference.encode_batch(cands), candidate.encode_batch(cands))]

    # Shrink one mismatch per label first, so the repros cover different bugs, then fill up in order.
    first_of_label = {}
    for m in mismatches:
        first_of_label.setdefault(m.label, m)
    queue = list(first_of_label.values()) + [m for m in mismatches if first_of_label[m.label] is not m]
    seen: set[str] = set()
    done = 0
    for m in queue:
        if done >= minimize_n:
            break
        if any(s in m.text for s in seen):  # already explained by a known minimal repro
            continue
        done += 1
        try:
            m.minimal = minimize(m.text, differs)
        except ValueError:  # does not differ when encoded alone (backend depends on the batch)
            m.tags = ["batch-dependent"]
            continue
        seen.add(m.minimal)
        m.minimal_reference = reference.encode_batch([m.minimal])[0]
        m.minimal_candidate = candidate.encode_batch([m.minimal])[0]
        m.tags = tags(m.minimal)
    return Result(reference.name, candidate.name, len(texts), time.time() - t0, mismatches, by_label)
