"""Markdown reports from one or more saved results."""

from __future__ import annotations

import json


def _esc(text: str) -> str:
    """Show a repro string unambiguously: escape whitespace and anything invisible or combining."""
    import unicodedata

    out = []
    for ch in text:
        cat = unicodedata.category(ch)
        if ch == " ":
            out.append(" ")
        elif ch.isascii() and ch.isprintable() and ch not in "`|":
            out.append(ch)
        elif cat[0] in "MCZ" or ch in "`|":
            out.append(f"\\u{ord(ch):04x}" if ord(ch) <= 0xFFFF else f"\\U{ord(ch):08x}")
        else:
            out.append(ch)
    return "".join(out)


def single(result: dict, model: str, top: int = 10) -> str:
    n, bad = result["n_texts"], result["n_mismatches"]
    lines = [
        f"## {model}: {result['candidate']} vs {result['reference']}",
        "",
        f"{bad} of {n} texts tokenize differently ({100 * bad / max(n, 1):.2f}%).",
        "",
    ]
    worst = sorted(((v[1], v[0], k) for k, v in result["by_label"].items() if v[1]), reverse=True)[:top]
    if worst:
        lines += ["| label | differ | of |", "|---|---:|---:|"]
        lines += [f"| {k} | {b} | {t} |" for b, t, k in worst]
        lines.append("")
    repros = [m for m in result["mismatches"] if m.get("minimal") is not None]
    if repros:
        lines += ["Minimal repros:", "", "| input | reference | candidate | tags |", "|---|---|---|---|"]
        for m in repros:
            lines.append(
                f"| `{_esc(m['minimal'])}` | `{m['minimal_reference']}` | `{m['minimal_candidate']}` "
                f"| {', '.join(m['tags'])} |"
            )
        lines.append("")
    return "\n".join(lines)


def summary(paths: list[str]) -> str:
    rows = ["| model | candidate | texts | differ | % |", "|---|---|---:|---:|---:|"]
    for p in paths:
        r = json.loads(open(p, encoding="utf-8").read())
        n, bad = r["n_texts"], r["n_mismatches"]
        rows.append(f"| {r.get('model', p)} | {r['candidate']} | {n} | {bad} | {100 * bad / max(n, 1):.2f} |")
    return "\n".join(rows)
