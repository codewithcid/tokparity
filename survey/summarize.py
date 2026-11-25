"""Turn survey/results/*.json into a Markdown matrix (model x backend) and a list of minimal repros.

python survey/summarize.py survey/results > RESULTS.md
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

from tokparity.report import _esc


def main(results_dir: str) -> None:
    rows: dict[str, dict[str, dict]] = defaultdict(dict)
    backends: list[str] = []
    for path in sorted(Path(results_dir).glob("*.json")):
        r = json.loads(path.read_text(encoding="utf-8"))
        rows[r["model"]][r["backend"]] = r
        if r["backend"] not in backends:
            backends.append(r["backend"])
    order = ["llamacpp", "ollama-main", "ollama-fix", "tokenizersjs-release", "tokenizersjs-pr57"]
    backends = [b for b in order if b in backends] + [b for b in backends if b not in order]

    any_r = next(r for m in rows.values() for r in m.values() if "n_texts" in r)
    corpora = ", ".join(Path(c).stem if Path(c).suffix else c for c in any_r["corpora"])
    print(f"Texts per model: {any_r['n_texts']} ({corpora}).")
    print("Each cell: share of texts whose token ids differ from Hugging Face `tokenizers`.")
    print("`n/a` = the backend cannot load this tokenizer.\n")
    print("| model | " + " | ".join(backends) + " |")
    print("|---|" + "---:|" * len(backends))
    for model, by_backend in rows.items():
        cells = []
        for b in backends:
            r = by_backend.get(b)
            if r is None:
                cells.append("")
            elif "error" in r:
                cells.append("n/a")
            else:
                pct = 100 * r["n_mismatches"] / r["n_texts"]
                cells.append(
                    "0"
                    if r["n_mismatches"] == 0
                    else f"**{pct:.2f}%**"
                    if pct >= 1
                    else f"{pct:.2f}%"
                    if pct >= 0.01
                    else "<0.01%"
                )
        print(f"| {model} | " + " | ".join(cells) + " |")

    print("\n## Minimal repros\n")
    for model, by_backend in rows.items():
        for b in backends:
            r = by_backend.get(b)
            if not r or "error" in r or not r["n_mismatches"]:
                continue
            repros = [m for m in r["mismatches"] if m.get("minimal") is not None][:5]
            print(f"### {model} / {b}: {r['n_mismatches']} of {r['n_texts']}\n")
            print("| input | `tokenizers` | backend | tags |\n|---|---|---|---|")
            for m in repros:
                print(
                    f"| `{_esc(m['minimal'])}` | `{m['minimal_reference']}` | `{m['minimal_candidate']}` "
                    f"| {', '.join(m['tags'])} |"
                )
            print()


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main(sys.argv[1])
