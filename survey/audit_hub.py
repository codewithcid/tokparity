"""Audit Hugging Face `tokenizer.json` files against the original tokenizers they were converted from.

1. Scan the most downloaded models on the Hub and keep those that ship both a tokenizer.json and an
   original tokenizer (SentencePiece model, tiktoken rank file, or Mistral tekken.json).
2. Group repos whose two files are byte-identical (fine-tunes and mirrors share tokenizers), adding up
   their downloads.
3. For each group, compare the original (reference) with `tokenizers` on tokenizer.json (candidate).

    python survey/audit_hub.py --scan 3000 --corpus stress --corpus sib200 --out survey/audit
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download

from tokparity import corpora
from tokparity.backends.hf import HFTokenizer
from tokparity.engine import compare
from tokparity.originals import load_original, original_file


def scan(api: HfApi, n: int) -> list[dict]:
    found = []
    for m in api.list_models(sort="downloads", limit=n, expand=["siblings", "downloads", "gated"]):
        files = {s.rfilename for s in (m.siblings or [])}
        orig = original_file(files)
        if "tokenizer.json" in files and orig and not m.gated:
            found.append({"repo": m.id, "downloads": m.downloads or 0, "original": orig})
    return found


def blob_ids(api: HfApi, repo: str, names: list[str]) -> tuple | None:
    try:
        info = api.model_info(repo, files_metadata=True)
    except Exception:
        return None
    by_name = {s.rfilename: (s.lfs.sha256 if s.lfs else s.blob_id) for s in info.siblings or []}
    return tuple(by_name.get(n) for n in names)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--scan", type=int, default=3000, help="how many of the most downloaded models to scan")
    p.add_argument("--corpus", action="append", required=True)
    p.add_argument("--minimize", type=int, default=5)
    p.add_argument("--out", required=True)
    args = p.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    api = HfApi()

    candidates = scan(api, args.scan)
    print(
        f"{len(candidates)} of the top {args.scan} models ship tokenizer.json and an original tokenizer",
        flush=True,
    )
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for c in candidates:
        key = blob_ids(api, c["repo"], ["tokenizer.json", c["original"]])
        if key and all(key):
            groups[key].append(c)
    print(f"{len(groups)} distinct (tokenizer.json, original) pairs", flush=True)
    (out / "groups.json").write_text(json.dumps([g for g in groups.values()], indent=1), encoding="utf-8")

    texts, labels = [], []
    for spec in args.corpus:
        for t, lab in corpora.load(spec):
            texts.append(t)
            labels.append(lab)

    for members in sorted(groups.values(), key=lambda g: -sum(m["downloads"] for m in g)):
        rep = max(members, key=lambda m: m["downloads"])
        total = sum(m["downloads"] for m in members)
        safe = rep["repo"].replace("/", "__")
        try:
            repo_dir = Path(hf_hub_download(rep["repo"], "tokenizer.json")).parent
            hf_hub_download(rep["repo"], rep["original"])
            try:
                hf_hub_download(rep["repo"], "config.json")
            except Exception:
                pass
            reference = load_original(repo_dir, rep["original"])
            result = compare(reference, HFTokenizer(repo_dir), texts, labels, minimize_n=args.minimize)
            data = result.to_dict()
        except Exception as e:
            data = {"error": f"{type(e).__name__}: {str(e).splitlines()[0][:300] if str(e) else ''}"}
        data |= {
            "model": rep["repo"],
            "original": rep["original"],
            "group_downloads": total,
            "group": [m["repo"] for m in sorted(members, key=lambda m: -m["downloads"])],
            "corpora": args.corpus,
        }
        (out / f"{safe}.json").write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
        status = data.get("error") or f"{data['n_mismatches']:6d} / {data['n_texts']}"
        print(f"{rep['repo']:60} {rep['original']:26} {total:>12,} dl  {status}", flush=True)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
