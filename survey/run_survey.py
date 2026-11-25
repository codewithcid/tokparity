"""Run tokparity over many models and backends and write one JSON result per model and backend.

    python survey/run_survey.py --hf-dir hf --gguf-dir gguf --llama-lib vendor/llama-b12728 \
        --ollama-bin main=vendor/ollama-encode-main.exe \
        --tokenizersjs release=PATH/TO/@huggingface/tokenizers/dist/tokenizers.mjs \
        --corpus stress --corpus sib200 --out survey/results

`--hf-dir/NAME` must hold the tokenizer files of each model (see models.txt) and `--gguf-dir/NAME.gguf`
its vocab-only GGUF (`tokparity.models.vocab_gguf` makes both from the repo ids).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from tokparity import corpora
from tokparity.backends.hf import HFTokenizer
from tokparity.engine import compare


def read_models(path: Path) -> list[tuple[str, str]]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip() and not line.startswith("#"):
            name, repo = line.split()
            rows.append((name, repo))
    return rows


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--models", default=str(Path(__file__).with_name("models.txt")))
    p.add_argument("--hf-dir", required=True)
    p.add_argument("--gguf-dir")
    p.add_argument("--llama-lib")
    p.add_argument("--ollama-bin", action="append", default=[], help="LABEL=PATH, repeatable")
    p.add_argument("--tokenizersjs", action="append", default=[], help="LABEL=MODULE, repeatable")
    p.add_argument("--corpus", action="append", required=True)
    p.add_argument("--minimize", type=int, default=10)
    p.add_argument("--out", required=True)
    args = p.parse_args()

    texts, labels = [], []
    for spec in args.corpus:
        for t, lab in corpora.load(spec):
            texts.append(t)
            labels.append(lab)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    candidates = []
    if args.llama_lib:
        candidates.append(("llamacpp", None))
    candidates += [("ollama-" + spec.split("=")[0], spec.split("=")[1]) for spec in args.ollama_bin]
    candidates += [("tokenizersjs-" + spec.split("=")[0], spec.split("=")[1]) for spec in args.tokenizersjs]

    for name, repo in read_models(Path(args.models)):
        model_dir = Path(args.hf_dir) / name
        try:
            reference = HFTokenizer(model_dir)
        except FileNotFoundError:
            print(f"{name:22} no tokenizer.json, skipped", flush=True)
            continue
        for backend, extra in candidates:
            try:
                if backend == "llamacpp":
                    from tokparity.backends.llamacpp import LlamaCpp

                    cand = LlamaCpp(Path(args.gguf_dir) / f"{name}.gguf", args.llama_lib)
                elif backend.startswith("ollama"):
                    from tokparity.backends.ollama import Ollama

                    cand = Ollama(model_dir, extra)
                else:
                    from tokparity.backends.tokenizersjs import TokenizersJS

                    cand = TokenizersJS(model_dir, extra)
                result = compare(reference, cand, texts, labels, minimize_n=args.minimize)
            except Exception as e:  # unsupported tokenizer in this backend, missing GGUF, ...
                status = {
                    "model": name,
                    "repo": repo,
                    "backend": backend,
                    "error": str(e).splitlines()[0][:300],
                }
                (out / f"{name}__{backend}.json").write_text(json.dumps(status), encoding="utf-8")
                print(f"{name:22} {backend:14} ERROR {status['error'][:90]}", flush=True)
                continue
            data = result.to_dict() | {
                "model": name,
                "repo": repo,
                "backend": backend,
                "corpora": args.corpus,
            }
            (out / f"{name}__{backend}.json").write_text(
                json.dumps(data, ensure_ascii=False), encoding="utf-8"
            )
            print(
                f"{name:22} {backend:14} {result.n_mismatches:6d} / {result.n_texts}  "
                f"tags={dict(result.tag_counts().most_common(4))}",
                flush=True,
            )


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
