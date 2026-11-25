"""tokparity command line."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from tokparity import corpora, report
from tokparity.engine import compare


def _candidate(args, model: str):
    if args.backend == "llamacpp":
        from tokparity.backends.llamacpp import LlamaCpp
        from tokparity.models import vocab_gguf

        if not args.llama_lib:
            sys.exit(
                "--llama-lib DIR (llama.cpp shared library / binaries) is required for --backend llamacpp"
            )
        if args.gguf:
            gguf = args.gguf
        elif args.llama_cpp_src:
            gguf = vocab_gguf(model, args.llama_cpp_src)
        else:
            sys.exit("give --gguf FILE, or --llama-cpp-src DIR to convert the tokenizer with llama.cpp")
        return LlamaCpp(gguf, args.llama_lib, mode=args.llama_mode)
    if args.backend == "ollama":
        from tokparity.backends.ollama import Ollama
        from tokparity.models import model_dir

        return Ollama(model_dir(model), args.ollama_bin)
    if args.backend == "tokenizersjs":
        from tokparity.backends.tokenizersjs import TokenizersJS
        from tokparity.models import model_dir

        return TokenizersJS(model_dir(model), args.tokenizersjs_module)
    sys.exit(f"unknown backend {args.backend}")


def cmd_run(args) -> int:
    from tokparity.backends.hf import HFTokenizer
    from tokparity.models import model_dir

    texts, labels = [], []
    for spec in args.corpus or ["stress"]:
        for text, label in corpora.load(spec):
            texts.append(text)
            labels.append(label)
    reference = HFTokenizer(model_dir(args.model))
    candidate = _candidate(args, args.model)
    result = compare(reference, candidate, texts, labels, minimize_n=args.minimize)
    data = result.to_dict()
    data["model"] = args.model
    data["corpora"] = args.corpus or ["stress"]
    md = report.single(data, args.model)
    if args.out:
        out = Path(args.out)
        out.mkdir(parents=True, exist_ok=True)
        stem = f"{args.model.replace('/', '__')}__{args.backend}"
        (out / f"{stem}.json").write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
        (out / f"{stem}.md").write_text(md, encoding="utf-8")
    sys.stdout.reconfigure(encoding="utf-8")
    print(md)
    return 1 if result.n_mismatches and args.fail_on_mismatch else 0


def cmd_build_ollama(args) -> int:
    from tokparity.backends.ollama import build

    print(build(args.ollama_src, args.out))
    return 0


def cmd_setup_tokenizersjs(args) -> int:
    from tokparity.backends.tokenizersjs import setup

    print(setup(args.version))
    return 0


def cmd_report(args) -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    print(report.summary(args.results))
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="tokparity", description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)

    run = sub.add_parser("run", help="compare a backend with Hugging Face tokenizers")
    run.add_argument("model", help="Hugging Face repo id or local directory with tokenizer.json")
    run.add_argument("--backend", choices=["llamacpp", "ollama", "tokenizersjs"], required=True)
    run.add_argument(
        "--corpus",
        action="append",
        help="'stress' (default), 'sib200[:N]', or a .txt/.json/.jsonl file; repeatable",
    )
    run.add_argument("--minimize", type=int, default=20, help="shrink the first N mismatches (default 20)")
    run.add_argument("--out", help="directory for the JSON and Markdown results")
    run.add_argument("--fail-on-mismatch", action="store_true", help="exit with status 1 on any mismatch")
    run.add_argument("--gguf", help="llamacpp: vocab-only GGUF to use")
    run.add_argument("--llama-cpp-src", help="llamacpp: llama.cpp checkout, to convert the tokenizer")
    run.add_argument("--llama-lib", help="llamacpp: directory with llama.cpp's shared library / binaries")
    run.add_argument("--llama-mode", choices=["lib", "cli"], default="lib")
    run.add_argument("--ollama-bin", help="ollama: harness binary (default: built by build-ollama)")
    run.add_argument(
        "--tokenizersjs-module",
        help="tokenizersjs: ESM entry to test, e.g. a checkout's dist/tokenizers.mjs (default: npm package)",
    )
    run.set_defaults(func=cmd_run)

    b = sub.add_parser("build-ollama", help="compile the Ollama tokenizer harness (needs Go)")
    b.add_argument("--ollama-src", required=True, help="path to an ollama/ollama checkout")
    b.add_argument("--out", help="where to write the binary")
    b.set_defaults(func=cmd_build_ollama)

    js = sub.add_parser("setup-tokenizersjs", help="install @huggingface/tokenizers from npm (needs Node.js)")
    js.add_argument("--version", default="latest")
    js.set_defaults(func=cmd_setup_tokenizersjs)

    r = sub.add_parser("report", help="summary table of saved results")
    r.add_argument("results", nargs="+")
    r.set_defaults(func=cmd_report)

    args = p.parse_args(argv)
    try:
        return args.func(args)
    except (FileNotFoundError, RuntimeError) as e:
        print(f"tokparity: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
