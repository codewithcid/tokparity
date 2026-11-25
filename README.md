# tokparity

**Differential testing for tokenizer ports.** tokparity runs Hugging Face [`tokenizers`](https://github.com/huggingface/tokenizers) (the reference) and a re-implementation of the same tokenizer on the same text, reports every input where the token ids differ, and shrinks each difference to a minimal repro.

Tokenizers are re-implemented again and again: in C++ for llama.cpp, in Go for Ollama, in JavaScript for transformers.js. A port that disagrees with the reference silently changes what the model sees. English test sentences rarely show it; decomposed accents, Indic vowel signs, zero-width joiners and non-ASCII digits do.

```text
$ tokparity run tiiuae/Falcon-H1-1B-Base --backend llamacpp --llama-lib ~/llama.cpp/build/bin \
    --llama-cpp-src ~/llama.cpp --corpus stress

## tiiuae/Falcon-H1-1B-Base: llama.cpp (lib) vs hf-tokenizers

22 of 1808 texts tokenize differently (1.22%).

Minimal repros:

| input | reference | candidate | tags |
|---|---|---|---|
| `ड\u093c` | `[20672]` | `[7095, 13373]` | combining-mark |
| `ש\u05c1` | `[669, 2580, 734]` | `[669, 626, 669, 734]` | combining-mark |
| `  #` | `[767, 526]` | `[731, 731, 526]` | punctuation, whitespace-run, leading-space, ascii-only |
```

## Bugs found with it

| project | what | status |
|---|---|---|
| [Ollama](https://github.com/ollama/ollama) (Go, MLX engine) | `Sequence` normalizers ignored: Llama 2, TinyLlama, CodeLlama and Phi-3 lose the leading `▁`, so every input differs | fix: [ollama#19910](https://github.com/ollama/ollama/pull/19910) |
| [tokenizers.js](https://github.com/huggingface/tokenizers.js) | `Precompiled` normalizer approximated with NFKC: decomposed Latin, Greek, Hangul and Indic text tokenized differently for XLM-R, bge-m3, multilingual-e5, mT5, NLLB, DeBERTa-v2 | fix: [tokenizers.js#57](https://github.com/huggingface/tokenizers.js/pull/57); issues [#58](https://github.com/huggingface/tokenizers.js/issues/58)-[#62](https://github.com/huggingface/tokenizers.js/issues/62) |
| [tokenizers.js](https://github.com/huggingface/tokenizers.js) | `Metaspace` pre-tokenizer ignores `split`: runs of spaces merge into one token for mmBERT (10.5% of texts differ) | see [RESULTS.md](RESULTS.md) |
| [tokenizers.js](https://github.com/huggingface/tokenizers.js) | `\w` in pre-tokenizer regexes was ASCII-only; Indic and Cyrillic words split | fix: [tokenizers.js#44](https://github.com/huggingface/tokenizers.js/pull/44) (parity data in the PR) |
| [llama.cpp](https://github.com/ggml-org/llama.cpp) | Qwen2.5/3/3.5, MiniMax-M2: the BPE path ignores the NFC normalizer flag | see [RESULTS.md](RESULTS.md) |
| [llama.cpp](https://github.com/ggml-org/llama.cpp) | Falcon-H1, Falcon3: mapped to the Llama 3 pre-tokenizer; Falcon-H1 splits combining marks off words (53% of Hindi/Malayalam texts differ) | see [RESULTS.md](RESULTS.md); related: [llama.cpp#32953](https://github.com/ggml-org/llama.cpp/issues/32953) |
| [llama.cpp](https://github.com/ggml-org/llama.cpp) | Kimi-K2: custom splitter drops `\p{M}`; SmolLM3: detected as `smaug-bpe` without `ignore_merges` | see [RESULTS.md](RESULTS.md) |

[RESULTS.md](RESULTS.md) has the full matrix: 37 tokenizers x llama.cpp, Ollama and tokenizers.js, on 33,007 texts each, with minimal repros.

## Install

```bash
pip install git+https://github.com/codewithcid/tokparity
```

Each backend needs its own runtime:

| backend | needs | setup |
|---|---|---|
| `llamacpp` | a llama.cpp build (shared library + `llama-tokenize`), and a checkout for `convert_hf_to_gguf.py` and its requirements | `--llama-lib DIR --llama-cpp-src DIR` (or `--gguf FILE`) |
| `ollama` | Go and an `ollama/ollama` checkout | `tokparity build-ollama --ollama-src DIR` |
| `tokenizersjs` | Node.js | `tokparity setup-tokenizersjs` (or `--tokenizersjs-module` to test a local build) |

Only tokenizer files are downloaded, never model weights. llama.cpp is called through ctypes on a vocab-only GGUF; the struct layout is pinned to release b12728, and `--llama-mode cli` uses `llama-tokenize` instead for other releases.

## Usage

```bash
# any Hugging Face repo id, or a local directory with tokenizer.json
tokparity run Qwen/Qwen3-8B --backend llamacpp --llama-lib ~/llama.cpp/build/bin \
    --llama-cpp-src ~/llama.cpp --corpus stress --corpus sib200 --out results/

tokparity run NousResearch/Llama-2-7b-hf --backend ollama --corpus my_texts.txt

# test a tokenizers.js branch before sending the PR
tokparity run BAAI/bge-m3 --backend tokenizersjs --tokenizersjs-module ../tokenizers.js/dist/tokenizers.mjs

tokparity report results/*.json      # summary table
```

`--fail-on-mismatch` makes `run` exit with status 1, for CI.

### Auditing tokenizer.json itself

`tokparity audit` turns the comparison around: the reference is the tokenizer the model was trained with, and the candidate is the model's `tokenizer.json` run by `tokenizers`. The original is detected from the repo files: a SentencePiece model (`tokenizer.model`, `spiece.model`, `sentencepiece.bpe.model`), Meta's tiktoken file (`original/tokenizer.model`) or Mistral's `tekken.json`.

```bash
pip install "tokparity[originals] @ git+https://github.com/codewithcid/tokparity"
tokparity audit baidu/ERNIE-4.5-21B-A3B-Thinking --corpus stress --corpus sib200
```

`survey/audit_hub.py` runs the audit over the most downloaded Hub models. Of the top 5,000, 290 ship both a `tokenizer.json` and an original tokenizer; they share 146 distinct file pairs, and 47 of those match exactly. Among the rest:

| models | difference |
|---|---|
| umT5, MADLAD-400 | tabs, newlines and repeated spaces are lost: the SentencePiece models keep whitespace (`remove_extra_whitespaces: false`, byte fallback) |
| Sarvam-1, Salamandra | the SentencePiece normalization (`nmt_nfkc` and custom rules) is not applied: `ＡＢＣ ﬁle` is 3 tokens in Sarvam-1's SentencePiece model and 15 byte tokens in `tokenizer.json` |
| ERNIE-4.5-21B-A3B-Thinking | a `▁` is prepended to every text segment, also after special tokens in chat prompts; the original has no dummy prefix, nor do the other ERNIE-4.5 repos |
| XLM-R family (bge-m3, multilingual-e5, ...), T5, NLLB | the `Precompiled` normalizer works per grapheme, unlike SentencePiece (known: [tokenizers#2116](https://github.com/huggingface/tokenizers/issues/2116)) |

Some differences are intended: ALBERT, XLNet and SigLIP lowercase or strip accents in `tokenizer.json` on purpose.

### Corpora

- `stress` (default): 1,808 generated strings aimed at the places ports go wrong: canonically decomposed text in every script (and characters NFC itself changes), every combining mark after a letter of its block, ZWJ/ZWNJ inside words, the decimal digits of every script, whitespace runs before punctuation, case-split words and contractions, compatibility characters and code-mixed text. Generated from the Unicode database at run time, so it is deterministic for a given Python version.
- `sib200[:N]`: the first N (default 60) sentences of each of the 205 languages of [SIB-200](https://huggingface.co/datasets/Davlan/sib200) (FLORES-200 sentences, CC BY-SA 4.0).
- a `.txt` (one text per line), `.json` (strings, or `{"text", "label"}` objects) or `.jsonl` file.

### How it works

1. Both tokenizers encode every text without special tokens; texts whose ids differ are mismatches.
2. Up to `--minimize N` mismatches are shrunk with delta debugging: trim the ends, then remove ever smaller chunks while the tokenizers still disagree. One mismatch per corpus label goes first, and mismatches that contain an already-found repro are skipped, so the repros cover different bugs. Candidates are sent to the backend a round at a time; the Ollama and tokenizers.js harnesses stay running and answer one JSON line per round.
3. Each repro is tagged with the Unicode features it contains (`not-nfc`, `combining-mark`, `zero-width-joiner`, `non-ascii-digit`, `whitespace-run`, ...) to group mismatches by likely cause.

### Python API

```python
from tokparity import compare
from tokparity.backends.hf import HFTokenizer
from tokparity.backends.llamacpp import LlamaCpp
from tokparity.corpora import stress

texts, labels = zip(*stress())
result = compare(HFTokenizer("path/to/model"), LlamaCpp("vocab.gguf", "llama.cpp/build/bin"), texts, labels)
for m in result.mismatches:
    if m.minimal:
        print(repr(m.minimal), m.minimal_reference, m.minimal_candidate, m.tags)
```

A backend is any object with `name`, `encode_batch(texts) -> list[list[int]]` and `piece(id) -> str`.

## Scope

- The reference is the model's `tokenizer.json` run by `tokenizers`. Models that only ship a SentencePiece or tiktoken file are skipped.
- Ids are compared without special tokens, so BOS/EOS handling and chat templates are out of scope.
- A difference is a disagreement with `tokenizers`, which is what the models were trained with in almost every case. It is not always a bug in the port: some `tokenizer.json` files themselves disagree with the original tokenizer (Mistral-Small-3.x's regex, for example).

## License

Apache-2.0. SIB-200 / FLORES-200 data is downloaded at run time and is not redistributed here.
