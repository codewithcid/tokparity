# Results

37 tokenizers from the Hugging Face Hub (every model in `survey/models.txt` that ships a `tokenizer.json`), compared with Hugging Face `tokenizers` 0.23.3 on 33,007 texts each:

- `stress`: the 1,808 generated edge cases (see the README);
- `sib200`: 60 sentences in each of 205 languages from SIB-200 / FLORES-200 (12,300);
- `indic`: 18,899 Hindi and Malayalam texts, code-mixed with English, in Latin and native scripts (YouTube comments from DravidianCodeMix and SMS-style messages; not redistributed).

Backends, as of 2026-11-26:

| column | what |
|---|---|
| `llamacpp` | llama.cpp release b12728, vocab-only GGUFs made with its own `convert_hf_to_gguf.py` |
| `ollama-main` | Ollama's Go tokenizer (`mlxrunner/tokenizer`) at `784dd54` |
| `ollama-fix` | the same with [ollama#19910](https://github.com/ollama/ollama/pull/19910) |
| `tokenizersjs-release` | `@huggingface/tokenizers` 0.2.0 from npm |
| `tokenizersjs-pr57` | tokenizers.js with [tokenizers.js#57](https://github.com/huggingface/tokenizers.js/pull/57) |

Reproduce with `survey/run_survey.py` and `survey/summarize.py`.

## Matrix

Texts per model: 33007 (stress, sib200, indic_18899).
Each cell: share of texts whose token ids differ from Hugging Face `tokenizers`.
`n/a` = the backend cannot load this tokenizer.

| model | llamacpp | ollama-main | ollama-fix | tokenizersjs-release | tokenizersjs-pr57 |
|---|---:|---:|---:|---:|---:|
| bge-m3 | **1.84%** | n/a | n/a | **1.22%** | <0.01% |
| codellama | 0 | **100.00%** | 0 | 0 | 0 |
| deepseek-r1-qwen | **4.43%** | 0 | 0 | 0 | 0 |
| deepseek-v3 | 0 | 0 | 0 | 0 | 0 |
| exaone4 | 0 | 0 | 0 | 0 | 0 |
| falcon-h1 | **35.55%** | n/a | n/a | 0 | 0 |
| falcon3 | **1.68%** | n/a | n/a | 0 | 0 |
| gemma2 | 0 | 0 | 0 | 0 | 0 |
| gemma3 | 0 | 0 | 0 | 0 | 0 |
| glm45 | 0 | 0 | 0 | 0 | 0 |
| gpt-oss | 0 | 0 | 0 | 0 | 0 |
| granite-embed-multi | 0 | 0 | 0 | 0 | 0 |
| granite4 | 0 | 0 | 0 | 0 | 0 |
| hunyuan | 0 | 0 | 0 | 0 | 0 |
| llama2 | 0 | **100.00%** | 0 | 0 | 0 |
| llama3 | 0 | 0 | 0 | 0 | 0 |
| llama4 | 0 | 0 | 0 | 0 | 0 |
| me5-base | **1.17%** | n/a | n/a | **1.22%** | <0.01% |
| minimax-m2 | **4.43%** | 0 | 0 | 0 | 0 |
| mistral-nemo | 0 | 0 | 0 | 0 | 0 |
| mistral7b-v03 | 0.06% | 0 | 0 | 0 | 0 |
| mmbert | 0 | n/a | n/a | **10.54%** | **10.54%** |
| nemotron-h | 0 | 0 | 0 | 0 | 0 |
| olmo2 | 0 | 0 | 0 | 0 | 0 |
| phi3 | 0 | **100.00%** | 0 | 0 | 0 |
| phi4-mini | 0 | 0 | 0 | 0 | 0 |
| poro | 0 | n/a | n/a | 0 | 0 |
| qwen25 | **4.43%** | 0 | 0 | 0 | 0 |
| qwen3-embed | **4.43%** | 0 | 0 | 0 | 0 |
| qwen35 | **4.43%** | 0 | 0 | 0 | 0 |
| qwen3 | **4.43%** | 0 | 0 | 0 | 0 |
| sarvam-1 | 0.07% | 0 | 0 | 0 | 0 |
| sarvam-m | 0 | 0 | 0 | 0 | 0 |
| sarvam-moe | 0 | 0 | 0 | 0 | 0 |
| smollm3 | 0.36% | 0 | 0 | 0 | 0 |
| t5 | 0.09% | n/a | n/a | 0.10% | <0.01% |
| tinyllama | 0 | **100.00%** | 0 | 0 | 0 |

## Findings

### llama.cpp: the NFC normalizer is ignored for BPE vocabularies

Qwen2.5, Qwen3, Qwen3.5, Qwen3-Embedding, DeepSeek-R1-Distill-Qwen and MiniMax-M2 declare an `NFC` normalizer. The GGUF records it (`tokenizer.ggml.normalizer.nfc`), but only the Unigram (UGM) path applies it, so text that is not already NFC is tokenized differently: 1,462 of 33,007 texts. In SIB-200 the worst languages are Manipuri (53 of 60 sentences), Kashmiri in Devanagari (49), Assamese (44), Bengali (40), Punjabi (30) and Yoruba (21). Normalizing the text to NFC before calling llama.cpp removes every one of these differences.

| input | `tokenizers` | llama.cpp |
|---|---|---|
| `A\u0300` | `[71953]` | `[32, 96549]` |
| `Α\u0301` (Greek) | `[146488]` | `[144842, 53839]` |
| `ا\u0653` (Arabic) | `[124340]` | `[5703, 149, 241]` |

### llama.cpp: Falcon-H1 and Falcon3 use the Llama 3 pre-tokenizer

`falcon-h1` and `falcon3` are mapped to `LLAMA_VOCAB_PRE_TYPE_LLAMA3`. Falcon-H1's `tokenizer.json` instead runs `Punctuation`, then a Tekken-style regex whose letter classes include `\p{M}`, so combining marks stay inside words; the Llama 3 regex splits every vowel sign and nukta off its letter. 10,141 of the 18,899 Hindi/Malayalam texts and all 60 sentences of Hindi, Marathi, Nepali, Sanskrit, Punjabi, Malayalam, Thai and Burmese differ. Emulating "`Punctuation`, then the Tekken regex" with llama.cpp gives 0 differences on all 33,007 texts; "`Punctuation`, then the Llama 3 regex" (the closed llama.cpp#32954) still leaves 10,034 Hindi/Malayalam texts different. [llama.cpp#32953](https://github.com/ggml-org/llama.cpp/issues/32953) reports the whitespace-before-punctuation part.

Falcon3 has `Punctuation`, `ByteLevel` with the GPT-2 regex, and `Digits`; "`Punctuation`, then the GPT-2 regex" brings its Hindi/Malayalam differences from 233 to 0.

| model | input | `tokenizers` | llama.cpp |
|---|---|---|---|
| Falcon-H1 | `ड\u093c` | `[20672]` | `[7095, 13373]` |
| Falcon-H1 | `  #` | `[767, 526]` | `[731, 731, 526]` |

### llama.cpp: Kimi-K2's pre-tokenizer drops combining marks

Kimi-K2 ships a tiktoken file, not `tokenizer.json`, so it is not in the matrix. Against its own `TikTokenTokenizer`, 9,881 of 18,899 Hindi/Malayalam texts and 1,693 of 12,300 SIB-200 sentences differ, and every differing text contains a combining mark: `unicode_regex_split_custom_kimi_k2` tests only `is_letter`, while the pattern's letter classes include `\p{M}` (`अरे`: `[89750, 159170]` vs `[89750, 9261, 7614]`).

### llama.cpp: SmolLM3 is detected as `smaug-bpe`

SmolLM3 and Smaug-Llama-3 have the same vocabulary and merges, so the converter's hash gives `smaug-bpe`, which runs without `ignore_merges`. SmolLM3's `tokenizer.json` sets `ignore_merges: true` (Smaug's does not), and 119 of 12,300 SIB-200 sentences differ, mostly Vietnamese, Turkish, Czech and Arabic. Converting with `llama-bpe` gives 0.

### llama.cpp: Unigram normalization (bge-m3, multilingual-e5, T5)

Decomposed text with more than one combining mark can come out as `<unk>`: `U\u0308\u0304` is `[8578]` (`Ǖ`) in `tokenizers` and `[6, 3]` in llama.cpp for bge-m3. Some trailing whitespace is also dropped where `tokenizers` emits `▁`. bge-m3: 607 differing texts, multilingual-e5: 385.

### Ollama: `Sequence` normalizers are ignored

Fixed by [ollama#19910](https://github.com/ollama/ollama/pull/19910). Llama 2, TinyLlama, CodeLlama and Phi-3 use `Sequence[Prepend("▁"), Replace(" ", "▁")]`, which `mlxrunner/tokenizer` ignored, so the first word of every input lost its `▁`: 100% of texts differ on `main`, 0 with the fix.

### tokenizers.js: `Precompiled` normalizer

Fixed by [tokenizers.js#57](https://github.com/huggingface/tokenizers.js/pull/57): bge-m3 and multilingual-e5 go from 403 differing texts to 1, T5 from 33 to 1. The one left in each is a tie between equally scored Unigram segmentations (`w99999`), reported as [tokenizers.js#60](https://github.com/huggingface/tokenizers.js/issues/60).

### tokenizers.js: the Metaspace pre-tokenizer ignores `split`

mmBERT's `tokenizer.json` has a `Metaspace` pre-tokenizer without a `split` field, which `tokenizers` reads as `split: true`: the text is split before every `▁`. tokenizers.js never splits, so runs of spaces merge into one token (`"  "`: `[235248, 235248]` vs `[139]`): 3,264 of 18,899 Hindi/Malayalam texts and 10.5% of all texts differ.

### Smaller differences

- llama.cpp, Mistral-7B-v0.3 (19 texts) and Sarvam-1 (24): whitespace-only input and whitespace before a word (`" "`, `"\tb"`).
- Ollama cannot load some tokenizers at all (`Punctuation` pre-tokenizer, Unigram models, legacy `Metaspace` fields, a nested character class in Poro's regex). These are load errors, not wrong ids, and are shown as `n/a`.

## All minimal repros

Up to five per model and backend, from `--minimize 10`.

### bge-m3 / llamacpp: 607 of 33007

| input | `tokenizers` | backend | tags |
|---|---|---|---|
| `U\u0308\u0304` | `[8578]` | `[6, 3]` | not-nfc, combining-mark |
| `ι\u0308\u0301` | `[6, 1473, 246635, 4868]` | `[6, 78877]` | not-nfc, combining-mark |
| `न\u093c` | `[1618, 27885]` | `[6, 244401]` | not-nfc, combining-mark |
| `ஒ\u0bd7` | `[34681, 3]` | `[6, 249684]` | not-nfc, combining-mark |
| `ဥ\u102e` | `[6, 41845, 24226]` | `[6, 245096]` | not-nfc, combining-mark |

### bge-m3 / tokenizersjs-release: 403 of 33007

| input | `tokenizers` | backend | tags |
|---|---|---|---|
| `U\u0308\u0304` | `[8578]` | `[6, 3]` | not-nfc, combining-mark |
| `ι\u0308\u0301` | `[6, 1473, 246635, 4868]` | `[6, 78877]` | not-nfc, combining-mark |
| `न\u093c` | `[1618, 27885]` | `[6, 244401]` | not-nfc, combining-mark |
| `ஒ\u0bd7` | `[34681, 3]` | `[6, 249684]` | not-nfc, combining-mark |
| `ဥ\u102e` | `[6, 41845, 24226]` | `[6, 245096]` | not-nfc, combining-mark |

### bge-m3 / tokenizersjs-pr57: 1 of 33007

| input | `tokenizers` | backend | tags |
|---|---|---|---|
| `w99999` | `[148, 65040, 5046]` | `[148, 5046, 65040]` | ascii-only |

### codellama / ollama-main: 33007 of 33007

| input | `tokenizers` | backend | tags |
|---|---|---|---|
| `\u0300` | `[29871, 30712]` | `[30712]` | combining-mark |
| `ʹ` | `[29871, 208, 183]` | `[208, 183]` | not-nfc |
| `\u0301` | `[29871, 30103]` | `[30103]` | combining-mark |
| `\u0653` | `[29871, 220, 150]` | `[220, 150]` | combining-mark |
| `\u093c` | `[29871, 31821]` | `[31821]` | combining-mark |

### deepseek-r1-qwen / llamacpp: 1462 of 33007

| input | `tokenizers` | backend | tags |
|---|---|---|---|
| `A\u0300` | `[71953]` | `[32, 96549]` | not-nfc, combining-mark |
| `ʹ` | `[145217]` | `[150354]` | not-nfc |
| `Α\u0301` | `[146488]` | `[144842, 53839]` | not-nfc, combining-mark |
| `Е\u0300` | `[147089]` | `[55091, 96549]` | not-nfc, combining-mark |
| `ا\u0653` | `[124340]` | `[5703, 149, 241]` | not-nfc, combining-mark |

### falcon-h1 / llamacpp: 11734 of 33007

| input | `tokenizers` | backend | tags |
|---|---|---|---|
| ` ا\u0653` | `[55931, 752]` | `[2717, 671, 752]` | not-nfc, combining-mark, leading-space |
| `ड\u093c` | `[20672]` | `[7095, 13373]` | combining-mark |
| `য\u09bc` | `[48870]` | `[48971, 20459]` | combining-mark |
| `フ\u3099` | `[8376, 758]` | `[4685, 872, 758]` | not-nfc, combining-mark |
| `ש\u05c1` | `[669, 2580, 734]` | `[669, 626, 669, 734]` | combining-mark |

### falcon3 / llamacpp: 554 of 33007

| input | `tokenizers` | backend | tags |
|---|---|---|---|
| `ĉ` | `[13]` | `[2150, 2237]` |  |
| `\u1063ၡ` | `[2179, 2229, 2120, 2179, 2229, 2118]` | `[2179, 2229, 94681, 2229, 2118]` | combining-mark |
| `\u1923ᤞ` | `[2179, 2121, 2120, 2179, 2121, 2258]` | `[2179, 2121, 94681, 2121, 2258]` | combining-mark |
| `\u1a1bᨖ` | `[2179, 2125, 2255, 2179, 2125, 2250]` | `[2179, 2125, 111043, 2125, 2250]` | combining-mark |
| `\u1a5bᩔ` | `[2179, 2126, 2255, 2179, 2126, 2248]` | `[2179, 2126, 111043, 2126, 2248]` | combining-mark |

### llama2 / ollama-main: 33007 of 33007

| input | `tokenizers` | backend | tags |
|---|---|---|---|
| `\u0300` | `[29871, 30712]` | `[30712]` | combining-mark |
| `ʹ` | `[29871, 208, 183]` | `[208, 183]` | not-nfc |
| `\u0301` | `[29871, 30103]` | `[30103]` | combining-mark |
| `\u0653` | `[29871, 220, 150]` | `[220, 150]` | combining-mark |
| `\u093c` | `[29871, 31821]` | `[31821]` | combining-mark |

### me5-base / llamacpp: 385 of 33007

| input | `tokenizers` | backend | tags |
|---|---|---|---|
| `U\u0308\u0304` | `[8578]` | `[6, 3]` | not-nfc, combining-mark |
| `ι\u0308\u0301` | `[6, 1473, 246635, 4868]` | `[6, 78877]` | not-nfc, combining-mark |
| `न\u093c` | `[1618, 27885]` | `[6, 244401]` | not-nfc, combining-mark |
| `ஒ\u0bd7` | `[34681, 3]` | `[6, 249684]` | not-nfc, combining-mark |
| `ဥ\u102e` | `[6, 41845, 24226]` | `[6, 245096]` | not-nfc, combining-mark |

### me5-base / tokenizersjs-release: 403 of 33007

| input | `tokenizers` | backend | tags |
|---|---|---|---|
| `U\u0308\u0304` | `[8578]` | `[6, 3]` | not-nfc, combining-mark |
| `ι\u0308\u0301` | `[6, 1473, 246635, 4868]` | `[6, 78877]` | not-nfc, combining-mark |
| `न\u093c` | `[1618, 27885]` | `[6, 244401]` | not-nfc, combining-mark |
| `ஒ\u0bd7` | `[34681, 3]` | `[6, 249684]` | not-nfc, combining-mark |
| `ဥ\u102e` | `[6, 41845, 24226]` | `[6, 245096]` | not-nfc, combining-mark |

### me5-base / tokenizersjs-pr57: 1 of 33007

| input | `tokenizers` | backend | tags |
|---|---|---|---|
| `w99999999` | `[148, 65040, 5046, 65040]` | `[148, 5046, 65040, 65040]` | ascii-only |

### minimax-m2 / llamacpp: 1462 of 33007

| input | `tokenizers` | backend | tags |
|---|---|---|---|
| `A\u0300` | `[21459]` | `[65, 133135]` | not-nfc, combining-mark |
| `ʹ` | `[202, 185]` | `[205, 180]` | not-nfc |
| `Α\u0301` | `[206, 134]` | `[98059, 21213]` | not-nfc, combining-mark |
| `Е\u0300` | `[208, 128]` | `[13734, 133135]` | not-nfc, combining-mark |
| `ا\u0653` | `[23003]` | `[349, 217, 147]` | not-nfc, combining-mark |

### mistral7b-v03 / llamacpp: 19 of 33007

| input | `tokenizers` | backend | tags |
|---|---|---|---|
| ` ` | `[29473]` | `[1027]` | leading-space, trailing-space, ascii-only |

### mmbert / tokenizersjs-release: 3478 of 33007

| input | `tokenizers` | backend | tags |
|---|---|---|---|
| `  ` | `[235248, 235248]` | `[139]` | whitespace-run, leading-space, trailing-space, ascii-only |

### mmbert / tokenizersjs-pr57: 3478 of 33007

| input | `tokenizers` | backend | tags |
|---|---|---|---|
| `  ` | `[235248, 235248]` | `[139]` | whitespace-run, leading-space, trailing-space, ascii-only |

### phi3 / ollama-main: 33007 of 33007

| input | `tokenizers` | backend | tags |
|---|---|---|---|
| `\u0300` | `[29871, 30712]` | `[30712]` | combining-mark |
| `ʹ` | `[29871, 208, 183]` | `[208, 183]` | not-nfc |
| `\u0301` | `[29871, 30103]` | `[30103]` | combining-mark |
| `\u0653` | `[29871, 220, 150]` | `[220, 150]` | combining-mark |
| `\u093c` | `[29871, 31821]` | `[31821]` | combining-mark |

### qwen25 / llamacpp: 1462 of 33007

| input | `tokenizers` | backend | tags |
|---|---|---|---|
| `A\u0300` | `[71953]` | `[32, 96549]` | not-nfc, combining-mark |
| `ʹ` | `[145217]` | `[150354]` | not-nfc |
| `Α\u0301` | `[146488]` | `[144842, 53839]` | not-nfc, combining-mark |
| `Е\u0300` | `[147089]` | `[55091, 96549]` | not-nfc, combining-mark |
| `ا\u0653` | `[124340]` | `[5703, 149, 241]` | not-nfc, combining-mark |

### qwen3-embed / llamacpp: 1462 of 33007

| input | `tokenizers` | backend | tags |
|---|---|---|---|
| `A\u0300` | `[71953]` | `[32, 96549]` | not-nfc, combining-mark |
| `ʹ` | `[145217]` | `[150354]` | not-nfc |
| `Α\u0301` | `[146488]` | `[144842, 53839]` | not-nfc, combining-mark |
| `Е\u0300` | `[147089]` | `[55091, 96549]` | not-nfc, combining-mark |
| `ا\u0653` | `[124340]` | `[5703, 149, 241]` | not-nfc, combining-mark |

### qwen35 / llamacpp: 1462 of 33007

| input | `tokenizers` | backend | tags |
|---|---|---|---|
| `A\u0300` | `[69508]` | `[32, 93213]` | not-nfc, combining-mark |
| `ʹ` | `[134, 117]` | `[137, 112]` | not-nfc |
| `Α\u0301` | `[169347]` | `[153061, 52033]` | not-nfc, combining-mark |
| `Е\u0300` | `[140, 222]` | `[53241, 93213]` | not-nfc, combining-mark |
| `ا\u0653` | `[148975]` | `[5525, 149, 241]` | not-nfc, combining-mark |

### qwen3 / llamacpp: 1462 of 33007

| input | `tokenizers` | backend | tags |
|---|---|---|---|
| `A\u0300` | `[71953]` | `[32, 96549]` | not-nfc, combining-mark |
| `ʹ` | `[145217]` | `[150354]` | not-nfc |
| `Α\u0301` | `[146488]` | `[144842, 53839]` | not-nfc, combining-mark |
| `Е\u0300` | `[147089]` | `[55091, 96549]` | not-nfc, combining-mark |
| `ا\u0653` | `[124340]` | `[5703, 149, 241]` | not-nfc, combining-mark |

### sarvam-1 / llamacpp: 24 of 33007

| input | `tokenizers` | backend | tags |
|---|---|---|---|
| ` ` | `[67476]` | `[4363]` | leading-space, trailing-space, ascii-only |
| `\u0009b` | `[4105, 67543]` | `[4105, 4403]` | leading-space, ascii-only |
| `\u000an` | `[4103, 67483]` | `[4103, 4531]` | leading-space, ascii-only |
| `\u000aN` | `[4103, 67758]` | `[4103, 4916]` | leading-space, ascii-only |
| `\u000aw` | `[4103, 67556]` | `[4103, 4388]` | leading-space, ascii-only |

### smollm3 / llamacpp: 120 of 33007

| input | `tokenizers` | backend | tags |
|---|---|---|---|
| ` جديدة` | `[119846]` | `[83268, 80920, 102539]` | leading-space |
| ` الحياة` | `[112224]` | `[100688, 101160, 26957]` | leading-space |
| ` اطلاع` | `[124833]` | `[102992, 82070, 24102]` | leading-space |
| ` görüntü` | `[115385]` | `[58804, 2448, 406, 2448]` | leading-space, mixed-script |
| ` daleko` | `[127232]` | `[294, 1604, 9509]` | leading-space, ascii-only |

### t5 / llamacpp: 29 of 33007

| input | `tokenizers` | backend | tags |
|---|---|---|---|
| `U\u0308\u0304` | `[3, 14858]` | `[3, 2]` | not-nfc, combining-mark |
| `u\u0308\u0304` | `[3, 1272]` | `[3, 2]` | not-nfc, combining-mark |
| `U\u0308\u0301` | `[3, 14858]` | `[3, 2]` | not-nfc, combining-mark |
| `u\u0308\u0301` | `[3, 1272]` | `[3, 2]` | not-nfc, combining-mark |
| `U\u0308\u030c` | `[3, 14858]` | `[3, 2]` | not-nfc, combining-mark |

### t5 / tokenizersjs-release: 33 of 33007

| input | `tokenizers` | backend | tags |
|---|---|---|---|
| `U\u0308\u0304` | `[3, 14858]` | `[3, 2]` | not-nfc, combining-mark |
| `u\u0308\u0304` | `[3, 1272]` | `[3, 2]` | not-nfc, combining-mark |
| `U\u0308\u0301` | `[3, 14858]` | `[3, 2]` | not-nfc, combining-mark |
| `u\u0308\u0301` | `[3, 1272]` | `[3, 2]` | not-nfc, combining-mark |
| `U\u0308\u030c` | `[3, 14858]` | `[3, 2]` | not-nfc, combining-mark |

### t5 / tokenizersjs-pr57: 1 of 33007

| input | `tokenizers` | backend | tags |
|---|---|---|---|
| `loadingggg` | `[12115, 4102, 122]` | `[12115, 122, 4102]` | ascii-only |

### tinyllama / ollama-main: 33007 of 33007

| input | `tokenizers` | backend | tags |
|---|---|---|---|
| `\u0300` | `[29871, 30712]` | `[30712]` | combining-mark |
| `ʹ` | `[29871, 208, 183]` | `[208, 183]` | not-nfc |
| `\u0301` | `[29871, 30103]` | `[30103]` | combining-mark |
| `\u0653` | `[29871, 220, 150]` | `[220, 150]` | combining-mark |
| `\u093c` | `[29871, 31821]` | `[31821]` | combining-mark |

