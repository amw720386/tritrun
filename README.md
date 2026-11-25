# tritrun

A small CUDA inference engine for **Ternary-Bonsai-2.5-27B** (PTQ1_0, 1.75 bits per weight), built for
bandwidth-bound cards. It loads the official GGUF, reproduces the PrismML llama.cpp fork's greedy output,
and runs the model's built-in MTP head for speculative decoding.

Validation: 24/24 greedy tokens match the fork in a teacher-forced comparison on a T4, top-10
log-probabilities differ by about 0.03 on average, plain decode is deterministic and bit-identical between
1-token and multi-token steps, and speculative output equals plain greedy output token for token.

## Results

Fork: `prism-b12178-03ec600` (`-fa on`, PTQ1_0-mtp file). Same container, same GPU, same prompt token ids,
greedy, 200 new tokens, 4 prompts, 4096-token context, decode time only. Medians of 2 repetitions.

| GPU | Plain decode, tok/s (fork → ours) | MTP decode, tok/s (fork → ours, 3 drafts) | Plain VRAM, MiB | MTP VRAM, MiB |
|---|---|---|---|---|
| T4 (16 GB) | 33.2 → 32.0 | 46.7 → **51.6 (+10%)** | 6174 → **5962 (−3%)** | 6978 → **6382 (−9%)** |
| L4 (24 GB) | 36.1 → 34.3 | 63.9 → 61.0 | 6277 → 6533* | 7089 → 6793* |
| A10 (24 GB) | 61.4 → 52.6 | 99.3 → 81.2 | 6329 → 6577* | 7149 → 6837* |

`*` measured with an earlier build, before weights moved to a single arena. The T4 row is the current build;
its VRAM numbers are the ones to read. On the weakest card the MTP path is ahead; on L4 plain and MTP are
within 5%, and on A10 both are behind by 14–18%. A10-class cards are limited by kernel-launch overhead and
compute rather than memory bandwidth, and that is not fixed here.

Prefill: the batched path processes about 125 tokens/s on a T4 (8-token steps). The fork processes ~350
(`llama-bench pp512`). Long prompts are slower here.

### Small-memory cards

An emulated 6 GB card (a ballast process on a T4 leaving 5856 MiB free), 1024-token context:

| | fork, plain | tritrun, plain |
|---|---|---|
| starts and generates | fails (compute buffer allocation out of memory, fit check disabled) | yes, 30–31 tok/s |

tritrun's MTP mode does not fit in that budget, and at 5400 MiB free neither engine starts.

## How it works

* **Ternary matrix-vector kernel.** Each 128-weight block is decoded once per row (multiply-by-3 digit steps
  on bytes held in 16-bit lanes) and dotted with int8 activations (quantised per 32 elements, as in the fork)
  through `dp4a`. The weight read is shared by all tokens of a step, so a 4-token verification step costs
  about 1.4x a single-token step on a T4.
* **Speculation.** The MTP head runs on the GPU; drafts are verified in one K≤4 step; the recurrent state
  (conv + gated delta net) is double-buffered and rejected drafts are rolled back by replaying recorded layer
  inputs, instead of keeping per-token snapshots.
* **Memory.** Token embeddings are looked up on the CPU; weights live in one exactly sized arena; small arrays
  share slabs; gate projections are FP16. Plain mode keeps a single copy of the recurrent state.
* **Reproducibility.** Residual-norm sums use 64-bit fixed-point atomics, so results do not depend on
  scheduling.

## Build and run

Linux, CUDA 12.x toolkit, Python 3.10+, `pip install tokenizers numpy`.

```bash
bash build.sh                       # produces tritrun/libtritrun.so
python -m tritrun.cli Ternary-Bonsai-2.5-27B-PTQ1_0-mtp.gguf "Write a haiku about autumn." --drafts 3
```

`--drafts 0` runs plain decode and works with or without the `-mtp` file. `--raw` skips the chat template.
From Python:

```python
from tritrun import Engine
e = Engine("Ternary-Bonsai-2.5-27B-PTQ1_0-mtp.gguf", ctx=4096, mtp=True)
text = e.chat("Explain quicksort briefly.", max_new=256)
ids, stats = e.generate(e.tok.encode("Hello"), max_new=64, drafts=3)
```

The tokenizer is built from the vocabulary in the GGUF and matches the fork's tokenizer on the test strings
(`bench/e2e.py`).

## Limits

* Greedy decoding only. One sequence, text only. Context at most 8192. Linux + NVIDIA only.
* MTP needs the `-mtp` GGUF. KV cache is FP16; a 4-bit or 8-bit KV cache is not implemented.
* Prefill is slow (see above). Large GPUs are slower than the fork.
* Speeds above are for Turing/Ada/Ampere classes tested here; no claim is made for other GPUs.

## Repository layout

* `src/engine.cu` engine (kernels, loader, MTP, speculation); `src/gguf.hpp` GGUF parser.
* `tritrun/` Python package (ctypes wrapper, tokenizer, CLI). `build.sh` builds the library.
* `bench/` Modal harnesses: `paired.py` (fork vs this engine in one container), `validate.py`,
  `consistency.py` (K-batched determinism), `spec_test.py`, `e2e.py`, `six_gb.py`, `profile.py`.
