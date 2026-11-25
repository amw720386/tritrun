# tritrun

A standalone CUDA inference engine for **Ternary-Bonsai-2.5-27B** (PTQ1_0 GGUF), aimed at bandwidth-bound
cards. It uses less VRAM than the PrismML llama.cpp fork on every card tested, and decodes faster than it on
the weakest one (T4) when the model's MTP draft head is on. On L4 and A10G the fork is faster.

## Results

Fork: `prism-b12178-03ec600`, flash attention, FP16 KV, 4096 context, one slot. Both engines ran in the same
container on the same GPU with the same prompt token ids: 4 short raw prompts, greedy, 200 new tokens,
decode time only, 2 repetitions (the value shown is the mean). The fork column is its best setting among
those tried: plain GGUF for plain decode, `-ub 512` and `-ub 64`, and MTP with 3 drafts at cut-off 0.1 and 0.

| GPU | Plain tok/s, fork / ours | MTP tok/s, fork / ours | Plain VRAM MiB, fork `-ub 64` / ours | MTP VRAM MiB, fork `-ub 64` / ours |
|---|---|---|---|---|
| T4 | 32.9 / 31.7 (-3%) | 46.1 / **50.5 (+10%)** | 6038 / **5962** (-1%) | 6738 / **6382** (-5%) |
| L4 | 37.0 / 34.9 (-6%) | 66.4 / 62.5 (-6%) | 6144 / **6052** (-1%) | 6852 / **6472** (-6%) |
| A10G | 63.9 / 54.1 (-15%) | 107.5 / 83.4 (-22%) | 6220 / **6114** (-2%) | 6938 / **6534** (-6%) |

VRAM is peak `nvidia-smi` memory. At the fork's default `-ub 512` its figures are 136 MiB higher in plain mode
and 232 to 240 MiB higher with MTP.

**Small cards.** With a ballast process leaving 5.86 GB free (a T4 standing in for a 6 GB card) at 1024
context, plain mode runs at 28 tok/s here and 26 tok/s on the fork with `-ub 128`; the fork's default batch
size does not fit. Both run at 6.0 GB free, neither at 5.6 GB, and MTP mode here does not fit in 6.0 GB.
This is not a capability the fork lacks.

**Prefill is slower.** An 8-token step takes about 64 ms on a T4 (roughly 125 tok/s), against 350 tok/s for
the fork's `llama-bench` pp512. Long prompts are a poor fit for this engine today.

## Validation

* Against the fork: 24 of 24 greedy tokens agree on one prompt (teacher forced, T4). Top-10 log-probabilities
  differ by 0.03 on average, worst case 0.43. This is a small sample, and no quality benchmark has been run.
* Steps of 1, 2 and 4 tokens give bit-identical logits, and repeated runs are bit-identical.
* Speculative output equals plain greedy output for 1, 2 and 3 drafts on 3 prompts of 200 tokens.
* The tokenizer built from the GGUF agrees with the fork's on 7 test strings.

## How it works

* Each 128-weight block is decoded once per row and dotted with the int8 activations of every token in the
  step (`dp4a`), so one read of the weights serves several tokens. On a T4 a 4-token step costs about 1.4x a
  1-token step, which is where the MTP gain comes from.
* Rejected drafts roll back by replaying recorded layer inputs from a checkpoint of the recurrent state,
  rather than keeping one state snapshot per token.
* Token embeddings stay in host memory, and weights sit in one exactly sized allocation.

The weights themselves do not compress further: the trits are uniformly distributed (1.585 bits of entropy),
and a byte-lookup-table kernel was tried first and dropped because its cost grew with every extra token.

## Build and run

Linux, NVIDIA GPU, CUDA 12 toolkit, Python 3.10+ with `pip install -r requirements.txt`.

```bash
bash build.sh
python -m tritrun.cli Ternary-Bonsai-2.5-27B-PTQ1_0-mtp.gguf "Write a haiku about autumn." --drafts 3
```

`--drafts 0` is plain decode. The prompt goes through a minimal ChatML template, not the full template in the
GGUF (no tools, no reasoning controls).

## Limits

Greedy only, one sequence, text only, context up to 8192, no KV quantization, no server. `build.sh` (the
multi-architecture build) has not been run; the benchmarks compile one architecture at a time with the same
flags. The L4 and A10G rows were measured for speed and VRAM only; correctness was checked on a T4.

`bench/` holds the Modal harnesses and `results/` the raw logs behind the numbers above.
