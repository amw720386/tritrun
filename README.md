# tritrun

A standalone CUDA inference engine for **Ternary-Bonsai-2.5-27B** (PTQ1_0 GGUF), aimed at bandwidth-bound
cards. It uses less VRAM than the PrismML llama.cpp fork on every card tested, and decodes faster than it on
the weakest one (T4) when the model's MTP draft head is on. On L4 and A10G the fork decodes faster, and the
fork is faster at prefilling long prompts everywhere.

## Results

Fork: `prism-b12178-03ec600`, flash attention, FP16 KV, 4096 context, one slot. Both engines ran in the same
container on the same GPU with the same prompt token ids.

### Decode and VRAM

4 short raw prompts, greedy, 200 new tokens, decode time only, mean of 2 repetitions. The fork column is its
best setting among those tried: plain GGUF for plain decode, `-ub 512` and `-ub 64`, and MTP with 3 drafts at
cut-off 0.1 and 0.

| GPU | Plain tok/s, fork / ours | MTP tok/s, fork / ours | Plain VRAM MiB, fork `-ub 64` / ours | MTP VRAM MiB, fork `-ub 64` / ours |
|---|---|---|---|---|
| T4 | 32.8 / 31.6 (-4%) | 46.1 / **51.3 (+11%)** | 6038 / **6012** (-0.4%) | 6738 / **6384** (-5%) |
| L4 | 36.3 / 34.2 (-6%) | 64.6 / 61.6 (-5%) | 6141 / **6099** (-0.7%) | 6849 / **6471** (-6%) |
| A10G | 64.7 / 53.5 (-17%) | 109.2 / 84.4 (-23%) | 6219 / **6163** (-0.9%) | 6929 / **6535** (-6%) |

VRAM is peak `nvidia-smi` memory. At the fork's default `-ub 512` its figures are 136 MiB higher in plain mode
and 240 MiB higher with MTP.

### Prefill

Real text, MTP off, best of 3, tok/s. Ours is the wall time of the whole prefill plus the first token; the
fork's is its own `prompt_per_second` at its default batch size. The first generated token agreed in every
case. Wide steps handle up to 128 tokens.

| GPU | 64 tokens, fork / ours | 512 tokens, fork / ours | 2048 tokens, fork / ours |
|---|---|---|---|
| T4 | 206 / **226** | 326 / 249 (76%) | 336 / 240 (71%) |
| L4 | 440 / **473** | 786 / 562 (71%) | 837 / 540 (64%) |
| A10 | 499 / 496 | 896 / 578 (65%) | 924 / 560 (61%) |

With MTP on, prefill here costs 1 to 4% more. Before the batched path existed, the T4 figures were 125 tok/s
at 512 tokens and 109 at 2048.

## Validation

* Against the fork on real 300- and 1000-token prompts (T4): 96 of 96 teacher-forced greedy tokens agree.
  The mean top-10 log-probability difference is 0.025 to 0.050, whether the prompt is prefilled in 128-token
  steps or one token at a time. The fork compared with itself at two batch sizes differs by 0.026 to 0.049.
  This is a small sample, and no quality benchmark has been run.
* Steps of 1, 2, 4 and 8 tokens give bit-identical logits, and repeated runs are bit-identical. Wider steps
  add the group sums in a different order, so int8 activations can round differently: single kernels agree
  to 1e-6 or better, full logits to about 0.1.
* Speculative output equals plain greedy output for 1, 2 and 3 drafts on 3 prompts of 200 tokens.
* The tokenizer built from the GGUF agrees with the fork's on 7 test strings.

## How it works

* **Decode.** Each 128-weight block is decoded once per row and dotted with the int8 activations of every
  token in the step (`dp4a`), so one read of the weights serves several tokens. On a T4 a 4-token step costs
  about 1.4x a 1-token step, which is where the MTP gain comes from.
* **Prefill.** Steps of more than 8 tokens decode 64 rows of trits into shared memory and evaluate them
  against up to 128 tokens with the INT8 tensor-core instruction (`mma.sync`; m8n8k16 on Turing, m16n8k32 on
  Ampere and later). Attention shares KV reads across 8 tokens and evaluates the softmax online.
* **Rollback.** Rejected drafts are undone by replaying recorded layer inputs from a checkpoint of the
  recurrent state, rather than keeping one state snapshot per token.
* **Memory.** Token embeddings stay in host memory, and weights sit in one exactly sized allocation.

The weights themselves do not compress further: the trits are uniformly distributed (1.585 bits of entropy).

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
flags. Correctness was checked on a T4; on L4 and A10/A10G only speed, VRAM and the first generated token were.

`bench/` holds the Modal harnesses and `results/` the raw logs behind the numbers above.
