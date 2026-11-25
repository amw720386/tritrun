# tritrun

A standalone CUDA inference engine for **Ternary-Bonsai-2.5-27B** (PTQ1_0 GGUF), aimed at bandwidth-bound
cards. On the cards tested (T4, L4, A10) it decodes faster than the PrismML llama.cpp fork and uses less VRAM.
The fork is still faster at prefilling long prompts.

## Results

Fork: `prism-b12178-03ec600`, flash attention, FP16 KV, 4096 context, one slot. Both engines ran in the same
container on the same GPU with the same prompt token ids.

### Decode and VRAM

4 short raw prompts, greedy, 200 new tokens, decode time only, mean of 2 repetitions. The fork column is its
best setting among those tried: plain GGUF for plain decode, `-ub 512` and `-ub 64`, and MTP with 3 drafts at
cut-off 0.1 and 0.

| GPU | Plain tok/s, fork / ours | MTP tok/s, fork / ours | Plain VRAM MiB, fork `-ub 64` / ours | MTP VRAM MiB, fork `-ub 64` / ours |
|---|---|---|---|---|
| T4 | 33.2 / **35.2 (+6%)** | 47.1 / **60.7 (+29%)** | 6038 / **6012** (-0.4%) | 6738 / **6384** (-5%) |
| L4 | 35.9 / **37.6 (+5%)** | 63.5 / **72.8 (+15%)** | 6140 / **6098** (-0.7%) | 6848 / **6470** (-6%) |
| A10 | 61.4 / **62.0 (+1%)** | 99.4 / **109.7 (+10%)** | 6191 / **6141** (-0.8%) | 6899 / **6513** (-6%) |

The A10 row is the middle one of three machines; across them the margins were +1 to +2% plain and +10 to +12%
with MTP. A10G machines, which an earlier revision of this table used, were not measured again.

VRAM is peak `nvidia-smi` memory. At the fork's default `-ub 512` its figures are 136 MiB higher in plain mode
and 240 MiB higher with MTP.

### Prefill

Real text, MTP off, best of 3, tok/s. Ours is the wall time of the whole prefill plus the first token; the
fork's is its own `prompt_per_second` at its default batch size. The first generated token agreed in every
case. Wide steps handle up to 128 tokens.

| GPU | 64 tokens, fork / ours | 512 tokens, fork / ours | 2048 tokens, fork / ours |
|---|---|---|---|
| T4 | 203 / **224** | 321 / 260 (81%) | 329 / 250 (76%) |
| L4 | 376 / **498** | 761 / 582 (77%) | 806 / 556 (69%) |
| A10 | 515 / 526 | 919 / 621 (68%) | 954 / 598 (63%) |

With MTP on, prefill of 512 or more tokens costs 3 to 4% more. Before the batched path existed, the T4 figures were 125 tok/s
at 512 tokens and 109 at 2048.

## Validation

* Against the fork on real 300- and 1000-token prompts, teacher-forced greedy tokens that agree, out of 96, for
  128-token and one-token prefill: T4 95 / 95, L4 95 / 94, A10 96 / 95. The fork does not always agree with
  itself either: changing its batch size changed its free-running output on one of the three prompts on T4 and
  on one on L4. The mean top-10 log-probability difference per prompt is 0.022 to 0.048; for the fork against
  itself it is 0.026 to 0.049. The quality section below uses larger samples.
* On a short prompt, 24 of 24 greedy tokens agree with the fork on each of the three cards.
* Steps of 1, 2, 4 and 8 tokens give bit-identical logits, and repeated runs are bit-identical, on all three
  cards. Wider steps add the group sums in a different order, so int8 activations can round differently: single
  kernels agree to 1e-6 or better, full logits to about 0.1. Attention agrees with a double-precision CPU
  reference to 2e-6 (T4).
* Speculative output equals plain greedy output for 1, 2 and 3 drafts on 3 prompts of 200 tokens, on all three
  cards.
* The tokenizer built from the GGUF agrees with the fork's on 7 test strings.

## Quality

Same model file, compared with the fork and with a control: the fork against itself at another physical batch
size, which shows how far its own rounding moves the results. Prompts use the fork's chat template with thinking
disabled and the fork's tokenizer, so both engines get identical ids.

| Test | Fork | Fork, `-ub 64` | Ours | Hardware |
|---|---|---|---|---|
| WikiText-2 perplexity, 10,200 scored tokens | 9.7154 | 9.7145 | 9.7131 | T4 |
| | 9.7147 | 9.7167 | 9.7175 | L4 |
| | 9.7160 | 9.7156 | 9.7169 | A10G |
| MMLU, 1,000 random test questions, correct | 778 | 775 | 778 | A10G |
| (second run) | 777 | 778 | 778 | A10 |
| GSM8K, 300 random test questions, 2,048-token limit, correct | 282 | 283 | 281 | A10 |
| GSM8K, same questions, 768-token limit, correct | 269 | 272 | 269 | A10G |

* **MMLU choices.** Ours picks the fork's answer on 1000 and 998 of 1000 questions; the control on 997 and 999.
* **GSM8K answers.** With the 2,048-token limit, ours gives the fork's final number on 291 of 300 (control 292) and
  a byte-identical answer on 212 (control 214). Greedy text diverges after one near-tie, so the rest differ in
  wording. All three of the questions that only the fork got right are ones where the answer tail shows ours
  still re-checking its working, not stating a final answer. Runs hit the limit 7 (fork), 7 (control) and 8
  (ours) times. MTP with 3 drafts was used.
* **Perplexity.** Mean negative log-likelihood, ours minus the fork: -0.00023 to +0.00029, each within about one
  standard error (0.0003 to 0.00045) of zero. Top-1 agreement is 99.3 to 99.4% (control 99.4 to 99.5%).
* **KL divergence from the fork.** Median 9e-5 in both ours and the control, 99th percentile 0.0018 (control
  0.0016 to 0.0021). The mean is 0.00032 to 0.00043 (control 0.00021 to 0.00031); without each run's single worst
  token it is 0.00022 to 0.00024 (control 0.00020 to 0.00023). That worst token is the same one in ours on all three cards (KL 1.0 to 2.0) and is numerically
  unstable: the fork's log-probability for it ranges from -1.24 to -0.05 (T4) and -1.18 to -0.07 (L4) across
  eleven batch sizes. The neighbouring tokens are normal.
* **No systematic offset found** (T4, L4; all 10,200 tokens; 8 fork batch sizes against 5 of our step splits).
  Mean absolute difference in the true token's log-probability: fork to fork 0.0134 (L4) and 0.0124 (T4), fork
  to ours 0.0135 and 0.0134; on T4 several fork batch sizes give identical results, which lowers its own
  figure. The squared gap left after removing each engine's own variation is inside the range obtained by
  splitting the fork's own arms in two, on both cards. On T4 and L4, 400 randomly placed tokens scored from a
  wide step differ from the fork by about as much as it differs from itself. Of four tokens picked because they
  looked bad, one stayed outside the fork's range on T4 for every step split: log-probability -0.26 to -0.34
  against -0.03 to -0.24.

What this does not show: accuracy differences of about a point or less on these samples; results with thinking
enabled, with long contexts, or with other samplers; MMLU and GSM8K on T4 and L4. `bench/quality.py`,
`bench/offset.py` and `bench/outlier.py` reproduce the numbers, and `results/quality_*.log` holds the output.

## How it works

* **Decode.** Each 128-weight block is decoded once per row and dotted with the int8 activations of every
  token in the step (`dp4a`), so one read of the weights serves several tokens. A 4-token step costs 1.15x (L4)
  to 1.4x (T4) a 1-token step, which is where the MTP gain comes from. Matmul blocks are small (256 or 512 rows) with
  shared memory sized per launch, so many fit on an SM at once.
* **Few launches.** The RMS norm factor is a scalar, so it is applied to the matmul output; that lets one kernel
  do the residual add, sum of squares, Hadamard transform and quantisation. A 1-token step is about 640 launches.
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
flags. Correctness
was checked on T4, L4 and A10 with the tests above.

`bench/` holds the Modal harnesses and `results/` the raw logs behind the numbers above.
