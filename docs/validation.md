# Validation and quality

Everything behind the README's correctness and quality claims. Raw output is in `results/`.

## Validation

* Against the fork on real 300- and 1000-token prompts, teacher-forced greedy tokens that agree, out of 96, for
  128-token and one-token prefill: T4 95 / 95, L4 95 / 94, A10 96 / 95. The fork does not always agree with
  itself either: changing its batch size changed its free-running output on one of the three prompts on T4 and
  on one on L4. The mean top-10 log-probability difference per prompt is 0.022 to 0.048; for the fork against
  itself it is 0.026 to 0.049. The quality section below uses larger samples.
* On a short prompt, 24 of 24 greedy tokens agree with the fork on each of the three cards.
* Steps of 1, 2, 4 and 8 tokens give bit-identical logits, and repeated runs are bit-identical, on T4, L4, A10
  and RTX 5090. (On the 5090 this first failed: under `--use_fast_math` the RMS scale's `1/sqrtf` compiled to
  different instruction sequences in different kernels; it now uses the correctly rounded `__frsqrt_rn`.) Wider steps add the group sums in a different order, so int8 activations can round differently: single
  kernels agree to 1e-6 or better, full logits to about 0.1. Attention agrees with a double-precision CPU
  reference to 2e-6 for steps of up to 8 tokens and to 3e-4 for wider steps, whose attention uses FP16 queries
  and probabilities (T4).
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

The tables were measured before the prefill and decode changes listed under "How it works" in the README. Re-run on the current
version (`results/quality_current.log`), fork / fork control / ours:

| Test | Fork | Control | Ours | Hardware |
|---|---|---|---|---|
| WikiText-2 perplexity | 9.7154 | 9.7145 | 9.7150 | T4 |
| | 9.7147 | 9.7167 | 9.7195 | L4 |
| | 9.7161 | 9.7170 | 9.7146 | A10 |
| | 9.7107 | 9.7167 | 9.7173 | RTX 5090 |
| MMLU, 1,000 questions, correct | 777 | 778 / 776 | 778 | A10 |
| | 777 | 776 / 779 | 778 | RTX 5090 |
| GSM8K, 300 questions, 2,048-token limit, correct | 282 | 283 | 280 | A10 |
| | 282 | 282 | 284 | RTX 5090 |

Median KL 9e-5 on every card (control 9e-5), 99th percentile 0.0018 to 0.0019 (control 0.0018 to 0.0021), top-1
agreement 99.3 to 99.4% (control 99.3 to 99.5%). Mean negative log-likelihood, ours minus the fork: -0.00014 to
+0.00068, at most 1.7 standard errors; on the RTX 5090 the control is 0.0006 from the fork too. MMLU: the fork's
choice on 998 (A10) and 996 (RTX 5090) of 1000 questions, controls 998 to 999. GSM8K: the fork's final number on
291 (A10) and 295 (RTX 5090) of 300, controls 292 and 295. On A10 the three questions that only the fork got
right again end with ours still checking its working at the length limit.

What this does not show: accuracy differences of about a point or less on these samples; results with thinking
enabled, with long contexts, or with other samplers; MMLU and GSM8K on T4 and L4. `bench/quality.py`,
`bench/offset.py` and `bench/outlier.py` reproduce the numbers, and `results/quality_*.log` holds the output.

