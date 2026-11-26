# Benchmarks

The full conditions behind the README's speed and memory tables. `tools/release_numbers.py` reads the logs named here and writes `results/release.json`, which the README reads.

Fork: `prism-b12178-03ec600`, flash attention, FP16 KV, 4096 context, one slot. Both engines ran in the same
container on the same GPU with the same prompt token ids.

## Decode and VRAM

4 short raw prompts, greedy, 200 new tokens, decode time only, the better of 2 repetitions. The fork column is its
best setting among those tried: plain GGUF for plain decode, `-ub 512` and `-ub 64`, and MTP with 3 drafts at
cut-off 0.1 and 0. Logs: `results/paired_graphs.log`, `results/rtx5090_graphs.log`.

| GPU | Plain tok/s, fork / ours | MTP tok/s, fork / ours | Plain VRAM MiB, fork `-ub 64` / ours | MTP VRAM MiB, fork `-ub 64` / ours |
|---|---|---|---|---|
| T4 | 33.1 / **35.6 (+8%)** | 46.8 / **60.1 (+28%)** | 6038 / **6024** | 6738 / **6396** (-5%) |
| L4 | 36.0 / **38.8 (+8%)** | 63.7 / **73.7 (+16%)** | 6145 / **6115** | 6853 / **6487** (-5%) |
| A10 | 61.6 / **66.4 (+8%)** | 99.5 / **112.7 (+13%)** | 6191 / **6153** | 6899 / **6525** (-5%) |
| RTX 5090 | 161.8 / **179.3 (+11%)** | 241.4 / **323.6 (+34%)** | 6562 / **6418** (-2%) | 7284 / **6790** (-7%) |

One machine per card. Machines of one type differ: on an earlier A10 the margins were +1 to +2% plain and +10
to +12% with MTP (before the two decode changes below), and on another RTX 5090 the fork reached 171.8 plain
(ours 168.0 before those changes, `results/rtx5090.log`). VRAM is peak `nvidia-smi` memory. At the fork's
default `-ub 512` its figures are 136 MiB higher in plain mode and 240 MiB higher with MTP.

## Prefill

Real text, best of 3, tok/s, both engines in the same container. Ours is the wall time of the whole prefill plus
the first token; the fork's is its own `prompt_per_second` at its default batch size. The first generated token
agreed in every case. Wide steps handle up to 128 tokens.

| GPU | 64 tokens, fork / ours | 512 tokens, fork / ours | 2048 tokens, fork / ours |
|---|---|---|---|
| T4 | 208 / **352** | 331 / **412 (+24%)** | 341 / **403 (+18%)** |
| L4 | 367 / **745** | 741 / **827 (+12%)** | 788 / **811 (+3%)** |
| A10 | 500 / **998** | 913 / **994 (+9%)** | 951 / 966 (+2%) |
| RTX 5090 | 553 / **3461** | 2899 / **4043 (+39%)** | 3174 / **3531 (+11%)** |

The 2048-token margin on A10 is within machine variation: in another A10 container the fork reached 1127 and
ours 1060. Timing the previous and the current version of this engine in one container showed no change
(A10, 2048 tokens: 949 and 966). With MTP on, both engines prefill a few percent slower.

