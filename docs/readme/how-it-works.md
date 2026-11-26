
* **Decode.** Each 128-weight block is decoded once per row and dotted with the int8 activations of every
  token in the step (`dp4a`), so one read of the weights serves several tokens. A 4-token step costs 1.15x (L4)
  to 1.4x (T4) a 1-token step, which is where the MTP gain comes from. Matmul blocks are small (256 or 512 rows) with
  shared memory sized per launch, so many fit on an SM at once.
* **Few launches.** The RMS norm factor is a scalar, so it is applied to the matmul output; that lets one kernel
  do the residual add, sum of squares, Hadamard transform and quantisation. A 1-token step is about 640 launches,
  replayed as one CUDA graph per step width (the position is read from device memory, so the graph is reused).
* **Column partials.** Matmuls split their columns into partitions so every SM has work; the next kernel adds
  the partial rows. Cards with many SMs use many partitions (34 for a 5120-row matrix on an RTX 5090), so the
  adds fetch 8 partitions of 4 elements per thread at once. On the 5090 this was worth 7% of plain decode,
  graphs about 2%.
* **Prefill.** Steps of more than 8 tokens decode 128 rows of trits into shared memory and evaluate them
  against up to 128 tokens with the INT8 tensor-core instruction (`mma.sync`; m8n8k16 on Turing, m16n8k32 on
  Ampere and later). Shared memory delivers about 64 bytes per clock and SM on these cards, which was the
  limit, so a warp owns 64 rows x 32 tokens and each fragment is one wide load; prep writes the activations
  in the order the kernel stores them, so loading them is a plain copy. Small matrices are split by columns
  to fill all SMs. The recurrent layers stage a chunk of tokens in shared memory once per block, and attention
  runs on FP16 tensor cores with an online softmax. `bench/tile_lab.py` times the matmul alone with parts of it
  switched off, which is how the limits were found.
* **Rollback.** Rejected drafts are undone by replaying recorded layer inputs from a checkpoint of the
  recurrent state, rather than keeping one state snapshot per token.
* **Memory.** Token embeddings stay in host memory, and weights sit in one exactly sized allocation.

The weights themselves do not compress further: the trits are uniformly distributed (1.585 bits of entropy).

