// tritrun: a CUDA decode engine for PTQ1_0 (ternary, group-128) Bonsai 2.5 27B.
//
// Matrix-vector products: each 128-weight block is decoded once per row (multiply-by-3 digit steps on
// bytes held in 16-bit lanes), and every digit word is dotted with the int8 activations of 1..K tokens
// (dp4a).  Activations are quantised per 32 elements, as in the reference fork, so the in-group sums are
// exact integers.  K tokens share one read of the weights, which is what makes MTP verification cheap on
// bandwidth-bound cards.  Recurrent state (conv + gated delta net) is double-buffered so a rejected draft
// rolls back by replaying recorded inputs instead of storing per-token snapshots.
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <string>
#include <thread>
#include <vector>
#include <algorithm>
#include <chrono>
#include <map>
#include "gguf.hpp"

typedef uint32_t u32;

#define CK(x) do { cudaError_t _e = (x); if (_e != cudaSuccess) { fprintf(stderr, "CUDA error %s at %s:%d: %s\n", #x, __FILE__, __LINE__, cudaGetErrorString(_e)); abort(); } } while (0)

static const int CH = 1024;            // rows per chunk
#ifndef PARTS_HALF
#define PARTS_HALF 2                   // column partitions x chunks per SM, in halves
#endif
static const int KREC = 4;             // tokens per step that can be rolled back
static const int KMAX = 128;         // tokens per step (prefill width)
static const int KLOG = 4;           // tokens per step that can return logits (wider steps return the last token only)
static const int D_MODEL = 5120, D_INNER = 6144, D_FF = 17408;
static const int QKV = 10240, GATE96 = 96;

// ---------------------------------------------------------------------------------------------
// device helpers
// ---------------------------------------------------------------------------------------------
__device__ __forceinline__ float warp_sum(float v) {
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xFFFFFFFFu, v, o, 32);
    return v;
}
typedef unsigned long long ssq_t;                // sums of squares in 2^-16 fixed point: order-independent atomic accumulation
__device__ __forceinline__ float ssq_rms_inv(const ssq_t v, const int n) {
    // correctly rounded: with --use_fast_math, 1/sqrtf compiles to different instruction sequences in different kernels on
    // sm_120, which broke bit-identity between the fused (K <= 2) and separate (K = 4, 8) gate paths
    return __frsqrt_rn((float) v * (1.0f / 65536.0f / n) + 1e-6f);
}
__device__ __forceinline__ float silu_f(float x) { return x / (1.0f + expf(-x)); }
__device__ __forceinline__ float softplus_f(float x) { return x > 20.0f ? x : logf(1.0f + expf(x)); }

// element (0..127) read by trit n of byte b of a 128-weight block (PTQ1_0 packing order)
__device__ __forceinline__ int trit_elem(int b, int n) {
    if (b < 16) return b + 16 * n;
    if (b < 24) return 80 + (b - 16) + 8 * n;
    return 120 + (b - 24) + 2 * n;
}

// ---------------------------------------------------------------------------------------------
// prep: elementwise source -> signs -> Hadamard(1024) -> per-group int8 quantisation
// mode 0: norm        x[k][i] += partial sums of the previous matrix (if any), sum of squares -> ssq (if given), v = x * w.
//                     The RMS factor is a scalar, so it is applied to the matmul output instead (see mvm_kernel): no extra kernel
// mode 1: swiglu      v = silu(gate) * up, read from the mvm partials
// mode 2: gdn output  per-head RMS norm of x, silu(z) gate from the mvm partials, tiled heads permuted to grouped order
// mode 3: plain       K x n vector
// ---------------------------------------------------------------------------------------------
// y[p0 * stride], y[(p0 + 1) * stride], ...: n partial results summed in order, requested 8 at a time so their latencies overlap
__device__ __forceinline__ float part_sum(const float* y, const int n, const int stride) {
    float t = 0;
    for (int p = 0; p < n; p += 8) {
        float b[8];
#pragma unroll
        for (int q = 0; q < 8; ++q) b[q] = p + q < n ? y[(size_t) (p + q) * stride] : 0.0f;
#pragma unroll
        for (int q = 0; q < 8; ++q) if (p + q < n) t += b[q];
    }
    return t;
}
// part_sum of N elements at once: all their loads of a batch of 8 partitions are in flight together (the sums are added in the
// same order as part_sum, so results are bit-identical).  Cards with many SMs use many partitions, where this matters.
#ifndef PS_BATCH
#define PS_BATCH 8
#endif
template <int N, int B = PS_BATCH>
__device__ __forceinline__ void part_sum_n(const float* const (&y)[N], const int n, const int stride, float (&t)[N]) {
#pragma unroll
    for (int i = 0; i < N; ++i) t[i] = 0;
    for (int p = 0; p < n; p += B) {
        float b[N][B];
#pragma unroll
        for (int q = 0; q < B; ++q)
#pragma unroll
            for (int i = 0; i < N; ++i) b[i][q] = p + q < n ? y[i][(size_t) (p + q) * stride] : 0.0f;
#pragma unroll
        for (int i = 0; i < N; ++i)
#pragma unroll
            for (int q = 0; q < B; ++q) if (p + q < n) t[i] += b[i][q];
    }
}

static const int F_MAGIC_I = 0x4B400000;                        // 1.5 * 2^23 as float: adding an integer to the bits adds it to the value
#define F_MAGIC_F 12582912.0f
struct PrepArgs {
    float* src; ssq_t* ssq; const float* w; const float* signs;
    const float* yp; int parts; int yrows; int upoff;
    int n; int G; int wide;
    signed char* A; float* d8; int* sa;     // A [k][G][128] int8 planes, d8 [k][G][4] sub-block scales, sa [k][G][4] sub-block sums
                                            // wide: A in the word order of mvm_tile_kernel, sa slot 0 = group correction (float)
};

// byte position inside the 128-byte activation record of one group: words permuted to match the trit decode order
__device__ __forceinline__ int plane_slot(int e) {
    if (e < 80) {                                   // 5 steps x 16 elements: word n*4+j, byte lane perm {0,2,1,3}
        const int n = e >> 4, j = (e & 15) >> 2, l = e & 3;
        const int lane = l == 1 ? 2 : (l == 2 ? 1 : l);
        return (n * 4 + j) * 4 + lane;
    }
    if (e < 120) {                                  // 5 steps x 8 elements
        const int n = (e - 80) >> 3, r = (e - 80) & 7, j = r >> 2, l = r & 3;
        const int lane = l == 1 ? 2 : (l == 2 ? 1 : l);
        return (20 + n * 2 + j) * 4 + lane;
    }
    return 30 * 4 + (e - 120);                      // last 8 elements in natural order (words 30, 31)
}

__global__ __launch_bounds__(256) void prep_kernel(const PrepArgs a, const int mode) {
    __shared__ float s[1024];
    const int k = blockIdx.y, seg = blockIdx.x, tid = threadIdx.x;
    const int base = seg * 1024;
    float v[4];                                                 // elements tid + 256 i
    if (mode == 0) {
        float acc = 0, t[4];
        {
            const float* yk = a.yp + (size_t) k * a.parts * a.yrows + base + tid;
            const float* const ys[4] = { yk, yk + 256, yk + 512, yk + 768 };
            part_sum_n<4>(ys, a.parts, a.yrows, t);
        }
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            const int e = base + tid + 256 * i;
            const float x = a.src[(size_t) k * a.n + e] + t[i];
            if (a.parts) a.src[(size_t) k * a.n + e] = x;
            acc += x * x;
            v[i] = x * a.w[e];
        }
        if (a.ssq) {
            acc = warp_sum(acc);
            if ((tid & 31) == 0) atomicAdd(a.ssq + k, (ssq_t) llrint((double) acc * 65536.0));
        }
    } else if (mode == 1) {
        const float* yk = a.yp + (size_t) k * a.parts * a.yrows + base + tid, *yu = yk + a.upoff;
        const float* const ys[8] = { yk, yk + 256, yk + 512, yk + 768, yu, yu + 256, yu + 512, yu + 768 };
        float t[8];
        part_sum_n<8>(ys, a.parts, a.yrows, t);
#pragma unroll
        for (int i = 0; i < 4; ++i) v[i] = silu_f(t[i]) * t[4 + i];
    } else if (mode == 2) {
        float z[4];
        {
            const float* ys[4];
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                const int q = seg * 8 + 2 * i + (tid >> 7), h = q / 3 + 16 * (q % 3);
                ys[i] = a.yp + (size_t) k * a.parts * a.yrows + QKV + h * 128 + (tid & 127);
            }
            part_sum_n<4>(ys, a.parts, a.yrows, z);
        }
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            const int q = seg * 8 + 2 * i + (tid >> 7), h = q / 3 + 16 * (q % 3);
            v[i] = a.src[(size_t) k * a.n + h * 128 + (tid & 127)];
            const float ss = warp_sum(v[i] * v[i]);
            if ((tid & 31) == 0) s[i * 8 + (tid >> 5)] = ss;
        }
        __syncthreads();
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            const float* ps = s + i * 8 + (tid >> 7) * 4;
            const float irms = 1.0f / sqrtf((ps[0] + ps[1] + ps[2] + ps[3]) / 128.0f + 1e-6f);
            v[i] = silu_f(z[i]) * (v[i] * irms * a.w[tid & 127]);
        }
        __syncthreads();
    } else {
#pragma unroll
        for (int i = 0; i < 4; ++i) v[i] = a.src[(size_t) k * a.n + base + tid + 256 * i];
    }
#pragma unroll
    for (int i = 0; i < 4; ++i) v[i] *= a.signs[base + tid + 256 * i];
    // radix-4 Hadamard, high address bits first, so every thread ends with 4 consecutive elements
#define HAD4(v) { const float a0_ = v[0] + v[1], a1_ = v[0] - v[1], a2_ = v[2] + v[3], a3_ = v[2] - v[3]; v[0] = a0_ + a2_; v[1] = a1_ + a3_; v[2] = a0_ - a2_; v[3] = a1_ - a3_; }
    HAD4(v)
#pragma unroll
    for (int i = 0; i < 4; ++i) s[tid + 256 * i] = v[i];
    __syncthreads();
#pragma unroll
    for (int sh = 6; sh >= 0; sh -= 2) {
        const int b = ((tid >> sh) << (sh + 2)) | (tid & ((1 << sh) - 1));
#pragma unroll
        for (int i = 0; i < 4; ++i) v[i] = s[b + (i << sh)];
        HAD4(v)
        if (sh) {
#pragma unroll
            for (int i = 0; i < 4; ++i) s[b + (i << sh)] = v[i];
            __syncthreads();
        }
    }
    // elements 4 tid .. 4 tid + 3; 8 threads share one 32-element quantisation block
    float m = 0;
#pragma unroll
    for (int i = 0; i < 4; ++i) { v[i] *= 0.03125f; m = fmaxf(m, fabsf(v[i])); }
#pragma unroll
    for (int o = 4; o > 0; o >>= 1) m = fmaxf(m, __shfl_xor_sync(0xFFFFFFFFu, m, o));
    int q[4], sum = 0;
#pragma unroll
    for (int i = 0; i < 4; ++i) { q[i] = m > 0 ? (int) lrintf(v[i] * (127.0f / m)) : 0; sum += q[i]; }
#pragma unroll
    for (int o = 4; o > 0; o >>= 1) sum += __shfl_xor_sync(0xFFFFFFFFu, sum, o);
    const int gg = base / 128 + (tid >> 5), el = (tid & 31) * 4;
    const int q1 = el < 120 ? q[2] : q[1], q2 = el < 120 ? q[1] : q[2];
    int wd = plane_slot(el) >> 2;
    if (a.wide) wd = (wd >> 3) * 8 + (wd & 3) * 2 + ((wd >> 2) & 1);      // [sub-block][k & 3][k >> 2]: a token's record is copied verbatim into shared memory
    ((u32*) a.A)[(size_t) (k * a.G + gg) * 32 + wd] = (q[0] & 255) | ((q1 & 255) << 8) | ((q2 & 255) << 16) | ((u32) q[3] << 24);
    if (a.wide) {                                               // slot 0 of sa: sum over the sub-blocks of scale * sum (the digits are 0..2, so this is subtracted once per group)
        float c = (tid & 7) == 0 ? (m / 127.0f) * (float) sum : 0.0f;
        for (int o = 16; o >= 8; o >>= 1) c += __shfl_xor_sync(0xFFFFFFFFu, c, o);
        if ((tid & 7) == 0) a.d8[(size_t) (k * a.G + gg) * 4 + (el >> 5)] = m / 127.0f;
        if ((tid & 31) == 0) ((float*) a.sa)[(size_t) (k * a.G + gg) * 4] = c;
    } else if ((tid & 7) == 0) {
        a.d8[(size_t) (k * a.G + gg) * 4 + (el >> 5)] = m / 127.0f;
        a.sa[(size_t) (k * a.G + gg) * 4 + (el >> 5)] = sum;
    }
}

// ---------------------------------------------------------------------------------------------
// ternary matrix-vector product: decode the trits once per row, then one integer dot per token.
// W layout: [chunk][G][7 words][1024 rows] (the raw 28-byte PTQ1_0 block as 7 little-endian words).
// Digit step: multiply by 3 on bytes held in 16-bit lanes; the resulting digit word has byte order
// (b0, b2, b1, b3), and prep stores the activation planes in the same order.
// Y layout: [token k][part][yrows]
// ---------------------------------------------------------------------------------------------
#define DIGIT_STEP(lo, hi, D) { const u32 t1_ = (lo) * 3u, t2_ = (hi) * 3u; (lo) = t1_ & 0x00FF00FFu; (hi) = t2_ & 0x00FF00FFu; D = __byte_perm(t1_, t2_, 0x7531); }

// F32 beta / alpha gate projections (96 rows) of the same normalised input, evaluated by extra blocks of the mvm launch
struct GateArgs { const float* x; const float* nw; const half* wb; const half* wa; float* out; int gy; };   // gy: extra block rows (0 = none)

// ss: sums of squares of the input vectors (null = no RMS norm); the factor 1/rms is applied to the result
#define MVM_SMEM(K, G, parts) ((size_t) (K) * (((G) + (parts) - 1) / (parts)) * 160)
template <int K, int NT, int RPT, bool PF, bool GT>
__global__ __launch_bounds__(NT, 1) void mvm_kernel(const u32* W, const signed char* A, const float* d8, const int* sa, float* Y,
                                                    const int G, const int parts, const int yrows, const ssq_t* ss, const GateArgs ga) {
    // activation records of this block's groups, 16-byte aligned so the 4 words of a digit step come from one 128-bit load
    // the shared memory is sized by the launch (MVM_SMEM), so that small partitions let more blocks live on one SM
    extern __shared__ __align__(16) u32 sA[];
    const int gmax = (G + parts - 1) / parts;
    float* sds = (float*) (sA + K * gmax * 32);
    int* sss = (int*) (sds + K * gmax * 4);
    const int tid = threadIdx.x, rowb = blockIdx.x * NT * RPT;
    float ir[K];
    if (GT && (int) blockIdx.y < ga.gy) {                             // gate block: one row, 256 threads
        const int row = blockIdx.y * gridDim.x + blockIdx.x;
        if (row >= GATE96 || tid >= 256) return;
#pragma unroll
        for (int k = 0; k < K; ++k) ir[k] = ssq_rms_inv(ss[k], D_MODEL);
        const half* Wg = row < 48 ? ga.wb + (size_t) row * D_MODEL : ga.wa + (size_t) (row - 48) * D_MODEL;
        float acc[K];
#pragma unroll
        for (int t = 0; t < K; ++t) acc[t] = 0;
        for (int e = tid; e < D_MODEL; e += 256) {
            const float w = __half2float(Wg[e]), nwe = ga.nw[e];
#pragma unroll
            for (int t = 0; t < K; ++t) acc[t] += w * (ga.x[(size_t) t * D_MODEL + e] * ir[t] * nwe);
        }
#pragma unroll
        for (int t = 0; t < K; ++t) { const float a = warp_sum(acc[t]); if ((tid & 31) == 0) sds[t * 8 + (tid >> 5)] = a; }
        __syncthreads();
        if (tid < K) { float t = 0; for (int i = 0; i < 8; ++i) t += sds[tid * 8 + i]; ga.out[tid * GATE96 + row] = t; }
        return;
    }
    const int part = blockIdx.y - (GT ? ga.gy : 0);
    const int g0 = (int) ((long long) part * G / parts), g1 = (int) ((long long) (part + 1) * G / parts);
    const int Gp = g1 - g0;
    // the first weights are requested before the activations are copied, so the two latencies overlap
    const u32* Wc = W + (size_t) (rowb / CH) * G * 7 * CH + (rowb % CH);
    u32 wn[RPT][7];
#define LOADW(g) { const u32* Wg = Wc + (size_t) (g) * 7 * CH; _Pragma("unroll") for (int j = 0; j < RPT; ++j) _Pragma("unroll") for (int q = 0; q < 7; ++q) wn[j][q] = Wg[q * CH + tid + j * NT]; }
    if (PF) LOADW(g0)
#pragma unroll
    for (int k = 0; k < K; ++k) ir[k] = ss ? ssq_rms_inv(ss[k], D_MODEL) : 1.0f;
    const u32* Au = (const u32*) A;
    for (int i = tid; i < K * Gp * 32; i += NT) {
        const int k = i / (Gp * 32), r = i - k * Gp * 32;
        sA[i] = Au[(size_t) (k * G + g0) * 32 + r];
    }
    for (int i = tid; i < K * Gp * 4; i += NT) {
        const int k = i / (Gp * 4), r = i - k * Gp * 4;
        sds[i] = d8[(size_t) (k * G + g0) * 4 + r];
        sss[i] = sa[(size_t) (k * G + g0) * 4 + r];
    }
    __syncthreads();

    float acc[RPT][K];
#pragma unroll
    for (int j = 0; j < RPT; ++j)
#pragma unroll
        for (int k = 0; k < K; ++k) acc[j][k] = 0.0f;
    for (int g = g0; g < g1; ++g) {
        u32 w[RPT][7];
        if (!PF) LOADW(g)
#pragma unroll
        for (int j = 0; j < RPT; ++j)
#pragma unroll
            for (int q = 0; q < 7; ++q) w[j][q] = wn[j][q];
        if (PF && g + 1 < g1) LOADW(g + 1)
        const int gi = g - g0;
        float ds[K][4]; int ss[K][4];
#pragma unroll
        for (int k = 0; k < K; ++k)
#pragma unroll
            for (int q = 0; q < 4; ++q) { ds[k][q] = sds[(k * Gp + gi) * 4 + q]; ss[k][q] = sss[(k * Gp + gi) * 4 + q]; }

        int isum[RPT][K][4];
#pragma unroll
        for (int j = 0; j < RPT; ++j)
#pragma unroll
            for (int k = 0; k < K; ++k)
#pragma unroll
                for (int q = 0; q < 4; ++q) isum[j][k][q] = 0;
        u32 lo[RPT][4], hi[RPT][4], l2[RPT][2], h2[RPT][2];
#pragma unroll
        for (int j = 0; j < RPT; ++j) {
#pragma unroll
            for (int q = 0; q < 4; ++q) { lo[j][q] = w[j][q] & 0x00FF00FFu; hi[j][q] = (w[j][q] >> 8) & 0x00FF00FFu; }
#pragma unroll
            for (int q = 0; q < 2; ++q) { l2[j][q] = w[j][4 + q] & 0x00FF00FFu; h2[j][q] = (w[j][4 + q] >> 8) & 0x00FF00FFu; }
        }
        // bytes 0..15: 5 digit steps of 16 elements; one 128-bit activation load per token and step
#pragma unroll
        for (int n = 0; n < 5; ++n) {
            uint4 a4[K];
#pragma unroll
            for (int k = 0; k < K; ++k) a4[k] = ((const uint4*) sA)[(k * Gp + gi) * 8 + n];
#pragma unroll
            for (int j = 0; j < RPT; ++j) {
#pragma unroll
                for (int q = 0; q < 4; ++q) {
                    u32 D; DIGIT_STEP(lo[j][q], hi[j][q], D)
#pragma unroll
                    for (int k = 0; k < K; ++k) {
                        const u32 aw = q == 0 ? a4[k].x : (q == 1 ? a4[k].y : (q == 2 ? a4[k].z : a4[k].w));
                        isum[j][k][n >> 1] = __dp4a((int) D, (int) aw, isum[j][k][n >> 1]);
                    }
                }
            }
        }
        // bytes 16..23: 5 digit steps of 8 elements; 64-bit loads
#pragma unroll
        for (int n = 0; n < 5; ++n) {
            const int sb = n < 2 ? 2 : 3;
            uint2 a2[K];
#pragma unroll
            for (int k = 0; k < K; ++k) a2[k] = ((const uint2*) sA)[(k * Gp + gi) * 16 + 10 + n];
#pragma unroll
            for (int j = 0; j < RPT; ++j) {
#pragma unroll
                for (int q = 0; q < 2; ++q) {
                    u32 D; DIGIT_STEP(l2[j][q], h2[j][q], D)
#pragma unroll
                    for (int k = 0; k < K; ++k) isum[j][k][sb] = __dp4a((int) D, (int) (q == 0 ? a2[k].x : a2[k].y), isum[j][k][sb]);
                }
            }
        }
        // bytes 24, 25: 4 digits each -> elements 120 + 2n, 121 + 2n (words 30, 31)
        uint2 a3[K];
#pragma unroll
        for (int k = 0; k < K; ++k) a3[k] = ((const uint2*) sA)[(k * Gp + gi) * 16 + 15];
#pragma unroll
        for (int j = 0; j < RPT; ++j) {
            u32 v = (w[j][6] & 0xFFu) | ((w[j][6] & 0xFF00u) << 8);
#pragma unroll
            for (int n = 0; n < 4; ++n) {
                const u32 t = v * 3u;
                const int d24 = (t >> 8) & 3, d25 = (t >> 24) & 3;
                v = t & 0x00FF00FFu;
                const int sh = 8 * ((2 * n) & 3);
#pragma unroll
                for (int k = 0; k < K; ++k) {
                    const u32 aw = (n >> 1) == 0 ? a3[k].x : a3[k].y;
                    const int a0 = (int) (signed char) (aw >> sh), a1 = (int) (signed char) (aw >> (sh + 8));
                    isum[j][k][3] += d24 * a0 + d25 * a1;
                }
            }
            const float dh = __half2float(__ushort_as_half((unsigned short) (w[j][6] >> 16)));
#pragma unroll
            for (int k = 0; k < K; ++k) {
                float f = 0;
#pragma unroll
                for (int q = 0; q < 4; ++q) f += ds[k][q] * (float) (isum[j][k][q] - ss[k][q]);
                acc[j][k] += dh * f;
            }
        }
    }
#pragma unroll
    for (int k = 0; k < K; ++k)
#pragma unroll
        for (int j = 0; j < RPT; ++j)
            Y[(size_t) (k * parts + part) * yrows + rowb + tid + j * NT] = acc[j][k] * ir[k];
}

// ---------------------------------------------------------------------------------------------
// wide steps (prefill): a block owns R rows x TC tokens, a warp a tile of RW rows x 32 tokens (RW * 32 outputs, all accumulated in
// registers).  Per 128-column group the rows are decoded once into shared memory as digit words and the activation records are
// copied verbatim; evaluation is a product of INT8 fragments.  Shared memory delivers about 64 bytes per clock and SM, so the layouts
// are chosen to move little of it per product: a weight fragment is one 128-bit load (already the register quad the instruction wants),
// an activation fragment one 64-bit load, and a token's scale one load per sub-block shared by all rows.
// Integer sums are the decode kernel's; int -> float uses the 1.5 * 2^23 trick: F = M + I exactly, and fma(F, d, -M d) = round(I d).
// The digits are 0..2, so a group contributes d * scale * (sum digits * a); the sum of scale * a over the group is subtracted at its end.
// ---------------------------------------------------------------------------------------------
#ifndef TILE_RW
#define TILE_RW 64                                              // rows per warp (32 or 64)
#endif
static const int TILE_NT = 256;
static const int TILE_TT = 4;                                   // token fragments (of 8) per warp
#define TILE_RWT(TC) ((TC) == 32 && TILE_RW > 32 ? 32 : TILE_RW)              // rows per warp: the narrowest kernel would need too much shared memory
#define tile_rows(TC) (TILE_RWT(TC) * 8 * 32 / (TC))                          // rows per block: 8 warps = (R / RW) row groups x (TC / 32) token groups

__device__ __forceinline__ void mma_init(int (&D)[4], const u32 (&Af)[4], const u32 b0, const u32 b1) {
    const int m = F_MAGIC_I;
#if __CUDA_ARCH__ >= 800
    asm("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %10, %10, %10};"
        : "=r"(D[0]), "=r"(D[1]), "=r"(D[2]), "=r"(D[3])
        : "r"(Af[0]), "r"(Af[1]), "r"(Af[2]), "r"(Af[3]), "r"(b0), "r"(b1), "r"(m));
#else                                                           // the same fragment as four m8n8k16
    asm("mma.sync.aligned.m8n8k16.row.col.s32.s8.s8.s32 {%0, %1}, {%2}, {%3}, {%4, %4};" : "=r"(D[0]), "=r"(D[1]) : "r"(Af[0]), "r"(b0), "r"(m));
    asm("mma.sync.aligned.m8n8k16.row.col.s32.s8.s8.s32 {%0, %1}, {%2}, {%3}, {%4, %4};" : "=r"(D[2]), "=r"(D[3]) : "r"(Af[1]), "r"(b0), "r"(m));
    asm("mma.sync.aligned.m8n8k16.row.col.s32.s8.s8.s32 {%0, %1}, {%2}, {%3}, {%0, %1};" : "+r"(D[0]), "+r"(D[1]) : "r"(Af[2]), "r"(b1));
    asm("mma.sync.aligned.m8n8k16.row.col.s32.s8.s8.s32 {%0, %1}, {%2}, {%3}, {%0, %1};" : "+r"(D[2]), "+r"(D[3]) : "r"(Af[3]), "r"(b1));
#endif
}

// shared layouts (conflict-free for a whole warp):
//   sW [sub-block][16-row fragment][row & 7][k & 3] { row r word k, row r + 8 word k, row r word k + 4, row r + 8 word k + 4 }
//   sA [sub-block][token][k & 3] { word k, word k + 4 }
struct TileW { u32 w0, w1, w2, w3; };                           // half 0: words 0-3 of the block; half 1: words 4, 5, 6
__device__ __forceinline__ void tile_fetch_w(TileW& r, const u32* Wq, const int half_, const int row) {
    const u32* b = Wq + (half_ ? 4 : 0) * CH + row;
    r.w0 = b[0]; r.w1 = b[CH]; r.w2 = b[2 * CH];
    if (!half_) r.w3 = b[3 * CH];
}
template <int R>
__device__ __forceinline__ void tile_decode_w(const TileW& r, u32* sW, float* sh, const int half_, const int row) {
#define TSP(w) sW[((((w) >> 3) * (R / 16) + (row >> 4)) * 8 + (row & 7)) * 16 + ((w) & 3) * 4 + (((w) >> 2) & 1) * 2 + ((row >> 3) & 1)]
    if (!half_) {
        u32 l[4] = { r.w0 & 0x00FF00FFu, r.w1 & 0x00FF00FFu, r.w2 & 0x00FF00FFu, r.w3 & 0x00FF00FFu };
        u32 h[4] = { (r.w0 >> 8) & 0x00FF00FFu, (r.w1 >> 8) & 0x00FF00FFu, (r.w2 >> 8) & 0x00FF00FFu, (r.w3 >> 8) & 0x00FF00FFu };
#pragma unroll
        for (int n = 0; n < 5; ++n)
#pragma unroll
            for (int c = 0; c < 4; ++c) { u32 D; DIGIT_STEP(l[c], h[c], D) TSP(n * 4 + c) = D; }
    } else {
        u32 l0 = r.w0 & 0x00FF00FFu, h0 = (r.w0 >> 8) & 0x00FF00FFu, l1 = r.w1 & 0x00FF00FFu, h1 = (r.w1 >> 8) & 0x00FF00FFu;
#pragma unroll
        for (int n = 0; n < 5; ++n) { u32 D0, D1; DIGIT_STEP(l0, h0, D0) DIGIT_STEP(l1, h1, D1) TSP(20 + n * 2) = D0; TSP(20 + n * 2 + 1) = D1; }
        u32 v = (r.w2 & 0xFFu) | ((r.w2 & 0xFF00u) << 8);
        const u32 t0 = v * 3u; v = t0 & 0x00FF00FFu; const u32 t1 = v * 3u; v = t1 & 0x00FF00FFu; const u32 t2 = v * 3u; v = t2 & 0x00FF00FFu; const u32 t3 = v * 3u;
        TSP(30) = __byte_perm(t0, t1, 0x7531); TSP(31) = __byte_perm(t2, t3, 0x7531);
        sh[row] = __half2float(__ushort_as_half((unsigned short) (r.w2 >> 16)));
    }
#undef TSP
}

template <int TC>
__global__ __launch_bounds__(TILE_NT, 1) void mvm_tile_kernel(const u32* W, const signed char* A, const float* d8, const int* sa, float* Y,
                                                              const int G, const int yrows, const int K, const ssq_t* ss, const int parts) {
    constexpr int RW = TILE_RWT(TC), R = tile_rows(TC), NF = RW / 16;         // rows per block, row fragments per warp
    constexpr int NW = (2 * R + TILE_NT - 1) / TILE_NT;         // half-rows decoded per thread and group
    constexpr int NI = (TC * 8 + TILE_NT - 1) / TILE_NT;        // 16-byte activation items per thread and group (8 per token)
    __shared__ __align__(16) u32 sW[R * 32];
    __shared__ __align__(16) u32 sA[TC * 32];
    __shared__ __align__(16) float sd[4][TC];                   // scale [sub-block][token]
    __shared__ __align__(16) float sc[TC];                      // group correction [token]
    __shared__ float sh[R];                                     // group scale [row]
    const int tid = threadIdx.x, lane = tid & 31, wi = tid >> 5;
    const int tgw = wi & (TC / 32 - 1), rgw = wi / (TC / 32);
    const int row0 = blockIdx.x * R, tok0 = 0, part = blockIdx.y;              // all tokens in one block; blockIdx.y = column partition
    const int gper = (G + parts - 1) / parts, g0 = part * gper, g1 = min(G, g0 + gper);
    const u32* Wc = W + (size_t) (row0 / CH) * G * 7 * CH + (row0 % CH);
    const uint4* A0 = (const uint4*) A;
    const bool ls = tid < TC && tok0 + tid < K;
    TileW nw[NW]; uint4 na[NI]; float4 nd = make_float4(0, 0, 0, 0); float nc = 0;
#pragma unroll
    for (int j = 0; j < NW; ++j) nw[j].w0 = nw[j].w1 = nw[j].w2 = nw[j].w3 = 0;
#pragma unroll
    for (int j = 0; j < NI; ++j) na[j] = make_uint4(0, 0, 0, 0);
#define TILE_FETCH(g_) { \
        for (int j = 0; j < NW; ++j) { const int hr = tid + TILE_NT * j; if (hr < 2 * R) tile_fetch_w(nw[j], Wc + (size_t) (g_) * 7 * CH, hr / R, hr % R); } \
        for (int j = 0; j < NI; ++j) { const int it = tid + TILE_NT * j, tk = it >> 3; if (tk < TC && tok0 + tk < K) na[j] = A0[((size_t) (tok0 + tk) * G + (g_)) * 8 + (it & 7)]; } \
        if (ls) { const size_t o_ = ((size_t) (tok0 + tid) * G + (g_)) * 4; nd = *(const float4*) (d8 + o_); nc = ((const float*) sa)[o_]; } }
#define TILE_STORE { \
        for (int j = 0; j < NW; ++j) { const int hr = tid + TILE_NT * j; if (hr < 2 * R) tile_decode_w<R>(nw[j], sW, sh, hr / R, hr % R); } \
        for (int j = 0; j < NI; ++j) { const int it = tid + TILE_NT * j, tk = it >> 3, v_ = it & 7; if (tk < TC && tok0 + tk < K) ((uint4*) sA)[((v_ >> 1) * TC + tk) * 2 + (v_ & 1)] = na[j]; } \
        if (ls) { sd[0][tid] = nd.x; sd[1][tid] = nd.y; sd[2][tid] = nd.z; sd[3][tid] = nd.w; sc[tid] = nc; } }
    TILE_FETCH(g0)
    TILE_STORE
    if (g0 + 1 < g1) TILE_FETCH(g0 + 1)
    __syncthreads();

    float acc[NF][TILE_TT][4];                                  // [row fragment][token fragment][row half * 2 + column]
#pragma unroll
    for (int f = 0; f < NF; ++f)
#pragma unroll
        for (int t = 0; t < TILE_TT; ++t) acc[f][t][0] = acc[f][t][1] = acc[f][t][2] = acc[f][t][3] = 0.0f;
    const int kA = lane & 3, r0 = rgw * RW + (lane >> 2), tb0 = tgw * 32;
    for (int g = g0; g < g1; ++g) {
        float dA[NF][2], nA[NF][2];
#pragma unroll
        for (int f = 0; f < NF; ++f)
#pragma unroll
            for (int u = 0; u < 2; ++u) { dA[f][u] = sh[r0 + 16 * f + 8 * u]; nA[f][u] = -F_MAGIC_F * dA[f][u]; }
#pragma unroll
        for (int q = 0; q < 4; ++q) {
            u32 Af[NF][4];
#pragma unroll
            for (int f = 0; f < NF; ++f) {
                const uint4 v = ((const uint4*) sW)[((q * (R / 16) + (rgw * NF + f)) * 8 + (lane >> 2)) * 4 + kA];
                Af[f][0] = v.x; Af[f][1] = v.y; Af[f][2] = v.z; Af[f][3] = v.w;
            }
#pragma unroll
            for (int t = 0; t < TILE_TT; ++t) {
                const int tok = tb0 + t * 8;
                const uint2 b = ((const uint2*) sA)[(q * TC + tok + (lane >> 2)) * 4 + kA];
                const float2 ds = ((const float2*) sd[q])[(tok >> 1) + kA];
#pragma unroll
                for (int f = 0; f < NF; ++f) {
                    int D[4];
                    mma_init(D, Af[f], b.x, b.y);
#pragma unroll
                    for (int u = 0; u < 2; ++u) {
                        acc[f][t][u * 2]     = __fmaf_rn(__fmaf_rn(__int_as_float(D[u * 2]),     dA[f][u], nA[f][u]), ds.x, acc[f][t][u * 2]);
                        acc[f][t][u * 2 + 1] = __fmaf_rn(__fmaf_rn(__int_as_float(D[u * 2 + 1]), dA[f][u], nA[f][u]), ds.y, acc[f][t][u * 2 + 1]);
                    }
                }
            }
        }
#pragma unroll
        for (int t = 0; t < TILE_TT; ++t) {
            const float2 c = ((const float2*) sc)[((tb0 + t * 8) >> 1) + kA];
#pragma unroll
            for (int f = 0; f < NF; ++f)
#pragma unroll
                for (int u = 0; u < 2; ++u) { acc[f][t][u * 2] = __fmaf_rn(-dA[f][u], c.x, acc[f][t][u * 2]); acc[f][t][u * 2 + 1] = __fmaf_rn(-dA[f][u], c.y, acc[f][t][u * 2 + 1]); }
        }
        __syncthreads();
        if (g + 1 < g1) {
            TILE_STORE
            if (g + 2 < g1) TILE_FETCH(g + 2)
        }
        __syncthreads();
    }
#pragma unroll
    for (int t = 0; t < TILE_TT; ++t)
#pragma unroll
        for (int v = 0; v < 2; ++v) {
            const int tok = tok0 + tb0 + t * 8 + 2 * kA + v;
            if (tok < K) {
                const float ir = ss ? ssq_rms_inv(ss[tok], D_MODEL) : 1.0f;
#pragma unroll
                for (int f = 0; f < NF; ++f)
#pragma unroll
                    for (int u = 0; u < 2; ++u) Y[((size_t) tok * parts + part) * yrows + row0 + r0 + 16 * f + 8 * u] = acc[f][t][u * 2 + v] * ir;
            }
        }
#undef TILE_FETCH
#undef TILE_STORE
}

// token embedding rows (PTQ1_0, Hadamard-latent): decode, Hadamard, signs.  rows: [k][1120 bytes]
__global__ void embed_kernel(const unsigned char* rows, const float* signs, float* dst) {
    __shared__ float s[1024];
    const int seg = blockIdx.x, k = blockIdx.y, tid = threadIdx.x;
    for (int e = tid; e < 1024; e += 256) {
        const int g = seg * 8 + (e >> 7), i = e & 127;
        const unsigned char* b = rows + (size_t) k * 1120 + g * 28;
        int by, n;
        if (i < 80) { by = i & 15; n = i >> 4; } else if (i < 120) { by = 16 + ((i - 80) & 7); n = (i - 80) >> 3; } else { by = 24 + ((i - 120) & 1); n = (i - 120) >> 1; }
        u32 q = b[by];
        for (int r = 0; r < n; ++r) q = (q * 3) & 0xFF;
        const float d = __half2float(__ushort_as_half((unsigned short) (b[26] | (b[27] << 8))));
        s[e] = (float) ((int) ((q * 3) >> 8) - 1) * d;
    }
    __syncthreads();
    for (int st = 0; st < 10; ++st) {
        const int h = 1 << st;
        for (int idx = tid; idx < 512; idx += 256) {
            const int j = ((idx >> st) << (st + 1)) | (idx & (h - 1));
            const float x = s[j], y = s[j + h];
            s[j] = x + y; s[j + h] = x - y;
        }
        __syncthreads();
    }
    for (int e = tid; e < 1024; e += 256) dst[(size_t) k * D_MODEL + seg * 1024 + e] = s[e] * 0.03125f * signs[seg * 1024 + e];
}

// ---------------------------------------------------------------------------------------------
// x[k][i] += sum over parts of y[k][part][off + i]
// ---------------------------------------------------------------------------------------------
__global__ void addres_kernel(float* x, const float* yp, int n, int parts, int yrows, int off, ssq_t* ssq) {
    const int k = blockIdx.y;
    float acc = 0;
    for (int i = blockIdx.x * 256 + threadIdx.x; i < n; i += gridDim.x * 256) {
        float t = 0;
        for (int p = 0; p < parts; ++p) t += yp[(size_t) (k * parts + p) * yrows + off + i];
        const float v = x[(size_t) k * n + i] + t;
        x[(size_t) k * n + i] = v;
        acc += v * v;
    }
    acc = warp_sum(acc);
    if ((threadIdx.x & 31) == 0) atomicAdd(ssq + k, (ssq_t) llrint((double) acc * 65536.0));
}

// F32 beta / alpha gate projections of the RMS-normalised activation
__global__ void gate_kernel(const float* x, const ssq_t* ssq, const float* nw, const half* wb, const half* wa, float* out, int K) {
    __shared__ float part[8][8];
    const int k0 = blockIdx.y * 8, row = blockIdx.x;
    const int nk = min(8, K - k0);
    const half* W = row < 48 ? wb + (size_t) row * D_MODEL : wa + (size_t) (row - 48) * D_MODEL;
    float acc[8], ir[8];
    for (int t = 0; t < 8; ++t) { acc[t] = 0; ir[t] = t < nk ? ssq_rms_inv(ssq[k0 + t], D_MODEL) : 0.0f; }
    for (int e = threadIdx.x; e < D_MODEL; e += 256) {
        const float w = __half2float(W[e]), nwe = nw[e];
#pragma unroll
        for (int t = 0; t < 8; ++t) if (t < nk) acc[t] += w * (x[(size_t) (k0 + t) * D_MODEL + e] * ir[t] * nwe);
    }
#pragma unroll
    for (int t = 0; t < 8; ++t) if (t < nk) {
        const float a = warp_sum(acc[t]);
        if ((threadIdx.x & 31) == 0) part[t][threadIdx.x >> 5] = a;
    }
    __syncthreads();
    if (threadIdx.x < nk) { float t = 0; for (int i = 0; i < 8; ++i) t += part[threadIdx.x][i]; out[(k0 + threadIdx.x) * GATE96 + row] = t; }
}

// depthwise conv (4 taps) + silu + l2 norm of q/k heads; 80 blocks of 128 channels
// raw_in (optional): recorded qkv channel inputs [k][10240] from a previous run (rollback replay); rec_out: record them
__global__ void convqk_kernel(const float* yA, int parts, int yrows, const float* ck, const float* st_in, float* st_out, float* qkv,
                              const float* raw_in, float* rec_out, int K) {
    __shared__ float ps[4];
    const int b = blockIdx.x, tid = threadIdx.x, ch = b * 128 + tid;
    float w0 = st_in[3 * ch], w1 = st_in[3 * ch + 1], w2 = st_in[3 * ch + 2];
    const float c0 = ck[4 * ch], c1 = ck[4 * ch + 1], c2 = ck[4 * ch + 2], c3 = ck[4 * ch + 3];
    for (int k = 0; k < K; ++k) {
        float v = 0;
        if (raw_in) v = raw_in[(size_t) k * QKV + ch];
        else v = part_sum(yA + (size_t) k * parts * yrows + ch, parts, yrows);
        if (rec_out) rec_out[(size_t) k * QKV + ch] = v;
        float o = silu_f(w0 * c0 + w1 * c1 + w2 * c2 + v * c3);
        w0 = w1; w1 = w2; w2 = v;
        if (b < 32) {
            const float ss = warp_sum(o * o);
            if ((tid & 31) == 0) ps[tid >> 5] = ss;
            __syncthreads();
            o *= 1.0f / fmaxf(sqrtf(ps[0] + ps[1] + ps[2] + ps[3]), 1e-6f);
            __syncthreads();
        }
        qkv[(size_t) k * QKV + ch] = o;
    }
    st_out[3 * ch] = w0; st_out[3 * ch + 1] = w1; st_out[3 * ch + 2] = w2;
}

// gated delta net: 48 v-heads, 128x128 state per head (stored [head][col][row]); tokens are applied sequentially.
__global__ void gdn_kernel(const float* qkv, const float* g_raw, const float* dt_bias, const float* ssm_a,
                           const float* S_in, float* S_out, float* out, int K) {
    const int h = blockIdx.x, cgrp = blockIdx.y;
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int col = cgrp * 8 + warp;
    const int nk = h % 16;
    const float* Sr = S_in + ((size_t) h * 128 + col) * 128;
    float* Sw = S_out + ((size_t) h * 128 + col) * 128;
    float s_[4];
    for (int r = 0; r < 4; ++r) s_[r] = Sr[r * 32 + lane];
    for (int k = 0; k < K; ++k) {
        const float* qk = qkv + (size_t) k * QKV;
        const float bet = 1.0f / (1.0f + expf(-g_raw[k * GATE96 + h]));
        const float gv = expf(ssm_a[h] * softplus_f(g_raw[k * GATE96 + 48 + h] + dt_bias[h]));
        float kr[4], qr[4];
        for (int r = 0; r < 4; ++r) { kr[r] = qk[2048 + nk * 128 + r * 32 + lane]; qr[r] = qk[nk * 128 + r * 32 + lane]; }
        float kv = 0;
        for (int r = 0; r < 4; ++r) kv += s_[r] * kr[r];
        kv = warp_sum(kv);
        const float delta = (qk[4096 + h * 128 + col] - gv * kv) * bet;
        float acc = 0;
        for (int r = 0; r < 4; ++r) { s_[r] = gv * s_[r] + kr[r] * delta; acc += s_[r] * qr[r]; }
        acc = warp_sum(acc) * 0.088388347648318447f;        // 1 / sqrt(128)
        if (lane == 0 && out) out[(size_t) k * D_INNER + h * 128 + col] = acc;
    }
    for (int r = 0; r < 4; ++r) Sw[r * 32 + lane] = s_[r];
}

// ---------------------------------------------------------------------------------------------
// wide steps (prefill) of the recurrent layers.  The 1..8-token kernels above are kept as they are, so narrow steps stay bit-identical.
// ---------------------------------------------------------------------------------------------
// beta / alpha projections: a block evaluates 8 rows for 8 tokens, so each activation it reads serves 8 rows
__global__ __launch_bounds__(256) void gate_wide_kernel(const float* x, const ssq_t* ssq, const float* nw, const half* wb, const half* wa, float* out, int K) {
    __shared__ float part[8][64];
    const int r0 = blockIdx.x * 8, k0 = blockIdx.y * 8, tid = threadIdx.x;
    const int nk = min(8, K - k0);
    float acc[8][8], ir[8];
#pragma unroll
    for (int t = 0; t < 8; ++t) { ir[t] = t < nk ? ssq_rms_inv(ssq[k0 + t], D_MODEL) : 0.0f; for (int r = 0; r < 8; ++r) acc[r][t] = 0; }
    for (int e = tid; e < D_MODEL; e += 256) {
        float xn[8];
        const float nwe = nw[e];
#pragma unroll
        for (int t = 0; t < 8; ++t) xn[t] = t < nk ? x[(size_t) (k0 + t) * D_MODEL + e] * ir[t] * nwe : 0.0f;
#pragma unroll
        for (int r = 0; r < 8; ++r) {
            const int row = r0 + r;
            const float w = __half2float(row < 48 ? wb[(size_t) row * D_MODEL + e] : wa[(size_t) (row - 48) * D_MODEL + e]);
#pragma unroll
            for (int t = 0; t < 8; ++t) acc[r][t] += w * xn[t];
        }
    }
#pragma unroll
    for (int r = 0; r < 8; ++r)
#pragma unroll
        for (int t = 0; t < 8; ++t) { const float v = warp_sum(acc[r][t]); if ((tid & 31) == 0) part[tid >> 5][r * 8 + t] = v; }
    __syncthreads();
    if (tid < 64 && (tid & 7) < nk) { float v = 0; for (int w = 0; w < 8; ++w) v += part[w][tid]; out[(k0 + (tid & 7)) * GATE96 + r0 + (tid >> 3)] = v; }
}

// depthwise conv: no recurrence, so every (channel, token) is independent; the inputs before the step come from the state
__global__ __launch_bounds__(128) void convqk_wide_kernel(const float* yA, int parts, int yrows, const float* ck, const float* st_in, float* st_out, float* qkv, int K) {
    __shared__ float ps[4];
    const int b = blockIdx.x, k = blockIdx.y, tid = threadIdx.x, ch = b * 128 + tid;
    float x[4];
#pragma unroll
    for (int j = 0; j < 4; ++j) { const int kk = k - 3 + j; x[j] = kk >= 0 ? part_sum(yA + (size_t) kk * parts * yrows + ch, parts, yrows) : st_in[3 * ch + 3 + kk]; }
    float o = silu_f(x[0] * ck[4 * ch] + x[1] * ck[4 * ch + 1] + x[2] * ck[4 * ch + 2] + x[3] * ck[4 * ch + 3]);
    if (b < 32) {
        const float ss = warp_sum(o * o);
        if ((tid & 31) == 0) ps[tid >> 5] = ss;
        __syncthreads();
        o *= 1.0f / fmaxf(sqrtf(ps[0] + ps[1] + ps[2] + ps[3]), 1e-6f);
    }
    qkv[(size_t) k * QKV + ch] = o;
    if (k >= K - 3) st_out[3 * ch + k - (K - 3)] = x[3];
}

// gated delta net over many tokens: the keys, queries, values and gates of TT tokens are staged in shared memory once per block
// (they are the same for all 128 columns of a head), and the two reductions of a token are done together:
// with S the state before the token, out = g (S q) + delta (k q), delta = (v - g (S k)) beta
template <int TT>
__global__ __launch_bounds__(256) void gdn_wide_kernel(const float* qkv, const float* g_raw, const float* dt_bias, const float* ssm_a,
                                                       const float* S_in, float* S_out, float* out, int K) {
    __shared__ __align__(16) float sk[TT][128];
    __shared__ __align__(16) float sq[TT][128];
    __shared__ float sv[TT][32], sg[TT], sbt[TT], skq[TT];
    // block: 32 columns of one head; 8 lanes per column, lane j of a column holds rows 4 j + 32 i + (0..3), i = 0..3
    const int h = blockIdx.x, cgrp = blockIdx.y, tid = threadIdx.x, sub = tid & 7, cl = tid >> 3;
    const int col = cgrp * 32 + cl, nk = h % 16;
    const float* Sr = S_in + ((size_t) h * 128 + col) * 128;
    float s_[16];
#pragma unroll
    for (int i = 0; i < 4; ++i) { const float4 v = *(const float4*) (Sr + 32 * i + 4 * sub); s_[4 * i] = v.x; s_[4 * i + 1] = v.y; s_[4 * i + 2] = v.z; s_[4 * i + 3] = v.w; }
    for (int k0 = 0; k0 < K; k0 += TT) {
        const int nt = min(TT, K - k0);
        for (int i = tid; i < TT * 128; i += 256) {
            const int t = i >> 7, e = i & 127;
            if (t < nt) { const float* qk = qkv + (size_t) (k0 + t) * QKV; sk[t][e] = qk[2048 + nk * 128 + e]; sq[t][e] = qk[nk * 128 + e]; }
        }
        for (int i = tid; i < TT * 32; i += 256) if ((i >> 5) < nt) sv[i >> 5][i & 31] = qkv[(size_t) (k0 + (i >> 5)) * QKV + 4096 + h * 128 + cgrp * 32 + (i & 31)];
        if (tid < nt) {
            const int k = k0 + tid;
            sbt[tid] = 1.0f / (1.0f + expf(-g_raw[k * GATE96 + h]));
            sg[tid] = expf(ssm_a[h] * softplus_f(g_raw[k * GATE96 + 48 + h] + dt_bias[h]));
        }
        __syncthreads();
        for (int t = tid >> 5; t < nt; t += 8) {
            float p = 0;
#pragma unroll
            for (int r = 0; r < 4; ++r) p += sk[t][r * 32 + (tid & 31)] * sq[t][r * 32 + (tid & 31)];
            p = warp_sum(p);
            if ((tid & 31) == 0) skq[t] = p;
        }
        __syncthreads();
        for (int t = 0; t < nt; ++t) {
            float kr[16], a = 0, c = 0;
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                const float4 kv = *(const float4*) &sk[t][32 * i + 4 * sub], qv = *(const float4*) &sq[t][32 * i + 4 * sub];
                kr[4 * i] = kv.x; kr[4 * i + 1] = kv.y; kr[4 * i + 2] = kv.z; kr[4 * i + 3] = kv.w;
                a += s_[4 * i] * kv.x + s_[4 * i + 1] * kv.y + s_[4 * i + 2] * kv.z + s_[4 * i + 3] * kv.w;
                c += s_[4 * i] * qv.x + s_[4 * i + 1] * qv.y + s_[4 * i + 2] * qv.z + s_[4 * i + 3] * qv.w;
            }
#pragma unroll
            for (int o = 4; o > 0; o >>= 1) { a += __shfl_xor_sync(0xFFFFFFFFu, a, o); c += __shfl_xor_sync(0xFFFFFFFFu, c, o); }
            const float gv = sg[t], delta = (sv[t][cl] - gv * a) * sbt[t];
#pragma unroll
            for (int r = 0; r < 16; ++r) s_[r] = gv * s_[r] + kr[r] * delta;
            if (sub == 0 && out) out[(size_t) (k0 + t) * D_INNER + h * 128 + col] = (gv * c + delta * skq[t]) * 0.088388347648318447f;
        }
        __syncthreads();
    }
    float* Sw = S_out + ((size_t) h * 128 + col) * 128;
#pragma unroll
    for (int i = 0; i < 4; ++i) *(float4*) (Sw + 32 * i + 4 * sub) = make_float4(s_[4 * i], s_[4 * i + 1], s_[4 * i + 2], s_[4 * i + 3]);
}

// ---------------------------------------------------------------------------------------------
// full attention: prep (norms + rope, KV store) and the per-head softmax
// yA rows: q 12288 (24 heads x [256 q | 256 gate]) | k 1024 | v 1024
// KV layout per layer: [kvhead 4][ctx][256] fp16
// ---------------------------------------------------------------------------------------------
__global__ void attn_prep_kernel(const float* yA, int parts, int yrows, const float* qn, const float* kn,
                                 float* Q, float* gate, half* kc, half* vc, int ctx, int pos, float theta_base, int hb0, const int* posd = nullptr) {
    if (posd) pos = *posd;                                   // graph replay: the position lives in device memory
    __shared__ float sh[256];
    __shared__ float ps[8];
    const int hb = blockIdx.x + hb0, k = blockIdx.y, tid = threadIdx.x;   // 256 threads = 256 dims; hb0 = 24 skips the query heads
    const int src = hb < 24 ? hb * 512 + tid : (hb < 28 ? 12288 + (hb - 24) * 256 + tid : 13312 + (hb - 28) * 256 + tid);
    float v = part_sum(yA + (size_t) k * parts * yrows + src, parts, yrows);
    const bool has_kv = hb >= 24;
    if (hb < 24) {
        const float gt = part_sum(yA + (size_t) k * parts * yrows + hb * 512 + 256 + tid, parts, yrows);
        gate[(size_t) k * D_INNER + hb * 256 + tid] = 1.0f / (1.0f + expf(-gt));
    }
    if (hb < 28) {
        const float ss = warp_sum(v * v);
        if ((tid & 31) == 0) ps[tid >> 5] = ss;
        __syncthreads();
        float t = 0; for (int i = 0; i < 8; ++i) t += ps[i];
        v *= 1.0f / sqrtf(t / 256.0f + 1e-6f);
        v *= (hb < 24 ? qn[tid] : kn[tid]);
        sh[tid] = v;
        __syncthreads();
        if (tid < 64) {                         // rope on the first 64 dims: pairs (i, i + 32)
            const int i = tid & 31;
            const float th = (float) (pos + k) * powf(theta_base, -2.0f * i / 64.0f);
            const float c = cosf(th), s = sinf(th);
            const float x0 = sh[i], x1 = sh[i + 32];
            v = tid < 32 ? x0 * c - x1 * s : x0 * s + x1 * c;
        }
    }
    if (hb < 24) Q[((size_t) k * 24 + hb) * 256 + tid] = v;
    else if (hb < 28) kc[((size_t) (hb - 24) * ctx + (pos + k)) * 256 + tid] = __float2half(v);
    else vc[((size_t) (hb - 28) * ctx + (pos + k)) * 256 + tid] = __float2half(v);
    (void) has_kv;
}

// block (head, group of T tokens).  The KV entries are read once for all T tokens; softmax is evaluated online over chunks of 512
// positions (running maximum, rescaled accumulators), so no per-position storage of the whole context is needed.  The reduction
// order does not depend on T, so T = 1 / 4 / 8 give bit-identical results.
static const int ACH = 512;
template <int T>
__global__ __launch_bounds__(256, 1) void attn_kernel(const float* Q, const float* gate, const half* kc, const half* vc, int ctx, int pos, float* out, int K, int pstart,
                                                     const int* posd = nullptr) {
    if (posd) pos = *posd;
    __shared__ float sc[T][ACH];
    __shared__ float red[T][32];
    __shared__ float mxn[T], scl[T], sumt[T], mxo[T];
    const int h = blockIdx.x, k0 = blockIdx.y * T, tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    const int nt = min(T, K - k0), kvh = h / 6;
    const int L = pos + k0 + nt;                       // positions pstart..L-1 can be visible to some token of the group
    const half* Kp = kc + (size_t) kvh * ctx * 256;
    const half* Vp = vc + (size_t) kvh * ctx * 256;
    float qr[T][8];                                    // [token][dim slot]
#pragma unroll
    for (int t = 0; t < T; ++t)
#pragma unroll
        for (int m = 0; m < 8; ++m) qr[t][m] = t < nt ? Q[((size_t) (k0 + t) * 24 + h) * 256 + m * 32 + lane] : 0.0f;
    float acc[T];
#pragma unroll
    for (int t = 0; t < T; ++t) acc[t] = 0.0f;
    if (tid < T) { mxo[tid] = -1e30f; sumt[tid] = 0.0f; }
    __syncthreads();

    const int rt = tid & (T - 1), rp = tid / T;        // reduction role: token, partial
    for (int c0 = pstart; c0 < L; c0 += ACH) {
        const int n = min(ACH, L - c0);
        // scores: one warp per position
        for (int i = warp; i < n; i += 8) {
            const half* kr = Kp + (size_t) (c0 + i) * 256 + lane;
            float s[T];
            {
                float kv = __half2float(kr[0]);
#pragma unroll
                for (int t = 0; t < T; ++t) s[t] = qr[t][0] * kv;
#pragma unroll
                for (int m = 1; m < 8; ++m) {
                    kv = __half2float(kr[m * 32]);
#pragma unroll
                    for (int t = 0; t < T; ++t) s[t] += qr[t][m] * kv;
                }
            }
            // reduce T values over 32 lanes: recursive halving while more than one value is left, then plain xor rounds
            int o = 16, left = T, t = 0;
#pragma unroll
            for (int r = 0; r < 3; ++r) if (left > 1) {
                left >>= 1;
                const bool up = (lane & o) != 0;
#pragma unroll
                for (int x = 0; x < T / 2; ++x) if (x < left) {
                    const float snd = up ? s[x] : s[x + left];
                    s[x] = (up ? s[x + left] : s[x]) + __shfl_xor_sync(0xFFFFFFFFu, snd, o);
                }
                t = t * 2 + (up ? 1 : 0);
                o >>= 1;
            }
            float v = s[0];
            const int rest = o;
#pragma unroll
            for (int r = 0; r < 5; ++r) { if (o > 0) { v += __shfl_xor_sync(0xFFFFFFFFu, v, o); o >>= 1; } }
            if ((lane & (2 * rest - 1)) == 0) sc[t][i] = (t < nt && c0 + i <= pos + k0 + t) ? v * 0.0625f : -1e30f;
        }
        __syncthreads();
        // chunk maximum per token: 32 partial maxima, then T threads combine
        if (rp < 32) {
            float m = -1e30f;
            for (int i = rp; i < n; i += 32) m = fmaxf(m, sc[rt][i]);
            red[rt][rp] = m;
        }
        __syncthreads();
        if (tid < T) {
            float m = -1e30f;
            for (int p = 0; p < 32; ++p) m = fmaxf(m, red[tid][p]);
            const float nm = fmaxf(mxo[tid], m);
            mxn[tid] = nm; scl[tid] = expf(mxo[tid] - nm);
        }
        __syncthreads();
        if (rp < 32) {
            float su = 0;
            for (int i = rp; i < n; i += 32) { const float e = expf(sc[rt][i] - mxn[rt]); sc[rt][i] = e; su += e; }
            red[rt][rp] = su;
        }
        __syncthreads();
        if (tid < T) {
            float su = 0; for (int p = 0; p < 32; ++p) su += red[tid][p];
            sumt[tid] = sumt[tid] * scl[tid] + su; mxo[tid] = mxn[tid];
        }
#pragma unroll
        for (int t = 0; t < T; ++t) acc[t] *= scl[t];
        __syncthreads();
        for (int i = 0; i < n; ++i) {
            const float v = __half2float(Vp[(size_t) (c0 + i) * 256 + tid]);
#pragma unroll
            for (int t = 0; t < T; ++t) acc[t] += sc[t][i] * v;
        }
        __syncthreads();
    }
#pragma unroll
    for (int t = 0; t < T; ++t) if (t < nt) {
        const int k = k0 + t;
        out[(size_t) k * D_INNER + h * 256 + tid] = acc[t] / sumt[t] * gate[(size_t) k * D_INNER + h * 256 + tid];
    }
}

// packed argmax: key = (order-preserving value bits) << 32 | (~index), so ties keep the lowest index; one atomic per warp
__global__ void argmax_kernel(const float* y, int yrows, int n, ssq_t* out) {
    const int k = blockIdx.y;
    float bv = -1e38f; int bi = 0x7fffffff;
    for (int i = blockIdx.x * 256 + threadIdx.x; i < n; i += gridDim.x * 256) {
        const float v = y[(size_t) k * yrows + i];
        if (v > bv || (v == bv && i < bi)) { bv = v; bi = i; }
    }
    // reduce inside the warp
    for (int o = 16; o > 0; o >>= 1) {
        const float v2 = __shfl_xor_sync(0xFFFFFFFFu, bv, o); const int i2 = __shfl_xor_sync(0xFFFFFFFFu, bi, o);
        if (v2 > bv || (v2 == bv && i2 < bi)) { bv = v2; bi = i2; }
    }
    if ((threadIdx.x & 31) == 0) {
        const u32 b = (u32) __float_as_int(bv);
        const u32 ord = (b & 0x80000000u) ? ~b : (b | 0x80000000u);
        const ssq_t key = ((ssq_t) ord << 32) | (ssq_t) (u32) (~(u32) bi);
        atomicMax(out + k, key);
    }
}

// ---------------------------------------------------------------------------------------------
// plain float helpers used by the MTP head (single token)
// ---------------------------------------------------------------------------------------------
__global__ void norm_store_kernel(const float* x, const ssq_t* ssq, const float* w, float* dst, int n, int stride) {
    const int k = blockIdx.y;
    const float ir = ssq_rms_inv(ssq[k], n);
    const int i = blockIdx.x * 256 + threadIdx.x;
    if (i < n) dst[(size_t) k * stride + i] = x[(size_t) k * n + i] * ir * w[i];
}
// y = a (+ b): RMS norm of y into dst, y itself into sum_out, its sum of squares into slot; one block per token, one launch
__global__ __launch_bounds__(512) void norm_kernel(const float* a, const float* b, float* sum_out, const float* w, float* dst, ssq_t* slot, int n, int stride) {
    __shared__ float part[16];
    const int k = blockIdx.x, tid = threadIdx.x;
    float v[10], acc = 0;                                       // n = 5120
#pragma unroll
    for (int i = 0; i < 10; ++i) { const size_t e = (size_t) k * n + tid + 512 * i; v[i] = a[e] + (b ? b[e] : 0.0f); acc += v[i] * v[i]; if (sum_out) sum_out[e] = v[i]; }
    acc = warp_sum(acc);
    if ((tid & 31) == 0) part[tid >> 5] = acc;
    __syncthreads();
    float t = 0;
    for (int i = 0; i < 16; ++i) t += part[i];
    const float ir = 1.0f / sqrtf(t / n + 1e-6f);
#pragma unroll
    for (int i = 0; i < 10; ++i) dst[(size_t) k * stride + tid + 512 * i] = v[i] * ir * w[tid + 512 * i];
    if (tid == 0 && slot) slot[k] = (ssq_t) llrint((double) t * 65536.0);
}
__global__ void swiglu_kernel(const float* g, const float* u, float* out, int n) {
    const int i = blockIdx.x * 256 + threadIdx.x;
    if (i < n) out[i] = silu_f(g[i]) * u[i];
}
// Q4_0 matrix-vector for T tokens per block: nib [nblk][4][R] words, scale [nblk][R] fp16; x [k][C], y [k][ystride]; rows r0..r0+nr.
// A block owns 64 rows x 8 column parts, and each thread requests two 32-weight blocks at a time, to keep the memory system busy.
// wide steps: attention on FP16 tensor cores.  Block = (query head, 16 tokens), 4 warps; K/V chunks of 32 positions are staged in
// shared memory.  Warp w scores positions 8w..8w+7 of a chunk against all 16 tokens, the softmax is evaluated online across the warps,
// and warp w accumulates output dimensions 64w..64w+63.  Queries and probabilities are rounded to FP16 (as in the fork's kernels).
__device__ __forceinline__ void ldsm_x4(u32 (&r)[4], const void* p) {
    const u32 a = (u32) __cvta_generic_to_shared(p);
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];" : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(a));
}
__device__ __forceinline__ void ldsm_x4_t(u32 (&r)[4], const void* p) {
    const u32 a = (u32) __cvta_generic_to_shared(p);
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];" : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(a));
}
__device__ __forceinline__ void mma_f16(float (&d)[4], const u32 (&a)[4], const u32 b0, const u32 b1) {
#if __CUDA_ARCH__ >= 800
    asm("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]) : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
#else
    asm("mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32 {%0, %1, %2, %3}, {%4, %5}, {%6}, {%0, %1, %2, %3};"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]) : "r"(a[0]), "r"(a[1]), "r"(b0));
    asm("mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32 {%0, %1, %2, %3}, {%4, %5}, {%6}, {%0, %1, %2, %3};"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]) : "r"(a[2]), "r"(a[3]), "r"(b1));
#endif
}
__device__ __forceinline__ u32 pack_h2(float a, float b) { const __half2 h = __floats2half2_rn(a, b); return *(const u32*) &h; }

static const int AW_T = 16, AW_P = 32, AW_S = 256 + 8;          // tokens per block, positions per chunk, padded row (halves)
__global__ __launch_bounds__(128) void attn_wide_kernel(const float* Q, const float* gate, const half* kc, const half* vc, int ctx, int pos, float* out, int K) {
    __shared__ __align__(16) half sQ[AW_T][AW_S];
    __shared__ __align__(16) half sK[AW_P][AW_S];
    __shared__ __align__(16) half sV[AW_P][AW_S];
    __shared__ __align__(16) half sP[AW_T][AW_P + 8];
    __shared__ float smax[4][AW_T], ssum[4][AW_T];
    const int h = blockIdx.x, k0 = blockIdx.y * AW_T, tid = threadIdx.x, w = tid >> 5, lane = tid & 31, g = lane >> 2, t4 = lane & 3;
    const int kvh = h / 6, nt = min(AW_T, K - k0);
    const half* Kp = kc + (size_t) kvh * ctx * 256;
    const half* Vp = vc + (size_t) kvh * ctx * 256;
    for (int i = tid; i < AW_T * 128; i += 128) {                // queries as FP16 (rows of absent tokens are zero)
        const int r = i >> 7, c = (i & 127) * 2;
        float2 v = make_float2(0, 0);
        if (r < nt) v = *(const float2*) &Q[((size_t) (k0 + r) * 24 + h) * 256 + c];
        *(__half2*) &sQ[r][c] = __floats2half2_rn(v.x, v.y);
    }
    float o[8][4];
#pragma unroll
    for (int j = 0; j < 8; ++j) o[j][0] = o[j][1] = o[j][2] = o[j][3] = 0.0f;
    float m_[2] = { -1e30f, -1e30f }, l_[2] = { 0.0f, 0.0f };      // rows g and g + 8
    const int L = pos + k0 + nt;                                 // positions 0..L-1 are visible to some token of the block
    for (int c0 = 0; c0 < L; c0 += AW_P) {
        __syncthreads();
        for (int i = tid; i < AW_P * 32; i += 128) {             // 16-byte pieces of K and V rows (zero past L)
            const int r = i >> 5, c = (i & 31) * 8;
            uint4 kv = make_uint4(0, 0, 0, 0), vv = make_uint4(0, 0, 0, 0);
            if (c0 + r < L) { kv = *(const uint4*) &Kp[(size_t) (c0 + r) * 256 + c]; vv = *(const uint4*) &Vp[(size_t) (c0 + r) * 256 + c]; }
            *(uint4*) &sK[r][c] = kv; *(uint4*) &sV[r][c] = vv;
        }
        __syncthreads();
        float sc[4] = { 0, 0, 0, 0 };                            // scores: rows g, g + 8; positions 8w + 2 t4 + (0, 1)
#pragma unroll
        for (int ks = 0; ks < 16; ks += 2) {
            u32 a0[4], a1[4], b[4];
            ldsm_x4(a0, &sQ[lane & 15][ks * 16 + (lane >> 4) * 8]);
            ldsm_x4(a1, &sQ[lane & 15][(ks + 1) * 16 + (lane >> 4) * 8]);
            ldsm_x4(b, &sK[8 * w + (lane & 7)][ks * 16 + (lane >> 3) * 8]);
            mma_f16(sc, a0, b[0], b[1]);
            mma_f16(sc, a1, b[2], b[3]);
        }
        float pm[2];
#pragma unroll
        for (int u = 0; u < 2; ++u) {
            const int tok = k0 + g + 8 * u;
#pragma unroll
            for (int v = 0; v < 2; ++v) { const int pp = c0 + 8 * w + 2 * t4 + v; sc[2 * u + v] = pp <= pos + tok ? sc[2 * u + v] * 0.0625f : -1e30f; }
            pm[u] = fmaxf(sc[2 * u], sc[2 * u + 1]);
            pm[u] = fmaxf(pm[u], __shfl_xor_sync(0xFFFFFFFFu, pm[u], 1)); pm[u] = fmaxf(pm[u], __shfl_xor_sync(0xFFFFFFFFu, pm[u], 2));
        }
        if (t4 == 0) { smax[w][g] = pm[0]; smax[w][g + 8] = pm[1]; }
        __syncthreads();
        float al[2];
#pragma unroll
        for (int u = 0; u < 2; ++u) {
            const int r = g + 8 * u;
            const float cm = fmaxf(fmaxf(smax[0][r], smax[1][r]), fmaxf(smax[2][r], smax[3][r])), nm = fmaxf(m_[u], cm);
            al[u] = expf(m_[u] - nm); m_[u] = nm;
            const float p0 = expf(sc[2 * u] - nm), p1 = expf(sc[2 * u + 1] - nm);
            *(__half2*) &sP[r][8 * w + 2 * t4] = __floats2half2_rn(p0, p1);
            float ps = p0 + p1; ps += __shfl_xor_sync(0xFFFFFFFFu, ps, 1); ps += __shfl_xor_sync(0xFFFFFFFFu, ps, 2);
            if (t4 == 0) ssum[w][r] = ps;
        }
#pragma unroll
        for (int j = 0; j < 8; ++j) { o[j][0] *= al[0]; o[j][1] *= al[0]; o[j][2] *= al[1]; o[j][3] *= al[1]; }
        __syncthreads();
#pragma unroll
        for (int u = 0; u < 2; ++u) { const int r = g + 8 * u; l_[u] = l_[u] * al[u] + ssum[0][r] + ssum[1][r] + ssum[2][r] + ssum[3][r]; }
#pragma unroll
        for (int kk = 0; kk < AW_P / 16; ++kk) {
            u32 a[4];
            ldsm_x4(a, &sP[lane & 15][kk * 16 + (lane >> 4) * 8]);
#pragma unroll
            for (int j = 0; j < 8; j += 2) {
                u32 b[4];
                ldsm_x4_t(b, &sV[kk * 16 + (lane & 15)][64 * w + 8 * j + (lane >> 4) * 8]);
                mma_f16(o[j], a, b[0], b[1]);
                mma_f16(o[j + 1], a, b[2], b[3]);
            }
        }
    }
#pragma unroll
    for (int u = 0; u < 2; ++u) {
        const int r = g + 8 * u;
        if (r >= nt) continue;
        const int k = k0 + r;
        const float il = 1.0f / l_[u];
#pragma unroll
        for (int j = 0; j < 8; ++j)
#pragma unroll
            for (int v = 0; v < 2; ++v) {
                const int d = h * 256 + 64 * w + 8 * j + 2 * t4 + v;
                out[(size_t) k * D_INNER + d] = o[j][2 * u + v] * il * gate[(size_t) k * D_INNER + d];
            }
    }
}
template <int T>
__global__ __launch_bounds__(512, 1) void q4mv_kernel(const u32* nib, const half* sc, const float* x, float* y, int R, int r0, int nblk, int ystride, int K) {
    __shared__ float ps[T][8][64];
    const int tid = threadIdx.x, part = tid >> 6, rr = blockIdx.x * 64 + (tid & 63), r = r0 + rr;
    const int k0 = blockIdx.y * T, C = nblk * 32, np = nblk / 8;
    const float4* xf = (const float4*) x;
    float acc[T];
#pragma unroll
    for (int t = 0; t < T; ++t) acc[t] = 0;
    for (int b = part * np; b < (part + 1) * np; b += 2) {
        u32 nw[2][4]; float d[2];
#pragma unroll
        for (int u = 0; u < 2; ++u) {
#pragma unroll
            for (int w = 0; w < 4; ++w) nw[u][w] = nib[((size_t) (b + u) * 4 + w) * R + r];
            d[u] = __half2float(sc[(size_t) (b + u) * R + r]);
        }
#pragma unroll
        for (int u = 0; u < 2; ++u) {
            float s[T];
#pragma unroll
            for (int t = 0; t < T; ++t) s[t] = 0;
#pragma unroll
            for (int w = 0; w < 4; ++w) {
                const u32 v = nw[u][w];
                float lo[4], hi[4];
#pragma unroll
                for (int by = 0; by < 4; ++by) { lo[by] = (float) ((int) ((v >> (8 * by)) & 15) - 8); hi[by] = (float) ((int) ((v >> (8 * by + 4)) & 15) - 8); }
#pragma unroll
                for (int t = 0; t < T; ++t) if (T == 1 || k0 + t < K) {
                    const float4* xb = xf + ((size_t) (k0 + t) * C + (size_t) (b + u) * 32) / 4;
                    const float4 xl = xb[w], xh = xb[w + 4];
                    s[t] += lo[0] * xl.x + lo[1] * xl.y + lo[2] * xl.z + lo[3] * xl.w + hi[0] * xh.x + hi[1] * xh.y + hi[2] * xh.z + hi[3] * xh.w;
                }
            }
#pragma unroll
            for (int t = 0; t < T; ++t) acc[t] += s[t] * d[u];
        }
    }
#pragma unroll
    for (int t = 0; t < T; ++t) ps[t][part][tid & 63] = acc[t];
    __syncthreads();
    if (tid < 64) {
#pragma unroll
        for (int t = 0; t < T; ++t) if (k0 + t < K) { float a = 0; for (int p = 0; p < 8; ++p) a += ps[t][p][tid]; y[(size_t) (k0 + t) * ystride + rr] = a; }
    }
}

// ---------------------------------------------------------------------------------------------
// host engine
// ---------------------------------------------------------------------------------------------
struct Mvm { u32* w = nullptr; int rows = 0, cols = 0, chunks = 0, parts = 1, yrows = 0; size_t bytes = 0; };

// device allocator: one exactly-sized arena for the weights and 8 MB slabs for the rest, because many separate
// cudaMalloc calls waste up to 2 MB each
struct Pool {
    char* base = nullptr; size_t cap = 0, used = 0, total = 0;
    static cudaError_t raw(void** p, size_t n) { return (cudaMalloc)(p, n); }
    void arena(size_t n) { CK(raw((void**) &base, n)); cap = n; used = 0; total = n; }
    void* take(size_t n) {
        n = (n + 255) & ~(size_t) 255;
        if (used + n > cap) {
            if (n > (size_t) (6u << 20)) { char* q; CK(raw((void**) &q, n)); total += n; return q; }
            const size_t c = (size_t) 8 << 20; CK(raw((void**) &base, c)); cap = c; used = 0; total += c;
        }
        char* q = base + used; used += n; return q;
    }
};
static Pool g_pool;                                      // misc arrays
static Pool g_wpool;                                     // weights
static cudaError_t pool_malloc(void** p, size_t n) { *p = g_pool.take(n); return cudaSuccess; }
#define cudaMalloc(p, n) pool_malloc((void**) (p), (n))

struct Layer {
    Mvm A, B, C, D;
    float* attn_norm = nullptr; float* post_norm = nullptr;
    float* qnorm = nullptr; float* knorm = nullptr;                       // attention layers
    float* ssm_norm = nullptr; float* dt_bias = nullptr; float* ssm_a = nullptr; float* conv_k = nullptr;
    half* wb = nullptr; half* wa = nullptr;
    float* conv[2] = {nullptr, nullptr}; float* gdn[2] = {nullptr, nullptr};   // recurrent state: two copies (checkpoint / new)
    float* rec_conv = nullptr; float* g_raw = nullptr;                          // recorded channel inputs [KMAX][10240], gates [KMAX][96]
    half* kc = nullptr; half* vc = nullptr;
    bool recr = false;
};

// plain Q4_0 matrix, stored planar for coalesced row access
struct Q4 { u32* nib = nullptr; half* sc = nullptr; int rows = 0, nblk = 0, cols = 0; };

struct Mtp {
    Q4 eh, qkv, wo, gu, down;                                                   // qkv and gate + up are stacked
    float* enorm = nullptr; float* hnorm = nullptr; float* attn_norm = nullptr; float* post_norm = nullptr; float* head_norm = nullptr;
    float* qnorm = nullptr; float* knorm = nullptr;
    half* kc = nullptr; half* vc = nullptr;
    float* b_e = nullptr; float* b_en = nullptr; float* b_cat = nullptr; float* y0 = nullptr; float* yn = nullptr; float* yq = nullptr;
    float* y1 = nullptr; float* pn = nullptr; float* fg = nullptr; float* fu = nullptr; float* fs = nullptr; float* y2 = nullptr; float* h_next = nullptr;
    float* Q = nullptr; float* gateb = nullptr; float* attn_out = nullptr; float* wo_out = nullptr; ssq_t* slots = nullptr;   // slots[16]
    float* y_logits = nullptr;
    bool ok = false;
};

static bool g_prof = getenv("TR_PROF") != nullptr;
static const int g_rep = getenv("TR_REP") ? std::max(1, atoi(getenv("TR_REP"))) : 1;   // profiling: launches per sample, so that sync latency averages out
static std::map<std::string, std::pair<double, int>> g_times;
#define PROF_BEGIN(st) std::chrono::steady_clock::time_point _t0; if (g_prof) { CK(cudaStreamSynchronize(st)); _t0 = std::chrono::steady_clock::now(); }
#define PROF_END(st, name) if (g_prof) { CK(cudaStreamSynchronize(st)); auto& _r = g_times[name]; _r.first += std::chrono::duration<double>(std::chrono::steady_clock::now() - _t0).count() / g_rep; _r.second++; }

static half* upload_half(const std::vector<float>& v) {
    std::vector<half> h(v.size());
    for (size_t i = 0; i < v.size(); ++i) h[i] = __float2half(v[i]);
    half* d; CK(cudaMalloc(&d, v.size() * sizeof(half))); CK(cudaMemcpy(d, h.data(), v.size() * sizeof(half), cudaMemcpyHostToDevice)); return d;
}
static float* upload_f32(const void* src, size_t n) {
    float* d; CK(cudaMalloc(&d, n * sizeof(float))); CK(cudaMemcpy(d, src, n * sizeof(float), cudaMemcpyHostToDevice)); return d;
}
static std::vector<float> to_f32(const gguf::Tensor& t) {
    size_t n = 1; for (auto d : t.ne) n *= d;
    std::vector<float> v(n);
    if (t.type == gguf::T_F32) memcpy(v.data(), t.data, n * 4);
    else if (t.type == gguf::T_F16) for (size_t i = 0; i < n; ++i) { uint16_t h; memcpy(&h, t.data + 2 * i, 2); uint32_t s = (h & 0x8000) << 16, e = (h >> 10) & 31, m = h & 1023, o; if (e == 0) { o = s; if (m) { float f = std::ldexp((float) m, -24); memcpy(&o, &f, 4); o |= s; } } else if (e == 31) o = s | 0x7f800000 | (m << 13); else o = s | ((e + 112) << 23) | (m << 13); memcpy(&v[i], &o, 4); }
    else throw std::runtime_error("unsupported float tensor type " + t.name);
    return v;
}

struct Engine {
    gguf::File f;
    int ctx = 4096, n_vocab = 248320, n_sm = 40, wide_cap = 1 << 30;
    std::vector<Layer> L;
    Mvm out;                                         // lm head
    Mtp mtp;
    float* out_norm = nullptr; float* signs[3] = {nullptr, nullptr, nullptr};   // 5120, 6144, 17408
    std::vector<float> sign_host[3];
    float* rx = nullptr; float* irms = nullptr; signed char* A = nullptr; float* d8 = nullptr; int* sa = nullptr;
    float* ybuf = nullptr; float* y_logits = nullptr; float* qkv = nullptr; float* gout = nullptr; float* gated = nullptr; float* Q = nullptr; float* gate = nullptr; float* attn_out = nullptr;
    float* hlast = nullptr;                          // [KMAX][5120] final-normed hidden of the last step (MTP input)
    ssq_t* am = nullptr; ssq_t* am_host = nullptr;    // argmax keys
    float* emb_host = nullptr; float* logits_host = nullptr;
    cudaStream_t st = nullptr;
    ssq_t* ssq = nullptr;                            // [2*64+2][KMAX] sums of squares, one slot per norm input
    ssq_t* slot(int s) { return ssq + (size_t) s * KMAX; }
    float rope_base = 1e7f;
    int cur = 0;                                     // active recurrent-state copy
    size_t dev_bytes = 0;

    float* dalloc(size_t n) { float* d; CK(cudaMalloc(&d, n * 4)); dev_bytes += n * 4; return d; }

    int auto_parts(int chunks, int G, int lo) const { int want = (PARTS_HALF * n_sm / 2 + chunks - 1) / chunks; if (want < lo) want = lo; if (want > G) want = G; return want; }

    // fused matrix: rows of several tensors stacked, all sharing the same column count
    void load_mvm(Mvm& m, const std::vector<const gguf::Tensor*>& ts, int parts) {
        int rows = 0, cols = (int) ts[0]->ne[0]; for (auto* t : ts) rows += (int) t->ne[1];
        m.rows = rows; m.cols = cols; m.chunks = (rows + CH - 1) / CH;
        m.parts = auto_parts(m.chunks, cols / 128, parts); if (m.chunks > 100) m.parts = 1;
        m.yrows = m.chunks * CH;
        const int G = cols / 128;
        const size_t words = (size_t) m.chunks * G * 7 * CH;
        std::vector<u32> host(words, 0u);
        std::vector<const uint8_t*> rowptr(rows);
        { int r = 0; for (auto* t : ts) for (int i = 0; i < (int) t->ne[1]; ++i) rowptr[r++] = t->data + (size_t) i * G * 28; }
        const int nth = std::max(1u, std::thread::hardware_concurrency());
        std::vector<std::thread> th;
        for (int tt = 0; tt < nth; ++tt) th.emplace_back([&, tt] {
            for (int c = tt; c < m.chunks; c += nth) {
                u32* dst = host.data() + (size_t) c * G * 7 * CH;
                for (int r = 0; r < CH; ++r) {
                    const int gr = c * CH + r; if (gr >= rows) break;
                    const uint8_t* s = rowptr[gr];
                    for (int g = 0; g < G; ++g) {
                        const u32* q = (const u32*) (s + (size_t) g * 28);
                        for (int wi = 0; wi < 7; ++wi) { u32 v; memcpy(&v, q + wi, 4); dst[((size_t) g * 7 + wi) * CH + r] = v; }
                    }
                }
            }
        });
        for (auto& t : th) t.join();
        m.w = (u32*) g_wpool.take(words * 4); m.bytes = words * 4; dev_bytes += words * 4;
        CK(cudaMemcpy(m.w, host.data(), words * 4, cudaMemcpyHostToDevice));
    }
    Q4 load_q4(const std::vector<const gguf::Tensor*>& ts) {
        Q4 q; q.cols = (int) ts[0]->ne[0]; q.nblk = q.cols / 32; q.rows = 0; for (auto* t : ts) q.rows += (int) t->ne[1];
        std::vector<const uint8_t*> rowptr; for (auto* t : ts) for (int i = 0; i < (int) t->ne[1]; ++i) rowptr.push_back(t->data + (size_t) i * q.nblk * 18);
        const int R = q.rows, N = q.nblk;
        std::vector<u32> nib((size_t) N * 4 * R); std::vector<half> sc((size_t) N * R);
        std::vector<std::thread> th; const int nth = std::max(1u, std::thread::hardware_concurrency());
        for (int tt = 0; tt < nth; ++tt) th.emplace_back([&, tt] {
            for (int r = tt; r < R; r += nth) {
                const uint8_t* s = rowptr[r];
                for (int b = 0; b < N; ++b) {
                    uint16_t h; memcpy(&h, s + b * 18, 2); __half_to_halfword(sc, (size_t) b * R + r, h);
                    for (int w = 0; w < 4; ++w) { u32 v; memcpy(&v, s + b * 18 + 2 + w * 4, 4); nib[((size_t) b * 4 + w) * R + r] = v; }
                }
            }
        });
        for (auto& x : th) x.join();
        q.nib = (u32*) g_wpool.take(nib.size() * 4); CK(cudaMemcpy(q.nib, nib.data(), nib.size() * 4, cudaMemcpyHostToDevice));
        q.sc = (half*) g_wpool.take(sc.size() * 2); CK(cudaMemcpy(q.sc, sc.data(), sc.size() * 2, cudaMemcpyHostToDevice));
        dev_bytes += nib.size() * 4 + sc.size() * 2;
        return q;
    }
    static void __half_to_halfword(std::vector<half>& v, size_t i, uint16_t bits) { memcpy(&v[i], &bits, 2); }

    void load(const char* path, int ctx_, bool with_mtp) {
        f.open(path);
        ctx = ctx_;
        { cudaDeviceProp pr; int dev; CK(cudaGetDevice(&dev)); CK(cudaGetDeviceProperties(&pr, dev)); n_sm = pr.multiProcessorCount; }
        rope_base = (float) f.num("qwen35.rope.freq_base", 1e7);
        const auto& vw = f.kv.at("prism.hadamard.sign_widths").nums; const auto& sv = f.kv.at("prism.hadamard.sign_values").nums;
        size_t o = 0;
        for (size_t i = 0; i < vw.size(); ++i) {
            const int w = (int) vw[i]; const int idx = w == 5120 ? 0 : (w == 6144 ? 1 : 2);
            std::vector<float> s(w); for (int j = 0; j < w; ++j) s[j] = (float) sv[o + j]; o += w;
            sign_host[idx] = s; signs[idx] = upload_f32(s.data(), w);
        }
        CK(cudaStreamCreate(&st));
        const int NL = 64;
        L.resize(NL);
        {
            size_t need = 0;
            auto mvm_b = [&](std::initializer_list<const gguf::Tensor*> ts) { int rows = 0; for (auto* t : ts) rows += (int) t->ne[1]; const size_t G = (*ts.begin())->ne[0] / 128; const size_t b = (size_t) ((rows + CH - 1) / CH) * G * 7 * CH * 4; need += (b + 255) & ~(size_t) 255; };
            auto q4_b = [&](std::initializer_list<const gguf::Tensor*> ts) { size_t R = 0; for (auto* t : ts) R += t->ne[1]; const size_t N = (*ts.begin())->ne[0] / 32; need += ((N * 4 * R * 4) + 255) & ~(size_t) 255; need += ((N * R * 2) + 255) & ~(size_t) 255; };
            for (int il = 0; il < NL; ++il) {
                const std::string p = "blk." + std::to_string(il) + "."; auto T = [&](const std::string& n) { return &f.get(p + n); };
                if (((il + 1) % 4) != 0) { mvm_b({ T("attn_qkv.weight"), T("attn_gate.weight") }); mvm_b({ T("ssm_out.weight") }); }
                else { mvm_b({ T("attn_q.weight"), T("attn_k.weight"), T("attn_v.weight") }); mvm_b({ T("attn_output.weight") }); }
                mvm_b({ T("ffn_gate.weight"), T("ffn_up.weight") }); mvm_b({ T("ffn_down.weight") });
            }
            mvm_b({ &f.get("output.weight") });
            if (with_mtp && f.find("blk.64.nextn.eh_proj.weight")) {
                const std::string p = "blk.64.";
                auto T = [&](const char* n) { return &f.get(p + n); };
                q4_b({ T("nextn.eh_proj.weight") }); q4_b({ T("attn_q.weight"), T("attn_k.weight"), T("attn_v.weight") }); q4_b({ T("attn_output.weight") });
                q4_b({ T("ffn_gate.weight"), T("ffn_up.weight") }); q4_b({ T("ffn_down.weight") });
            }
            g_wpool.arena(need);
        }
        auto up = [&](const gguf::Tensor* t) { auto v = to_f32(*t); dev_bytes += v.size() * 4; return upload_f32(v.data(), v.size()); };
        for (int il = 0; il < NL; ++il) {
            Layer& y = L[il]; const std::string p = "blk." + std::to_string(il) + ".";
            auto T = [&](const std::string& n) -> const gguf::Tensor* { return &f.get(p + n); };
            y.attn_norm = up(T("attn_norm.weight")); y.post_norm = up(T("post_attention_norm.weight"));
            const bool recr = ((il + 1) % 4) != 0; y.recr = recr;
            if (recr) {
                load_mvm(y.A, { T("attn_qkv.weight"), T("attn_gate.weight") }, 1);
                load_mvm(y.B, { T("ssm_out.weight") }, 1);
                y.ssm_norm = up(T("ssm_norm.weight")); y.dt_bias = up(T("ssm_dt.bias")); y.ssm_a = up(T("ssm_a")); y.conv_k = up(T("ssm_conv1d.weight"));
                y.wb = upload_half(to_f32(*T("ssm_beta.weight"))); y.wa = upload_half(to_f32(*T("ssm_alpha.weight"))); dev_bytes += 2 * (size_t) 48 * D_MODEL * 2;
                const size_t cb = (size_t) 3 * QKV * 4, sbytes = (size_t) 48 * 128 * 128 * 4;
                // plain decode updates the state in place; speculation keeps a checkpoint, so it needs a second copy
                for (int c = 0; c < 2; ++c) {
                    if (c == 1 && !with_mtp) { y.conv[1] = y.conv[0]; y.gdn[1] = y.gdn[0]; break; }
                    CK(cudaMalloc(&y.conv[c], cb)); CK(cudaMemset(y.conv[c], 0, cb)); dev_bytes += cb;
                    CK(cudaMalloc(&y.gdn[c], sbytes)); CK(cudaMemset(y.gdn[c], 0, sbytes)); dev_bytes += sbytes;
                }
                if (with_mtp) { CK(cudaMalloc(&y.rec_conv, (size_t) KREC * QKV * 4)); dev_bytes += (size_t) KREC * QKV * 4; }
                CK(cudaMalloc(&y.g_raw, (size_t) KMAX * GATE96 * 4)); dev_bytes += (size_t) KMAX * GATE96 * 4;
            } else {
                load_mvm(y.A, { T("attn_q.weight"), T("attn_k.weight"), T("attn_v.weight") }, 1);
                load_mvm(y.B, { T("attn_output.weight") }, 1);
                y.qnorm = up(T("attn_q_norm.weight")); y.knorm = up(T("attn_k_norm.weight"));
                const size_t kvb = (size_t) 4 * ctx * 256 * 2; CK(cudaMalloc(&y.kc, kvb)); CK(cudaMalloc(&y.vc, kvb)); dev_bytes += 2 * kvb;
            }
            load_mvm(y.C, { T("ffn_gate.weight"), T("ffn_up.weight") }, 1);
            load_mvm(y.D, { T("ffn_down.weight") }, 1);
        }
        { auto v = to_f32(f.get("output_norm.weight")); out_norm = upload_f32(v.data(), v.size()); dev_bytes += v.size() * 4; }
        load_mvm(out, { &f.get("output.weight") }, 1);
        n_vocab = out.rows;
        rx = dalloc((size_t) KMAX * D_MODEL); irms = dalloc(KMAX); CK(cudaMalloc(&ssq, (size_t) (2 * 64 + 2) * KMAX * sizeof(ssq_t)));
        CK(cudaMalloc(&A, (size_t) KMAX * (D_FF / 128) * 128)); CK(cudaMalloc(&d8, (size_t) KMAX * (D_FF / 128) * 16)); CK(cudaMalloc(&sa, (size_t) KMAX * (D_FF / 128) * 16));
        wide_cap = 0; for (auto& y : L) for (Mvm* m : { &y.A, &y.B, &y.C, &y.D }) wide_cap = std::max(wide_cap, m->yrows);
        { size_t mx = 0; for (auto& y : L) for (Mvm* m : { &y.A, &y.B, &y.C, &y.D }) for (int k : { 4, 8, KMAX }) mx = std::max(mx, (size_t) eff_parts(*m, k) * m->yrows * k); ybuf = dalloc(mx); }
        y_logits = dalloc((size_t) KLOG * out.yrows);
        qkv = dalloc((size_t) KMAX * QKV); gout = dalloc((size_t) KMAX * D_INNER); gated = dalloc((size_t) KMAX * D_INNER);
        // attention layers use the recurrent layers' scratch (24 x 256 = 6144 = D_INNER values per token each)
        Q = gout; gate = gated; attn_out = qkv;
        hlast = dalloc((size_t) KMAX * D_MODEL);
        CK(cudaMalloc(&am, (size_t) KMAX * sizeof(ssq_t))); CK(cudaHostAlloc(&am_host, (size_t) KMAX * sizeof(ssq_t), cudaHostAllocDefault));
                CK(cudaHostAlloc(&logits_host, (size_t) KLOG * n_vocab * 4, cudaHostAllocDefault));
        if (with_mtp && f.find("blk.64.nextn.eh_proj.weight")) load_mtp();
    }

    void load_mtp() {
        Mtp& m = mtp; const std::string p = "blk.64.";
        auto T = [&](const std::string& n) -> const gguf::Tensor& { return f.get(p + n); };
        auto up = [&](const gguf::Tensor& t) { auto v = to_f32(t); dev_bytes += v.size() * 4; return upload_f32(v.data(), v.size()); };
        m.eh = load_q4({ &T("nextn.eh_proj.weight") }); m.qkv = load_q4({ &T("attn_q.weight"), &T("attn_k.weight"), &T("attn_v.weight") });
        m.wo = load_q4({ &T("attn_output.weight") }); m.gu = load_q4({ &T("ffn_gate.weight"), &T("ffn_up.weight") }); m.down = load_q4({ &T("ffn_down.weight") });
        m.enorm = up(T("nextn.enorm.weight")); m.hnorm = up(T("nextn.hnorm.weight")); m.attn_norm = up(T("attn_norm.weight")); m.post_norm = up(T("post_attention_norm.weight"));
        m.head_norm = up(T("nextn.shared_head_norm.weight")); m.qnorm = up(T("attn_q_norm.weight")); m.knorm = up(T("attn_k_norm.weight"));
        const size_t kvb = (size_t) 4 * ctx * 256 * 2; CK(cudaMalloc(&m.kc, kvb)); CK(cudaMalloc(&m.vc, kvb)); dev_bytes += 2 * kvb;
        m.b_e = dalloc(D_MODEL); m.b_en = dalloc(D_MODEL); m.b_cat = dalloc(2 * D_MODEL); m.y0 = dalloc(D_MODEL); m.yn = dalloc(D_MODEL); m.yq = dalloc(14336);
        m.y1 = dalloc(D_MODEL); m.pn = dalloc(D_MODEL); m.fg = dalloc(2 * (size_t) D_FF); m.fu = m.fg + D_FF; m.fs = dalloc(D_FF); m.y2 = dalloc(D_MODEL); m.h_next = dalloc(D_MODEL);
        m.Q = dalloc(24 * 256); m.gateb = dalloc(D_INNER); m.attn_out = dalloc(D_INNER); m.wo_out = dalloc(D_MODEL); CK(cudaMalloc(&m.slots, 16 * sizeof(ssq_t)));
        m.y_logits = dalloc(out.yrows);
        m.ok = true;
    }

    // host-side embedding lookup with the inverse Hadamard applied
    void embed(int tok, float* out_) {
        const gguf::Tensor& t = f.get("token_embd.weight");
        const uint8_t* row = t.data + (size_t) tok * 40 * 28;
        float x[D_MODEL];
        for (int g = 0; g < 40; ++g) {
            const uint8_t* b = row + g * 28; uint16_t h; memcpy(&h, b + 26, 2);
            float d; { uint32_t s = (h & 0x8000) << 16, e = (h >> 10) & 31, m = h & 1023, o; if (e == 0) o = s; else o = s | ((e + 112) << 23) | (m << 13); memcpy(&d, &o, 4); }
            for (int i = 0; i < 128; ++i) {
                int byte, n;
                if (i < 80) { byte = i & 15; n = i >> 4; } else if (i < 120) { byte = 16 + ((i - 80) & 7); n = (i - 80) >> 3; } else { byte = 24 + ((i - 120) & 1); n = (i - 120) >> 1; }
                uint32_t q = b[byte]; for (int r = 0; r < n; ++r) q = (q * 3) & 0xFF;
                x[g * 128 + i] = (float) ((int) ((q * 3) >> 8) - 1) * d;
            }
        }
        const float* sg = sign_host[0].data();
        for (int b = 0; b < 5; ++b) {
            float* s = x + b * 1024;
            for (int h = 1; h < 1024; h *= 2)
                for (int i = 0; i < 1024; i += 2 * h) for (int j = i; j < i + h; ++j) { const float p = s[j], q = s[j + h]; s[j] = p + q; s[j + h] = p - q; }
            for (int i = 0; i < 1024; ++i) out_[b * 1024 + i] = s[i] * 0.03125f * sg[b * 1024 + i];
        }
    }

    unsigned char* tok_rows = nullptr; unsigned char* tok_ring = nullptr; unsigned ring_n = 0; int* d_pos = nullptr;
    // upload the PTQ1_0 rows of K tokens (and, if pos >= 0, the step position to d_pos); the pinned ring has 8 slots and is drained
    // by a sync every 4 uses
    void embed_upload(const int* toks, int K, int pos) {
        const size_t rb = 1120, rows = (size_t) KMAX * rb, slotb = rows + 16;
        if (!tok_rows) { CK(cudaMalloc(&tok_rows, rows + 16)); d_pos = (int*) (tok_rows + rows); CK(cudaHostAlloc(&tok_ring, 8 * slotb, cudaHostAllocDefault)); }
        if ((ring_n & 3) == 0) CK(cudaStreamSynchronize(st));
        unsigned char* h = tok_ring + (size_t) (ring_n++ & 7) * slotb;
        const gguf::Tensor& t = f.get("token_embd.weight");
        for (int k = 0; k < K; ++k) memcpy(h + (size_t) k * rb, t.data + (size_t) toks[k] * rb, rb);
        CK(cudaMemcpyAsync(tok_rows, h, (size_t) K * rb, cudaMemcpyHostToDevice, st));
        if (pos >= 0) { memcpy(h + rows, &pos, 4); CK(cudaMemcpyAsync(d_pos, h + rows, 4, cudaMemcpyHostToDevice, st)); }
    }
    void embed_gpu(const int* toks, int K, float* dst) {
        embed_upload(toks, K, -1);
        embed_kernel<<<dim3(5, K), 256, 0, st>>>(tok_rows, signs[0], dst);
    }
    // --- launches ---
    void prep(int mode, const float* src, int n, int ncols, int parts, int yrows, const float* sgn, const float* w, int upoff, const float* yp, int K, ssq_t* ss = nullptr) {
        PrepArgs a; a.src = (float*) src; a.ssq = ss; a.w = w; a.signs = sgn; a.yp = yp; a.parts = parts; a.yrows = yrows; a.upoff = upoff; a.n = n; a.G = ncols / 128; a.wide = K > 8; a.A = A; a.d8 = d8; a.sa = sa;
        prep_kernel<<<dim3(n / 1024, K), 256, 0, st>>>(a, mode);
    }
    // residual update + RMS-norm input of the next block: adds the partial results of matrix m (null = nothing pending)
    void prep_norm(const Mvm* m, const float* w, int K, ssq_t* ss) { prep(0, rx, D_MODEL, D_MODEL, m ? eff_parts(*m, K) : 0, m ? m->yrows : 0, signs[0], w, 0, ybuf, K, ss); }
    // column partitions used for K tokens (wider steps allow fewer groups per block because shared memory holds K activation records)
    // wide steps: column partitions that fill the SMs best (a block per SM at a time), without more partial rows than the largest matrix has
    int wide_parts(const Mvm& m, int K) const {
        const int R = K <= 32 ? tile_rows(32) : (K <= 64 ? tile_rows(64) : tile_rows(128)), nb = m.yrows / R, G = m.cols / 128;
        int best = 1; double be = 0;
        for (int p = 1; p * m.yrows <= wide_cap && G / p >= 8; ++p) {
            const double e = (double) nb * p / ((double) n_sm * ((nb * p + n_sm - 1) / n_sm));
            if (e > be + 0.01) { be = e; best = p; }
        }
        return best;
    }
    int eff_parts(const Mvm& m, int K) const { if (K > 8) return wide_parts(m, K); const int G = m.cols / 128; const int lim = K > 4 ? 20 : 40; return std::min(G, std::max(m.parts, (G + lim - 1) / lim)); }
    void launch_mvm(const u32* w, float* y, int K, int G, int parts, int yrows, const ssq_t* ss, GateArgs ga = GateArgs{}) {
#define MVM_GO(KK, NT, RPT, PF) { const int nb = yrows / ((NT) * (RPT)); ga.gy = ga.out ? (GATE96 + nb - 1) / nb : 0; \
        if (ga.gy) mvm_kernel<KK, NT, RPT, PF, true><<<dim3(nb, parts + ga.gy), NT, MVM_SMEM(KK, G, parts), st>>>(w, A, d8, sa, y, G, parts, yrows, ss, ga); \
        else mvm_kernel<KK, NT, RPT, PF, false><<<dim3(nb, parts), NT, MVM_SMEM(KK, G, parts), st>>>(w, A, d8, sa, y, G, parts, yrows, ss, ga); }
        if (K == 1) MVM_GO(1, 256, 1, true)
        else if (K == 2) MVM_GO(2, 256, 1, true)
        else if (K <= 4) MVM_GO(4, 128, 4, false)     // small blocks: many fit on an SM
        else if (K <= 8) MVM_GO(8, 256, 4, false)
        else if (K <= 32) mvm_tile_kernel<32><<<dim3(yrows / tile_rows(32), parts), TILE_NT, 0, st>>>(w, A, d8, sa, y, G, yrows, K, ss, parts);
        else if (K <= 64) mvm_tile_kernel<64><<<dim3(yrows / tile_rows(64), parts), TILE_NT, 0, st>>>(w, A, d8, sa, y, G, yrows, K, ss, parts);
        else mvm_tile_kernel<128><<<dim3(yrows / tile_rows(128), parts), TILE_NT, 0, st>>>(w, A, d8, sa, y, G, yrows, K, ss, parts);
    }
    void mvm(Mvm& m, int K, const ssq_t* ss = nullptr, GateArgs ga = GateArgs{}) { launch_mvm(m.w, ybuf, K, m.cols / 128, eff_parts(m, K), m.yrows, ss, ga); }
    void lm_head(int K, const ssq_t* ss, float* y) { launch_mvm(out.w, y, K, 40, eff_parts(out, K), out.yrows, ss); }
    void addres(Mvm& m, int K, ssq_t* ss) { addres_kernel<<<dim3((D_MODEL + 1023) / 1024, K), 256, 0, st>>>(rx, ybuf, D_MODEL, eff_parts(m, K), m.yrows, 0, ss); }
    void gdn_part(int il, int K, const float* raw_in) {          // conv + gated delta net, reading conv/gdn from copy `cur`, writing copy cur^1
        Layer& y = L[il];
        convqk_kernel<<<80, 128, 0, st>>>(ybuf, eff_parts(y.A, K), y.A.yrows, y.conv_k, y.conv[cur], y.conv[cur ^ 1], qkv, raw_in, (raw_in || K > KREC) ? nullptr : y.rec_conv, K);
        gdn_kernel<<<dim3(48, 16), 256, 0, st>>>(qkv, y.g_raw, y.dt_bias, y.ssm_a, y.gdn[cur], y.gdn[cur ^ 1], gout, K);
    }

#define P(name, ...) { PROF_BEGIN(st); for (int _i = 0; _i < (g_prof ? g_rep : 1); ++_i) { __VA_ARGS__; } PROF_END(st, name); }
    void attn(const float* q_, const float* g_, const half* kc, const half* vc, int pos, float* o_, int K, int pstart, const int* posd = nullptr) {
        if (K == 1) attn_kernel<1><<<dim3(24, 1), 256, 0, st>>>(q_, g_, kc, vc, ctx, pos, o_, K, pstart, posd);
        else if (K <= 4) attn_kernel<4><<<dim3(24, 1), 256, 0, st>>>(q_, g_, kc, vc, ctx, pos, o_, K, pstart, posd);
        else if (K <= 8 || pstart) attn_kernel<8><<<dim3(24, (K + 7) / 8), 256, 0, st>>>(q_, g_, kc, vc, ctx, pos, o_, K, pstart, posd);
        else attn_wide_kernel<<<dim3(24, (K + AW_T - 1) / AW_T), 128, 0, st>>>(q_, g_, kc, vc, ctx, pos, o_, K);
    }
    void layer(int il, int K, int pos) {
        Layer& y = L[il];
        ssq_t* sA = slot(2 * il); ssq_t* sP = slot(2 * il + 1);
        P("prep_norm", prep_norm(il ? &L[il - 1].D : nullptr, y.attn_norm, K, sA));
        GateArgs ga{};
        const bool gfuse = y.recr && K <= 2;                    // wider steps: extra blocks cost more than a launch of their own

        if (gfuse) { ga.x = rx; ga.nw = y.attn_norm; ga.wb = y.wb; ga.wa = y.wa; ga.out = y.g_raw; }
        P("mvmA", mvm(y.A, K, sA, ga));
        if (y.recr) {
            if (K > 8) {                                        // wide steps: kernels that work across tokens (never rolled back)
                P("gate", (gate_wide_kernel<<<dim3(12, (K + 7) / 8), 256, 0, st>>>(rx, sA, y.attn_norm, y.wb, y.wa, y.g_raw, K)));
                P("conv", (convqk_wide_kernel<<<dim3(80, K), 128, 0, st>>>(ybuf, eff_parts(y.A, K), y.A.yrows, y.conv_k, y.conv[cur], y.conv[cur ^ 1], qkv, K)));
                P("gdn", (gdn_wide_kernel<16><<<dim3(48, 4), 256, 0, st>>>(qkv, y.g_raw, y.dt_bias, y.ssm_a, y.gdn[cur], y.gdn[cur ^ 1], gout, K)));
            } else {
                if (!gfuse) P("gate", (gate_kernel<<<dim3(96, (K + 7) / 8), 256, 0, st>>>(rx, sA, y.attn_norm, y.wb, y.wa, y.g_raw, K)));
                P("conv", (convqk_kernel<<<80, 128, 0, st>>>(ybuf, eff_parts(y.A, K), y.A.yrows, y.conv_k, y.conv[cur], y.conv[cur ^ 1], qkv, nullptr, K > KREC ? nullptr : y.rec_conv, K)));
                P("gdn", (gdn_kernel<<<dim3(48, 16), 256, 0, st>>>(qkv, y.g_raw, y.dt_bias, y.ssm_a, y.gdn[cur], y.gdn[cur ^ 1], gout, K)));
            }
            P("prep_ssm", prep(2, gout, D_INNER, D_INNER, eff_parts(y.A, K), y.A.yrows, signs[1], y.ssm_norm, 0, ybuf, K));
        } else {
            P("attn_prep", (attn_prep_kernel<<<dim3(32, K), 256, 0, st>>>(ybuf, eff_parts(y.A, K), y.A.yrows, y.qnorm, y.knorm, Q, gate, y.kc, y.vc, ctx, pos, rope_base, 0, d_pos)));
            P("attn", attn(Q, gate, y.kc, y.vc, pos, attn_out, K, 0, d_pos));
            P("prep_wo", prep(3, attn_out, D_INNER, D_INNER, 0, 0, signs[1], nullptr, 0, nullptr, K));
        }
        P("mvmB", mvm(y.B, K));
        P("prep_norm", prep_norm(&y.B, y.post_norm, K, sP));
        P("mvmC", mvm(y.C, K, sP));
        P("prep_swiglu", prep(1, nullptr, D_FF, D_FF, eff_parts(y.C, K), y.C.yrows, signs[2], nullptr, D_FF, ybuf, K));
        P("mvmD", mvm(y.D, K));
    }

    // runs K tokens at positions pos..; fills hlast, and logits_host if want_logits.  State written into copy cur^1 (not committed)
    // Steps of up to 8 tokens run as CUDA graphs: one per (K, state copy cur, logits mode), captured on first use.  Every pointer in a
    // step is fixed and the position is read from d_pos, so a graph replays exactly the launches of step_body.  TR_NOGRAPH=1 disables.
    std::map<int, cudaGraphExec_t> graphs;
    const bool use_graphs = getenv("TR_NOGRAPH") == nullptr;
    void step(const int* toks, int K, int pos, int want_logits) {
        embed_upload(toks, K, pos);
        if (K > 8 || g_prof || !use_graphs) { step_body(K, pos, want_logits); return; }
        const int key = (K * 2 + cur) * 4 + want_logits;
        auto it = graphs.find(key);
        if (it == graphs.end()) {
            cudaGraph_t g; cudaGraphExec_t ex;
            CK(cudaStreamBeginCapture(st, cudaStreamCaptureModeThreadLocal));
            step_body(K, pos, want_logits);
            CK(cudaStreamEndCapture(st, &g));
            CK(cudaGraphInstantiate(&ex, g, 0)); CK(cudaGraphDestroy(g));
            it = graphs.emplace(key, ex).first;
        }
        CK(cudaGraphLaunch(it->second, st));
    }
    void step_body(int K, int pos, int want_logits) {
        embed_kernel<<<dim3(5, K), 256, 0, st>>>(tok_rows, signs[0], rx);
        CK(cudaMemsetAsync(ssq, 0, (size_t) (2 * 64 + 2) * KMAX * sizeof(ssq_t), st));
        for (int il = 0; il < 64; ++il) layer(il, K, pos);
        addres(L[63].D, K, slot(128));
        norm_store_kernel<<<dim3((D_MODEL + 255) / 256, K), 256, 0, st>>>(rx, slot(128), out_norm, hlast, D_MODEL, D_MODEL);
        if (want_logits) {
            const int L0 = K > 4 ? K - 1 : 0, Kl = K - L0;          // wide (prefill) steps only need the last token's logits
            prep(0, rx + (size_t) L0 * D_MODEL, D_MODEL, D_MODEL, 0, 0, signs[0], out_norm, 0, nullptr, Kl);
            lm_head(Kl, slot(128) + L0, y_logits);
            if (want_logits == 2) argmax_gpu(y_logits, Kl, out.yrows);
            else for (int k = 0; k < Kl; ++k) CK(cudaMemcpyAsync(logits_host + (size_t) k * n_vocab, y_logits + (size_t) k * out.yrows, (size_t) n_vocab * 4, cudaMemcpyDeviceToHost, st));
        }
    }
    void sync() { CK(cudaStreamSynchronize(st)); }
    // GPU argmax of K logit rows; result rows in am_host after sync()
    void argmax_gpu(const float* y, int K, int yrows) {
        CK(cudaMemsetAsync(am, 0, (size_t) KMAX * sizeof(ssq_t), st));
        argmax_kernel<<<dim3(128, K), 256, 0, st>>>(y, yrows, n_vocab, am);
        CK(cudaMemcpyAsync(am_host, am, (size_t) K * sizeof(ssq_t), cudaMemcpyDeviceToHost, st));
    }
    static int argmax_tok(ssq_t key) { return (int) (u32) ~(u32) (key & 0xFFFFFFFFu); }
    // after step(): keep only the first `kept` of K tokens.  Full acceptance = swap; otherwise replay the recorded inputs.
    void commit(int K, int kept) {
        if (kept < K)
            for (int il = 0; il < 64; ++il) if (L[il].recr) gdn_part(il, kept, L[il].rec_conv);
        cur ^= 1;
    }
    void reset_state() {
        cur = 0;
        for (auto& y : L) if (y.recr) { CK(cudaMemset(y.conv[0], 0, (size_t) 3 * QKV * 4)); CK(cudaMemset(y.gdn[0], 0, (size_t) 48 * 128 * 128 * 4)); }
    }
    // API used by tests: evaluate K tokens, full commit
    void eval(const float* embeds, int K, int pos, float* logits_out) {
        (void) embeds; (void) K; (void) pos; (void) logits_out;
    }

    // -------- MTP head: one row (input token embedding + trunk hidden h), position = the input token's position
    // draft=false: only fill K/V (catch-up rows); returns argmax token of the draft logits when draft=true and leaves h_next in mtp.h_next
    void norm(const float* a, const float* b, float* sum_out, const float* w, float* dst, int stride, int K, ssq_t* slot_ = nullptr) {
        norm_kernel<<<K, 512, 0, st>>>(a, b, sum_out, w, dst, slot_, D_MODEL, stride);
    }
    void q4mv(const Q4& m, const float* x, float* y, int ystride = 0, int K = 1, int r0 = 0, int nr = 0) {
        if (!nr) nr = m.rows; if (!ystride) ystride = nr;
        if (K == 1) q4mv_kernel<1><<<(nr + 63) / 64, 512, 0, st>>>(m.nib, m.sc, x, y, m.rows, r0, m.nblk, ystride, 1);
        else q4mv_kernel<4><<<dim3((nr + 63) / 64, (K + 3) / 4), 512, 0, st>>>(m.nib, m.sc, x, y, m.rows, r0, m.nblk, ystride, K);
    }
    // MTP K/V rows for K consecutive tokens: (embedding of toks[k], h[k]) at positions pos..pos+K-1.  Uses trunk scratch, no extra VRAM
    void mtp_catchup(const float* h, const int* toks, int K, int pos) {
        if (K <= 0) return;
        Mtp& m = mtp;
        float* be = rx; float* cat = qkv; float* y0 = gout; float* yn = gated; float* ykv = Q;
        embed_gpu(toks, K, be);
        norm(be, nullptr, nullptr, m.enorm, cat, 2 * D_MODEL, K);
        norm(h, nullptr, nullptr, m.hnorm, cat + D_MODEL, 2 * D_MODEL, K);
        q4mv(m.eh, cat, y0, D_MODEL, K);
        norm(y0, nullptr, nullptr, m.attn_norm, yn, D_MODEL, K);
        q4mv(m.qkv, yn, ykv, 2048, K, 12288, 2048);
        attn_prep_kernel<<<dim3(8, K), 256, 0, st>>>(ykv - 12288, 1, 2048, m.qnorm, m.knorm, m.Q, m.gateb, m.kc, m.vc, ctx, pos, rope_base, 24);
    }
    int mtp_row(const float* h, int tok, int pos, bool draft) {
        Mtp& m = mtp;
        embed_gpu(&tok, 1, m.b_e);
        P("m_norm", norm(m.b_e, nullptr, nullptr, m.enorm, m.b_cat, D_MODEL, 1));
        P("m_norm", norm(h, nullptr, nullptr, m.hnorm, m.b_cat + D_MODEL, D_MODEL, 1));
        P("m_q4_eh", q4mv(m.eh, m.b_cat, m.y0));
        P("m_norm", norm(m.y0, nullptr, nullptr, m.attn_norm, m.yn, D_MODEL, 1));
        P("m_q4_qkv", q4mv(m.qkv, m.yn, m.yq));                 // q [0, 12288) | k | v
        attn_prep_kernel<<<dim3(32, 1), 256, 0, st>>>(m.yq, 1, 14336, m.qnorm, m.knorm, m.Q, m.gateb, m.kc, m.vc, ctx, pos, rope_base, 0);
        if (!draft) return 0;
        P("m_attn", attn(m.Q, m.gateb, m.kc, m.vc, pos, m.attn_out, 1, 1));
        P("m_q4_wo", q4mv(m.wo, m.attn_out, m.wo_out));
        P("m_norm", norm(m.y0, m.wo_out, m.y1, m.post_norm, m.pn, D_MODEL, 1));
        P("m_q4_ffn", q4mv(m.gu, m.pn, m.fg));
        P("m_swiglu", (swiglu_kernel<<<(D_FF + 255) / 256, 256, 0, st>>>(m.fg, m.fu, m.fs, D_FF)));
        P("m_q4_down", q4mv(m.down, m.fs, m.y2));
        P("m_norm", norm(m.y1, m.y2, m.y2, m.head_norm, m.h_next, D_MODEL, 1, m.slots + 4));
        prep(0, m.y2, D_MODEL, D_MODEL, 0, 0, signs[0], m.head_norm, 0, nullptr, 1);
        P("m_head", lm_head(1, m.slots + 4, m.y_logits));
        P("m_argmax", argmax_gpu(m.y_logits, 1, out.yrows));
        sync();
        return argmax_tok(am_host[0]);
    }
    int argmax(const float* l) const { int b = 0; float bv = l[0]; for (int i = 1; i < n_vocab; ++i) if (l[i] > bv) { bv = l[i]; b = i; } return b; }
};

__global__ void nop_kernel(float* x) { if (blockIdx.x > 1000000) x[0] = 1; }
extern "C" {
// cost of launches alone: n empty kernels back to back (microseconds per launch: enqueue only, and including execution)
void tr_launch_cost(void* p, int n, int blocks, double* out) {
    Engine* e = (Engine*) p; CK(cudaStreamSynchronize(e->st));
    const auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < n; ++i) nop_kernel<<<blocks, 256, 0, e->st>>>(e->rx);
    const auto t1 = std::chrono::steady_clock::now();
    CK(cudaStreamSynchronize(e->st));
    const auto t2 = std::chrono::steady_clock::now();
    out[0] = std::chrono::duration<double>(t1 - t0).count() / n * 1e6; out[1] = std::chrono::duration<double>(t2 - t0).count() / n * 1e6;
}
// read-only bandwidth: the whole weight arena in one kernel, and split like the engine's matmuls (n kernels of 1/n each); GB/s
__global__ void read_kernel(const uint4* w, size_t n, u32* out) {
    const size_t st = (size_t) gridDim.x * 256; u32 a = 0;
    for (size_t i = (size_t) blockIdx.x * 256 + threadIdx.x; i < n; i += st) { const uint4 v = w[i]; a += v.x ^ v.y ^ v.z ^ v.w; }
    if (a == 0x12345678u) out[0] = a;
}
void tr_read_bw(void* p, double* out) {
    Engine* e = (Engine*) p; const size_t tot = g_wpool.cap / 16;
    const int ns[3] = { 1, 64, 256 };
    for (int q = 0; q < 3; ++q) {
        const size_t per = tot / ns[q]; double best = 1e9;
        for (int r = 0; r < 4; ++r) {
            CK(cudaStreamSynchronize(e->st)); const auto t0 = std::chrono::steady_clock::now();
            for (int i = 0; i < ns[q]; ++i) read_kernel<<<e->n_sm * 4, 256, 0, e->st>>>((const uint4*) g_wpool.base + per * i, per, (u32*) e->rx);
            CK(cudaStreamSynchronize(e->st));
            best = std::min(best, std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count());
        }
        out[q] = (double) per * ns[q] * 16 / best / 1e9;
    }
}
// one MTP draft row, milliseconds (average of n)
double tr_mtp_row_ms(void* p, int n) {
    Engine* e = (Engine*) p; if (!e->mtp.ok) return -1;
    e->sync(); const auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < n; ++i) e->mtp_row(e->hlast, 3000 + i, 200 + i, true);
    return std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count() / n * 1e3;
}
// host time spent enqueueing one K-token step, and the whole step (milliseconds)
void tr_step_split(void* p, const int* toks, int K, int pos, double* out) {
    Engine* e = (Engine*) p; e->sync();
    const auto t0 = std::chrono::steady_clock::now();
    e->step(toks, K, pos, 2);
    const auto t1 = std::chrono::steady_clock::now();
    e->sync();
    const auto t2 = std::chrono::steady_clock::now();
    e->commit(K, K); e->sync();
    out[0] = std::chrono::duration<double>(t1 - t0).count() * 1e3; out[1] = std::chrono::duration<double>(t2 - t0).count() * 1e3;
}
void tr_prof_dump(int steps) {
    double tot = 0; for (auto& kv : g_times) tot += kv.second.first;
    fprintf(stderr, "kernel profile over %d steps (synchronised, launch latency included):\n", steps);
    for (auto& kv : g_times) fprintf(stderr, "  %-12s %8.3f ms/step  %6.1f us/call  (%5.1f%%)\n", kv.first.c_str(), kv.second.first / steps * 1e3, kv.second.first / kv.second.second * 1e6, kv.second.first / tot * 100);
    g_times.clear();
}
void* tr_load(const char* path, int ctx, int with_mtp) { auto* e = new Engine(); e->load(path, ctx, with_mtp != 0); return e; }
int tr_vocab(void* p) { return ((Engine*) p)->n_vocab; }
long long tr_device_bytes(void* p) { return (long long) ((Engine*) p)->dev_bytes; }
void tr_embed(void* p, int tok, float* out) { ((Engine*) p)->embed(tok, out); }
// evaluate K (<=4) tokens at positions pos.., committing state; logits (K x vocab) copied to `logits` if non-null
void tr_eval(void* p, const int* toks, int K, int pos, float* logits) {
    Engine* e = (Engine*) p;
    e->step(toks, K, pos, logits ? 1 : 0);
    e->sync();
    if (logits) memcpy(logits, e->logits_host, (size_t) (K > KLOG ? 1 : K) * e->n_vocab * 4);   // wide steps: last token only
    e->commit(K, K);
    e->sync();
}
// compares the wide kernels with the decode kernels on identical random inputs; returns the largest relative error
double tr_test_tile(void* p, int il, int K, int which) {
    Engine* e = (Engine*) p; Layer& y = e->L[il]; Mvm& m = which == 0 ? y.A : (which == 1 ? y.B : (which == 2 ? y.C : y.D));
    const int n = m.cols;
    std::vector<float> src((size_t) K * n); srand(1234 + which + il);
    for (auto& v : src) { float u = (rand() % 20000 - 10000) / 10000.0f; v = u * u * u; }
    float* d; CK((cudaMalloc)((void**) &d, src.size() * 4)); CK(cudaMemcpy(d, src.data(), src.size() * 4, cudaMemcpyHostToDevice));
    const int sg = n == 5120 ? 0 : (n == 6144 ? 1 : 2);
    e->prep(3, d, n, n, 0, 0, e->signs[sg], nullptr, 0, nullptr, K);
    e->mvm(m, K); e->sync();
    const int wp = e->eff_parts(m, K);
    std::vector<float> wpart((size_t) K * wp * m.yrows), wide((size_t) K * m.yrows, 0.0f); CK(cudaMemcpy(wpart.data(), e->ybuf, wpart.size() * 4, cudaMemcpyDeviceToHost));
    for (int k = 0; k < K; ++k) for (int q = 0; q < wp; ++q) for (int r = 0; r < m.yrows; ++r) wide[(size_t) k * m.yrows + r] += wpart[((size_t) k * wp + q) * m.yrows + r];
    double worst = 0, scale = 0; for (float v : wide) scale = std::max(scale, (double) fabsf(v));
    for (int k = 0; k < K; ++k) {
        e->prep(3, d + (size_t) k * n, n, n, 0, 0, e->signs[sg], nullptr, 0, nullptr, 1);       // narrow records of token k (the wide step used another format)
        { GateArgs ga{}; mvm_kernel<1, 512, 2, true, false><<<dim3(m.chunks, m.parts), 512, MVM_SMEM(1, n / 128, m.parts), e->st>>>(m.w, e->A, e->d8, e->sa, e->ybuf, n / 128, m.parts, m.yrows, nullptr, ga); }
        e->sync();
        std::vector<float> one((size_t) m.parts * m.yrows); CK(cudaMemcpy(one.data(), e->ybuf, one.size() * 4, cudaMemcpyDeviceToHost));
        for (int r = 0; r < m.rows; ++r) { float t = 0; for (int q = 0; q < m.parts; ++q) t += one[(size_t) q * m.yrows + r]; worst = std::max(worst, (double) fabsf(t - wide[(size_t) k * m.yrows + r]) / scale); }
    }
    CK(cudaFree(d));
    return worst;
}
// attention on random queries, gates and KV against a double-precision CPU reference; returns the largest relative error
double tr_test_attn(void* p, int il, int L, int K) {
    Engine* e = (Engine*) p; Layer& y = e->L[il]; if (y.recr) return -1;
    const size_t nq = (size_t) K * 24 * 256, nk = (size_t) 4 * e->ctx * 256;
    std::vector<float> q(nq), g(nq), a(nq); std::vector<float> kf(nk), vf(nk); std::vector<half> kh(nk), vh(nk); srand(77);
    for (auto& v : q) v = (rand() % 2000 - 1000) / 1000.0f; for (auto& v : g) v = (rand() % 1000) / 1000.0f;
    for (size_t i = 0; i < nk; ++i) { kh[i] = __float2half((rand() % 2000 - 1000) / 1000.0f * 1.5f); vh[i] = __float2half((rand() % 2000 - 1000) / 1000.0f); kf[i] = __half2float(kh[i]); vf[i] = __half2float(vh[i]); }
    CK(cudaMemcpy(e->Q, q.data(), nq * 4, cudaMemcpyHostToDevice)); CK(cudaMemcpy(e->gate, g.data(), nq * 4, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(y.kc, kh.data(), nk * 2, cudaMemcpyHostToDevice)); CK(cudaMemcpy(y.vc, vh.data(), nk * 2, cudaMemcpyHostToDevice));
    const int ps = L > 0 && K <= 8 ? 1 : 0;               // also exercises the first-visible-position argument (wide steps never use it)
    e->attn(e->Q, e->gate, y.kc, y.vc, L, e->attn_out, K, ps); e->sync(); CK(cudaMemcpy(a.data(), e->attn_out, nq * 4, cudaMemcpyDeviceToHost));
    double worst = 0, scale = 0;
    std::vector<double> ref(nq, 0.0);
    for (int h = 0; h < 24; ++h) for (int k = 0; k < K; ++k) {
        const float* qv = &q[((size_t) k * 24 + h) * 256];
        const float *kb = &kf[(size_t) (h / 6) * e->ctx * 256], *vb = &vf[(size_t) (h / 6) * e->ctx * 256];
        const int n = L + k + 1; std::vector<double> sc(n); double m = -1e300, su = 0;
        for (int t = ps; t < n; ++t) { double d = 0; for (int i = 0; i < 256; ++i) d += (double) qv[i] * kb[(size_t) t * 256 + i]; sc[t] = d / 16.0; m = std::max(m, sc[t]); }
        for (int t = ps; t < n; ++t) { sc[t] = std::exp(sc[t] - m); su += sc[t]; }
        double* r = &ref[(size_t) k * D_INNER + h * 256];
        for (int t = ps; t < n; ++t) for (int i = 0; i < 256; ++i) r[i] += sc[t] / su * vb[(size_t) t * 256 + i];
        for (int i = 0; i < 256; ++i) r[i] *= g[(size_t) k * D_INNER + h * 256 + i];
    }
    for (double v : ref) scale = std::max(scale, std::fabs(v));
    for (size_t i = 0; i < nq; ++i) worst = std::max(worst, std::fabs(a[i] - ref[i]) / scale);
    return worst;
}
void tr_reset(void* p) { ((Engine*) p)->reset_state(); }

// greedy generation.  prompt: n tokens; D drafts per round (0 = plain).  Returns number of generated tokens; stats: [rounds, accepted drafts, mtp calls]
int tr_generate(void* p, const int* prompt, int n, int max_new, const int* eos, int n_eos, int D, int* out, double* stats) {
    Engine* e = (Engine*) p;
    auto is_eos = [&](int t) { for (int i = 0; i < n_eos; ++i) if (eos[i] == t) return true; return false; };
    e->reset_state();
    if (D > 3) D = 3;
    if (D > 0 && !e->mtp.ok) D = 0;
    double rounds = 0, accepted = 0, drafts = 0, kept_total = 0;
    // ---- prefill
    int pos = 0; int last_logit_k = 0; int last_K = 1;
    while (pos < n) {
        int K = std::min(KMAX, n - pos);
        int toks[KMAX]; for (int k = 0; k < K; ++k) toks[k] = prompt[pos + k];
        bool last = pos + K == n; last_K = K;
        e->step(toks, K, pos, last ? 2 : 0);
        e->sync();
        e->commit(K, K);
        // MTP catch-up rows for positions with a known next token: row (t_{j+1}, h_j) at kv position j+1
        if (D > 0) e->mtp_catchup(e->hlast, prompt + pos + 1, std::min(K, n - 1 - pos), pos + 1);
        last_logit_k = K > 4 ? 0 : K - 1;
        pos += K;
    }
    int cur_tok = Engine::argmax_tok(e->am_host[last_logit_k]);
    const auto t_dec0 = std::chrono::steady_clock::now();
    int out_n = 0;
    out[out_n++] = cur_tok;
    bool done = is_eos(cur_tok);
    // hidden of the last processed trunk token is hlast[last_logit_k] -- copy to a stable buffer
    float* h_last; CK((cudaMalloc)((void**) &h_last, D_MODEL * 4));
    CK(cudaMemcpyAsync(h_last, e->hlast + (size_t) (last_K - 1) * D_MODEL, D_MODEL * 4, cudaMemcpyDeviceToDevice, e->st));
    while (!done && out_n < max_new) {
        if (D == 0) {
            e->step(&cur_tok, 1, pos, 2); e->sync(); e->commit(1, 1);
            cur_tok = Engine::argmax_tok(e->am_host[0]);
            pos++;
            out[out_n++] = cur_tok; done = is_eos(cur_tok);
            if (is_eos(cur_tok)) break;
            continue;
        }
        // drafts: rows at kv positions pos (embed cur_tok, h_last), pos+1, ...
        int drafted[3]; int dn = 0; const float* hp = h_last;
        for (int d = 0; d < D; ++d) {
            const int t = d == 0 ? cur_tok : drafted[d - 1];
            drafted[d] = e->mtp_row(hp, t, pos + d, true); drafts++;
            hp = e->mtp.h_next; dn++;
            // note: the h_next pointer is overwritten by the next row; it is consumed by the next row before being overwritten (hnorm reads it first)
        }
        int batch[4]; batch[0] = cur_tok; for (int d = 0; d < D; ++d) batch[1 + d] = drafted[d];
        const int K = 1 + D;
        e->step(batch, K, pos, 2); e->sync();
        // verify
        int a = 0; int next = 0;
        for (int k = 0; k < K; ++k) {
            next = Engine::argmax_tok(e->am_host[k]);
            if (k < D) { if (next == drafted[k]) a++; else break; }
        }
        // emitted tokens: drafted[0..a-1] then `next` (the argmax after the last kept token)
        const int kept = 1 + a;
        e->commit(K, kept);
        rounds++; accepted += a; kept_total += kept;
        // MTP catch-up for the accepted drafts using the trunk's hiddens (rows at kv pos pos+1..pos+a)
        // hidden of batch index k is hlast[k] (trunk hidden for pos+k); row for drafted[i] (position pos+1+i) uses hlast[i]
        e->mtp_catchup(e->hlast, drafted, a, pos + 1);
        CK(cudaMemcpyAsync(h_last, e->hlast + (size_t) a * D_MODEL, D_MODEL * 4, cudaMemcpyDeviceToDevice, e->st));
        pos += kept;
        // output: accepted drafts then `next`
        for (int i = 0; i < a && out_n < max_new; ++i) { out[out_n++] = drafted[i]; if (is_eos(drafted[i])) { done = true; break; } }
        if (!done) { cur_tok = next; if (out_n < max_new) out[out_n++] = cur_tok; done = is_eos(cur_tok); }
        }
    e->sync();
    CK(cudaFree(h_last));
    if (stats) { stats[0] = rounds; stats[1] = accepted; stats[2] = drafts; stats[3] = kept_total; stats[4] = std::chrono::duration<double>(std::chrono::steady_clock::now() - t_dec0).count(); }
    return out_n;
}
}
