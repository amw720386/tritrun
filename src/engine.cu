// tritrun: a CUDA decode engine for PTQ1_0 (ternary, group-128) Bonsai 2.5 27B.
//
// The trits are evaluated with per-byte lookup tables in shared memory (the T-MAC idea on a GPU):
// for each 128-column group a block builds 26 tables of partial sums of the quantised activation,
// and then every 5-trit byte costs one table read.  Activations are quantised to 12 bits per group,
// the in-group dot product is exact in int32, and the float scales are applied per group.
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

#ifndef MVM_NT
#define MVM_NT 512
#endif
static const int NT = MVM_NT;         // threads per mvm block
static const int RPT = 1024 / MVM_NT;              // rows per thread
static const int CH = NT * RPT;       // rows per chunk (1024)
static const int KMAX = 4;            // tokens per step (verify width)
static const int QMAX = 2047;         // activation quantisation level (12 bits)
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
    return 1.0f / sqrtf((float) ((double) v / 65536.0 / n) + 1e-6f);
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
// prep: elementwise source -> signs -> Hadamard(1024) -> per-group 12-bit quantisation
// mode 0: norm        v = x[k][i] * irms[k] * w[i]
// mode 1: swiglu      v = silu(gate) * up, read from the mvm partials
// mode 2: gdn output  K x 6144 vector, tiled heads permuted to grouped order
// mode 3: plain       K x n vector
// ---------------------------------------------------------------------------------------------
struct PrepArgs {
    const float* src; const ssq_t* irms; const float* w; const float* signs;
    const float* yp; int parts; int yrows; int upoff;
    int n; int G;
    signed char* A; float* d8; int* sa;     // A [k][G][128] int8 planes, d8 [k][G][4] sub-block scales, sa [k][G][4] sub-block sums
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

__global__ void prep_kernel(const PrepArgs a, const int mode) {
    __shared__ float s[1024];
    __shared__ float mx[32];
    __shared__ int   sm[32];
    const int k = blockIdx.y, seg = blockIdx.x, tid = threadIdx.x;
    const int base = seg * 1024;
    const float ir = mode == 0 ? ssq_rms_inv(a.irms[k], a.n) : 1.0f;
    for (int e = tid; e < 1024; e += 256) {
        const int i = base + e;
        float v;
        if (mode == 0) {
            v = a.src[(size_t) k * a.n + i] * ir * a.w[i];
        } else if (mode == 1) {
            float g = 0, u = 0;
            for (int p = 0; p < a.parts; ++p) {
                const float* yk = a.yp + (size_t) (k * a.parts + p) * a.yrows;
                g += yk[i]; u += yk[a.upoff + i];
            }
            v = silu_f(g) * u;
        } else if (mode == 2) {
            const int hd = i & 127, q = i >> 7, rep = q % 3, nk = q / 3;
            v = a.src[(size_t) k * a.n + hd + 128 * (nk + 16 * rep)];
        } else {
            v = a.src[(size_t) k * a.n + i];
        }
        if (a.signs) v *= a.signs[i];
        s[e] = v;
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
    if (tid < 32) { mx[tid] = 0.0f; sm[tid] = 0; }
    __syncthreads();
    for (int e = tid; e < 1024; e += 256)
        atomicMax((int*) &mx[e >> 5], __float_as_int(fabsf(s[e]) * 0.03125f));
    __syncthreads();
    for (int e = tid; e < 1024; e += 256) {
        const int g = e >> 7, el = e & 127;
        const float m = mx[e >> 5];
        const int q = m > 0 ? (int) lrintf((s[e] * 0.03125f) * (127.0f / m)) : 0;
        a.A[((size_t) (k * a.G + base / 128 + g)) * 128 + plane_slot(el)] = (signed char) q;
        atomicAdd(&sm[e >> 5], q);
    }
    __syncthreads();
    if (tid < 32) {
        const int gg = base / 128 + (tid >> 2);
        a.d8[(size_t) (k * a.G + gg) * 4 + (tid & 3)] = mx[tid] / 127.0f;
        a.sa[(size_t) (k * a.G + gg) * 4 + (tid & 3)] = sm[tid];
    }
}

// inverse RMS of the residual vector, one value per token
__global__ void sumsq_kernel(const float* x, int n, ssq_t* irms) {
    __shared__ float part[8];
    const int k = blockIdx.x;
    float acc = 0;
    for (int e = threadIdx.x; e < n; e += 256) { const float v = x[(size_t) k * n + e]; acc += v * v; }
    acc = warp_sum(acc);
    if ((threadIdx.x & 31) == 0) part[threadIdx.x >> 5] = acc;
    __syncthreads();
    if (threadIdx.x == 0) {
        float t = 0; for (int i = 0; i < 8; ++i) t += part[i];
        irms[k] = (ssq_t) llrint((double) t * 65536.0);
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

template <int K>
__global__ __launch_bounds__(NT, 1) void mvm_kernel(const u32* W, const signed char* A, const float* d8, const int* sa, float* Y,
                                                    int G, int cols, int parts, int yrows) {
    __shared__ u32 sA[K * 40 * 32];
    const int chunk = blockIdx.x, part = blockIdx.y, tid = threadIdx.x;
    const int g0 = (int) ((long long) part * G / parts), g1 = (int) ((long long) (part + 1) * G / parts);
    const int Gp = g1 - g0;
    const u32* Au = (const u32*) A;
    for (int i = tid; i < K * Gp * 32; i += NT) {
        const int k = i / (Gp * 32), r = i - k * Gp * 32;
        sA[i] = Au[(size_t) (k * G + g0) * 32 + r];
    }
    __syncthreads();

    float acc[RPT][K];
#pragma unroll
    for (int j = 0; j < RPT; ++j)
#pragma unroll
        for (int k = 0; k < K; ++k) acc[j][k] = 0.0f;
    const u32* Wc = W + (size_t) chunk * G * 7 * CH;

    for (int g = g0; g < g1; ++g) {
        const u32* Wg = Wc + (size_t) g * 7 * CH;
        u32 w[RPT][7];
#pragma unroll
        for (int j = 0; j < RPT; ++j)
#pragma unroll
            for (int q = 0; q < 7; ++q) w[j][q] = Wg[q * CH + tid + j * NT];
        const int gi = g - g0;
        float ds[K][4]; int ss[K][4];
#pragma unroll
        for (int k = 0; k < K; ++k)
#pragma unroll
            for (int q = 0; q < 4; ++q) { ds[k][q] = d8[(size_t) (k * G + g) * 4 + q]; ss[k][q] = sa[(size_t) (k * G + g) * 4 + q]; }

#pragma unroll
        for (int j = 0; j < RPT; ++j) {
            int isum[K][4];
#pragma unroll
            for (int k = 0; k < K; ++k)
#pragma unroll
                for (int q = 0; q < 4; ++q) isum[k][q] = 0;
            u32 lo[4], hi[4];
#pragma unroll
            for (int q = 0; q < 4; ++q) { lo[q] = w[j][q] & 0x00FF00FFu; hi[q] = (w[j][q] >> 8) & 0x00FF00FFu; }
#pragma unroll
            for (int n = 0; n < 5; ++n) {
#pragma unroll
                for (int q = 0; q < 4; ++q) {
                    u32 D; DIGIT_STEP(lo[q], hi[q], D)
#pragma unroll
                    for (int k = 0; k < K; ++k) isum[k][n >> 1] = __dp4a((int) D, (int) sA[k * Gp * 32 + gi * 32 + n * 4 + q], isum[k][n >> 1]);
                }
            }
            u32 l2[2], h2[2];
#pragma unroll
            for (int q = 0; q < 2; ++q) { l2[q] = w[j][4 + q] & 0x00FF00FFu; h2[q] = (w[j][4 + q] >> 8) & 0x00FF00FFu; }
#pragma unroll
            for (int n = 0; n < 5; ++n) {
                const int sb = n < 2 ? 2 : 3;
#pragma unroll
                for (int q = 0; q < 2; ++q) {
                    u32 D; DIGIT_STEP(l2[q], h2[q], D)
#pragma unroll
                    for (int k = 0; k < K; ++k) isum[k][sb] = __dp4a((int) D, (int) sA[k * Gp * 32 + gi * 32 + 20 + n * 2 + q], isum[k][sb]);
                }
            }
            {
                u32 v = (w[j][6] & 0xFFu) | ((w[j][6] & 0xFF00u) << 8);
#pragma unroll
                for (int n = 0; n < 4; ++n) {
                    const u32 t = v * 3u;
                    const int d24 = (t >> 8) & 3, d25 = (t >> 24) & 3;
                    v = t & 0x00FF00FFu;
                    const int sh = 8 * ((2 * n) & 3);
#pragma unroll
                    for (int k = 0; k < K; ++k) {
                        const u32 aw = sA[k * Gp * 32 + gi * 32 + 30 + (n >> 1)];
                        const int a0 = (int) (signed char) (aw >> sh), a1 = (int) (signed char) (aw >> (sh + 8));
                        isum[k][3] += d24 * a0 + d25 * a1;
                    }
                }
            }
            const float dh = __half2float(__ushort_as_half((unsigned short) (w[j][6] >> 16)));
#pragma unroll
            for (int k = 0; k < K; ++k) {
                float f = 0;
#pragma unroll
                for (int q = 0; q < 4; ++q) f += ds[k][q] * (float) (isum[k][q] - ss[k][q]);
                acc[j][k] += dh * f;
            }
        }
    }
#pragma unroll
    for (int k = 0; k < K; ++k)
#pragma unroll
        for (int j = 0; j < RPT; ++j)
            Y[(size_t) (k * parts + part) * yrows + chunk * CH + tid + j * NT] = acc[j][k];
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
__global__ void gate_kernel(const float* x, const ssq_t* ssq, const float* nw, const half* wb, const half* wa, float* out) {
    __shared__ float part[8];
    const int k = blockIdx.y, row = blockIdx.x;
    const float ir = ssq_rms_inv(ssq[k], D_MODEL);
    const half* W = row < 48 ? wb + (size_t) row * D_MODEL : wa + (size_t) (row - 48) * D_MODEL;
    float acc = 0;
    for (int e = threadIdx.x; e < D_MODEL; e += 256) acc += __half2float(W[e]) * (x[(size_t) k * D_MODEL + e] * ir * nw[e]);
    acc = warp_sum(acc);
    if ((threadIdx.x & 31) == 0) part[threadIdx.x >> 5] = acc;
    __syncthreads();
    if (threadIdx.x == 0) { float t = 0; for (int i = 0; i < 8; ++i) t += part[i]; out[k * GATE96 + row] = t; }
}

// depthwise conv (4 taps) + silu + l2 norm of q/k heads; 80 blocks of 128 channels
// raw_in (optional): recorded qkv channel inputs [k][10240] from a previous run (rollback replay); rec_out: record them
__global__ void convqk_kernel(const float* yA, int parts, int yrows, const float* ck, const float* st_in, float* st_out, float* qkv,
                              const float* raw_in, float* rec_out, int K) {
    __shared__ float ps[4];
    const int b = blockIdx.x, tid = threadIdx.x, ch = b * 128 + tid;
    float win[3 + KMAX];
    win[0] = st_in[3 * ch]; win[1] = st_in[3 * ch + 1]; win[2] = st_in[3 * ch + 2];
    for (int k = 0; k < K; ++k) {
        float v = 0;
        if (raw_in) v = raw_in[(size_t) k * QKV + ch];
        else for (int p = 0; p < parts; ++p) v += yA[(size_t) (k * parts + p) * yrows + ch];
        if (rec_out) rec_out[(size_t) k * QKV + ch] = v;
        win[3 + k] = v;
    }
    st_out[3 * ch] = win[K]; st_out[3 * ch + 1] = win[K + 1]; st_out[3 * ch + 2] = win[K + 2];
    const float* c = ck + 4 * ch;
    for (int k = 0; k < K; ++k) {
        float o = silu_f(win[k] * c[0] + win[k + 1] * c[1] + win[k + 2] * c[2] + win[k + 3] * c[3]);
        if (b < 32) {
            const float ss = warp_sum(o * o);
            if ((tid & 31) == 0) ps[tid >> 5] = ss;
            __syncthreads();
            o *= 1.0f / fmaxf(sqrtf(ps[0] + ps[1] + ps[2] + ps[3]), 1e-6f);
            __syncthreads();
        }
        qkv[(size_t) k * QKV + ch] = o;
    }
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

// per head RMS norm (128 dims) * weight, then silu(z) gating
__global__ void gdnpost_kernel(const float* gout, const float* nw, const float* yA, int parts, int yrows, float* dst) {
    __shared__ float ps[4];
    const int h = blockIdx.x, k = blockIdx.y, tid = threadIdx.x;
    const float v = gout[(size_t) k * D_INNER + h * 128 + tid];
    const float ss = warp_sum(v * v);
    if ((tid & 31) == 0) ps[tid >> 5] = ss;
    __syncthreads();
    const float irms = 1.0f / sqrtf((ps[0] + ps[1] + ps[2] + ps[3]) / 128.0f + 1e-6f);
    float z = 0;
    for (int p = 0; p < parts; ++p) z += yA[(size_t) (k * parts + p) * yrows + QKV + h * 128 + tid];
    dst[(size_t) k * D_INNER + h * 128 + tid] = silu_f(z) * (v * irms * nw[tid]);
}

// ---------------------------------------------------------------------------------------------
// full attention: prep (norms + rope, KV store) and the per-head softmax
// yA rows: q 12288 (24 heads x [256 q | 256 gate]) | k 1024 | v 1024
// KV layout per layer: [kvhead 4][ctx][256] fp16
// ---------------------------------------------------------------------------------------------
__global__ void attn_prep_kernel(const float* yA, int parts, int yrows, const float* qn, const float* kn,
                                 float* Q, float* gate, half* kc, half* vc, int ctx, int pos, float theta_base) {
    __shared__ float sh[256];
    __shared__ float ps[8];
    const int hb = blockIdx.x, k = blockIdx.y, tid = threadIdx.x;   // 256 threads = 256 dims
    const int src = hb < 24 ? hb * 512 + tid : (hb < 28 ? 12288 + (hb - 24) * 256 + tid : 13312 + (hb - 28) * 256 + tid);
    float v = 0;
    for (int p = 0; p < parts; ++p) v += yA[(size_t) (k * parts + p) * yrows + src];
    const bool has_kv = hb >= 24;
    if (hb < 24) {
        float gt = 0;
        for (int p = 0; p < parts; ++p) gt += yA[(size_t) (k * parts + p) * yrows + hb * 512 + 256 + tid];
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

__global__ void attn_kernel(const float* Q, const float* gate, const half* kc, const half* vc, int ctx, int pos, float* out, int pstart) {
    __shared__ float sc[8192];
    __shared__ float mx[256];
    const int h = blockIdx.x, k = blockIdx.y, tid = threadIdx.x;
    const int L = pos + k + 1;
    const int kvh = h / 6;
    const half* Kp = kc + (size_t) kvh * ctx * 256;
    const half* Vp = vc + (size_t) kvh * ctx * 256;
    const float* q = Q + ((size_t) k * 24 + h) * 256;
    {
        const int warp = tid >> 5, lane = tid & 31;
        float qr[8];
        for (int m = 0; m < 8; ++m) qr[m] = q[m * 32 + lane];
        for (int p = pstart + ((warp - (pstart & 7) + 8) & 7); p < L; p += 8) {
            float s = 0;
            for (int m = 0; m < 8; ++m) s += qr[m] * __half2float(Kp[(size_t) p * 256 + m * 32 + lane]);
            s = warp_sum(s) * 0.0625f;          // 1 / sqrt(256)
            if (lane == 0) sc[p] = s;
        }
    }
    __syncthreads();
    float m = -1e30f;
    for (int p = tid; p < L; p += 256) if (p >= pstart) m = fmaxf(m, sc[p]);
    mx[tid] = m;
    __syncthreads();
    if (tid == 0) { float g = mx[0]; for (int i = 1; i < 256; ++i) g = fmaxf(g, mx[i]); mx[0] = g; }
    __syncthreads();
    const float gmax = mx[0];
    float sum = 0;
    for (int p = tid; p < L; p += 256) if (p >= pstart) { const float e = expf(sc[p] - gmax); sc[p] = e; sum += e; }
    __syncthreads();
    mx[tid] = sum;
    __syncthreads();
    float t = 0;
    if (tid == 0) { for (int i = 0; i < 256; ++i) t += mx[i]; mx[0] = t; }
    __syncthreads();
    const float inv = 1.0f / mx[0];
    float acc = 0;
    for (int p = pstart; p < L; ++p) acc += sc[p] * __half2float(Vp[(size_t) p * 256 + tid]);
    out[(size_t) k * D_INNER + h * 256 + tid] = acc * inv * gate[(size_t) k * D_INNER + h * 256 + tid];
}

// ---------------------------------------------------------------------------------------------
// plain float helpers used by the MTP head (single token)
// ---------------------------------------------------------------------------------------------
__global__ void norm_store_kernel(const float* x, const ssq_t* ssq, const float* w, float* dst, int n) {
    const int k = blockIdx.y;
    const float ir = ssq_rms_inv(ssq[k], n);
    const int i = blockIdx.x * 256 + threadIdx.x;
    if (i < n) dst[(size_t) k * n + i] = x[(size_t) k * n + i] * ir * w[i];
}
__global__ void vec_add_kernel(const float* a, const float* b, float* out, int n) {
    const int i = blockIdx.x * 256 + threadIdx.x;
    if (i < n) out[i] = a[i] + b[i];
}
__global__ void swiglu_kernel(const float* g, const float* u, float* out, int n) {
    const int i = blockIdx.x * 256 + threadIdx.x;
    if (i < n) out[i] = silu_f(g[i]) * u[i];
}
// Q4_0 matrix-vector: nib [nblk][4][R] words (each block's 16 bytes), scale [nblk][R] fp16; x float [C]
__global__ void q4mv_kernel(const u32* nib, const half* sc, const float* x, float* y, int R, int nblk) {
    const int r = blockIdx.x * 256 + threadIdx.x;
    if (r >= R) return;
    float acc = 0;
    for (int b = 0; b < nblk; ++b) {
        float s = 0;
        const float* xb = x + (size_t) b * 32;
#pragma unroll
        for (int w = 0; w < 4; ++w) {
            const u32 v = nib[((size_t) b * 4 + w) * R + r];
#pragma unroll
            for (int by = 0; by < 4; ++by) {
                const int q = (v >> (8 * by)) & 0xFF;
                const int j = w * 4 + by;
                s += (float) ((q & 15) - 8) * xb[j] + (float) ((q >> 4) - 8) * xb[j + 16];
            }
        }
        acc += s * __half2float(sc[(size_t) b * R + r]);
    }
    y[r] = acc;
}

// ---------------------------------------------------------------------------------------------
// host engine
// ---------------------------------------------------------------------------------------------
struct Mvm { u32* w = nullptr; int rows = 0, cols = 0, chunks = 0, parts = 1, yrows = 0; size_t bytes = 0; };

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
    Q4 eh, q, k, v, wo, gate, up, down;
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
static std::map<std::string, std::pair<double, int>> g_times;
#define PROF_BEGIN(st) std::chrono::steady_clock::time_point _t0; if (g_prof) { CK(cudaStreamSynchronize(st)); _t0 = std::chrono::steady_clock::now(); }
#define PROF_END(st, name) if (g_prof) { CK(cudaStreamSynchronize(st)); auto& _r = g_times[name]; _r.first += std::chrono::duration<double>(std::chrono::steady_clock::now() - _t0).count(); _r.second++; }

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
    int ctx = 4096, n_vocab = 248320, n_sm = 40;
    std::vector<Layer> L;
    Mvm out;                                         // lm head
    Mtp mtp;
    float* out_norm = nullptr; float* signs[3] = {nullptr, nullptr, nullptr};   // 5120, 6144, 17408
    std::vector<float> sign_host[3];
    float* rx = nullptr; float* irms = nullptr; signed char* A = nullptr; float* d8 = nullptr; int* sa = nullptr;
    float* ybuf = nullptr; float* y_logits = nullptr; float* qkv = nullptr; float* gout = nullptr; float* gated = nullptr; float* Q = nullptr; float* gate = nullptr; float* attn_out = nullptr;
    float* hlast = nullptr;                          // [KMAX][5120] final-normed hidden of the last step (MTP input)
    float* emb_host = nullptr; float* logits_host = nullptr;
    cudaStream_t st = nullptr;
    ssq_t* ssq = nullptr;                            // [2*64+2][KMAX] sums of squares, one slot per norm input
    ssq_t* slot(int s) { return ssq + (size_t) s * KMAX; }
    float rope_base = 1e7f;
    int cur = 0;                                     // active recurrent-state copy
    size_t dev_bytes = 0;

    float* dalloc(size_t n) { float* d; CK(cudaMalloc(&d, n * 4)); dev_bytes += n * 4; return d; }

    int auto_parts(int chunks, int G, int lo) const { int want = (2 * n_sm + chunks - 1) / chunks; if (want < lo) want = lo; if (want > G) want = G; return want; }

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
        CK(cudaMalloc(&m.w, words * 4)); m.bytes = words * 4; dev_bytes += words * 4;
        CK(cudaMemcpy(m.w, host.data(), words * 4, cudaMemcpyHostToDevice));
    }
    Q4 load_q4(const gguf::Tensor& t) {
        Q4 q; q.cols = (int) t.ne[0]; q.rows = (int) t.ne[1]; q.nblk = q.cols / 32;
        const int R = q.rows, N = q.nblk;
        std::vector<u32> nib((size_t) N * 4 * R); std::vector<half> sc((size_t) N * R);
        std::vector<std::thread> th; const int nth = std::max(1u, std::thread::hardware_concurrency());
        for (int tt = 0; tt < nth; ++tt) th.emplace_back([&, tt] {
            for (int r = tt; r < R; r += nth) {
                const uint8_t* s = t.data + (size_t) r * N * 18;
                for (int b = 0; b < N; ++b) {
                    uint16_t h; memcpy(&h, s + b * 18, 2); __half_to_halfword(sc, (size_t) b * R + r, h);
                    for (int w = 0; w < 4; ++w) { u32 v; memcpy(&v, s + b * 18 + 2 + w * 4, 4); nib[((size_t) b * 4 + w) * R + r] = v; }
                }
            }
        });
        for (auto& x : th) x.join();
        CK(cudaMalloc(&q.nib, nib.size() * 4)); CK(cudaMemcpy(q.nib, nib.data(), nib.size() * 4, cudaMemcpyHostToDevice));
        CK(cudaMalloc(&q.sc, sc.size() * 2)); CK(cudaMemcpy(q.sc, sc.data(), sc.size() * 2, cudaMemcpyHostToDevice));
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
                for (int c = 0; c < 2; ++c) {
                    CK(cudaMalloc(&y.conv[c], cb)); CK(cudaMemset(y.conv[c], 0, cb)); dev_bytes += cb;
                    CK(cudaMalloc(&y.gdn[c], sbytes)); CK(cudaMemset(y.gdn[c], 0, sbytes)); dev_bytes += sbytes;
                }
                CK(cudaMalloc(&y.rec_conv, (size_t) KMAX * QKV * 4)); CK(cudaMalloc(&y.g_raw, (size_t) KMAX * GATE96 * 4)); dev_bytes += (size_t) KMAX * (QKV + GATE96) * 4;
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
        { size_t mx = 0; for (auto& y : L) for (Mvm* m : { &y.A, &y.B, &y.C, &y.D }) mx = std::max(mx, (size_t) m->parts * m->yrows); ybuf = dalloc((size_t) KMAX * mx); }
        y_logits = dalloc((size_t) KMAX * out.yrows);
        qkv = dalloc((size_t) KMAX * QKV); gout = dalloc((size_t) KMAX * D_INNER); gated = dalloc((size_t) KMAX * D_INNER);
        Q = dalloc((size_t) KMAX * 24 * 256); gate = dalloc((size_t) KMAX * D_INNER); attn_out = dalloc((size_t) KMAX * D_INNER);
        hlast = dalloc((size_t) KMAX * D_MODEL);
        CK(cudaHostAlloc(&emb_host, (size_t) KMAX * D_MODEL * 4, cudaHostAllocDefault));
        CK(cudaHostAlloc(&logits_host, (size_t) KMAX * n_vocab * 4, cudaHostAllocDefault));
        if (with_mtp && f.find("blk.64.nextn.eh_proj.weight")) load_mtp();
    }

    void load_mtp() {
        Mtp& m = mtp; const std::string p = "blk.64.";
        auto T = [&](const std::string& n) -> const gguf::Tensor& { return f.get(p + n); };
        auto up = [&](const gguf::Tensor& t) { auto v = to_f32(t); dev_bytes += v.size() * 4; return upload_f32(v.data(), v.size()); };
        m.eh = load_q4(T("nextn.eh_proj.weight")); m.q = load_q4(T("attn_q.weight")); m.k = load_q4(T("attn_k.weight")); m.v = load_q4(T("attn_v.weight"));
        m.wo = load_q4(T("attn_output.weight")); m.gate = load_q4(T("ffn_gate.weight")); m.up = load_q4(T("ffn_up.weight")); m.down = load_q4(T("ffn_down.weight"));
        m.enorm = up(T("nextn.enorm.weight")); m.hnorm = up(T("nextn.hnorm.weight")); m.attn_norm = up(T("attn_norm.weight")); m.post_norm = up(T("post_attention_norm.weight"));
        m.head_norm = up(T("nextn.shared_head_norm.weight")); m.qnorm = up(T("attn_q_norm.weight")); m.knorm = up(T("attn_k_norm.weight"));
        const size_t kvb = (size_t) 4 * ctx * 256 * 2; CK(cudaMalloc(&m.kc, kvb)); CK(cudaMalloc(&m.vc, kvb)); dev_bytes += 2 * kvb;
        m.b_e = dalloc(D_MODEL); m.b_en = dalloc(D_MODEL); m.b_cat = dalloc(2 * D_MODEL); m.y0 = dalloc(D_MODEL); m.yn = dalloc(D_MODEL); m.yq = dalloc(14336);
        m.y1 = dalloc(D_MODEL); m.pn = dalloc(D_MODEL); m.fg = dalloc(D_FF); m.fu = dalloc(D_FF); m.fs = dalloc(D_FF); m.y2 = dalloc(D_MODEL); m.h_next = dalloc(D_MODEL);
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

    // --- launches ---
    void prep(int mode, const float* src, int n, int ncols, int parts, int yrows, const float* sgn, const float* w, int upoff, const float* yp, int K, const ssq_t* ss = nullptr) {
        PrepArgs a; a.src = src; a.irms = ss; a.w = w; a.signs = sgn; a.yp = yp; a.parts = parts; a.yrows = yrows; a.upoff = upoff; a.n = n; a.G = ncols / 128; a.A = A; a.d8 = d8; a.sa = sa;
        prep_kernel<<<dim3(n / 1024, K), 256, 0, st>>>(a, mode);
    }
    void mvm(Mvm& m, int K) {
        const int G = m.cols / 128;
        const dim3 grid(m.chunks, m.parts);
        if (K == 1) mvm_kernel<1><<<grid, NT, 0, st>>>(m.w, A, d8, sa, ybuf, G, m.cols, m.parts, m.yrows);
        else if (K == 2) mvm_kernel<2><<<grid, NT, 0, st>>>(m.w, A, d8, sa, ybuf, G, m.cols, m.parts, m.yrows);
        else mvm_kernel<4><<<grid, NT, 0, st>>>(m.w, A, d8, sa, ybuf, G, m.cols, m.parts, m.yrows);
    }
    void lm_head(int K) {
        const int G = 40;
        if (K == 1) mvm_kernel<1><<<dim3(out.chunks, 1), NT, 0, st>>>(out.w, A, d8, sa, y_logits, G, D_MODEL, 1, out.yrows);
        else if (K == 2) mvm_kernel<2><<<dim3(out.chunks, 1), NT, 0, st>>>(out.w, A, d8, sa, y_logits, G, D_MODEL, 1, out.yrows);
        else mvm_kernel<4><<<dim3(out.chunks, 1), NT, 0, st>>>(out.w, A, d8, sa, y_logits, G, D_MODEL, 1, out.yrows);
    }
    void addres(Mvm& m, int K, ssq_t* ss) { addres_kernel<<<dim3((D_MODEL + 1023) / 1024, K), 256, 0, st>>>(rx, ybuf, D_MODEL, m.parts, m.yrows, 0, ss); }
    void gdn_part(int il, int K, const float* raw_in) {          // conv + gated delta net, reading conv/gdn from copy `cur`, writing copy cur^1
        Layer& y = L[il];
        convqk_kernel<<<80, 128, 0, st>>>(ybuf, y.A.parts, y.A.yrows, y.conv_k, y.conv[cur], y.conv[cur ^ 1], qkv, raw_in, raw_in ? nullptr : y.rec_conv, K);
        gdn_kernel<<<dim3(48, 16), 256, 0, st>>>(qkv, y.g_raw, y.dt_bias, y.ssm_a, y.gdn[cur], y.gdn[cur ^ 1], gout, K);
    }

#define P(name, ...) { PROF_BEGIN(st); __VA_ARGS__; PROF_END(st, name); }
    void layer(int il, int K, int pos) {
        Layer& y = L[il];
        ssq_t* sA = slot(2 * il); ssq_t* sP = slot(2 * il + 1); ssq_t* sN = slot(2 * il + 2);
        P("prep_norm", prep(0, rx, D_MODEL, D_MODEL, 0, 0, signs[0], y.attn_norm, 0, nullptr, K, sA));
        P("mvmA", mvm(y.A, K));
        if (y.recr) {
            P("gate", (gate_kernel<<<dim3(96, K), 256, 0, st>>>(rx, sA, y.attn_norm, y.wb, y.wa, y.g_raw)));
            P("convgdn", gdn_part(il, K, nullptr));
            P("gdnpost", (gdnpost_kernel<<<dim3(48, K), 128, 0, st>>>(gout, y.ssm_norm, ybuf, y.A.parts, y.A.yrows, gated)));
            P("prep_ssm", prep(2, gated, D_INNER, D_INNER, 0, 0, signs[1], nullptr, 0, nullptr, K));
        } else {
            P("attn_prep", (attn_prep_kernel<<<dim3(32, K), 256, 0, st>>>(ybuf, y.A.parts, y.A.yrows, y.qnorm, y.knorm, Q, gate, y.kc, y.vc, ctx, pos, rope_base)));
            P("attn", (attn_kernel<<<dim3(24, K), 256, 0, st>>>(Q, gate, y.kc, y.vc, ctx, pos, attn_out, 0)));
            P("prep_wo", prep(3, attn_out, D_INNER, D_INNER, 0, 0, signs[1], nullptr, 0, nullptr, K));
        }
        P("mvmB", mvm(y.B, K));
        P("addres", addres(y.B, K, sP));
        P("prep_norm", prep(0, rx, D_MODEL, D_MODEL, 0, 0, signs[0], y.post_norm, 0, nullptr, K, sP));
        P("mvmC", mvm(y.C, K));
        P("prep_swiglu", prep(1, nullptr, D_FF, D_FF, y.C.parts, y.C.yrows, signs[2], nullptr, D_FF, ybuf, K));
        P("mvmD", mvm(y.D, K));
        P("addres", addres(y.D, K, sN));
    }

    // runs K tokens at positions pos..; fills hlast, and logits_host if want_logits.  State written into copy cur^1 (not committed)
    void step(const int* toks, int K, int pos, bool want_logits) {
        for (int k = 0; k < K; ++k) embed(toks[k], emb_host + (size_t) k * D_MODEL);
        CK(cudaMemcpyAsync(rx, emb_host, (size_t) K * D_MODEL * 4, cudaMemcpyHostToDevice, st));
        CK(cudaMemsetAsync(ssq, 0, (size_t) (2 * 64 + 2) * KMAX * sizeof(ssq_t), st));
        sumsq_kernel<<<dim3(K), 256, 0, st>>>(rx, D_MODEL, slot(0));
        for (int il = 0; il < 64; ++il) layer(il, K, pos);
        norm_store_kernel<<<dim3((D_MODEL + 255) / 256, K), 256, 0, st>>>(rx, slot(128), out_norm, hlast, D_MODEL);
        if (want_logits) {
            prep(0, rx, D_MODEL, D_MODEL, 0, 0, signs[0], out_norm, 0, nullptr, K, slot(128));
            lm_head(K);
            for (int k = 0; k < K; ++k) CK(cudaMemcpyAsync(logits_host + (size_t) k * n_vocab, y_logits + (size_t) k * out.yrows, (size_t) n_vocab * 4, cudaMemcpyDeviceToHost, st));
        }
    }
    void sync() { CK(cudaStreamSynchronize(st)); }
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
    float* emb_ring = nullptr; unsigned ring_n = 0;        // pinned ring [32][5120]: async uploads must not reuse a slot before the copy landed
    void mtp_upload_embed(int tok) {
        if (!emb_ring) CK(cudaHostAlloc(&emb_ring, (size_t) 32 * D_MODEL * 4, cudaHostAllocDefault));
        if ((ring_n & 15) == 0) sync();
        float* dst = emb_ring + (size_t) (ring_n++ & 31) * D_MODEL;
        embed(tok, dst); CK(cudaMemcpyAsync(mtp.b_e, dst, D_MODEL * 4, cudaMemcpyHostToDevice, st));
    }
    static void slot_norm(Engine& e, const float* src, ssq_t* slot_, const float* w, float* dst, int n) {
        sumsq_kernel<<<1, 256, 0, e.st>>>(src, n, slot_);
        norm_store_kernel<<<dim3((n + 255) / 256, 1), 256, 0, e.st>>>(src, slot_, w, dst, n);
    }
    void q4mv(const Q4& m, const float* x, float* y) { q4mv_kernel<<<(m.rows + 255) / 256, 256, 0, st>>>(m.nib, m.sc, x, y, m.rows, m.nblk); }
    int mtp_row(const float* h, int tok, int pos, bool draft) {
        Mtp& m = mtp; Engine& e = *this;
        mtp_upload_embed(tok);
        slot_norm(e, m.b_e, m.slots + 0, m.enorm, m.b_cat, D_MODEL);
        slot_norm(e, h, m.slots + 1, m.hnorm, m.b_cat + D_MODEL, D_MODEL);
        q4mv(m.eh, m.b_cat, m.y0);
        slot_norm(e, m.y0, m.slots + 2, m.attn_norm, m.yn, D_MODEL);
        q4mv(m.q, m.yn, m.yq);  // [0,12288)
        q4mv(m.k, m.yn, m.yq + 12288);
        q4mv(m.v, m.yn, m.yq + 13312);
        attn_prep_kernel<<<dim3(32, 1), 256, 0, st>>>(m.yq, 1, 14336, m.qnorm, m.knorm, m.Q, m.gateb, m.kc, m.vc, ctx, pos, rope_base);
        if (!draft) return 0;
        attn_kernel<<<dim3(24, 1), 256, 0, st>>>(m.Q, m.gateb, m.kc, m.vc, ctx, pos, m.attn_out, 1);
        q4mv(m.wo, m.attn_out, m.wo_out);
        vec_add_kernel<<<(D_MODEL + 255) / 256, 256, 0, st>>>(m.y0, m.wo_out, m.y1, D_MODEL);
        slot_norm(e, m.y1, m.slots + 3, m.post_norm, m.pn, D_MODEL);
        q4mv(m.gate, m.pn, m.fg); q4mv(m.up, m.pn, m.fu);
        swiglu_kernel<<<(D_FF + 255) / 256, 256, 0, st>>>(m.fg, m.fu, m.fs, D_FF);
        q4mv(m.down, m.fs, m.y2);
        vec_add_kernel<<<(D_MODEL + 255) / 256, 256, 0, st>>>(m.y1, m.y2, m.y2, D_MODEL);
        slot_norm(e, m.y2, m.slots + 4, m.head_norm, m.h_next, D_MODEL);
        prep(0, m.y2, D_MODEL, D_MODEL, 0, 0, signs[0], m.head_norm, 0, nullptr, 1, m.slots + 4);
        mvm_kernel<1><<<dim3(out.chunks, 1), NT, 0, st>>>(out.w, A, d8, sa, m.y_logits, 40, D_MODEL, 1, out.yrows);
        CK(cudaMemcpyAsync(logits_host, m.y_logits, (size_t) n_vocab * 4, cudaMemcpyDeviceToHost, st));
        sync();
        return argmax(logits_host);
    }
    int argmax(const float* l) const { int b = 0; float bv = l[0]; for (int i = 1; i < n_vocab; ++i) if (l[i] > bv) { bv = l[i]; b = i; } return b; }
};

extern "C" {
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
    e->step(toks, K, pos, logits != nullptr);
    e->sync();
    if (logits) memcpy(logits, e->logits_host, (size_t) K * e->n_vocab * 4);
    e->commit(K, K);
    e->sync();
}
void tr_reset(void* p) { ((Engine*) p)->reset_state(); }

// greedy generation.  prompt: n tokens; D drafts per round (0 = plain).  Returns number of generated tokens; stats: [rounds, accepted drafts, mtp calls]
int tr_generate(void* p, const int* prompt, int n, int max_new, const int* eos, int n_eos, int D, int* out, double* stats) {
    Engine* e = (Engine*) p;
    const int NV = e->n_vocab;
    auto is_eos = [&](int t) { for (int i = 0; i < n_eos; ++i) if (eos[i] == t) return true; return false; };
    e->reset_state();
    if (D > 3) D = 3;
    if (D > 0 && !e->mtp.ok) D = 0;
    int produced = 0; double rounds = 0, accepted = 0, drafts = 0, kept_total = 0;
    // ---- prefill
    int pos = 0; int last_logit_k = 0;
    while (pos < n) {
        int K = std::min(4, n - pos);
        int toks[4]; for (int k = 0; k < K; ++k) toks[k] = prompt[pos + k];
        bool last = pos + K == n;
        e->step(toks, K, pos, last);
        e->sync();
        e->commit(K, K);
        // MTP catch-up rows for positions with a known next token: row (t_{j+1}, h_j) at kv position j+1
        if (D > 0) for (int k = 0; k < K; ++k) { const int j = pos + k; if (j + 1 < n) e->mtp_row(e->hlast + (size_t) k * D_MODEL, prompt[j + 1], j + 1, false); }
        last_logit_k = K - 1;
        pos += K;
    }
    int cur_tok = e->argmax(e->logits_host + (size_t) last_logit_k * NV);
    int out_n = 0;
    out[out_n++] = cur_tok;
    bool done = is_eos(cur_tok);
    // hidden of the last processed trunk token is hlast[last_logit_k] -- copy to a stable buffer
    float* h_last; CK(cudaMalloc(&h_last, D_MODEL * 4));
    CK(cudaMemcpyAsync(h_last, e->hlast + (size_t) last_logit_k * D_MODEL, D_MODEL * 4, cudaMemcpyDeviceToDevice, e->st));
    while (!done && out_n < max_new) {
        if (D == 0) {
            e->step(&cur_tok, 1, pos, true); e->sync(); e->commit(1, 1);
            cur_tok = e->argmax(e->logits_host);
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
        e->step(batch, K, pos, true); e->sync();
        // verify
        int a = 0; int next = 0;
        for (int k = 0; k < K; ++k) {
            next = e->argmax(e->logits_host + (size_t) k * NV);
            if (k < D) { if (next == drafted[k]) a++; else break; }
        }
        // emitted tokens: drafted[0..a-1] then `next` (the argmax after the last kept token)
        const int kept = 1 + a;
        e->commit(K, kept);
        rounds++; accepted += a; kept_total += kept;
        // MTP catch-up for the accepted drafts using the trunk's hiddens (rows at kv pos pos+1..pos+a)
        // hidden of batch index k is hlast[k] (trunk hidden for pos+k); row for drafted[i] (position pos+1+i) uses hlast[i]
        for (int i = 0; i < a; ++i) e->mtp_row(e->hlast + (size_t) i * D_MODEL, drafted[i], pos + 1 + i, false);
        CK(cudaMemcpyAsync(h_last, e->hlast + (size_t) a * D_MODEL, D_MODEL * 4, cudaMemcpyDeviceToDevice, e->st));
        pos += kept;
        // output: accepted drafts then `next`
        for (int i = 0; i < a && out_n < max_new; ++i) { out[out_n++] = drafted[i]; if (is_eos(drafted[i])) { done = true; break; } }
        if (!done) { cur_tok = next; if (out_n < max_new) out[out_n++] = cur_tok; done = is_eos(cur_tok); }
        }
    e->sync();
    CK(cudaFree(h_last));
    if (stats) { stats[0] = rounds; stats[1] = accepted; stats[2] = drafts; stats[3] = kept_total; }
    return out_n;
}
}
