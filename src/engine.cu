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
#define MVM_NT 256
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
    const float* src; const float* irms; const float* w; const float* signs;
    const float* yp; int parts; int yrows; int upoff;
    int n; int ncols;
    short* X; float* xs;
};

__global__ void prep_kernel(const PrepArgs a, const int mode) {
    __shared__ float s[1024];
    __shared__ float gm[8];
    const int k = blockIdx.y, seg = blockIdx.x, tid = threadIdx.x;
    const int base = seg * 1024;
    for (int e = tid; e < 1024; e += 256) {
        const int i = base + e;
        float v;
        if (mode == 0) {
            v = a.src[(size_t) k * a.n + i] * a.irms[k] * a.w[i];
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
    if (tid < 8) gm[tid] = 0.0f;
    __syncthreads();
    for (int e = tid; e < 1024; e += 256)
        atomicMax((int*) &gm[e >> 7], __float_as_int(fabsf(s[e]) * 0.03125f));
    __syncthreads();
    for (int e = tid; e < 1024; e += 256) {
        const float m = gm[e >> 7];
        const float q = m > 0 ? (s[e] * 0.03125f) * ((float) QMAX / m) : 0.0f;
        a.X[(size_t) k * a.ncols + base + e] = (short) lrintf(q);
    }
    if (tid < 8) a.xs[(size_t) k * (a.ncols / 128) + seg * 8 + tid] = gm[tid] / (float) QMAX;
}

// inverse RMS of the residual vector, one value per token
__global__ void sumsq_kernel(const float* x, int n, float* irms) {
    __shared__ float part[8];
    const int k = blockIdx.x;
    float acc = 0;
    for (int e = threadIdx.x; e < n; e += 256) { const float v = x[(size_t) k * n + e]; acc += v * v; }
    acc = warp_sum(acc);
    if ((threadIdx.x & 31) == 0) part[threadIdx.x >> 5] = acc;
    __syncthreads();
    if (threadIdx.x == 0) {
        float t = 0; for (int i = 0; i < 8; ++i) t += part[i];
        irms[k] = 1.0f / sqrtf(t / n + 1e-6f);
    }
}

// ---------------------------------------------------------------------------------------------
// lookup-table matrix-vector product.  W layout: [chunk][G][7 words][1024 rows] (see gguf words per 128-weight block)
// Y layout: [token k][part][yrows]
// ---------------------------------------------------------------------------------------------
#ifdef NOSYNC
#define MVM_SYNC()
#else
#define MVM_SYNC() __syncthreads()
#endif
template <int K>
__global__ __launch_bounds__(NT, 1) void mvm_kernel(const u32* W, const short* X, const float* xs, float* Y,
                                                     int G, int cols, int parts, int yrows) {
    __shared__ short T[K * 26 * 256];
    const int chunk = blockIdx.x, part = blockIdx.y, tid = threadIdx.x;
    const int g0 = (int) ((long long) part * G / parts), g1 = (int) ((long long) (part + 1) * G / parts);
    float acc[RPT][K];
#pragma unroll
    for (int j = 0; j < RPT; ++j)
#pragma unroll
        for (int k = 0; k < K; ++k) acc[j][k] = 0.0f;
    const u32* Wc = W + (size_t) chunk * G * 7 * CH;
    u32 w[RPT][7], wn[RPT][7];
    if (g0 < g1) {
        const u32* Wg = Wc + (size_t) g0 * 7 * CH;
#pragma unroll
        for (int j = 0; j < RPT; ++j)
#pragma unroll
            for (int q = 0; q < 7; ++q) w[j][q] = Wg[q * CH + tid + j * NT];
    }

    for (int g = g0; g < g1; ++g) {
        const bool have_next = g + 1 < g1;
        // issue the NEXT group's weight loads now: they run while this group's tables are built and read
        if (have_next) {
            const u32* Wn = Wc + (size_t) (g + 1) * 7 * CH;
#pragma unroll
            for (int j = 0; j < RPT; ++j)
#pragma unroll
                for (int q = 0; q < 7; ++q) wn[j][q] = Wn[q * CH + tid + j * NT];
        }

#ifndef NOBUILD
        if (tid < 234) {
            const int b = tid / 9, r = tid - b * 9;
            const int d0 = r / 3, d1 = r - d0 * 3;
#pragma unroll
            for (int k = 0; k < K; ++k) {
                const short* xk = X + (size_t) k * cols + g * 128;
                int x[5];
#pragma unroll
                for (int n = 0; n < 5; ++n) x[n] = (b >= 24 && n == 4) ? 0 : xk[trit_elem(b, n)];
                const int R = (d0 - 1) * x[0] + (d1 - 1) * x[1];
                for (int lo = 0; lo < 27; ++lo) {
                    const int e2 = lo / 9, e3 = (lo / 3) % 3, e4 = lo % 3;
                    const int q = r * 27 + lo;
                    const int code = (q * 256 + 242) / 243;      // the file's byte encoding of 5 trits
                    T[k * 26 * 256 + b * 256 + code] = (short) (R + (e2 - 1) * x[2] + (e3 - 1) * x[3] + (e4 - 1) * x[4]);
                }
            }
        }
#endif
        MVM_SYNC();
#pragma unroll
        for (int k = 0; k < K; ++k) {
            const float xsg = xs[(size_t) k * G + g];
#pragma unroll
            for (int j = 0; j < RPT; ++j) {
                int isum = 0;
#ifndef NOLOOK
#pragma unroll
                for (int b = 0; b < 26; ++b) {
                    const u32 byte = (w[j][b >> 2] >> (8 * (b & 3))) & 0xFFu;
                    isum += T[k * 26 * 256 + b * 256 + byte];
                }
#else
                isum = w[j][0] ^ w[j][1] ^ w[j][2] ^ w[j][3] ^ w[j][4] ^ w[j][5] ^ w[j][6];
#endif
                const half dh = __ushort_as_half((unsigned short) (w[j][6] >> 16));
                acc[j][k] += __half2float(dh) * xsg * (float) isum;
            }
        }
        MVM_SYNC();
        if (have_next) {
#pragma unroll
            for (int j = 0; j < RPT; ++j)
#pragma unroll
                for (int q = 0; q < 7; ++q) w[j][q] = wn[j][q];
        }
    }
    for (int k = 0; k < K; ++k)
#pragma unroll
        for (int j = 0; j < RPT; ++j)
            Y[(size_t) (k * parts + part) * yrows + chunk * CH + tid + j * NT] = acc[j][k];
}

// ---------------------------------------------------------------------------------------------
// x[k][i] += sum over parts of y[k][part][off + i]
// ---------------------------------------------------------------------------------------------
__global__ void addres_kernel(float* x, const float* yp, int n, int parts, int yrows, int off) {
    const int k = blockIdx.y;
    for (int i = blockIdx.x * 256 + threadIdx.x; i < n; i += gridDim.x * 256) {
        float t = 0;
        for (int p = 0; p < parts; ++p) t += yp[(size_t) (k * parts + p) * yrows + off + i];
        x[(size_t) k * n + i] += t;
    }
}

// F32 beta / alpha gate projections of the RMS-normalised activation
__global__ void gate_kernel(const float* x, const float* irms, const float* nw, const float* wb, const float* wa, float* out) {
    __shared__ float part[8];
    const int k = blockIdx.y, row = blockIdx.x;
    const float* W = row < 48 ? wb + (size_t) row * D_MODEL : wa + (size_t) (row - 48) * D_MODEL;
    float acc = 0;
    for (int e = threadIdx.x; e < D_MODEL; e += 256) acc += W[e] * (x[(size_t) k * D_MODEL + e] * irms[k] * nw[e]);
    acc = warp_sum(acc);
    if ((threadIdx.x & 31) == 0) part[threadIdx.x >> 5] = acc;
    __syncthreads();
    if (threadIdx.x == 0) { float t = 0; for (int i = 0; i < 8; ++i) t += part[i]; out[k * GATE96 + row] = t; }
}

// depthwise conv (4 taps) + silu + l2 norm of q/k heads; 80 blocks of 128 channels
__global__ void convqk_kernel(const float* yA, int parts, int yrows, const float* ck, float* state, float* qkv, int K) {
    __shared__ float ps[4];
    const int b = blockIdx.x, tid = threadIdx.x, ch = b * 128 + tid;
    float win[3 + KMAX];
    win[0] = state[3 * ch]; win[1] = state[3 * ch + 1]; win[2] = state[3 * ch + 2];
    for (int k = 0; k < K; ++k) {
        float v = 0;
        for (int p = 0; p < parts; ++p) v += yA[(size_t) (k * parts + p) * yrows + ch];
        win[3 + k] = v;
    }
    state[3 * ch] = win[K]; state[3 * ch + 1] = win[K + 1]; state[3 * ch + 2] = win[K + 2];
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
                           float* state, float* out, int K) {
    const int h = blockIdx.x, cgrp = blockIdx.y;
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int col = cgrp * 8 + warp;
    const int nk = h % 16;
    float* S = state + ((size_t) h * 128 + col) * 128;
    float s_[4];
    for (int r = 0; r < 4; ++r) s_[r] = S[r * 32 + lane];
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
        if (lane == 0) out[(size_t) k * D_INNER + h * 128 + col] = acc;
    }
    for (int r = 0; r < 4; ++r) S[r * 32 + lane] = s_[r];
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

__global__ void attn_kernel(const float* Q, const float* gate, const half* kc, const half* vc, int ctx, int pos, float* out) {
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
        for (int p = warp; p < L; p += 8) {
            float s = 0;
            for (int m = 0; m < 8; ++m) s += qr[m] * __half2float(Kp[(size_t) p * 256 + m * 32 + lane]);
            s = warp_sum(s) * 0.0625f;          // 1 / sqrt(256)
            if (lane == 0) sc[p] = s;
        }
    }
    __syncthreads();
    float m = -1e30f;
    for (int p = tid; p < L; p += 256) m = fmaxf(m, sc[p]);
    mx[tid] = m;
    __syncthreads();
    if (tid == 0) { float g = mx[0]; for (int i = 1; i < 256; ++i) g = fmaxf(g, mx[i]); mx[0] = g; }
    __syncthreads();
    const float gmax = mx[0];
    float sum = 0;
    for (int p = tid; p < L; p += 256) { const float e = expf(sc[p] - gmax); sc[p] = e; sum += e; }
    __syncthreads();
    mx[tid] = sum;
    __syncthreads();
    float t = 0;
    if (tid == 0) { for (int i = 0; i < 256; ++i) t += mx[i]; mx[0] = t; }
    __syncthreads();
    const float inv = 1.0f / mx[0];
    float acc = 0;
    for (int p = 0; p < L; ++p) acc += sc[p] * __half2float(Vp[(size_t) p * 256 + tid]);
    out[(size_t) k * D_INNER + h * 256 + tid] = acc * inv * gate[(size_t) k * D_INNER + h * 256 + tid];
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
    float* wb = nullptr; float* wa = nullptr; float* conv_state = nullptr; float* gdn_state = nullptr;
    half* kc = nullptr; half* vc = nullptr;
    bool recr = false;
};

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

static bool g_prof = getenv("TR_PROF") != nullptr;
static std::map<std::string, std::pair<double, int>> g_times;
#define PROF_BEGIN(st) std::chrono::steady_clock::time_point _t0; if (g_prof) { CK(cudaStreamSynchronize(st)); _t0 = std::chrono::steady_clock::now(); }
#define PROF_END(st, name) if (g_prof) { CK(cudaStreamSynchronize(st)); auto& _r = g_times[name]; _r.first += std::chrono::duration<double>(std::chrono::steady_clock::now() - _t0).count(); _r.second++; }

struct Engine {
    gguf::File f;
    int ctx = 4096, n_vocab = 248320;
    std::vector<Layer> L;
    Mvm out;                                         // lm head
    float* out_norm = nullptr; float* signs[3] = {nullptr, nullptr, nullptr};   // 5120, 6144, 17408
    std::vector<float> sign_host[3];
    // scratch
    float* rx = nullptr; float* irms = nullptr; short* X = nullptr; float* xs = nullptr;
    float* ybuf = nullptr; float* y_logits = nullptr; float* qkv = nullptr; float* gout = nullptr; float* gated = nullptr; float* Q = nullptr; float* gate = nullptr; float* attn_out = nullptr; float* g_raw = nullptr;
    float* emb_host = nullptr;                       // pinned [KMAX][5120]
    cudaStream_t st = nullptr;
    float rope_base = 1e7f;
    size_t dev_bytes = 0;

    float* dalloc(size_t n) { float* d; CK(cudaMalloc(&d, n * 4)); dev_bytes += n * 4; return d; }

    // fused matrix: rows of several tensors stacked, all sharing the same column count
    int n_sm = 40;
    int auto_parts(int chunks, int G, int lo) const { int want = (2 * n_sm + chunks - 1) / chunks; if (want < lo) want = lo; if (want > G) want = G; return want; }
    void load_mvm(Mvm& m, const std::vector<const gguf::Tensor*>& ts, int parts) {
        int rows = 0, cols = (int) ts[0]->ne[0]; for (auto* t : ts) rows += (int) t->ne[1];
        m.rows = rows; m.cols = cols; m.chunks = (rows + CH - 1) / CH;
        m.parts = auto_parts(m.chunks, cols / 128, parts); if (m.chunks > 100) m.parts = 1; m.yrows = m.chunks * CH;
        const int G = cols / 128;
        const size_t words = (size_t) m.chunks * G * 7 * CH;
        std::vector<u32> host(words, 0u);
        // row -> source pointer
        std::vector<const uint8_t*> rowptr(rows);
        { int r = 0; for (auto* t : ts) { for (int i = 0; i < (int) t->ne[1]; ++i) rowptr[r++] = t->data + (size_t) i * G * 28; } }
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

    void load(const char* path, int ctx_) {
        f.open(path);
        ctx = ctx_;
        { cudaDeviceProp pr; int dev; CK(cudaGetDevice(&dev)); CK(cudaGetDeviceProperties(&pr, dev)); n_sm = pr.multiProcessorCount; }
        rope_base = (float) f.num("qwen35.rope.freq_base", 1e7);
        // signs
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
        for (int il = 0; il < NL; ++il) {
            Layer& y = L[il]; const std::string p = "blk." + std::to_string(il) + ".";
            auto T = [&](const std::string& n) -> const gguf::Tensor* { return &f.get(p + n); };
            auto upn = [&](const gguf::Tensor* t) { auto v = to_f32(*t); dev_bytes += v.size() * 4; return upload_f32(v.data(), v.size()); };
            y.attn_norm = upn(T("attn_norm.weight")); y.post_norm = upn(T("post_attention_norm.weight"));
            const bool recr = ((il + 1) % 4) != 0; y.recr = recr;
            if (recr) {
                load_mvm(y.A, { T("attn_qkv.weight"), T("attn_gate.weight") }, 1);
                load_mvm(y.B, { T("ssm_out.weight") }, 1);
                y.ssm_norm = upn(T("ssm_norm.weight")); y.dt_bias = upn(T("ssm_dt.bias")); y.ssm_a = upn(T("ssm_a")); y.conv_k = upn(T("ssm_conv1d.weight"));
                y.wb = upn(T("ssm_beta.weight")); y.wa = upn(T("ssm_alpha.weight"));
                CK(cudaMalloc(&y.conv_state, (size_t) 3 * QKV * 4)); CK(cudaMemset(y.conv_state, 0, (size_t) 3 * QKV * 4)); dev_bytes += (size_t) 3 * QKV * 4;
                const size_t sbytes = (size_t) 48 * 128 * 128 * 4; CK(cudaMalloc(&y.gdn_state, sbytes)); CK(cudaMemset(y.gdn_state, 0, sbytes)); dev_bytes += sbytes;
            } else {
                load_mvm(y.A, { T("attn_q.weight"), T("attn_k.weight"), T("attn_v.weight") }, 1);
                load_mvm(y.B, { T("attn_output.weight") }, 1);
                y.qnorm = upn(T("attn_q_norm.weight")); y.knorm = upn(T("attn_k_norm.weight"));
                const size_t kvb = (size_t) 4 * ctx * 256 * 2; CK(cudaMalloc(&y.kc, kvb)); CK(cudaMalloc(&y.vc, kvb)); dev_bytes += 2 * kvb;
            }
            load_mvm(y.C, { T("ffn_gate.weight"), T("ffn_up.weight") }, 1);
            load_mvm(y.D, { T("ffn_down.weight") }, 1);
            if ((il & 15) == 0) { fprintf(stderr, "layer %d loaded\n", il); fflush(stderr); }
        }
        { auto v = to_f32(f.get("output_norm.weight")); out_norm = upload_f32(v.data(), v.size()); }
        load_mvm(out, { &f.get("output.weight") }, 1);
        n_vocab = out.rows;
        rx = dalloc((size_t) KMAX * D_MODEL); irms = dalloc(KMAX); X = nullptr;
        CK(cudaMalloc(&X, (size_t) KMAX * D_FF * sizeof(short))); xs = dalloc((size_t) KMAX * (D_FF / 128));
        { size_t mx = 0; for (auto& y : L) for (Mvm* m : { &y.A, &y.B, &y.C, &y.D }) mx = std::max(mx, (size_t) m->parts * m->yrows); ybuf = dalloc((size_t) KMAX * mx); fprintf(stderr, "partial buffer %zu floats/token, SMs %d\n", mx, n_sm); }
        y_logits = dalloc((size_t) KMAX * out.yrows);
        qkv = dalloc((size_t) KMAX * QKV); gout = dalloc((size_t) KMAX * D_INNER); gated = dalloc((size_t) KMAX * D_INNER);
        Q = dalloc((size_t) KMAX * 24 * 256); gate = dalloc((size_t) KMAX * D_INNER); attn_out = dalloc((size_t) KMAX * D_INNER); g_raw = dalloc(KMAX * GATE96);
        CK(cudaHostAlloc(&emb_host, (size_t) KMAX * D_MODEL * 4, cudaHostAllocDefault));
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
                // digit n of the byte: multiply by 3^n and take bits 8..9
                uint32_t q = b[byte]; for (int r = 0; r < n; ++r) q = (q * 3) & 0xFF;
                x[g * 128 + i] = (float) ((int) ((q * 3) >> 8) - 1) * d;
            }
        }
        // inverse: H then signs
        const float* sg = sign_host[0].data();
        for (int b = 0; b < 5; ++b) {
            float* s = x + b * 1024;
            for (int h = 1; h < 1024; h *= 2)
                for (int i = 0; i < 1024; i += 2 * h) for (int j = i; j < i + h; ++j) { const float p = s[j], q = s[j + h]; s[j] = p + q; s[j + h] = p - q; }
            for (int i = 0; i < 1024; ++i) out_[b * 1024 + i] = s[i] * 0.03125f * sg[b * 1024 + i];
        }
    }

    // --- launches ---
    void prep(int mode, const float* src, int n, int ncols, int parts, int yrows, const float* sgn, const float* w, int upoff, const float* yp, int K) {
        PrepArgs a; a.src = src; a.irms = irms; a.w = w; a.signs = sgn; a.yp = yp; a.parts = parts; a.yrows = yrows; a.upoff = upoff; a.n = n; a.ncols = ncols; a.X = X; a.xs = xs;
        prep_kernel<<<dim3(n / 1024, K), 256, 0, st>>>(a, mode);
    }
    void mvm(Mvm& m, int K) {
        const int G = m.cols / 128;
        const dim3 grid(m.chunks, m.parts);
        if (K == 1) mvm_kernel<1><<<grid, NT, 0, st>>>(m.w, X, xs, ybuf, G, m.cols, m.parts, m.yrows);
        else if (K == 2) mvm_kernel<2><<<grid, NT, 0, st>>>(m.w, X, xs, ybuf, G, m.cols, m.parts, m.yrows);
        else mvm_kernel<4><<<grid, NT, 0, st>>>(m.w, X, xs, ybuf, G, m.cols, m.parts, m.yrows);
    }
    void addres(Mvm& m, int K) {
        addres_kernel<<<dim3((D_MODEL + 1023) / 1024, K), 256, 0, st>>>(rx, ybuf, D_MODEL, m.parts, m.yrows, 0);
    }
    void sumsq(int K) { sumsq_kernel<<<dim3(K), 256, 0, st>>>(rx, D_MODEL, irms); }

    void layer(int il, int K, int pos) {
        Layer& y = L[il];
#define P(name, ...) { PROF_BEGIN(st); __VA_ARGS__; PROF_END(st, name); }
        P("sumsq", sumsq(K));
        P("prep_norm", prep(0, rx, D_MODEL, D_MODEL, 0, 0, signs[0], y.attn_norm, 0, nullptr, K));
        P("mvmA", mvm(y.A, K));
        if (y.recr) {
            P("gate", (gate_kernel<<<dim3(96, K), 256, 0, st>>>(rx, irms, y.attn_norm, y.wb, y.wa, g_raw)));
            P("convqk", (convqk_kernel<<<80, 128, 0, st>>>(ybuf, y.A.parts, y.A.yrows, y.conv_k, y.conv_state, qkv, K)));
            P("gdn", (gdn_kernel<<<dim3(48, 16), 256, 0, st>>>(qkv, g_raw, y.dt_bias, y.ssm_a, y.gdn_state, gout, K)));
            P("gdnpost", (gdnpost_kernel<<<dim3(48, K), 128, 0, st>>>(gout, y.ssm_norm, ybuf, y.A.parts, y.A.yrows, gated)));
            P("prep_ssm", prep(2, gated, D_INNER, D_INNER, 0, 0, signs[1], nullptr, 0, nullptr, K));
        } else {
            P("attn_prep", (attn_prep_kernel<<<dim3(32, K), 256, 0, st>>>(ybuf, y.A.parts, y.A.yrows, y.qnorm, y.knorm, Q, gate, y.kc, y.vc, ctx, pos, rope_base)));
            P("attn", (attn_kernel<<<dim3(24, K), 256, 0, st>>>(Q, gate, y.kc, y.vc, ctx, pos, attn_out)));
            P("prep_wo", prep(3, attn_out, D_INNER, D_INNER, 0, 0, signs[1], nullptr, 0, nullptr, K));
        }
        P("mvmB", mvm(y.B, K));
        P("addres", addres(y.B, K));
        P("sumsq", sumsq(K));
        P("prep_norm", prep(0, rx, D_MODEL, D_MODEL, 0, 0, signs[0], y.post_norm, 0, nullptr, K));
        P("mvmC", mvm(y.C, K));
        P("prep_swiglu", prep(1, nullptr, D_FF, D_FF, y.C.parts, y.C.yrows, signs[2], nullptr, D_FF, ybuf, K));
        P("mvmD", mvm(y.D, K));
        P("addres", addres(y.D, K));
    }

    // embeds: [K][5120] host floats.  logits_out: [K][n_vocab] host floats (may be null for a pure prompt step)
    void eval(const float* embeds, int K, int pos, float* logits_out) {
        memcpy(emb_host, embeds, (size_t) K * D_MODEL * 4);
        CK(cudaMemcpyAsync(rx, emb_host, (size_t) K * D_MODEL * 4, cudaMemcpyHostToDevice, st));
        for (int il = 0; il < 64; ++il) layer(il, K, pos);
        sumsq(K);
        prep(0, rx, D_MODEL, D_MODEL, 0, 0, signs[0], out_norm, 0, nullptr, K);
        if (logits_out) {
            const int G = 40;
            if (K == 1) mvm_kernel<1><<<dim3(out.chunks, 1), NT, 0, st>>>(out.w, X, xs, y_logits, G, D_MODEL, 1, out.yrows);
            else if (K == 2) mvm_kernel<2><<<dim3(out.chunks, 1), NT, 0, st>>>(out.w, X, xs, y_logits, G, D_MODEL, 1, out.yrows);
            else mvm_kernel<4><<<dim3(out.chunks, 1), NT, 0, st>>>(out.w, X, xs, y_logits, G, D_MODEL, 1, out.yrows);
            for (int k = 0; k < K; ++k) CK(cudaMemcpyAsync(logits_out + (size_t) k * n_vocab, y_logits + (size_t) k * out.yrows, (size_t) n_vocab * 4, cudaMemcpyDeviceToHost, st));
        }
        CK(cudaStreamSynchronize(st));
    }
    void reset() { for (auto& y : L) if (y.recr) { CK(cudaMemset(y.conv_state, 0, (size_t) 3 * QKV * 4)); CK(cudaMemset(y.gdn_state, 0, (size_t) 48 * 128 * 128 * 4)); } }
};

extern "C" {
void tr_prof_dump(int steps) {
    double tot = 0; for (auto& kv : g_times) tot += kv.second.first;
    fprintf(stderr, "kernel profile over %d steps (synchronised, launch latency included):\n", steps);
    for (auto& kv : g_times) fprintf(stderr, "  %-12s %8.3f ms/step  %6.1f us/call  (%5.1f%%)\n", kv.first.c_str(), kv.second.first / steps * 1e3, kv.second.first / kv.second.second * 1e6, kv.second.first / tot * 100);
    g_times.clear();
}
void* tr_load(const char* path, int ctx) { auto* e = new Engine(); e->load(path, ctx); return e; }
int tr_vocab(void* p) { return ((Engine*) p)->n_vocab; }
long long tr_device_bytes(void* p) { return (long long) ((Engine*) p)->dev_bytes; }
void tr_embed(void* p, int tok, float* out) { ((Engine*) p)->embed(tok, out); }
// toks: K token ids; logits: host [K][vocab] or null
void tr_eval(void* p, const int* toks, int K, int pos, float* logits) {
    Engine* e = (Engine*) p;
    std::vector<float> em((size_t) K * D_MODEL);
    for (int k = 0; k < K; ++k) e->embed(toks[k], em.data() + (size_t) k * D_MODEL);
    e->eval(em.data(), K, pos, logits);
}
void tr_reset(void* p) { ((Engine*) p)->reset(); }
}
