"""Ablations of the wide-step matmul kernel: times mvm_tile_kernel alone on random data, with parts of it switched off by text replacement.

Numbers are for the (34816 x 5120) FFN matrix and 128 tokens.  A switched-off part gives wrong results on purpose: only the time matters.
"""
import os, re, subprocess, sys
import modal
from modal_common import image, gpu_arch
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
app = modal.App("bonsai25-engine-tilelab", image=image.add_local_python_source("modal_common").add_local_dir(os.path.join(ROOT, "src"), "/src"))

MAIN = r'''
#include "engine.cu"
#include <ctime>
int main() {
    const int rows = 34816, G = 40, K = 128, yrows = rows;
    u32* W; signed char* A; float* d8; int* sa; float* Y; ssq_t* ss;
    std::vector<u32> hw((size_t) rows * G * 7); srand(1);
    for (auto& v : hw) v = ((u32) rand() << 17) ^ ((u32) rand() << 2) ^ rand();
    std::vector<signed char> ha((size_t) K * G * 128); for (auto& v : ha) v = (signed char) (rand() % 255 - 127);
    std::vector<float> hd((size_t) K * G * 4, 0.01f); std::vector<int> hs((size_t) K * G * 4, 5); std::vector<ssq_t> hq(K, 1ull << 30);
    CK(cudaMalloc(&W, hw.size() * 4)); CK(cudaMalloc(&A, ha.size())); CK(cudaMalloc(&d8, hd.size() * 4)); CK(cudaMalloc(&sa, hs.size() * 4));
    CK(cudaMalloc(&Y, (size_t) K * yrows * 4)); CK(cudaMalloc(&ss, K * 8));
    CK(cudaMemcpy(W, hw.data(), hw.size() * 4, cudaMemcpyHostToDevice)); CK(cudaMemcpy(A, ha.data(), ha.size(), cudaMemcpyHostToDevice));
    CK(cudaMemcpy(d8, hd.data(), hd.size() * 4, cudaMemcpyHostToDevice)); CK(cudaMemcpy(sa, hs.data(), hs.size() * 4, cudaMemcpyHostToDevice)); CK(cudaMemcpy(ss, hq.data(), K * 8, cudaMemcpyHostToDevice));
    cudaStream_t st; CK(cudaStreamCreate(&st));
    for (int warm = 0; warm < 2; ++warm) {
        const int N = warm ? 600 : 400;
        CK(cudaStreamSynchronize(st)); const auto t0 = std::chrono::steady_clock::now();
        for (int i = 0; i < N; ++i) { LAUNCH; }
        CK(cudaStreamSynchronize(st));
        const double us = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count() / N * 1e6;
        if (warm) printf("TIME %.1f us  %.2f TMAC/s\n", us, (double) rows * G * 128 * K / us / 1e6);
    }
}
'''
LAUNCH = "mvm_tile_kernel<128><<<dim3(yrows / tile_rows(128), 1), TILE_NT, 0, st>>>(W, A, d8, sa, Y, G, yrows, K, ss, 1)"
def rep(*ps):
    def f(s):
        for a, b in ps:
            if isinstance(a, str):
                assert a in s, a; s = s.replace(a, b)
            else:
                assert a.search(s), a.pattern; s = a.sub(b, s)
        return s
    return f
RW = re.compile(r"#define TILE_RW 64")
LB2 = ("__launch_bounds__(TILE_NT, 1) void mvm_tile_kernel", "__launch_bounds__(TILE_NT, 2) void mvm_tile_kernel")
R32 = (RW, "#define TILE_RW 32")
VARIANTS = {"db rw64_b1": rep(), "db rw32_b2": rep(R32, LB2), "db rw32_b1": rep(R32), "db rw64_b2": rep(LB2)}

def run(names):
    g = subprocess.check_output("nvidia-smi --query-gpu=name,power.limit,clocks.max.sm --format=csv,noheader", shell=True, text=True).strip()
    src = open("/src/engine.cu").read()
    for n in (names.split(",") if names else VARIANTS):
        n0 = n; n = n.replace(" ", "_"); os.makedirs(f"/tmp/{n}", exist_ok=True)
        open(f"/tmp/{n}/engine.cu", "w").write(VARIANTS[n0](src)); open(f"/tmp/{n}/gguf.hpp", "w").write(open("/src/gguf.hpp").read())
        open(f"/tmp/{n}/main.cu", "w").write(MAIN.replace("LAUNCH", LAUNCH))
        r = subprocess.run(f"cd /tmp/{n} && nvcc -O3 -std=c++17 -arch={gpu_arch()} --use_fast_math -Xptxas -v -o m main.cu", shell=True, capture_output=True, text=True)
        if r.returncode: print(n, "BUILD", r.stderr[-1500:], flush=True); continue
        p = subprocess.Popen(f"/tmp/{n}/m", stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, shell=True)
        rg = re.findall(r"mvm_tile_kernelILi128E.*?Used (\d+) registers", r.stdout + r.stderr, re.S); samples = []
        import time
        while p.poll() is None:
            time.sleep(0.15); samples.append(subprocess.check_output("nvidia-smi --query-gpu=power.draw,clocks.sm --format=csv,noheader,nounits", shell=True, text=True).strip())
        pw = [tuple(float(x) for x in s.split(",")) for s in samples[-5:]]
        print(f"LAB [{g}] {n}:", p.stdout.read().strip(), "regs", rg[-1:], "power/clock", pw[-1] if pw else None, flush=True)

OPT = dict(timeout=1200, cpu=4, memory=8192)
@app.function(gpu="T4", **OPT)
def t4(names): run(names)
@app.function(gpu="L4", **OPT)
def l4(names): run(names)
@app.function(gpu="A10", **OPT)
def a10(names): run(names)
@app.local_entrypoint()
def main(which: str = "t4,l4,a10", names: str = ""):
    for h in [{"t4": t4, "l4": l4, "a10": a10}[n].spawn(names) for n in which.split(",")]:
        try: h.get()
        except Exception as ex: print("FAILED", repr(ex)[:400])
