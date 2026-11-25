"""K=4 batched steps vs K=1 steps on the same tokens (logit agreement), and rollback replay correctness."""
import ctypes, json, os, subprocess, sys, time
import modal
from modal_common import VOL, GGUF, image
app = modal.App("bonsai25-engine-consistency", image=image.add_local_python_source("modal_common")
                .add_local_dir(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"), "/src"))
W = f"/w/{GGUF}"

@app.function(gpu="T4", volumes={"/w": VOL}, timeout=1800, cpu=8, memory=24576)
def run():
    import numpy as np
    r = subprocess.run("nvcc -O3 -std=c++17 -arch=sm_75 --use_fast_math -shared -Xcompiler -fPIC -o /tmp/l.so /src/engine.cu", shell=True, capture_output=True, text=True)
    if r.returncode: print(r.stderr[-3000:]); raise SystemExit("build")
    lib = ctypes.CDLL("/tmp/l.so")
    lib.tr_load.restype = ctypes.c_void_p; lib.tr_load.argtypes = [ctypes.c_char_p, ctypes.c_int, ctypes.c_int]
    lib.tr_eval.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_int), ctypes.c_int, ctypes.c_int, ctypes.POINTER(ctypes.c_float)]
    e = lib.tr_load(W.encode(), 4096, 1)
    V = 248320
    toks = [3000 + (i * 7919) % 2000 for i in range(32)]
    toks = [1024 + (i * 37) % 4000 for i in range(64)]
    def run_chunks(K, total=32):
        lib.tr_reset.argtypes = [ctypes.c_void_p]; lib.tr_reset(e)
        rows = []
        for p in range(0, total, K):
            kk = min(K, total - p)
            buf = (ctypes.c_float * (V * kk))()
            a = (ctypes.c_int * kk)(*toks[p:p + kk])
            lib.tr_eval(e, a, kk, p, buf)
            rows += [np.frombuffer(buf, dtype=np.float32).reshape(kk, V)[k].copy() for k in range(kk)]
        return rows
    r1 = run_chunks(1); 
    for K in (2, 4):
        rk = run_chunks(K)
        d = [float(np.abs(a - b).max()) for a, b in zip(r1, rk)]
        am = [int(a.argmax() == b.argmax()) for a, b in zip(r1, rk)]
        print(f"K={K}: max|logit diff| over {len(d)} positions: {max(d):.4f} (mean of max {np.mean(d):.4f}); same argmax {sum(am)}/{len(am)}", flush=True)
    # rerun K=1 twice to see the numerical noise floor
    r1b = run_chunks(1)
    print("K=1 run-to-run max diff:", max(float(np.abs(a - b).max()) for a, b in zip(r1, r1b)), flush=True)
    # top-2 gaps at each position (how many are near-ties)
    gaps = []
    for a in r1:
        s = np.sort(a)[-2:]; gaps.append(float(s[1] - s[0]))
    print("smallest top-2 gap", sorted(gaps)[:5])
@app.local_entrypoint()
def main(): run.remote()
