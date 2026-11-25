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
    lib.tr_test_tile.restype = ctypes.c_double; lib.tr_test_tile.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_int]
    lib.tr_test_attn.restype = ctypes.c_double; lib.tr_test_attn.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_int]
    for il in (0, 3):
        print(f"layer {il}: wide matmul vs decode matmul, max error / max|y|:", {"ABCD"[w]: [f"{lib.tr_test_tile(e, il, k, w):.1e}" for k in (9, 37, 64, 128)] for w in range(4)}, flush=True)
    print("wide attention vs per-token attention:", {L: f"{lib.tr_test_attn(e, 3, L, 37):.2e}" for L in (0, 5, 300, 511, 512, 1500, 3000)}, flush=True)
    toks = [1024 + (i * 37) % 4000 for i in range(200)]
    lib.tr_reset.argtypes = [ctypes.c_void_p]
    def run_chunks(sizes):
        """Evaluate chunks of the given sizes; returns {position: logits} for every position whose logits are returned."""
        lib.tr_reset(e); got = {}; p = 0
        for kk in sizes:
            rows = kk if kk <= 4 else 1
            buf = (ctypes.c_float * (V * rows))()
            lib.tr_eval(e, (ctypes.c_int * kk)(*toks[p:p + kk]), kk, p, buf)
            a = np.frombuffer(buf, dtype=np.float32).reshape(rows, V)
            for r in range(rows): got[p + kk - rows + r] = a[r].copy()
            p += kk
        return got
    ref = run_chunks([1] * 140)
    ref2 = run_chunks([1] * 140)
    print("K=1 run-to-run max diff", max(float(np.abs(ref[i] - ref2[i]).max()) for i in ref), flush=True)
    for name, sizes in {"K=2": [2] * 70, "K=4": [4] * 35, "K=8": [8] * 17, "K=9": [9] * 15, "K=33": [33] * 4, "K=64": [64, 64],
                        "64+36+then 1s": [64, 36] + [1] * 40, "17+then 4s": [17] + [4] * 30}.items():
        g = run_chunks(sizes)
        d = {i: float(np.abs(g[i] - ref[i]).max()) for i in g}
        am = sum(int(g[i].argmax() == ref[i].argmax()) for i in g)
        print(f"{name}: positions {len(g)} max|logit diff| {max(d.values()):.6f} same argmax {am}/{len(g)}", flush=True)
@app.local_entrypoint()
def main(): run.remote()
