"""Build + profile per kernel on a Modal GPU (TR_PROF=1)."""
import ctypes, os, subprocess, sys, time
import modal
from modal_common import VOL, GGUF, image
app = modal.App("bonsai25-engine-profile", image=image.add_local_python_source("modal_common")
                .add_local_dir(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"), "/src"))
W = f"/w/{GGUF}"
ARCH = {"T4": "sm_75", "A10G": "sm_86", "L4": "sm_89"}

def body(gpu, prof, flags=""):
    import numpy as np
    arch = {"Tesla T4": "sm_75", "NVIDIA A10": "sm_86", "NVIDIA A10G": "sm_86", "NVIDIA L4": "sm_89"}.get(subprocess.check_output("nvidia-smi --query-gpu=name --format=csv,noheader", shell=True, text=True).strip(), "sm_75")
    r = subprocess.run(f"nvcc -O3 -std=c++17 -arch={arch} {flags} --use_fast_math -shared -Xcompiler -fPIC -o /tmp/l.so /src/engine.cu", shell=True, capture_output=True, text=True)
    if r.returncode: print(r.stderr[-3000:]); return
    if prof: os.environ["TR_PROF"] = "1"
    lib = ctypes.CDLL("/tmp/l.so")
    lib.tr_load.restype = ctypes.c_void_p; lib.tr_load.argtypes = [ctypes.c_char_p, ctypes.c_int, ctypes.c_int]
    lib.tr_eval.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_int), ctypes.c_int, ctypes.c_int, ctypes.POINTER(ctypes.c_float)]
    lib.tr_prof_dump.argtypes = [ctypes.c_int]
    e = lib.tr_load(W.encode(), 4096, 0)
    buf = (ctypes.c_float * 248320)()
    def ev(t, pos, w=True):
        a = (ctypes.c_int * 1)(t); lib.tr_eval(e, a, 1, pos, buf if w else None)
    for p in range(20): ev(1000 + p, p)
    lib.tr_prof_dump(20) if prof else None
    n = 60; t0 = time.time()
    for i in range(n): ev(2000 + i, 20 + i)
    dt = time.time() - t0
    print(f"{gpu} prof={prof} K=1: {dt/n*1e3:.2f} ms/token = {n/dt:.2f} tok/s", flush=True)
    if prof: lib.tr_prof_dump(n)
    # K-token steps (verification width): time per step, not per token
    for K in (4,):
        bufk = (ctypes.c_float * (248320 * K))()
        pos = 200
        def step(p, want):
            a = (ctypes.c_int * K)(*[3000 + p * 7 + q for q in range(K)])
            lib.tr_eval(e, a, K, p, bufk if (want and K <= 4) else None)
        for w in range(3): step(pos + w * K, False)
        if prof: lib.tr_prof_dump(3)
        t0 = time.time(); m = 20
        for s_ in range(m): step(pos + 100 + s_ * K, True)
        ms = (time.time() - t0) / m * 1e3
        if prof: lib.tr_prof_dump(m)
        print(f"{gpu} K={K}: {ms:.2f} ms/step -> {ms/K:.2f} ms per verified token, {1000*K/ms:.1f} tok/s at full acceptance", flush=True)
    print(subprocess.run("nvidia-smi --query-gpu=memory.used --format=csv,noheader", shell=True, capture_output=True, text=True).stdout)

@app.function(gpu="T4", volumes={"/w": VOL}, timeout=1200, cpu=8, memory=24576)
def t4(prof=True, variants=""):
    for v in (variants.split(";") if variants else [""]):
        print("=== variant", v, flush=True)
        import multiprocessing as mp
        p = mp.get_context("spawn").Process(target=body, args=("T4", prof, v)); p.start(); p.join()

@app.function(gpu="A10", volumes={"/w": VOL}, timeout=1200, cpu=8, memory=24576)
def a10(prof=True, variants=""):
    for v in (variants.split(";") if variants else [""]):
        print("=== variant", v, flush=True)
        import multiprocessing as mp
        p = mp.get_context("spawn").Process(target=body, args=("A10", prof, v)); p.start(); p.join()

@app.local_entrypoint()
def main(prof: bool = True, variants: str = "", gpu: str = "t4"): {"t4": t4, "a10": a10}[gpu].remote(prof, variants)
