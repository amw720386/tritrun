"""Speculative (MTP) vs plain greedy decode on a Modal GPU: same outputs, acceptance, tok/s."""
import ctypes, json, os, subprocess, sys, time
import modal
from modal_common import VOL, GGUF, image
app = modal.App("bonsai25-engine-spec", image=image.add_local_python_source("modal_common")
                .add_local_dir(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"), "/src"))
W = f"/w/{GGUF}"
ARCHS = {"Tesla T4": "sm_75", "NVIDIA A10G": "sm_86", "NVIDIA L4": "sm_89"}
PROMPTS = ["Implement quicksort in Python and explain its complexity.",
           "Write a short story about a lighthouse keeper who finds a message in a bottle.",
           "A train leaves at 3pm travelling 60 mph; another leaves at 4pm at 80 mph on the same track. When does the second catch the first? Show your work."]

def body(max_new=200, ctx=4096, drafts=(0, 1, 2, 3)):
    import numpy as np
    gname = subprocess.check_output("nvidia-smi --query-gpu=name --format=csv,noheader", shell=True, text=True).strip()
    r = subprocess.run(f"nvcc -O3 -std=c++17 -arch={ARCHS.get(gname, 'sm_75')} --use_fast_math -shared -Xcompiler -fPIC -o /tmp/l.so /src/engine.cu", shell=True, capture_output=True, text=True)
    if r.returncode: print(r.stderr[-3000:]); raise SystemExit("build")
    # tokenize with the fork (no server compute: tokenize endpoint needs the model; do it once)
    ls = subprocess.check_output("find /opt/prism -name llama-tokenize -type f | head -1", shell=True, text=True).strip()
    toks = []
    for q in PROMPTS:
        out = subprocess.run(f"LD_LIBRARY_PATH=$(dirname {ls}) {ls} -m {W} -p {json.dumps(q)} --ids 2>/dev/null", shell=True, capture_output=True, text=True).stdout
        toks.append(json.loads(out.strip().splitlines()[-1]))
    lib = ctypes.CDLL("/tmp/l.so")
    lib.tr_load.restype = ctypes.c_void_p; lib.tr_load.argtypes = [ctypes.c_char_p, ctypes.c_int, ctypes.c_int]
    lib.tr_device_bytes.restype = ctypes.c_longlong; lib.tr_device_bytes.argtypes = [ctypes.c_void_p]
    lib.tr_generate.restype = ctypes.c_int
    lib.tr_generate.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_int), ctypes.c_int, ctypes.c_int, ctypes.POINTER(ctypes.c_int), ctypes.c_int, ctypes.c_int, ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_double)]
    e = lib.tr_load(W.encode(), ctx, 1)
    print(gname, "device GiB", lib.tr_device_bytes(e) / 2**30, flush=True)
    refs = {}
    for D in drafts:
        tot_t, tot_n, tot_r, tot_a, tot_d, tot_dec = 0, 0, 0, 0, 0, 0
        for pi, t in enumerate(toks):
            arr = (ctypes.c_int * len(t))(*t); out = (ctypes.c_int * (max_new + 8))(); st = (ctypes.c_double * 5)()
            t0 = time.time()
            n = lib.tr_generate(e, arr, len(t), max_new, None, 0, D, out, st)
            dt = time.time() - t0
            seq = [out[i] for i in range(n)]
            if D == 0: refs[pi] = seq
            else: print(f"  D={D} prompt {pi}: match first-divergence at {next((i for i,(a,b) in enumerate(zip(seq, refs[pi])) if a != b), min(len(seq), len(refs[pi])))} of {len(seq)}", flush=True)
            tot_t += dt; tot_n += n; tot_dec += st[4]; tot_r += st[0]; tot_a += st[1]; tot_d += st[2]
        print(f"D={D}: {tot_n} tokens in {tot_t:.2f}s (incl. prefill) = {tot_n/tot_t:.2f} tok/s; decode-only {(tot_n - len(toks)) / tot_dec:.2f} tok/s; rounds {tot_r:.0f} accepted {tot_a:.0f} drafted {tot_d:.0f}" + (f"; accept rate {tot_a/max(1,tot_d):.2f}, tokens/round {1 + tot_a/max(1,tot_r):.2f}" if D else ""), flush=True)
    print(subprocess.run("nvidia-smi --query-gpu=memory.used --format=csv,noheader", shell=True, capture_output=True, text=True).stdout, flush=True)

@app.function(gpu="T4", volumes={"/w": VOL}, timeout=1800, cpu=8, memory=24576)
def t4(): body()
@app.function(gpu="L4", volumes={"/w": VOL}, timeout=1800, cpu=8, memory=24576)
def l4(): body()
@app.function(gpu="A10", volumes={"/w": VOL}, timeout=1800, cpu=8, memory=24576)
def a10(): body()

@app.local_entrypoint()
def main(which: str = "t4"): {"t4": t4, "l4": l4, "a10": a10}[which].remote()
