"""Build the engine on a Modal T4, compare against the PrismML fork on identical tokens (greedy tokens + top-10 logprobs)."""
import ctypes, json, os, subprocess, sys, time
import modal
from modal_common import VOL, GGUF, image

app = modal.App("bonsai25-engine-validate", image=image.add_local_python_source("modal_common")
                .add_local_dir(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"), "/src"))
W = f"/w/{GGUF}"
PROMPT = "Explain in three sentences why the sky is blue, and then write a small Python function that computes the nth Fibonacci number iteratively."

def build(arch="sm_75"):
    cmd = f"nvcc -O3 -std=c++17 -arch={arch} --use_fast_math -shared -Xcompiler -fPIC -o /tmp/libtritrun.so /src/engine.cu"
    print("$", cmd, flush=True)
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if r.returncode:
        print(r.stderr[-4000:], flush=True); raise SystemExit("build failed")
    print("build ok", r.stderr[-500:], flush=True)

def find(n):
    return subprocess.check_output(f"find /opt/prism -name {n} -type f | head -1", shell=True, text=True).strip()

def ref_run(n_gen):
    import requests
    ls = find("llama-server")
    os.environ["LD_LIBRARY_PATH"] = os.path.dirname(ls) + ":" + os.environ.get("LD_LIBRARY_PATH", "")
    p = subprocess.Popen(f"{ls} -m {W} -ngl 99 -fa on -c 4096 -np 1 --port 8099 --cache-ram 0 > /tmp/s.log 2>&1", shell=True)
    for _ in range(600):
        try:
            if requests.get("http://127.0.0.1:8099/health", timeout=2).status_code == 200: break
        except Exception: pass
        time.sleep(1)
    toks = requests.post("http://127.0.0.1:8099/tokenize", json={"content": PROMPT, "add_special": False}).json()["tokens"]
    r = requests.post("http://127.0.0.1:8099/completion", json={"prompt": toks, "n_predict": n_gen, "temperature": 0, "n_probs": 10, "return_tokens": True, "cache_prompt": False}, timeout=600).json()
    subprocess.run("pkill -9 llama-server", shell=True); p.wait(); time.sleep(3)
    gen = r["tokens"]
    probs = []
    for e in r.get("completion_probabilities", []):
        top = e.get("top_logprobs") or e.get("top_probs")
        probs.append([(t["id"], t["logprob"] if "logprob" in t else t.get("prob")) for t in top])
    return toks, gen, probs

@app.function(gpu="T4", volumes={"/w": VOL}, timeout=3000, cpu=8, memory=24576)
def validate(n_gen=24):
    import numpy as np
    build("sm_75")
    toks, gen, probs = ref_run(n_gen)
    print("prompt toks", len(toks), "ref gen", gen[:10], flush=True)
    lib = ctypes.CDLL("/tmp/libtritrun.so")
    lib.tr_load.restype = ctypes.c_void_p; lib.tr_load.argtypes = [ctypes.c_char_p, ctypes.c_int, ctypes.c_int]
    lib.tr_eval.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_int), ctypes.c_int, ctypes.c_int, ctypes.POINTER(ctypes.c_float)]
    lib.tr_device_bytes.restype = ctypes.c_longlong; lib.tr_device_bytes.argtypes = [ctypes.c_void_p]
    t0 = time.time()
    e = lib.tr_load(W.encode(), 4096, 0)
    print("loaded", round(time.time()-t0, 1), "s; device GiB", lib.tr_device_bytes(e)/2**30, flush=True)
    nv = lib.tr_vocab(ctypes.c_void_p(e)) if False else 248320
    buf = (ctypes.c_float * nv)()
    def ev(tok, pos, want):
        a = (ctypes.c_int * 1)(tok)
        lib.tr_eval(e, a, 1, pos, buf if want else None)
    my_toks, lp_err, agree, t_dec = [], [], 0, []
    cur = None
    all_toks = toks
    for pos, t in enumerate(all_toks):
        ev(t, pos, pos == len(all_toks) - 1)
    print("prompt done", flush=True)
    pos = len(all_toks)
    for i in range(n_gen):
        lg = np.ctypeslib.as_array(buf).copy()
        lp = lg - lg.max(); lp = lp - np.log(np.exp(lp).sum())
        tid = int(lg.argmax())
        my_toks.append(tid)
        ref_next = gen[i] if i < len(gen) else None
        if i < len(probs):
            rt = dict(probs[i])
            mine = {t: lp[t] for t in rt}
            d = [abs(mine[t] - v) for t, v in rt.items()]
            lp_err.append((i, max(d), float(np.mean(d))))
        agree += int(tid == ref_next)
        # feed the REFERENCE token (teacher forced) so later positions stay comparable
        t0 = time.time()
        ev(ref_next if ref_next is not None else tid, pos, True); t_dec.append(time.time() - t0); pos += 1
    print("greedy agreement", agree, "/", n_gen)
    print("my greedy ", my_toks)
    print("ref greedy", gen)
    for i, mx, mn in lp_err: print(f"pos {i}: max|dlogprob| {mx:.4f} mean {mn:.4f}", flush=True)
    print("decode s/token", np.mean(t_dec[2:]), "->", 1/np.mean(t_dec[2:]), "tok/s", flush=True)
    r = subprocess.run("nvidia-smi --query-gpu=memory.used,memory.total --format=csv", shell=True, capture_output=True, text=True); print(r.stdout)
    return True

@app.local_entrypoint()
def main(): validate.remote()
