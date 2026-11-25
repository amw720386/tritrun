"""Long-prompt agreement with the PrismML fork: prefill real text (wide 128-token steps, and 1-token steps), then teacher-force
the fork's greedy continuation and compare top-10 log-probabilities and argmax.  Shows whether the wide path is any further from
the fork than the one-token path (and than the fork is from itself at a different batch size)."""
import ctypes, json, os, subprocess, sys, time
import modal
from modal_common import VOL, GGUF, image, GPU, gpu_arch
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
app = modal.App("bonsai25-engine-validate-long", image=image.add_local_python_source("modal_common")
                .add_local_dir(os.path.join(ROOT, "src"), "/src"))
W = f"/w/{GGUF}"

@app.function(gpu=GPU, volumes={"/w": VOL}, timeout=3000, cpu=8, memory=24576)
def run(n_gen=32):
    import numpy as np, requests, glob
    r = subprocess.run(f"nvcc -O3 -std=c++17 -arch={gpu_arch()} --use_fast_math -shared -Xcompiler -fPIC -o /tmp/l.so /src/engine.cu", shell=True, capture_output=True, text=True)
    if r.returncode: print(r.stderr[-2000:]); raise SystemExit("build")
    ls = subprocess.check_output("find /opt/prism -name llama-server -type f | head -1", shell=True, text=True).strip()
    env = {**os.environ, "LD_LIBRARY_PATH": os.path.dirname(ls)}
    text = ""
    for f in sorted(glob.glob("/usr/share/common-licenses/*")):
        try: text += open(f, encoding="utf8", errors="replace").read()
        except Exception: pass
    open("/tmp/text.txt", "w").write(text[:40000])
    o = subprocess.run(f"{os.path.join(os.path.dirname(ls), 'llama-tokenize')} -m {W} -f /tmp/text.txt --ids --no-bos 2>/dev/null", shell=True, capture_output=True, text=True, env=env).stdout
    toks = json.loads(o.strip().splitlines()[-1])
    refs = {}
    p = subprocess.Popen(f"{ls} -m {W} -ngl 99 -fa on -c 4096 -np 1 --port 8099 --cache-ram 0 > /tmp/s.log 2>&1", shell=True, env=env)
    for _ in range(600):
        try:
            if requests.get("http://127.0.0.1:8099/health", timeout=2).status_code == 200: break
        except Exception: pass
        time.sleep(0.5)
    cases = [(300, 0), (1000, 0), (1000, 2000)]
    for n, off in cases:
        j = requests.post("http://127.0.0.1:8099/completion", json={"prompt": toks[off:off + n], "n_predict": n_gen, "temperature": 0, "ignore_eos": True, "n_probs": 10, "return_tokens": True, "cache_prompt": False}, timeout=300).json()
        refs[(n, off)] = (j["tokens"], [[(t["id"], t["logprob"]) for t in e["top_logprobs"]] for e in j["completion_probabilities"]])
    # the fork against itself with a different physical batch size: its own rounding noise
    subprocess.run("pkill -9 llama-server", shell=True); p.wait(); time.sleep(3)
    p = subprocess.Popen(f"{ls} -m {W} -ngl 99 -fa on -c 4096 -ub 64 -np 1 --port 8099 --cache-ram 0 > /tmp/s.log 2>&1", shell=True, env=env)
    for _ in range(600):
        try:
            if requests.get("http://127.0.0.1:8099/health", timeout=2).status_code == 200: break
        except Exception: pass
        time.sleep(0.5)
    for key in cases:
        # teacher-force the fork's own ub=512 continuation through the ub=64 server by scoring it as a prompt is not supported; compare its free-running output instead
        n, off = key
        j = requests.post("http://127.0.0.1:8099/completion", json={"prompt": toks[off:off + n], "n_predict": n_gen, "temperature": 0, "ignore_eos": True, "n_probs": 10, "return_tokens": True, "cache_prompt": False}, timeout=300).json()
        g0, pr0 = refs[key]
        ag = next((i for i, (a, b) in enumerate(zip(g0, j["tokens"])) if a != b), n_gen)
        d = []
        for i in range(min(ag + 1, n_gen)):
            m = dict((t["id"], t["logprob"]) for t in j["completion_probabilities"][i]["top_logprobs"])
            d += [abs(m[t] - v) for t, v in pr0[i] if t in m]
        print(f"fork ub=512 vs fork ub=64, prompt {n}@{off}: same tokens for {ag}/{n_gen}, mean |dlogprob| {np.mean(d):.4f} max {np.max(d):.3f}", flush=True)
    subprocess.run("pkill -9 llama-server", shell=True); p.wait(); time.sleep(3)

    lib = ctypes.CDLL("/tmp/l.so")
    lib.tr_load.restype = ctypes.c_void_p; lib.tr_load.argtypes = [ctypes.c_char_p, ctypes.c_int, ctypes.c_int]
    lib.tr_eval.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_int), ctypes.c_int, ctypes.c_int, ctypes.POINTER(ctypes.c_float)]
    lib.tr_reset.argtypes = [ctypes.c_void_p]
    e = lib.tr_load(W.encode(), 4096, 0)
    V = 248320
    for (n, off), (gen, probs) in refs.items():
        pr = toks[off:off + n]
        for name, step in (("wide 128-token steps", 128), ("4-token steps", 4), ("1-token steps", 1)):
            lib.tr_reset(e); pos = 0
            buf = (ctypes.c_float * (V * 4))()
            while pos < n:
                k = min(step, n - pos)
                lib.tr_eval(e, (ctypes.c_int * k)(*pr[pos:pos + k]), k, pos, buf if pos + k == n else None)
                pos += k
            last_k = k
            arr = np.frombuffer(buf, dtype=np.float32).copy()
            lg = arr[:V] if last_k > 4 else arr[(last_k - 1) * V:last_k * V]
            agree = 0; diffs = []
            for i in range(n_gen):
                l = lg - lg.max(); l = l - np.log(np.exp(l).sum())
                agree += int(int(lg.argmax()) == gen[i])
                diffs += [abs(float(l[t]) - v) for t, v in probs[i]]
                lib.tr_eval(e, (ctypes.c_int * 1)(gen[i]), 1, pos, buf); pos += 1
                lg = np.frombuffer(buf, dtype=np.float32)[:V].copy()
            print(f"ours vs fork, prompt {n}@{off}, {name}: argmax agreement {agree}/{n_gen}, mean |dlogprob| {np.mean(diffs):.4f} median {np.median(diffs):.4f} max {np.max(diffs):.3f}", flush=True)

@app.local_entrypoint()
def main(): run.remote()
