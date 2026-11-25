"""Paired benchmark: PrismML fork (llama-server) vs tritrun in the SAME container and GPU, alternating, same prompt tokens.
Reports decode-only tok/s and peak VRAM per configuration (nvidia-smi polled)."""
import ctypes, json, os, subprocess, sys, time, threading
import modal
from modal_common import VOL, GGUF, image, ARCHS
_img = image.add_local_python_source("modal_common").add_local_dir(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"), "/src")
if os.environ.get("BASE_SRC"): _img = _img.add_local_dir(os.environ["BASE_SRC"], "/src_base")   # optional: an older src/, measured next to the current one
app = modal.App("bonsai25-engine-paired", image=_img)
W = f"/w/{GGUF}"                                   # -mtp file
WP = "/w/Ternary-Bonsai-2.5-27B-PTQ1_0.gguf"        # plain file: the right one for fork plain mode (no draft head in VRAM)
PROMPTS = ["Implement quicksort in Python and explain its complexity.",
           "Write a short story about a lighthouse keeper who finds a message in a bottle.",
           "A train leaves at 3pm travelling 60 mph; another leaves at 4pm at 80 mph on the same track. When does the second catch the first? Show your work.",
           "Summarise the main causes of the French Revolution in five bullet points."]
N_NEW = 200
CTX = 4096

def find(n):
    return subprocess.check_output(f"find /opt/prism -name {n} -type f | head -1", shell=True, text=True).strip()

class Vram(threading.Thread):
    def __init__(self):
        super().__init__(daemon=True); self.peak = 0; self.stop = False
    def run(self):
        while not self.stop:
            try: self.peak = max(self.peak, int(subprocess.check_output("nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits", shell=True).split()[0]))
            except Exception: pass
            time.sleep(0.3)

def fork_run(toks, spec, pmin=0.1, ub=512):
    import requests
    ls = find("llama-server"); os.environ["LD_LIBRARY_PATH"] = os.path.dirname(ls) + ":" + os.environ.get("LD_LIBRARY_PATH", "")
    extra = f"--spec-type draft-mtp --spec-draft-n-min 0 --spec-draft-n-max 3 --spec-draft-p-min {pmin}" if spec else ""
    v = Vram(); v.start()
    p = subprocess.Popen(f"{ls} -m {W if spec else WP} -ngl 99 -fa on -c {CTX} -ub {ub} -np 1 --port 8099 --cache-ram 0 {extra} > /tmp/s.log 2>&1", shell=True)
    for _ in range(600):
        try:
            if requests.get("http://127.0.0.1:8099/health", timeout=2).status_code == 200: break
        except Exception: pass
        time.sleep(0.5)
    n = t = 0.0; pn = pt = 0.0
    for tk in toks:
        r = requests.post("http://127.0.0.1:8099/completion", json={"prompt": tk, "n_predict": N_NEW, "temperature": 0, "ignore_eos": True, "cache_prompt": False, "return_tokens": True}, timeout=900).json()
        tm = r["timings"]
        n += tm["predicted_n"]; t += tm["predicted_ms"] / 1e3; pn += tm["prompt_n"]; pt += tm["prompt_ms"] / 1e3
    subprocess.run("pkill -9 llama-server", shell=True); p.wait(); time.sleep(3); v.stop = True; v.join()
    return {"decode_tps": n / t, "prefill_tps": pn / pt, "vram_mib": v.peak}

def ours_setup():
    gname = subprocess.check_output("nvidia-smi --query-gpu=name --format=csv,noheader", shell=True, text=True).strip()
    r = subprocess.run(f"nvcc -O3 -std=c++17 -arch={ARCHS[gname]} --use_fast_math -shared -Xcompiler -fPIC -o /tmp/l.so /src/engine.cu", shell=True, capture_output=True, text=True)
    if r.returncode: print(r.stderr[-3000:]); raise SystemExit("build")
    if os.path.exists("/src_base/engine.cu"):
        r = subprocess.run(f"nvcc -O3 -std=c++17 -arch={ARCHS[gname]} --use_fast_math -shared -Xcompiler -fPIC -o /tmp/l_base.so /src_base/engine.cu", shell=True, capture_output=True, text=True)
        if r.returncode: print(r.stderr[-3000:]); raise SystemExit("build base")

# the engine is run in a child process so its VRAM is measured in isolation
CHILD = r'''
import ctypes, json, sys, time
toks = json.load(open("/tmp/toks.json")); D = int(sys.argv[1]); NN = int(sys.argv[2]); ctx = int(sys.argv[3])
lib = ctypes.CDLL(sys.argv[4])
lib.tr_load.restype = ctypes.c_void_p; lib.tr_load.argtypes = [ctypes.c_char_p, ctypes.c_int, ctypes.c_int]
lib.tr_generate.restype = ctypes.c_int
lib.tr_generate.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_int), ctypes.c_int, ctypes.c_int, ctypes.POINTER(ctypes.c_int), ctypes.c_int, ctypes.c_int, ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_double)]
e = lib.tr_load(b"%s", ctx, 1 if D > 0 else 0)
n = 0; dec = 0.0; pre = 0.0; pn = 0; ac = dr = rn = 0.0
for t in toks:
    arr = (ctypes.c_int * len(t))(*t); out = (ctypes.c_int * (NN + 8))(); st = (ctypes.c_double * 5)()
    t0 = time.time(); m = lib.tr_generate(e, arr, len(t), NN, None, 0, D, out, st); tt = time.time() - t0
    n += m; dec += st[4]; pre += tt - st[4]; pn += len(t); rn += st[0]; ac += st[1]; dr += st[2]
print("RES", json.dumps({"decode_tps": (n - len(toks)) / dec, "prefill_tps": pn / pre, "accept": ac / max(1, dr), "tok_per_round": 1 + ac / max(1, rn)}), flush=True)
''' % W

def ours_run(D, lib="/tmp/l.so"):
    v = Vram(); v.start()
    open("/tmp/child.py", "w").write(CHILD)
    out = subprocess.run([sys.executable, "/tmp/child.py", str(D), str(N_NEW), str(CTX), lib], capture_output=True, text=True)
    v.stop = True; v.join()
    line = [l for l in out.stdout.splitlines() if l.startswith("RES")]
    if not line: print(out.stdout[-1500:], out.stderr[-3000:]); raise SystemExit("ours failed")
    r = json.loads(line[0][4:]); r["vram_mib"] = v.peak
    return r

@app.function(volumes={"/w": VOL}, timeout=3600, cpu=4)
def fetch_plain():
    from huggingface_hub import hf_hub_download
    from modal_common import REPO, REV
    if not os.path.exists(WP):
        hf_hub_download(REPO, os.path.basename(WP), revision=REV, local_dir="/w"); VOL.commit()
    print("plain file", os.path.getsize(WP))

def run_all(reps=2):
    ls = subprocess.check_output("find /opt/prism -name llama-tokenize -type f | head -1", shell=True, text=True).strip()
    os.environ["LD_LIBRARY_PATH"] = os.path.dirname(ls)
    toks = []
    for q in PROMPTS:
        o = subprocess.run(f"{ls} -m {W} -p {json.dumps(q)} --ids 2>/dev/null", shell=True, capture_output=True, text=True).stdout
        toks.append(json.loads(o.strip().splitlines()[-1]))
    json.dump(toks, open("/tmp/toks.json", "w"))
    ours_setup()
    gname = subprocess.check_output("nvidia-smi --query-gpu=name,memory.total --format=csv,noheader", shell=True, text=True).strip()
    print("GPU", gname, flush=True)
    res = {}
    for rep in range(reps):
        for key, fn in [("fork_plain", lambda: fork_run(toks, False)), ("fork_plain_ub64", lambda: fork_run(toks, False, ub=64)),
                        ("ours_plain", lambda: ours_run(0)),
                        ("fork_mtp_p0.1", lambda: fork_run(toks, True)), ("fork_mtp_p0", lambda: fork_run(toks, True, pmin=0)),
                        ("fork_mtp_p0_ub64", lambda: fork_run(toks, True, pmin=0, ub=64)),
                        ("ours_mtp3", lambda: ours_run(3))] + ([("base_plain", lambda: ours_run(0, "/tmp/l_base.so")), ("base_mtp3", lambda: ours_run(3, "/tmp/l_base.so"))] if os.path.exists("/tmp/l_base.so") else []):
            r = fn(); res.setdefault(key, []).append(r); print(f"[{gname}] rep{rep} {key}: {json.dumps(r)}", flush=True)
    print("SUMMARY", json.dumps({k: {m: sorted(x[m] for x in v)[len(v) // 2] for m in ("decode_tps", "prefill_tps", "vram_mib")} for k, v in res.items()}, indent=None), flush=True)
    return res

@app.function(gpu="T4", volumes={"/w": VOL}, timeout=3000, cpu=8, memory=24576)
def t4(): run_all()
@app.function(gpu="L4", volumes={"/w": VOL}, timeout=3000, cpu=8, memory=24576)
def l4(): run_all()
@app.function(gpu="A10", volumes={"/w": VOL}, timeout=3000, cpu=8, memory=24576)
def a10(): run_all()

@app.local_entrypoint()
def main(which: str = "t4,l4,a10"):
    fetch_plain.remote()
    hs = {n: {"t4": t4, "l4": l4, "a10": a10}[n].spawn() for n in which.split(",")}
    for n, h in hs.items():
        try: h.get()
        except Exception as ex: print(n, "FAILED", ex)
