"""End-to-end prefill, fork vs tritrun in one container: 512 and 2048 token prompts of real text, MTP off and on.

Fork: llama-server /completion, timings.prompt_per_second.  Ours: wall time of tr_generate(max_new=1), which is the
whole prefill plus the first token.  Also checks that the first token agrees.
"""
import ctypes, json, os, subprocess, sys, time
import modal
from modal_common import VOL, GGUF, image, ARCHS
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
app = modal.App("bonsai25-engine-prefill", image=image.add_local_python_source("modal_common")
                .add_local_dir(os.path.join(ROOT, "src"), "/src"))
W = f"/w/{GGUF}"
WP = "/w/Ternary-Bonsai-2.5-27B-PTQ1_0.gguf"

CHILD = r'''
import ctypes, json, sys, time
toks = json.load(open("/tmp/ptoks.json")); mtp = int(sys.argv[1])
lib = ctypes.CDLL("/tmp/l.so")
lib.tr_load.restype = ctypes.c_void_p; lib.tr_load.argtypes = [ctypes.c_char_p, ctypes.c_int, ctypes.c_int]
lib.tr_generate.restype = ctypes.c_int
lib.tr_generate.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_int), ctypes.c_int, ctypes.c_int, ctypes.POINTER(ctypes.c_int), ctypes.c_int, ctypes.c_int, ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_double)]
lib.tr_prof_dump.argtypes = [ctypes.c_int]
e = lib.tr_load(b"@W@", 4096, mtp)
res = {}
for n in (64, 512, 2048):
    t = toks[:n]; arr = (ctypes.c_int * n)(*t); out = (ctypes.c_int * 64)(); st = (ctypes.c_double * 8)()
    best = 1e9
    for rep in range(3):
        t0 = time.time(); lib.tr_generate(e, arr, n, 1, None, 0, 3 if mtp else 0, out, st); best = min(best, time.time() - t0)
    res[n] = {"tps": n / best, "first": out[0]}
    if len(sys.argv) > 2: lib.tr_prof_dump(1)
print("RES", json.dumps(res), flush=True)
'''.replace("@W@", W)

def run(prof=False, fork=True):
    import requests
    gname = subprocess.check_output("nvidia-smi --query-gpu=name --format=csv,noheader", shell=True, text=True).strip()
    r = subprocess.run(f"nvcc -O3 -std=c++17 -arch={ARCHS[gname]} --use_fast_math -shared -Xcompiler -fPIC -o /tmp/l.so /src/engine.cu", shell=True, capture_output=True, text=True)
    if r.returncode: print(r.stderr[-3000:]); raise SystemExit("build")
    ls = subprocess.check_output("find /opt/prism -name llama-server -type f | head -1", shell=True, text=True).strip()
    env = {**os.environ, "LD_LIBRARY_PATH": os.path.dirname(ls)}
    # real text: the CUDA licence/readme shipped in the image
    import glob
    text = ""
    for f in sorted(glob.glob("/usr/share/common-licenses/*")):
        try: text += open(f, encoding="utf8", errors="replace").read()
        except Exception: pass
        if len(text) > 40000: break
    open("/tmp/text.txt", "w").write(text[:40000])
    lt = os.path.join(os.path.dirname(ls), "llama-tokenize")
    o = subprocess.run(f"{lt} -m {W} -f /tmp/text.txt --ids --no-bos 2>/dev/null", shell=True, capture_output=True, text=True, env=env).stdout
    toks = json.loads(o.strip().splitlines()[-1])[:2048]
    json.dump(toks, open("/tmp/ptoks.json", "w"))
    open("/tmp/child.py", "w").write(CHILD)
    print("GPU", gname, "tokens", len(toks), flush=True)
    for mtp in (0, 1):
        o = subprocess.run([sys.executable, "/tmp/child.py", str(mtp)] + (["prof"] if prof else []), capture_output=True, text=True, env={**os.environ, **({"TR_PROF": "1"} if prof else {})})
        line = [l for l in o.stdout.splitlines() if l.startswith("RES")]
        print(f"[{gname}] ours mtp={mtp}:", line[0][4:] if line else o.stderr[-1500:], flush=True)
        if prof: print(o.stderr[-9000:], flush=True)
    if not fork: return
    for mtp in (0, 1):
        extra = "--spec-type draft-mtp --spec-draft-n-min 0 --spec-draft-n-max 3 --spec-draft-p-min 0" if mtp else ""
        p = subprocess.Popen(f"{ls} -m {W if mtp else WP} -ngl 99 -fa on -c 4096 -np 1 --port 8099 --cache-ram 0 {extra} > /tmp/s.log 2>&1", shell=True, env=env)
        for _ in range(600):
            try:
                if requests.get("http://127.0.0.1:8099/health", timeout=2).status_code == 200: break
            except Exception: pass
            time.sleep(0.5)
        res = {}
        for n in (64, 512, 2048):
            best, first = 0, None
            for rep in range(3):
                j = requests.post("http://127.0.0.1:8099/completion", json={"prompt": toks[:n], "n_predict": 1, "temperature": 0, "ignore_eos": True, "cache_prompt": False, "return_tokens": True}, timeout=300).json()
                best = max(best, j["timings"]["prompt_per_second"]); first = j["tokens"][0] if j.get("tokens") else None
            res[n] = {"tps": best, "first": first}
        print(f"[{gname}] fork mtp={mtp}:", json.dumps(res), flush=True)
        subprocess.run("pkill -9 llama-server", shell=True); p.wait(); time.sleep(3)

@app.function(gpu="T4", volumes={"/w": VOL}, timeout=2400, cpu=8, memory=24576)
def t4(prof=False, fork=True): run(prof, fork)
@app.function(gpu="L4", volumes={"/w": VOL}, timeout=2400, cpu=8, memory=24576)
def l4(prof=False, fork=True): run(prof, fork)
@app.function(gpu="A10", volumes={"/w": VOL}, timeout=2400, cpu=8, memory=24576)
def a10(prof=False, fork=True): run(prof, fork)

@app.local_entrypoint()
def main(which: str = "t4", prof: bool = False, fork: bool = True):
    hs = [{"t4": t4, "l4": l4, "a10": a10}[n].spawn(prof, fork) for n in which.split(",")]
    for h in hs: h.get()
