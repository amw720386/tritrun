"""Where a decode step goes: cost of an empty launch, host enqueue time vs step time, and steady per-kernel cost
(TR_REP launches per sample, so synchronisation latency averages out)."""
import ctypes, os, subprocess, sys
import modal
from modal_common import VOL, GGUF, image, ARCHS
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
app = modal.App("bonsai25-engine-launch", image=image.add_local_python_source("modal_common").add_local_dir(os.path.join(ROOT, "src"), "/src"))
W = f"/w/{GGUF}"
CHILD = r'''
import ctypes, os, sys, time
prof = os.environ.get("TR_PROF")
lib = ctypes.CDLL("/tmp/l.so")
lib.tr_load.restype = ctypes.c_void_p; lib.tr_load.argtypes = [ctypes.c_char_p, ctypes.c_int, ctypes.c_int]
D = ctypes.POINTER(ctypes.c_double)
lib.tr_launch_cost.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int, D]
lib.tr_step_split.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_int), ctypes.c_int, ctypes.c_int, D]
lib.tr_read_bw.argtypes = [ctypes.c_void_p, D]
lib.tr_mtp_row_ms.restype = ctypes.c_double; lib.tr_mtp_row_ms.argtypes = [ctypes.c_void_p, ctypes.c_int]
lib.tr_prof_dump.argtypes = [ctypes.c_int]
e = lib.tr_load(b"@W@", 4096, 1)
o = (ctypes.c_double * 2)()
if not prof:
    for b in (1, 40, 160):
        for r in range(2): lib.tr_launch_cost(e, 4000, b, o)
        print(f"empty kernel, {b} blocks: enqueue {o[0]:.2f} us, total {o[1]:.2f} us per launch", flush=True)
if not prof:
    b = (ctypes.c_double * 3)(); lib.tr_read_bw(e, b); print("read bandwidth GB/s: one kernel %.0f, 64 kernels %.0f, 256 kernels %.0f" % tuple(b), flush=True)
lib.tr_mtp_row_ms(e, 4)
if prof: lib.tr_prof_dump(1); print("--- MTP row", file=sys.stderr)
r = lib.tr_mtp_row_ms(e, 30)
if prof: lib.tr_prof_dump(30)
else: print(f"MTP draft row: {r:.2f} ms", flush=True)
for K in (1, 4):
    pos = 200; en = tot = 0; n = 10 if prof else 60
    for i in range(n + 8):
        lib.tr_step_split(e, (ctypes.c_int * K)(*[3000 + i * 7 + q for q in range(K)]), K, pos, o); pos += K
        if i == 7 and prof: lib.tr_prof_dump(8); sys.stderr.write("--- above: warm-up\n")
        if i >= 8: en += o[0]; tot += o[1]
    if prof: sys.stderr.write(f"--- K={K}\n"); sys.stderr.flush(); lib.tr_prof_dump(n)
    else: print(f"K={K}: enqueue {en/n:.2f} ms, step {tot/n:.2f} ms ({tot/n/K:.2f} per token)", flush=True)
'''.replace("@W@", W)

def run(flags="", prof=True):
    """flags: ';'-separated compile-flag variants, each measured in turn"""
    g = subprocess.check_output("nvidia-smi --query-gpu=name --format=csv,noheader", shell=True, text=True).strip()
    vs = flags.split(";")
    ps = [subprocess.Popen(f"nvcc -O3 -std=c++17 -arch={ARCHS[g]} {v} --use_fast_math -shared -Xcompiler -fPIC -o /tmp/l{i}.so /src/engine.cu", shell=True, stderr=subprocess.PIPE, text=True) for i, v in enumerate(vs)]
    err = [p.communicate()[1] for p in ps]
    open("/tmp/c.py", "w").write(CHILD)
    out = f"===== {g}\n"
    for i, v in enumerate(vs):
        if ps[i].returncode: out += f"build failed: {v}\n" + err[i][-3000:]; continue
        subprocess.run(f"cp /tmp/l{i}.so /tmp/l.so", shell=True)
        out += f"=== variant [{v}]\n"
        for env in ({}, {"TR_PROF": "1", "TR_REP": "16"})[:2 if prof else 1]:
            p = subprocess.run([sys.executable, "/tmp/c.py"], capture_output=True, text=True, env={**os.environ, **env})
            out += p.stdout + p.stderr[-7000:]
            if p.returncode: out += f"exit {p.returncode}\n"
    print(out, flush=True)

@app.function(gpu="T4", volumes={"/w": VOL}, timeout=2400, cpu=8, memory=32768)
def t4(flags="", prof=True): run(flags, prof)
@app.function(gpu="L4", volumes={"/w": VOL}, timeout=2400, cpu=8, memory=32768)
def l4(flags="", prof=True): run(flags, prof)
@app.function(gpu="A10", volumes={"/w": VOL}, timeout=2400, cpu=8, memory=32768)
def a10(flags="", prof=True): run(flags, prof)
@app.local_entrypoint()
def main(which: str = "t4,l4,a10", flags: str = "", prof: bool = True):
    hs = [{"t4": t4, "l4": l4, "a10": a10}[n].spawn(flags, prof) for n in which.split(",")]
    for h in hs: h.get()
