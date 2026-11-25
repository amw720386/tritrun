"""End-to-end on a GPU: build the shared library, check the tokenizer against the fork, run the CLI path, compare plain vs MTP text."""
import ctypes, json, os, subprocess, sys
import modal
from modal_common import VOL, GGUF, image
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
app = modal.App("bonsai25-engine-e2e", image=image.pip_install("tokenizers").add_local_python_source("modal_common")
                .add_local_dir(os.path.join(ROOT, "src"), "/src").add_local_dir(os.path.join(ROOT, "tritrun"), "/opt/tritrun_pkg/tritrun"))
W = f"/w/{GGUF}"
TEXTS = ["Hello, world! 12345 don't 日本語テスト", "def f(x):\n    return x ** 2  # square\n", "   spaces\tand\nnewlines\n\n  x=1;y=2", "The year 2026, I'm thinking…",
         "emoji 😀 and accents éàü ½ — “quotes”", "<|im_start|>user\nhi<|im_end|>\n", "a" * 300]

@app.function(gpu="T4", volumes={"/w": VOL}, timeout=1800, cpu=8, memory=24576)
def e2e():
    sys.path.insert(0, "/opt/tritrun_pkg")
    r = subprocess.run("nvcc -O3 -std=c++17 -arch=sm_75 --use_fast_math -shared -Xcompiler -fPIC -o /opt/tritrun_pkg/tritrun/libtritrun.so /src/engine.cu", shell=True, capture_output=True, text=True)
    print("build", r.returncode, r.stderr[-800:])
    from tritrun import Tokenizer
    tok = Tokenizer(W)
    ls = subprocess.check_output("find /opt/prism -name llama-tokenize -type f | head -1", shell=True, text=True).strip()
    bad = 0
    for t in TEXTS:
        open("/tmp/t.txt", "w").write(t)
        o = subprocess.run(f"LD_LIBRARY_PATH=$(dirname {ls}) {ls} -m {W} -f /tmp/t.txt --ids --no-bos 2>/dev/null", shell=True, capture_output=True, text=True).stdout
        ref = json.loads(o.strip().splitlines()[-1]); mine = tok.encode(t)
        ok = ref == mine; bad += (not ok)
        print("tok", "OK " if ok else "DIFF", repr(t[:40]), "" if ok else (ref[:12], mine[:12]), flush=True)
    print("tokenizer mismatches:", bad, "of", len(TEXTS), " eos", tok.eos_id)
    import io
    for d in (0, 3):
        p = subprocess.run([sys.executable, "-m", "tritrun.cli", W, "Write a haiku about autumn, then explain it in one sentence.", "--max-new", "120", "--drafts", str(d)],
                           capture_output=True, text=True, env={**os.environ, "PYTHONPATH": "/opt/tritrun_pkg", "PYTHONIOENCODING": "utf-8"})
        print(f"--- CLI drafts={d} rc={p.returncode}\n{p.stdout}\n{p.stderr[-500:]}", flush=True)
@app.local_entrypoint()
def main(): e2e.remote()
