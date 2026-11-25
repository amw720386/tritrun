"""Spread of the log-probability at chosen tokens over many batch splits, both engines (chunk, position, next token).

The tokens were picked because their KL between the engines was large, so this says how unstable they are, not how typical.
"""
import json, math, os, subprocess
import modal, quality as Q
from modal_common import VOL
app = modal.App("bonsai25-engine-outlier", image=Q.image.pip_install("pyarrow").add_local_python_source("modal_common", "quality").add_local_dir(os.path.join(Q.ROOT, "src"), "/src"))
PTS = [(18, 434, 15), (1, 303, 654), (25, 479, 9104), (22, 492, 66158)]

def run():
    import numpy as np
    g = Q.gpu_name(); Q.build()
    open("/tmp/wiki.txt", "w", encoding="utf8").write("".join(r["text"] for r in Q.rows("wikitext")))
    exe = subprocess.check_output("find /opt/prism -name llama-perplexity -type f | head -1", shell=True, text=True).strip()
    env = {**os.environ, "LD_LIBRARY_PATH": os.path.dirname(exe)}
    r = subprocess.run(f"{exe} -m {Q.WP} -f /tmp/wiki.txt -ngl 99 -fa on -c 512 -b 512 -ub 512 --chunks 26 --kl-divergence-base /tmp/b.bin", shell=True, capture_output=True, text=True, env=env)
    if r.returncode: print(r.stderr[-2000:]); raise SystemExit("perplexity")
    raw = open("/tmp/b.bin", "rb").read(20 + 26 * 512 * 4)
    T = np.frombuffer(raw[20:], dtype=np.int32).reshape(26, 512)
    P = [(c, p, y, T[c][:p + 1].tolist()) for c, p, y in PTS]
    for c, p, y, _ in P: assert int(T[c][p + 1]) == y, (c, p, int(T[c][p + 1]))
    F = {i: {} for i in range(len(P))}; O = {i: {} for i in range(len(P))}
    for ub in (8, 16, 24, 32, 48, 64, 100, 128, 200, 256, 512):
        sv = Q.Server(f"-b 512 -ub {ub}")
        for i, (c, p, y, t) in enumerate(P):
            j = sv.post("completion", prompt=t, n_predict=1, temperature=0, n_probs=40, cache_prompt=False)
            d = {x["id"]: x["logprob"] for x in j["completion_probabilities"][0]["top_logprobs"]}
            F[i][ub] = round(d.get(y, float("nan")), 3)
        sv.stop()
    e = Q.Ours(576, 0)
    def ours(t, k, tail):                               # steps of k, then single-token steps for the last `tail` tokens
        e.reset(); n = len(t); pos = 0
        while pos < n:
            left = n - pos; s = k if left > tail + k else (left - tail if left > tail else 1)
            s = max(1, min(s, left)); r = e.eval(t[pos:pos + s], pos, pos + s == n); pos += s
        l = r[-1].astype(np.float64); return l - (l.max() + math.log(np.exp(l - l.max()).sum()))
    for i, (c, p, y, t) in enumerate(P):
        for k in (1, 2, 4, 5, 8, 16, 24, 32, 48, 64, 100, 128):
            for tail in (0, 1, 33):
                O[i][f"{k}/{tail}"] = round(float(ours(t, k, tail)[y]), 3)
        for nm, v in (("fork", F[i]), ("ours", O[i])):
            a = np.array(list(v.values())); print("SPREAD", g, (c, p, y), nm, f"n={len(a)} min {a.min():.3f} median {np.median(a):.3f} max {a.max():.3f} sd {a.std():.3f}", json.dumps(v), flush=True)

OPT = dict(volumes={"/w": VOL}, timeout=2400)
@app.function(gpu="T4", **OPT)
def t4(): run()
@app.function(gpu="L4", **OPT)
def l4(): run()
@app.local_entrypoint()
def main():
    for h in [t4.spawn(), l4.spawn()]: h.get()
