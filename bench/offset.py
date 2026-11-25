"""Is there a systematic difference between the two engines, beyond rounding noise?

Every token is scored by the fork at several physical batch sizes and by ours at several splits between wide and narrow steps.
Pairwise differences within and between engines are compared, and the part of the between-engine gap that within-engine noise
does not explain is estimated as (mean_f - mean_o)^2 - var_f / n_f - var_o / n_o. The arms of one engine are not independent
draws (many splits give bit-identical logits), so this estimate leans towards finding a difference.
"""
import itertools, json, math, os, subprocess
import modal, quality as Q
from modal_common import VOL
app = modal.App("bonsai25-engine-offset", image=Q.image.pip_install("pyarrow").add_local_python_source("modal_common", "quality").add_local_dir(os.path.join(Q.ROOT, "src"), "/src"))
CH, CTX, FIRST, PER = 40, 512, 256, 255
V = Q.V
NV = 2 * ((V + 1) // 2) + 4
UBS = (512, 256, 200, 128, 100, 64, 32, 16)
SPLITS = (0, 64, 128, 192, 256)

def run():
    import numpy as np
    g = Q.gpu_name(); Q.build()
    open("/tmp/wiki.txt", "w", encoding="utf8").write("".join(r["text"] for r in Q.rows("wikitext")))
    exe = subprocess.check_output("find /opt/prism -name llama-perplexity -type f | head -1", shell=True, text=True).strip()
    env = {**os.environ, "LD_LIBRARY_PATH": os.path.dirname(exe)}
    F = np.zeros((len(UBS), CH, PER)); T = None
    for a, ub in enumerate(UBS):
        r = subprocess.run(f"{exe} -m {Q.WP} -f /tmp/wiki.txt -ngl 99 -fa on -c {CTX} -b {CTX} -ub {ub} --chunks {CH} --kl-divergence-base /tmp/b.bin", shell=True, capture_output=True, text=True, env=env)
        if r.returncode: print(r.stderr[-2000:]); raise SystemExit("perplexity")
        raw = np.memmap("/tmp/b.bin", dtype=np.uint8, mode="r"); off = 20 + CH * CTX * 4
        T = np.frombuffer(bytes(raw[20:off]), dtype=np.int32).reshape(CH, CTX); b = raw[off:].view(np.uint16)
        for c in range(CH):
            for i in range(PER):
                s = (c * PER + i) * NV; sc, mn = np.frombuffer(np.array(b[s:s + 4]).tobytes(), dtype=np.float32)
                F[a, c, i] = float(b[s + 4 + int(T[c, FIRST + i + 1])]) * sc + mn
        del raw, b; os.remove("/tmp/b.bin"); print(f"[{g}] fork ub={ub} ppl {math.exp(-F[a].mean()):.4f}", flush=True)
    e = Q.Ours(CTX + 64, 0)
    def lp(l, y):                                       # same clamp as the saved file
        l = l.astype(np.float64); m = l.max(); return max(l[y], m - 16.0) - (m + math.log(np.exp(l - m).sum()))
    O = np.zeros((len(SPLITS), CH, PER))
    for a, sp in enumerate(SPLITS):
        for c in range(CH):
            t = T[c].tolist(); e.reset(); p = 0
            while p < CTX:
                k = min(128, sp - p) if p < sp else 4
                w = p >= FIRST; lg = e.eval(t[p:p + k], p, w)
                if w:
                    for j in range(k):
                        if p + j - FIRST < PER: O[a, c, p + j - FIRST] = lp(lg[j], t[p + j + 1])
                p += k
        print(f"[{g}] ours wide to {sp} ppl {math.exp(-O[a].mean()):.4f}", flush=True)
    rng = np.random.RandomState(0); sub = [(c, int(i)) for c in range(CH) for i in rng.choice(PER, 10, replace=False)]
    W = np.array([lp(e.last_logits(T[c, :FIRST + i + 1].tolist()), T[c, FIRST + i + 1]) for c, i in sub])   # scored from a wide step
    Fs = np.array([F[:, c, i] for c, i in sub]).T; Os = np.array([O[:, c, i] for c, i in sub]).T
    def stats(A, B, same):
        ps = [(x, y) for x in range(len(A)) for y in range(len(B)) if not same or x < y]
        d = np.array([np.abs(A[x] - B[y]) for x, y in ps])
        return {"pairs": len(ps), "mean_abs": round(float(d.mean()), 5), "median_abs": round(float(np.median(d)), 5),
                "p99_abs": round(float(np.percentile(d, 99)), 4), "rms": round(float(np.sqrt((d ** 2).mean())), 5)}
    f, o = F.reshape(len(UBS), -1), O.reshape(len(SPLITS), -1)
    sysv = (f.mean(0) - o.mean(0)) ** 2 - f.var(0, ddof=1) / len(f) - o.var(0, ddof=1) / len(o)
    nul = []                                            # the same estimate for every balanced split of the fork arms into two halves
    for h in itertools.combinations(range(len(f)), len(f) // 2):
        A, B = f[list(h)], f[[x for x in range(len(f)) if x not in h]]
        nul.append(float(((A.mean(0) - B.mean(0)) ** 2 - A.var(0, ddof=1) / len(A) - B.var(0, ddof=1) / len(B)).mean()))
    def outside(x, rest): return float(((x < rest.min(0)) | (x > rest.max(0))).mean())
    res = {"gpu": g, "tokens": int(f.shape[1]), "fork_fork": stats(f, f, True), "fork_ours": stats(f, o, False), "ours_ours": stats(o, o, True),
           "sys_ms_gap": float(sysv.mean()), "sys_ms_gap_median": float(np.median(sysv)), "null_ms_gap_min_max": [min(nul), max(nul)],
           "mean_signed_gap": float((o.mean(0) - f.mean(0)).mean()),
           "fork_arm_means": [round(float(x), 5) for x in f.mean(1)], "ours_arm_means": [round(float(x), 5) for x in o.mean(1)],
           "outside_range_fork_heldout": [round(outside(f[x], np.delete(f, x, 0)), 4) for x in range(len(f))],
           "outside_range_ours_vs_7_fork": [round(float(np.mean([outside(o[x], np.delete(f, y, 0)) for y in range(len(f))])), 4) for x in range(len(o))],
           "wide_scored_n": len(sub), "wide_vs_fork": stats(Fs, W[None], False), "narrow_vs_fork_same_tokens": stats(Fs, Os, False), "fork_fork_same_tokens": stats(Fs, Fs, True)}
    big = np.argsort(-np.abs(f.mean(0) - o.mean(0)))[:8]
    res["largest"] = [{"tok": [int(x) // PER, FIRST + int(x) % PER], "fork": [round(float(v), 3) for v in f[:, x]], "ours": [round(float(v), 3) for v in o[:, x]]} for x in big]
    print("OFFSET", json.dumps(res), flush=True)

OPT = dict(volumes={"/w": VOL}, timeout=5400, cpu=8, memory=32768)
@app.function(gpu="T4", **OPT)
def t4(): run()
@app.function(gpu="L4", **OPT)
def l4(): run()
@app.local_entrypoint()
def main():
    for h in [t4.spawn(), l4.spawn()]:
        try: h.get()
        except Exception as ex: print("FAILED", repr(ex)[:500])
