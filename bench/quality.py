"""Quality against the PrismML fork, always with a control arm: the fork at another physical batch size (its own rounding noise).

lm     WikiText-2 test: perplexity, KL divergence and top-1 agreement on identical tokens, from the fork's saved full distributions.
mmlu   random sample of MMLU test questions, zero-shot, scored on the next-token log-probabilities of A/B/C/D.
gsm8k  random sample of GSM8K test questions, greedy generation with MTP drafts, scored on the final number.

Prompts are rendered by the fork's chat template (thinking disabled) and tokenised by the fork, so both engines get the same ids.
"""
import ctypes, json, math, os, re, subprocess, sys, time
import modal
from modal_common import VOL, GGUF, image, gpu_arch
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
app = modal.App("bonsai25-engine-quality", image=image.pip_install("pyarrow").add_local_python_source("modal_common")
                .add_local_dir(os.path.join(ROOT, "src"), "/src"))
W = f"/w/{GGUF}"
WP = "/w/Ternary-Bonsai-2.5-27B-PTQ1_0.gguf"
V = 248320
DATA = {"wikitext": ("Salesforce/wikitext", "wikitext-2-raw-v1/test-00000-of-00001.parquet"),
        "mmlu": ("cais/mmlu", "all/test-00000-of-00001.parquet"),
        "gsm8k": ("openai/gsm8k", "main/test-00000-of-00001.parquet")}

def gpu_name():
    return subprocess.check_output("nvidia-smi --query-gpu=name --format=csv,noheader", shell=True, text=True).strip()

def build():
    r = subprocess.run(f"nvcc -O3 -std=c++17 -arch={gpu_arch()} --use_fast_math -shared -Xcompiler -fPIC -o /tmp/l.so /src/engine.cu", shell=True, capture_output=True, text=True)
    if r.returncode: print(r.stderr[-3000:]); raise SystemExit("build")

def rows(name):
    from huggingface_hub import hf_hub_download
    import pyarrow.parquet as pq
    return pq.read_table(hf_hub_download(DATA[name][0], DATA[name][1], repo_type="dataset")).to_pylist()

def sample(xs, n):
    import random
    i = list(range(len(xs))); random.Random(0).shuffle(i)
    return [xs[j] for j in i[:n]]

class Ours:
    def __init__(self, ctx, mtp):
        L = self.L = ctypes.CDLL("/tmp/l.so")
        L.tr_load.restype = ctypes.c_void_p; L.tr_load.argtypes = [ctypes.c_char_p, ctypes.c_int, ctypes.c_int]
        L.tr_eval.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_int), ctypes.c_int, ctypes.c_int, ctypes.POINTER(ctypes.c_float)]
        L.tr_reset.argtypes = [ctypes.c_void_p]
        L.tr_generate.restype = ctypes.c_int
        L.tr_generate.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_int), ctypes.c_int, ctypes.c_int, ctypes.POINTER(ctypes.c_int), ctypes.c_int, ctypes.c_int,
                                  ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_double)]
        self.e = L.tr_load(W.encode(), ctx, mtp); self.buf = (ctypes.c_float * (V * 4))()
    def reset(self): self.L.tr_reset(self.e)
    def eval(self, toks, pos, want):
        """logits of the last token (steps wider than 4) or of every token"""
        import numpy as np
        k = len(toks)
        self.L.tr_eval(self.e, (ctypes.c_int * k)(*toks), k, pos, self.buf if want else None)
        if want: return np.frombuffer(self.buf, dtype=np.float32)[: V * (1 if k > 4 else k)].reshape(-1, V)
    def last_logits(self, toks):
        self.reset(); pos = 0; n = len(toks)
        while pos < n:
            k = min(128, n - pos); r = self.eval(toks[pos:pos + k], pos, pos + k == n); pos += k
        return r[-1]
    def generate(self, toks, max_new, eos, drafts):
        out = (ctypes.c_int * (max_new + 8))(); st = (ctypes.c_double * 8)()
        n = self.L.tr_generate(self.e, (ctypes.c_int * len(toks))(*toks), len(toks), max_new, (ctypes.c_int * len(eos))(*eos), len(eos), drafts, out, st)
        return list(out[:n])

class Server:
    def __init__(self, extra="", model=WP):
        ls = subprocess.check_output("find /opt/prism -name llama-server -type f | head -1", shell=True, text=True).strip()
        self.env = {**os.environ, "LD_LIBRARY_PATH": os.path.dirname(ls)}
        self.p = subprocess.Popen(f"{ls} -m {model} -ngl 99 -fa on -c 4096 -np 1 --port 8099 --cache-ram 0 --jinja {extra} > /tmp/s.log 2>&1", shell=True, env=self.env)
        import requests
        for _ in range(600):
            try:
                if requests.get("http://127.0.0.1:8099/health", timeout=2).status_code == 200: return
            except Exception: pass
            time.sleep(0.5)
        print(open("/tmp/s.log").read()[-3000:]); raise SystemExit("server did not start")
    def post(self, path, **j):
        import requests
        r = requests.post(f"http://127.0.0.1:8099/{path}", json=j, timeout=900); r.raise_for_status(); return r.json()
    def chat_ids(self, user):
        t = self.post("apply-template", messages=[{"role": "user", "content": user}], chat_template_kwargs={"enable_thinking": False})["prompt"]
        return t, self.post("tokenize", content=t, add_special=False, parse_special=True)["tokens"]
    def stop(self):
        subprocess.run("pkill -9 llama-server", shell=True); self.p.wait(); time.sleep(3)

# ------------------------------------------------------------------------------------------------ perplexity and KL
def lm(chunks=40, n_ctx=512):
    import numpy as np
    g = gpu_name(); build()
    text = "".join(r["text"] for r in rows("wikitext"))
    open("/tmp/wiki.txt", "w", encoding="utf8").write(text)
    exe = subprocess.check_output("find /opt/prism -name llama-perplexity -type f | head -1", shell=True, text=True).strip()
    env = {**os.environ, "LD_LIBRARY_PATH": os.path.dirname(exe)}
    printed = {}
    for ub in (512, 64):
        r = subprocess.run(f"{exe} -m {WP} -f /tmp/wiki.txt -ngl 99 -fa on -c {n_ctx} -b {n_ctx} -ub {ub} --chunks {chunks} --kl-divergence-base /tmp/b{ub}.bin", shell=True, capture_output=True, text=True, env=env)
        print(f"[{g}] fork ub={ub}:", [l for l in (r.stdout + r.stderr).splitlines() if "Final estimate" in l or "error" in l.lower()][-3:], flush=True)
        if r.returncode: print(r.stderr[-2000:]); raise SystemExit("perplexity")
        printed["fork" if ub == 512 else "fork64"] = float(re.findall(r"Final estimate: PPL = ([\d.]+)", r.stdout + r.stderr)[-1])
    first, per = n_ctx // 2, n_ctx - 1 - n_ctx // 2
    nv = 2 * ((V + 1) // 2) + 4
    def open_base(path):
        raw = np.memmap(path, dtype=np.uint8, mode="r")
        assert bytes(raw[:8]) == b"_logits_", bytes(raw[:8])
        nc, nvoc, nch = np.frombuffer(bytes(raw[8:20]), dtype=np.int32)
        assert (nc, nvoc, nch) == (n_ctx, V, chunks), (nc, nvoc, nch)
        off = 20 + int(nch) * int(nc) * 4
        assert len(raw) == off + int(nch) * per * nv * 2, (len(raw), off, per, nv)
        return np.frombuffer(bytes(raw[20:off]), dtype=np.int32).reshape(nch, nc), raw[off:].view(np.uint16) if off % 2 == 0 else None
    toks, b0 = open_base("/tmp/b512.bin"); t1, b1 = open_base("/tmp/b64.bin")
    assert (toks == t1).all()
    def base_row(b, i):
        r = np.array(b[i * nv:(i + 1) * nv])
        sc, mn = np.frombuffer(r[:4].tobytes(), dtype=np.float32)
        return r[4:4 + V].astype(np.float32) * sc + mn
    def clamp(l):                                       # the saved file clamps logits to max - 16 after normalising: same here
        l = l.astype(np.float32); m = float(l.max())
        z = m + math.log(float(np.exp((l - m).astype(np.float64)).sum()))
        return np.maximum(l, m - 16.0) - z, float(z)
    def kl(a, b):                                       # generalised KL(a || b), non-negative although the clamped rows sum to slightly more than 1
        p = np.exp(a.astype(np.float64)); return float((p * (a - b).astype(np.float64) - p + np.exp(b.astype(np.float64))).sum())
    exact = np.zeros((chunks, per))                     # ours without the clamp, comparable with the PPL the fork prints
    e = Ours(n_ctx + 64, 0)
    nll = {k: np.zeros((chunks, per)) for k in ("fork", "fork64", "ours")}
    kls = {k: np.zeros((chunks, per)) for k in ("fork64", "ours")}
    top = {k: np.zeros((chunks, per)) for k in ("fork64", "ours")}
    for c in range(chunks):
        t = toks[c].tolist(); e.reset()
        for p in range(0, first, 128): e.eval(t[p:p + min(128, first - p)], p, False)
        for p in range(first, n_ctx, 4):
            lg = e.eval(t[p:p + 4], p, True)
            for j in range(4):
                i = p + j - first
                if i >= per: break
                row = c * per + i; y = t[p + j + 1]
                o, z = clamp(lg[j]); f = base_row(b0, row); f6 = base_row(b1, row)
                nll["ours"][c, i] = -float(o[y]); exact[c, i] = z - float(lg[j][y]); nll["fork"][c, i] = -f[y]; nll["fork64"][c, i] = -f6[y]
                for k, q in (("ours", o), ("fork64", f6)):
                    kls[k][c, i] = kl(f, q); top[k][c, i] = int(q.argmax()) == int(f.argmax())
        if c % 10 == 9: print(f"[{g}] chunk {c + 1}: ppl so far", {k: round(math.exp(v[:c + 1].mean()), 4) for k, v in nll.items()}, flush=True)
    res = {"gpu": g, "tokens": chunks * per, "ppl_exact": {**printed, "ours": math.exp(exact.mean())}, "ppl_clamped": {k: math.exp(v.mean()) for k, v in nll.items()}}
    for k in ("ours", "fork64"):
        d = (nll[k] - nll["fork"]).mean(axis=1)         # per chunk: tokens of a chunk are correlated
        res[k + "_vs_fork"] = {"nll_diff": float(d.mean()), "nll_diff_se": float(d.std(ddof=1) / math.sqrt(chunks)), "kl_mean": float(kls[k].mean()), "kl_median": float(np.median(kls[k])),
                               "kl_p99": float(np.percentile(kls[k], 99)), "kl_max": float(kls[k].max()), "top1_agree": float(top[k].mean())}
    # the largest KL values: is it the same token in the control arm, and does one-token evaluation from scratch reproduce ours?
    def top3(l): j = np.argsort(-l)[:3]; return [(int(a), round(float(l[a]), 3)) for a in j]
    cand = set(np.argsort(-kls["ours"].ravel())[:4].tolist()) | set(np.argsort(-kls["fork64"].ravel())[:2].tolist())
    for row in sorted(cand, key=lambda r: -kls["ours"].ravel()[r]):
        c, i = divmod(int(row), per); t = toks[c].tolist(); end = first + i + 1
        f, f6 = base_row(b0, row), base_row(b1, row)
        e.reset()
        for p in range(end): lg = e.eval(t[p:p + 1], p, p == end - 1)
        one, _ = clamp(lg[0]); wide, _ = clamp(e.last_logits(t[:end]))
        print("OUTLIER", json.dumps({"gpu": g, "chunk": c, "pos": first + i, "next_tok": int(t[end]), "kl_ours": round(float(kls["ours"][c, i]), 4), "kl_fork64": round(float(kls["fork64"][c, i]), 4),
              "kl_ours_1tok": round(kl(f, one), 4), "kl_ours_all_wide": round(kl(f, wide), 4), "neighbour_kl_ours": [round(float(x), 5) for x in kls["ours"][c, max(0, i - 3):i + 4]],
              "fork": top3(f), "fork64": top3(f6), "ours_1tok": top3(one), "ours_wide": top3(wide), "last_ids": t[end - 6:end]}), flush=True)
    for k in ("ours", "fork64"): res[k + "_vs_fork"]["kl_mean_without_max"] = float((kls[k].sum() - kls[k].max()) / (kls[k].size - 1))
    print("LM", json.dumps(res), flush=True)
    return res

# ------------------------------------------------------------------------------------------------ multiple choice
def mmlu(n=1000):
    import numpy as np
    g = gpu_name(); build()
    qs = sample(rows("mmlu"), n)
    sv = Server()
    ids = []
    for i, q in enumerate(qs):
        u = q["question"].strip() + "\n\n" + "\n".join(f"{l}. {c}" for l, c in zip("ABCD", q["choices"])) + "\n\nAnswer with the letter of the correct option only."
        t, x = sv.chat_ids(u)
        if i == 0: print("prompt example:", json.dumps(t), flush=True)
        ids.append(x)
    letters = [sv.post("tokenize", content=l, add_special=False)["tokens"] for l in "ABCD"]
    assert all(len(l) == 1 for l in letters); letters = [l[0] for l in letters]
    def fork_arm(s):
        out = []
        for x in ids:
            j = s.post("completion", prompt=x, n_predict=1, temperature=0, n_probs=200, cache_prompt=False)
            m = {t["id"]: t["logprob"] for t in j["completion_probabilities"][0]["top_logprobs"]}
            out.append([m.get(l, -99.0) for l in letters])
        return out
    lp = {"fork": fork_arm(sv)}; sv.stop()
    for ub in (64, 16):
        s2 = Server(f"-ub {ub}"); lp[f"fork{ub}"] = fork_arm(s2); s2.stop()
    e = Ours(4096, 0); o = []
    for x in ids:
        l = e.last_logits(x).astype(np.float64); l = l - l.max(); l = l - math.log(np.exp(l).sum()); o.append([float(l[t]) for t in letters])
    lp["ours"] = o
    gold = [q["answer"] for q in qs]
    ok = {k: [int(int(np.argmax(v)) == a) for v, a in zip(x, gold)] for k, x in lp.items()}
    res = {"gpu": g, "n": n, "prompt_tokens_mean": float(np.mean([len(x) for x in ids])), "prompt_tokens_max": max(len(x) for x in ids), "correct": {k: sum(v) for k, v in ok.items()}}
    for k in ("ours", "fork64", "fork16"):
        a, b = np.array(ok["fork"]), np.array(ok[k])
        dd = [abs(p - q) for u, v in zip(lp["fork"], lp[k]) for p, q in zip(u, v) if p > -20 and q > -20]
        # a prompt that fits in one physical batch is evaluated identically by both fork arms, so also split by prompt length
        by = {}
        for lo, hi in ((0, 64), (64, 128), (128, 9999)):
            z = [abs(p - q) for x, u, v in zip(ids, lp["fork"], lp[k]) if lo < len(x) <= hi for p, q in zip(u, v) if p > -20 and q > -20]
            by[f"{lo + 1}-{hi}"] = [len(z) // 4, float(np.mean(z)), float(np.median(z))]
        res[k + "_vs_fork"] = {"same_choice": int(sum(int(np.argmax(u)) == int(np.argmax(v)) for u, v in zip(lp["fork"], lp[k]))), "fork_only_correct": int(((a == 1) & (b == 0)).sum()),
                               "arm_only_correct": int(((a == 0) & (b == 1)).sum()), "mean_abs_dlogprob": float(np.mean(dd)), "by_prompt_len_n_mean_median": by}
    print("MMLU", json.dumps(res), flush=True)
    return res

# ------------------------------------------------------------------------------------------------ generation
NUM = r"-?\$?\d[\d,]*(?:\.\d+)?"
def final_number(t):
    m = re.findall(r"[Aa]nswer:?\**:?\s*\**\s*(" + NUM + ")", t) or re.findall(NUM, t)
    if not m: return None
    try: return float(m[-1].replace(",", "").replace("$", ""))
    except ValueError: return None

def gsm8k(n=300, max_new=2048):
    import numpy as np
    g = gpu_name(); build()
    qs = sample(rows("gsm8k"), n)
    spec = "--spec-type draft-mtp --spec-draft-n-min 0 --spec-draft-n-max 3 --spec-draft-p-min 0"
    sv = Server(spec, W)
    ids = []
    for i, q in enumerate(qs):
        t, x = sv.chat_ids(q["question"].strip() + "\n\nSolve the problem step by step, then finish with a line of the form \"Answer: <number>\".")
        if i == 0: print("prompt example:", json.dumps(t), flush=True)
        ids.append(x)
    eos = [sv.post("tokenize", content=s, add_special=False, parse_special=True)["tokens"] for s in ("<|im_end|>", "<|endoftext|>")]
    assert all(len(x) == 1 for x in eos); eos = [x[0] for x in eos]
    def fork_arm(s):
        out = []
        for x in ids:
            j = s.post("completion", prompt=x, n_predict=max_new, temperature=0, cache_prompt=False, return_tokens=True)
            out.append((j["content"], [t for t in j["tokens"] if t not in eos], j.get("stop_type")))
        return out
    arms = {"fork": fork_arm(sv)}
    print("fork done", flush=True)
    e = Ours(4096, 1); o = []
    for x in ids:
        t = e.generate(x, max_new, eos, 3); stop = "eos" if t and t[-1] in eos else "limit"
        t = [z for z in t if z not in eos]
        o.append((sv.post("detokenize", tokens=t)["content"], t, stop))
    arms["ours"] = o; sv.stop()
    print("ours done", flush=True)
    s2 = Server(spec + " -ub 64", W); arms["fork64"] = fork_arm(s2); s2.stop()
    gold = [float(q["answer"].split("####")[-1].strip().replace(",", "")) for q in qs]
    pred = {k: [final_number(t) for t, _, _ in v] for k, v in arms.items()}
    ok = {k: [int(p is not None and abs(p - a) < 1e-6) for p, a in zip(v, gold)] for k, v in pred.items()}
    print("sample output (ours):", json.dumps(o[0][0][-400:]), "gold", gold[0], flush=True)
    res = {"gpu": g, "n": n, "correct": {k: sum(v) for k, v in ok.items()}, "hit_length_limit": {k: sum(s == "limit" for _, _, s in v) for k, v in arms.items()},
           "no_answer_line": {k: sum(not re.search(r"[Aa]nswer:", t) for t, _, _ in v) for k, v in arms.items()}, "mean_new_tokens": {k: float(np.mean([len(t) for _, t, _ in v])) for k, v in arms.items()}}
    for k in ("ours", "fork64"):
        a, b = np.array(ok["fork"]), np.array(ok[k])
        res[k + "_vs_fork"] = {"identical_output": sum(x[1] == y[1] for x, y in zip(arms["fork"], arms[k])), "same_final_number": sum(p == q for p, q in zip(pred["fork"], pred[k])),
                               "fork_only_correct": int(((a == 1) & (b == 0)).sum()), "arm_only_correct": int(((a == 0) & (b == 1)).sum())}
    res["wrong_in_ours_only"] = [{"gold": gold[i], "fork": pred["fork"][i], "ours": pred["ours"][i], "ours_tail": o[i][0][-160:]} for i in range(n) if ok["fork"][i] and not ok["ours"][i]]
    print("GSM8K", json.dumps(res), flush=True)
    return res

RUN = {"lm": lm, "mmlu": mmlu, "gsm8k": gsm8k}
def go(what, n): return RUN[what](n) if n else RUN[what]()
OPT = dict(volumes={"/w": VOL}, timeout=7200, cpu=8, memory=32768)
@app.function(gpu="T4", **OPT)
def t4(what, n): return go(what, n)
@app.function(gpu="L4", **OPT)
def l4(what, n): return go(what, n)
@app.function(gpu="A10", **OPT)
def a10(what, n): return go(what, n)

@app.local_entrypoint()
def main(jobs: str = "lm:t4,lm:l4,lm:a10", n: int = 0):
    hs = [(j, {"t4": t4, "l4": l4, "a10": a10}[j.split(":")[1]].spawn(j.split(":")[0], n)) for j in jobs.split(",")]
    for j, h in hs:
        try: h.get()
        except Exception as ex: print(j, "FAILED", repr(ex)[:500])
