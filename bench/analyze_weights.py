"""Measure trit and scale statistics of the PTQ1_0 file: how many bits are really needed."""
import modal
from modal_common import VOL, GGUF, image
app = modal.App("bonsai25-engine-analyze", image=image.pip_install("gguf").add_local_python_source("modal_common"))

@app.function(volumes={"/w": VOL}, timeout=3000, cpu=8, memory=16384)
def analyze():
    import numpy as np, collections
    from gguf import GGUFReader
    import struct
    # header parse via a light reader (PTQ1_0 type id is fork-private, so do not use GGUFReader tensors)
    import os, time
    for _ in range(120):
        VOL.reload()
        if os.path.exists(f"/w/{GGUF}") and os.path.getsize(f"/w/{GGUF}") > 6.2e9: break
        time.sleep(10)
    f = open(f"/w/{GGUF}", "rb"); b = f.read(60_000_000)
    o = 0
    def rd(fmt):
        nonlocal o
        s = struct.calcsize("<" + fmt); v = struct.unpack_from("<" + fmt, b, o); o += s; return v[0]
    def rs():
        nonlocal o
        n = rd("Q"); s = b[o:o+n]; o += n; return s.decode("utf8", "replace")
    T = {0:"B",1:"b",2:"H",3:"h",4:"I",5:"i",6:"f",7:"?",10:"Q",11:"q",12:"d"}
    def rv(t):
        if t == 8: return rs()
        if t == 9:
            et = rd("I"); n = rd("Q")
            return [rv(et) for _ in range(n)]
        return rd(T[t])
    assert b[:4] == b"GGUF"; o = 4; rd("I"); nt = rd("Q"); nk = rd("Q")
    meta = {}
    for _ in range(nk):
        k = rs(); t = rd("I"); v = rv(t); meta[k] = v
    for k, v in meta.items():
        if not isinstance(v, list): print(k, "=", str(v)[:200])
    infos = []
    for _ in range(nt):
        n = rs(); nd = rd("I"); d = [rd("Q") for _ in range(nd)]; ty = rd("I"); off = rd("Q"); infos.append((n, d, ty, off))
    al = meta.get("general.alignment", 32); data = (o + al - 1)//al*al
    print("tensors", nt, "data at", data)
    for n, d, ty, off in infos:
        if n.startswith(("blk.0.", "blk.3.", "blk.64.", "output", "token")): print(n, d, ty, off)
    by = collections.Counter()
    # PTQ1_0: 28 bytes per 128 weights: 24 B (5 trits) + 2 B (4 trits) + fp16 scale
    pow_ = np.array([1,3,9,27,81])
    tot = collections.Counter()
    for n, d, ty, off in infos:
        ne = int(np.prod(d)); by[ty] += ne
    print("types (id -> elements)", dict(by))
    rng = np.random.default_rng(0)
    H0 = []
    for n, d, ty, off in infos:
        if len(d) != 2 or ty in (0, 1): continue
        ne = int(np.prod(d)); nb = ne//128
        mm = np.memmap(f"/w/{GGUF}", dtype=np.uint8, mode="r", offset=data+off, shape=(nb*28,)).reshape(nb, 28)
        idx = np.sort(rng.choice(nb, min(nb, 200000), replace=False))
        s = np.asarray(mm[idx]); q = s[:, :24]; d16 = s[:, 26:28].copy().view("<f2").ravel()
        # decode 5 trits per byte with the fork's multiply-high convention is (byte*3^k) -- compare distribution only
        hist = np.zeros(3)
        x = q.astype(np.uint16)
        for k in range(5):
            x = x * 3
            hist += np.bincount((x >> 8).ravel().astype(np.int64), minlength=3)[:3]
            x = x & 0xFF
        p = hist/hist.sum(); ent = -(p*np.log2(p+1e-12)).sum()
        uq = np.unique(d16.view("<u2")).size
        tot[n.split(".")[-2] if n.startswith("blk") else n] += 0
        if n.startswith(("blk.0.", "blk.30.", "blk.64.", "output", "token")):
            print(f"{n:32s} {str(d):18s} ty{ty} trit p={np.round(p,3)} H={ent:.3f} scales: uniq {uq} of {d16.size} mean {np.abs(d16).mean():.4f} sd {d16.std():.4f}", flush=True)
        H0.append((n, ent, ne))
    e = sum(h*n for _, h, n in H0)/sum(n for *_, n in H0)
    print("weighted trit entropy bits", e)

@app.local_entrypoint()
def main(): analyze.remote()
