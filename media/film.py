"""The explainer at the middle of the launch video, square (media/edit.py cuts it between the bonsai scenes of
media/film_bonsai.py). Rates come from results/release.json (built from the logs by tools/release_numbers.py);
the layer count and block size from src/engine.cu. The trits on screen are drawn uniformly, like the real ones, but are
not the file's, and the token widths on the two pages are illustrative (both pages show the same sequence, as greedy
decoding of one model file does).

    python tools/release_numbers.py
    modal run <toolkit>/film/render_modal.py --film media/film.py --data results/release.json --out <dir> --w 1080 --h 1080 --t1 16.2

 0.0  one block of 128 weights: +1 stands up, 0 lies flat, -1 sinks
 3.4  a read sweeps the block once, and four tokens light together (one token and three drafts)
 6.6  pull back: the block is one of many in a layer, and every step reads all 64 layers
10.6  cut: the same text decoded on one RTX 5090 by both engines, a tile per token at the measured rates, in real time
14.9  the race ends; the counts hold
16.2  (the video cuts back to the bonsai here)
"""
import math

import numpy as np

try:
    import bpy
    import kit
    from mathutils import Vector
except ImportError:          # the overlay runs without Blender
    kit = None

DURATION = 17.5
S = {}
LAYERS, BLOCK = 64, 128          # src/engine.cu: NL = 64, 128-weight trit blocks
COLS, ROWS = 16, 8               # the block's tiles
P = 0.1                          # tile pitch
BX, BY = 12, 8                   # blocks per layer plate (illustrative)
BW, BD = COLS * P + 0.2, ROWS * P + 0.2
GAP = 0.26                       # between layer plates
HERO = (5, 3)                    # the block we start on
SWEEP0, SWEEP1, LIGHT = 3.7, 5.0, 5.05
PULL0, PULL1 = 6.4, 9.4
PASSES = (8.9, 9.75)
PASS = 0.75
CUT = 10.6
RACE0, RACE_LEN = 10.9, 4.0
PAGE_W, LINE_H, TILE_H = 6.0, 0.2, 0.12
RACE_X = 400.0                   # the race stage is far from the tower
PAGE_GAP = 1.6


def rate(data):
    d = {r["gpu"]: r for r in data["decode"]}["RTX 5090"]
    return d["fork_mtp"], d["ours_mtp"]


def hero_origin():
    """World position of the hero block's first tile centre (the top plate's top is z = 0)."""
    x0 = -BX * BW / 2 + HERO[0] * BW + 0.1 + P / 2
    y0 = -BY * BD / 2 + HERO[1] * BD + 0.1 + P / 2
    return np.array([x0, y0, 0.0])


def hero_centre():
    return hero_origin() + np.array([(COLS - 1) * P / 2, (ROWS - 1) * P / 2, 0])


def tiles(trits):
    o = hero_origin()
    c = np.array([[o[0] + (k % COLS) * P, o[1] + (k // COLS) * P, 0.0] for k in range(BLOCK)])
    top = np.where(trits > 0, 0.07, np.where(trits < 0, -0.05, 0.012))
    bot = -0.07
    c[:, 2] = (top + bot) / 2
    s = np.stack([np.full(BLOCK, 0.084), np.full(BLOCK, 0.084), top - bot], 1)
    return c, s


def page_layout(n, seed=7):
    """Token tiles in reading order on a page: (x0, width, row) per token."""
    rng = np.random.default_rng(seed)
    w = rng.choice([0.07, 0.1, 0.13, 0.16, 0.2, 0.26, 0.32], n, p=[.1, .2, .22, .18, .15, .1, .05])
    out, x, row = [], 0.0, 0
    for k in range(n):
        if x + w[k] > PAGE_W:
            x, row = 0.0, row + 1
        out.append((x, w[k], row))
        x += w[k] + 0.05
    return out


def build(assets, data):
    sc = kit.stage(1080, 1080, 64)
    cam = sc.camera
    cam.data.lens = 50
    S["cam"] = cam
    rng = np.random.default_rng(3)
    clay = kit.clay("clay", rough=0.55)
    blue = kit.flat("blue", kit.BLUE, 1.0)
    S["trits"] = trits = rng.integers(0, 3, BLOCK) - 1

    def box(name, c, s, mat, shade):
        v, f = kit.boxes_mesh(c, s)
        o = kit.mesh_object(name, v, f.tolist(), mat)
        o.color = (shade, shade, shade, 1)
        return o

    # the hero block's tiles, one object per trit value, and a blue copy that shows what the read has touched
    c, s = tiles(trits)
    for val, shade in ((1, 0.62), (0, 0.10), (-1, 0.26)):
        m = trits == val
        box(f"t{val}", c[m], s[m], clay, shade)
    S["tc"], S["ts"] = c, s
    S["lit"] = kit.Boxes("lit", BLOCK, blue)
    S["bar"] = kit.Boxes("bar", 1, blue)
    # the block's tray, and the four tokens it feeds
    hc = hero_centre()
    box("tray", [hc + np.array([0, 0, -0.1])], [[COLS * P + 0.06, ROWS * P + 0.06, 0.06]], clay, 0.05)
    S["tok_c"] = [hc + np.array([COLS * P / 2 + 0.75, (k - 1.5) * 0.24, 0.02]) for k in range(4)]
    S["tok_grey"] = kit.Boxes("tokg", 4, clay)
    S["tok_grey"].o.color = (0.22, 0.22, 0.22, 1)
    S["tok_blue"] = kit.Boxes("tokb", 4, blue)
    S["beams"] = kit.Boxes("beams", 4, blue)

    # 64 layer plates of block slabs; the top plate leaves a hole for the hero block
    S["plates"], S["plates_b"] = [], []
    for L in range(LAYERS):
        z = -L * GAP
        cs, ss = [], []
        for i in range(BX):
            for j in range(BY):
                if L == 0 and (i, j) == HERO:
                    continue
                cs.append([-BX * BW / 2 + (i + 0.5) * BW, -BY * BD / 2 + (j + 0.5) * BD, z - 0.05])
                ss.append([BW - 0.14, BD - 0.14, 0.08])
        sh = 0.13 if L % 4 == 3 else 0.085          # every 4th layer is attention
        S["plates"].append(box(f"p{L}", cs, ss, clay, sh))
        b = box(f"pb{L}", cs, [[a, b_, 0.085] for a, b_, _ in ss], blue, 1.0)
        b.hide_render = True
        S["plates_b"].append(b)

    # the race: two pages, the same token sequence on each
    fork, ours = rate(data)
    n = int(ours * RACE_LEN) + 2
    S["layout"] = page_layout(n)
    S["pages"] = {}
    for key, x in (("fork", -PAGE_W / 2 - PAGE_GAP / 2), ("ours", PAGE_W / 2 + PAGE_GAP / 2)):
        ox = RACE_X + x - PAGE_W / 2
        S["pages"][key] = {"ox": ox, "grey": kit.Boxes(f"{key}g", n, clay), "blue": kit.Boxes(f"{key}b", n, blue),
                           "rate": ours if key == "ours" else fork}
        S["pages"][key]["grey"].o.color = (0.55, 0.57, 0.6, 1)
        box(f"{key}panel", [[RACE_X + x, 0.25, 4.0]], [[PAGE_W + 0.6, 0.05, 10.6]], clay, 0.035)

    # light: suns, so the tower and the close-up are lit the same at any distance
    for name, rot, e in (("key", (math.radians(50), 0, math.radians(-38)), 1.4), ("rim", (math.radians(66), 0, math.radians(150)), 1.0),
                         ("fill", (math.radians(75), 0, math.radians(60)), 0.18)):
        L = bpy.data.lights.new(name, "SUN")
        L.energy, L.angle = e, math.radians(6)
        o = bpy.data.objects.new(name, L)
        sc.collection.objects.link(o)
        o.rotation_euler = rot
    return sc


def frame(t):
    cam = S["cam"]
    hc = Vector(hero_centre())
    tc, ts = S["tc"], S["ts"]
    # the read: a bar crosses the block; tiles it has just passed show blue
    u = kit.clamp((t - SWEEP0) / (SWEEP1 - SWEEP0))
    x0, x1 = tc[0, 0] - 0.08, tc[-1, 0] + 0.08
    bx = kit.lerp(x0, x1, kit.ease(u))
    if SWEEP0 <= t <= SWEEP1 + 0.1:
        S["bar"].set([[bx, hc.y, 0.09]], [[0.018, ROWS * P + 0.12, 0.012]])
        near = (tc[:, 0] < bx + 0.01) & (tc[:, 0] > bx - 0.3)
        top = tc[near] + np.stack([np.zeros(near.sum()), np.zeros(near.sum()), ts[near, 2] / 2 + 0.004], 1)
        S["lit"].set(top, np.tile([0.088, 0.088, 0.01], (near.sum(), 1)))
    else:
        S["bar"].set([], [])
        S["lit"].set([], [])
    # four tokens: grey until the read ends, then lit together; beams from the block's edge
    g = kit.ease((t - 3.0) / 0.5) * (1 - kit.ramp(t, PULL1 - 0.6, 0.6))
    on = LIGHT <= t < PULL1 - 0.3
    tok = [list(c) for c in S["tok_c"]]
    sz = [[0.16 * g, 0.16 * g, 0.16 * g]] * 4
    S["tok_grey"].set([] if on else tok, [] if on else sz)
    S["tok_blue"].set(tok if on else [], sz if on else [])
    bk = kit.ease((t - LIGHT + 0.05) / 0.25) * (1 - kit.ramp(t, 6.3, 0.4))
    if bk > 0.01:
        ex = hc.x + COLS * P / 2 + 0.04
        beams, bs = [], []
        for c in S["tok_c"]:
            L_ = (c[0] - 0.08 - ex) * bk
            beams.append([ex + L_ / 2, c[1], 0.02])
            bs.append([L_, 0.012, 0.012])
        S["beams"].set(beams, bs)
    else:
        S["beams"].set([], [])
    # every step reads all 64 layers, top to bottom
    lit = set()
    for p0 in PASSES:
        if p0 <= t < p0 + PASS + 0.12:
            k = (t - p0) / PASS * LAYERS
            lit |= {L for L in range(LAYERS) if k - 6 <= L <= k}
    for L, b in enumerate(S["plates_b"]):
        b.hide_render = L not in lit
    # the race
    for key, pg in S["pages"].items():
        el = kit.clamp(t - RACE0, 0, RACE_LEN)
        n = int(pg["rate"] * el) if t >= CUT else 0
        cg, sg, cb, sb = [], [], [], []
        for k, (x, w, row) in enumerate(S["layout"][:n]):
            c = [pg["ox"] + x + w / 2, 0.2, 9.0 - row * LINE_H]
            s = [w, 0.04, TILE_H]
            if el - k / pg["rate"] < 0.22 and el < RACE_LEN:
                cb.append(c), sb.append(s)
            else:
                cg.append(c), sg.append(s)
        pg["grey"].set(cg, sg)
        pg["blue"].set(cb, sb)
    # camera
    if t < CUT:
        keys = [(0.0, hc + Vector((-0.95, -1.75, 1.55))), (3.3, hc + Vector((-0.1, -2.5, 1.8))), (6.2, hc + Vector((0.5, -3.15, 2.25))),
                (PULL0, hc + Vector((0.55, -3.25, 2.35))), (8.0, Vector((-9, -17, 10))), (PULL1, Vector((-22, -31, 13))), (CUT, Vector((-24, -33, 13.5)))]
        ats = [(0.0, hc + Vector((0, 0.05, -0.12))), (3.3, hc + Vector((0.5, 0.05, -0.15))), (6.2, hc + Vector((0.85, 0.05, -0.2))),
               (PULL0, hc + Vector((0.85, 0.05, -0.2))), (8.0, Vector((0, 0, -5.5))), (PULL1, Vector((0, 0, -10.5))), (CUT, Vector((0, 0, -10.8)))]
        kit.look(cam, kit.path(t, keys), kit.path(t, ats))
    else:
        loc = kit.path(t, [(CUT, (RACE_X - 0.5, -27.5, 4.6)), (DURATION, (RACE_X + 0.3, -26.2, 4.3))])
        kit.look(cam, loc, Vector((RACE_X, 0, 3.7)))
    return {"fork": (S["pages"]["fork"]["ox"] + PAGE_W / 2, 0, 9.75), "ours": (S["pages"]["ours"]["ox"] + PAGE_W / 2, 0, 9.75)}


# --- overlay (runs after rendering, without Blender) ---
def _e(x):
    x = max(0.0, min(1.0, x))
    return x * x * (3 - 2 * x)


def _fade(t, a, b, d=0.35):
    return _e((t - a) / d) * (1 - _e((t - b + d) / d))


CUT, RACE0, RACE_LEN = 10.6, 10.9, 4.0       # repeated: the overlay runs on its own
POSTER = 15.2
CLIPS = {"inside": (3.2, 6.6, 720), "layers": (6.4, 10.5, 720), "against": (10.6, 15.2, 720)}
MUTED = "#9aa5b6"


def overlay(t, pts, data):
    d = {r["gpu"]: r for r in data["decode"]}["RTX 5090"]
    fork, ours = d["fork_mtp"], d["ours_mtp"]
    out = []

    def say(a, b, label, text):
        k = _fade(t, a, b) if a > 0 else 1 - _e((t - b + 0.35) / 0.35)
        out.append({"text": label, "x": 0.06, "y": 0.81, "size": 28, "mono": True, "color": MUTED, "a": k})
        out.append({"text": text, "x": 0.06, "y": 0.88, "size": 64, "a": k})
    say(1.0, 3.35, "EVERY WEIGHT IS −1, 0 OR +1", "27B weights in 6 GB.")
    say(3.5, 6.5, "ONE BLOCK OF 128 WEIGHTS", "Read once, for four tokens.")
    say(6.7, 10.45, "EVERY STEP READS ALL 64 LAYERS", "Four tokens, barely more work.")
    if t >= CUT:
        el = max(0.0, min(RACE_LEN, t - RACE0))
        k = _e((t - CUT) / 0.3)
        for key, name, tps in (("fork", "PrismML's llama.cpp fork", fork), ("ours", "tritrun", ours)):
            if key in pts:
                x, y = pts[key]
                o = key == "ours"
                out.append({"text": name, "x": x, "y": y - 0.05, "size": 30, "weight": "Regular", "anchor": "c", "a": k, "color": "#e8edf4" if o else "#a3afc0"})
                out.append({"text": f"{int(tps * el):,} tokens", "x": x, "y": y - 0.005, "size": 46, "anchor": "c", "a": k, "tnum": True,
                            "color": "#3b82f6" if o and el > 0 else "#e8edf4" if o else "#a3afc0"})
        out.append({"text": f"{el:.2f} s", "x": 0.5, "y": 0.04, "size": 34, "anchor": "c", "a": k, "tnum": True})
        say(CUT, 16.2, "SAME RTX 5090, SAME MODEL FILE", "Both, in real time.")
    return out
