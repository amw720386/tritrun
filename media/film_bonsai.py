"""The bonsai scenes that open and close the launch video (media/edit.py cuts them around media/film.py). Rates come
from results/release.json; the layer count and block size from src/engine.cu. The trits are drawn uniformly, like the
real ones, but are not the file's.

    modal run <toolkit>/film/render_modal.py --film media/film_bonsai.py --data results/release.json --out <dir> --w 1080 --h 1080 --t1 6.4
    modal run <toolkit>/film/render_modal.py --film media/film_bonsai.py --data results/release.json --out <dir> --w 1080 --h 1080 --t0 20.4 --t1 26.5

 0.0  one bonsai in a shallow pot, the title; tokens open on it as blossoms, one per step
 4.6  fly into one twig: a block of 128 leaves, each -1 (down), 0 (absent) or +1 (up)   [the video cuts to film.py here]
15.0  two bonsai on one RTX 5090, a blossom per token at the measured rates, in real time, for 5 s
21.4  the two crowns after the same 5 s; what ours made beyond the fork's count turns blue
"""
import math

import numpy as np

try:
    import kit
    from mathutils import Vector
except ImportError:          # the overlay runs without Blender
    kit = None

DURATION = 27.0
S = {}
LAYERS, BLOCK = 64, 128          # src/engine.cu: NL = 64 (every 4th layer is attention), 128-weight trit blocks
TWIGS = 7                        # twigs per branch pad (illustrative)
SLOTS = 1800                     # blossom places on a crown
DRAFTS = 3
SWEEP0, SWEEP1 = 6.45, 7.75      # the twig is read
BUDS, PRUNE = 7.95, 9.15         # four buds open together; one draft is rejected
PASSES = (12.0, 13.55)          # two steps climb the 64 layers
PASS = 1.2
CUT = 15.0
RACE0, RACE_LEN = 15.7, 5.0
END = 21.4
RACE_X = 1.42                    # fork on the left, ours on the right


def rate(data):
    d = {r["gpu"]: r for r in data["decode"]}["RTX 5090"]
    return d["fork_mtp"], d["ours_mtp"]


# --- geometry -------------------------------------------------------------------------------------------------------
def spline(C, n):
    C = np.asarray(C, float)
    P = np.vstack([2 * C[0] - C[1], C, 2 * C[-1] - C[-2]])
    out = []
    for u in np.linspace(0, len(C) - 1, n):
        i = min(int(u), len(C) - 2)
        f = u - i
        p0, p1, p2, p3 = P[i], P[i + 1], P[i + 2], P[i + 3]
        out.append(0.5 * ((2 * p1) + (-p0 + p2) * f + (2 * p0 - 5 * p1 + 4 * p2 - p3) * f * f + (-p0 + 3 * p1 - 3 * p2 + p3) * f ** 3))
    return np.array(out)


def unit(v):
    return v / np.maximum(np.linalg.norm(v, axis=-1, keepdims=True), 1e-9)


def tube(P, R, ring=10):
    P, R = np.asarray(P, float), np.asarray(R, float)
    n = len(P)
    T = unit(np.gradient(P, axis=0))
    a = np.array([0, 0, 1.0]) if abs(T[0, 2]) < 0.9 else np.array([1.0, 0, 0])
    N = unit(np.cross(T[0], a))
    Ns = []
    for i in range(n):
        N = unit(N - T[i] * np.dot(N, T[i]))
        Ns.append(N)
    Ns = np.array(Ns)
    B = np.cross(T, Ns)
    th = np.linspace(0, 2 * np.pi, ring, endpoint=False)
    V = P[:, None] + R[:, None, None] * (np.cos(th)[None, :, None] * Ns[:, None] + np.sin(th)[None, :, None] * B[:, None])
    F = [[i * ring + j, i * ring + (j + 1) % ring, (i + 1) * ring + (j + 1) % ring, (i + 1) * ring + j] for i in range(n - 1) for j in range(ring)]
    F.append(list(range((n - 1) * ring, n * ring)))
    return V.reshape(-1, 3), F


LEAF_F = np.array([[0, 2, 4], [0, 4, 3], [0, 3, 5], [0, 5, 2], [1, 4, 2], [1, 3, 4], [1, 5, 3], [1, 2, 5]])


def leaves(p, d, n, L, W):
    """Needle leaves (6 verts each): base p, direction d, face normal n, length L, width W."""
    u = unit(np.cross(n, d))
    m = p + 0.42 * L[:, None] * d
    V = np.stack([p, p + L[:, None] * d, m + 0.5 * W[:, None] * u, m - 0.5 * W[:, None] * u,
                  m + 0.18 * W[:, None] * n, m - 0.18 * W[:, None] * n], 1)
    return V


class Mesh:
    def __init__(self):
        self.V, self.F, self.n = [], [], 0

    def add(self, V, F):
        V = np.asarray(V, float).reshape(-1, 3)
        self.V.append(V)
        self.F += (np.asarray(F) + self.n).tolist() if isinstance(F, np.ndarray) else [[x + self.n for x in f] for f in F]
        self.n += len(V)

    def make(self, name, mat, smooth=False):
        return kit.mesh_object(name, np.vstack(self.V), self.F, mat, smooth)


def twig(rng, start, d, length, trits=None):
    """One block: a twig with 128 leaf places. Returns (stem verts/faces, leaf verts (128,6,3), trits, curve, normals)."""
    u = (np.arange(BLOCK) + 0.5) / BLOCK
    bend = rng.uniform(-0.12, 0.12)
    side = unit(np.cross(d, [0, 0, 1.0]))
    C = start[None] + length * (u[:, None] * d[None] + (0.10 * u * (1 - u))[:, None] * np.array([0, 0, 1.0]) + (bend * u * u)[:, None] * side)
    T = unit(np.gradient(C, axis=0))
    N = unit(np.array([0, 0, 1.0]) - T * T[:, 2:3])
    B = np.cross(T, N)
    s = np.where(np.arange(BLOCK) % 2 == 0, 1.0, -1.0)
    if trits is None:
        trits = rng.integers(0, 3, BLOCK) - 1          # uniform over -1, 0, +1, as in the file
    fw, tilt = math.radians(50), np.radians(38) * trits
    dd = math.cos(fw) * T + math.sin(fw) * s[:, None] * B
    dl = np.cos(tilt)[:, None] * dd + np.sin(tilt)[:, None] * N
    nl = unit(np.cross(dl, T)) * s[:, None]
    L = np.where(trits == 0, 0.035, 0.24) * length * rng.uniform(0.9, 1.1, BLOCK)
    W = np.where(trits == 0, 0.016, 0.028) * length * np.ones(BLOCK)
    Lv = leaves(C, dl, nl, L, W)
    sv, sf = tube(C[::8].tolist() + [C[-1]], np.linspace(0.016, 0.006, len(C[::8]) + 1) * length, 6)
    return sv, sf, Lv, trits, C, N


def build_tree(rng):
    """The tree, in world coordinates around the origin. Pot top at z = 0.24."""
    tr = {}
    Ct = [(0, 0, 0.18), (0.13, 0.03, 0.55), (-0.13, -0.02, 1.0), (0.07, 0.03, 1.42), (-0.03, 0.0, 1.8), (0.02, 0, 2.08)]
    P = spline(Ct, 70)
    s = np.linspace(0, 1, len(P))
    R = 0.13 * (1 - s) ** 1.15 + 0.016 + 0.09 * np.clip(1 - s / 0.07, 0, 1) ** 2
    tr["trunk"] = tube(P, R, 16)
    tr["trunk_at"] = lambda f: P[min(len(P) - 1, int(f * (len(P) - 1)))]
    tr["branches"], tr["pads"] = [], []
    for i in range(LAYERS):
        f = i / (LAYERS - 1)
        base = tr["trunk_at"](0.26 + 0.70 * f)
        a = i * 2.39996 + 0.9
        h = np.array([math.cos(a), math.sin(a), 0.0])
        L = (0.74 - 0.58 * f ** 0.85) * rng.uniform(0.88, 1.1)
        rise = rng.uniform(0.10, 0.28)
        C = [base, base + 0.34 * L * h + [0, 0, 0.05 * L], base + 0.68 * L * h + [0, 0, rise * 0.4 * L], base + L * h + [0, 0, rise * L]]
        Pb = spline(C, 14)
        Rb = np.linspace(0.034 - 0.018 * f, 0.008, len(Pb))
        tr["branches"].append(tube(Pb, Rb, 8))
        E = Pb[-1]
        rp = 0.34 - 0.19 * f
        tw = []
        for j in range(TWIGS):
            phi = a + (j / (TWIGS - 1) - 0.5) * 2.7 + rng.uniform(-0.12, 0.12)
            d = unit(np.array([math.cos(phi), math.sin(phi), rng.uniform(0.04, 0.22)]))
            st = E - 0.03 * h + [0, 0, rng.uniform(-0.03, 0.03)]
            ln = rp * rng.uniform(0.85, 1.12) * (1 - 0.18 * abs(j / (TWIGS - 1) - 0.5) * 2)
            tw.append((st, d, ln))
        tr["pads"].append({"E": E, "h": h, "a": a, "rp": rp, "twigs": tw, "attn": (i + 1) % 4 == 0})
    return tr


def pick_hero(tr):
    """A low branch facing the camera (-y); its twig that runs most across the view."""
    best = min(range(3, 16), key=lambda i: tr["pads"][i]["E"][1] + 0.4 * abs(tr["pads"][i]["E"][0]))
    pad = tr["pads"][best]
    j = min(range(TWIGS), key=lambda j: abs(pad["twigs"][j][1][1]) - 0.3 * pad["twigs"][j][1][0])
    return best, j


SPH_V, SPH_F = None, None


def sphere(seg=10, rings=6):
    V = [(0, 0, -1.0)]
    for r in range(1, rings):
        ph = -math.pi / 2 + math.pi * r / rings
        for k in range(seg):
            th = 2 * math.pi * k / seg
            V.append((math.cos(ph) * math.cos(th), math.cos(ph) * math.sin(th), math.sin(ph)))
    V.append((0, 0, 1.0))
    F = []
    top = len(V) - 1
    for k in range(seg):
        F.append([0, 1 + (k + 1) % seg, 1 + k])
    for r in range(rings - 2):
        for k in range(seg):
            a, b = 1 + r * seg + k, 1 + r * seg + (k + 1) % seg
            F.append([a, b, b + seg, a + seg])
    for k in range(seg):
        a, b = 1 + (rings - 2) * seg + k, 1 + (rings - 2) * seg + (k + 1) % seg
        F.append([a, b, top])
    return np.array(V) * [1, 1, 0.8], F


class Inst:
    """n copies of a small mesh, moved every frame; unused copies collapse out of view."""

    def __init__(self, name, bv, bf, n, mat, smooth=True):
        self.bv, self.n, self.nb = np.asarray(bv, float), n, len(bv)
        F = [[x + c * self.nb for x in f] for c in range(n) for f in bf]
        self.o = kit.mesh_object(name, np.zeros((n * self.nb, 3)), F, mat, smooth)

    def set(self, c, s):
        V = np.full((self.n, self.nb, 3), -400.0)
        c = np.asarray(c, float).reshape(-1, 3)
        s = np.broadcast_to(np.asarray(s, float), (len(c),))
        k = min(len(c), self.n)
        if k:
            V[:k] = self.bv[None] * s[:k, None, None] + c[:k, None]
        self.o.data.vertices.foreach_set("co", V.astype("f4").ravel())
        self.o.data.update()


def build(assets, data):
    import bpy
    sc = kit.stage()
    S.update(sc=sc, cam=sc.camera, data=data)
    rng = np.random.default_rng(27)
    clay, blue = kit.clay("clay", rough=0.62), kit.flat("blue", kit.BLUE, 1.0)
    cclay = kit.clay("cclay", rough=0.62, clip=True)     # the crown is cut open in front of the twig
    S["blue"], S["cclay"] = blue, cclay
    grey_flash = kit.flat("greyflash", (0.42, 0.44, 0.47, 1), 1.0)
    tr = build_tree(rng)
    hero, hj = pick_hero(tr)
    S["hero"] = hero

    def obj(name, mesh, color, smooth=False, mat=clay):
        if isinstance(mesh, Mesh):
            o = mesh.make(name, mat, smooth)
        else:
            o = kit.mesh_object(name, mesh[0], mesh[1], mat, smooth)
        o.color = color
        return o
    BARK, LEAF, LEAF_A, POT = (.115, .112, .108, 1), (.19, .195, .20, 1), (.25, .255, .26, 1), (.06, .061, .064, 1)
    tree = {"trunk": obj("trunk", tr["trunk"], BARK, True), "branch": [], "pad": [], "blue": []}
    # the pot: a shallow tapered tray with a lip and feet, and dark soil
    pm = Mesh()

    def tray(x0, x1, y0, y1, z0, z1, k):
        V = [(x0, y0, z0), (x1, y0, z0), (x1, y1, z0), (x0, y1, z0), (x0 - k, y0 - k, z1), (x1 + k, y0 - k, z1), (x1 + k, y1 + k, z1), (x0 - k, y1 + k, z1)]
        pm.add(V, [[0, 3, 2, 1], [4, 5, 6, 7], [0, 1, 5, 4], [1, 2, 6, 5], [2, 3, 7, 6], [3, 0, 4, 7]])
    tray(-0.70, 0.70, -0.42, 0.42, 0.03, 0.21, 0.04)
    tray(-0.76, 0.76, -0.48, 0.48, 0.21, 0.245, 0.0)
    for fx in (-0.55, 0.55):
        for fy in (-0.30, 0.30):
            tray(fx - 0.07, fx + 0.07, fy - 0.05, fy + 0.05, -0.01, 0.035, 0.0)
    tree["pot"] = obj("pot", pm, POT)
    sm = Mesh()
    sm.add([(-0.72, -0.44, 0.235), (0.72, -0.44, 0.235), (0.72, 0.44, 0.235), (-0.72, 0.44, 0.235)], [[0, 1, 2, 3]])
    tree["soil"] = obj("soil", sm, (.03, .03, .03, 1))
    slots = []
    for i, (bv, pad) in enumerate(zip(tr["branches"], tr["pads"])):
        tree["branch"].append(obj(f"br{i}", bv, BARK, True, clay if i == hero else cclay))
        m = Mesh()
        for j, (st, d, ln) in enumerate(pad["twigs"]):
            sv, sf, Lv, trits, C, N = twig(rng, st, d, ln)
            if i == hero and j == hj:
                S["hero_twig"] = (sv, sf, Lv, trits, C, N, ln)
                continue
            m.add(sv, sf)
            m.add(Lv.reshape(-1, 3), (LEAF_F[None] + 6 * np.arange(BLOCK)[:, None, None]).reshape(-1, 3))
            for _ in range(4):
                k = int(rng.integers(30, BLOCK))
                slots.append(C[k] + N[k] * 0.022 * ln / 0.25 + unit(np.cross(d, [0, 0, 1.0])) * rng.uniform(-0.05, 0.05) * ln)
        tree["pad"].append(obj(f"pad{i}", m, LEAF_A if pad["attn"] else LEAF, mat=cclay))
    sv, sf, Lv, trits, C, N, ln = S["hero_twig"]
    hm = Mesh()
    hm.add(sv, sf)
    hm.add(Lv.reshape(-1, 3), (LEAF_F[None] + 6 * np.arange(BLOCK)[:, None, None]).reshape(-1, 3))
    tree["hero_static"] = obj("hero_static", hm, LEAF)
    # blue copies (same mesh, the accent material) for a layer being read
    for i in range(LAYERS):
        group = [tree["branch"][i], tree["pad"][i]] + ([tree["hero_static"]] if i == hero else [])
        cp = []
        for o in group:
            c = o.copy()
            sc.collection.objects.link(c)
            c.material_slots[0].link = "OBJECT"
            c.material_slots[0].material = blue
            c.color = (1, 1, 1, 1)
            c.hide_render = True
            cp.append(c)
        tree["blue"].append(cp)
    S["tree"] = tree
    S["base_col"] = [(o, tuple(o.color)) for o in [tree["trunk"], tree["pot"]] + tree["branch"] + tree["pad"]]
    # the hero twig, rebuilt every frame while it is read: grey and blue halves
    S["hstem"] = kit.mesh_object("hstem", sv, sf, clay)
    S["hstem"].color = LEAF
    S["hstem_b"] = kit.mesh_object("hstemb", sv, sf, blue)
    S["hleaf"] = kit.mesh_object("hleaf", Lv.reshape(-1, 3), (LEAF_F[None] + 6 * np.arange(BLOCK)[:, None, None]).reshape(-1, 3).tolist(), clay)
    S["hleaf"].color = LEAF
    S["hleaf_b"] = kit.mesh_object("hleafb", Lv.reshape(-1, 3), (LEAF_F[None] + 6 * np.arange(BLOCK)[:, None, None]).reshape(-1, 3).tolist(), blue)
    S["Lv"] = Lv
    # the four buds at the twig's tip: one token and three drafts
    T = unit(C[-1] - C[-12])
    side = unit(np.cross(T, [0, 0, 1.0]))
    S["bud_pos"] = [C[-1] + T * 0.02 * ln / 0.3 + N[-1] * 0.012] + [C[-1 - 12 * k] + N[-1 - 12 * k] * 0.035 * ln / 0.3 + side * (0.035 if k % 2 else -0.035) * ln / 0.3 for k in range(1, 4)]
    S["bud_r"] = 0.016 * ln / 0.3
    S["H"] = C[BLOCK // 2]
    S["Hd"] = unit(C[-1] - C[0])
    S["Hn"] = N[BLOCK // 2]
    sv_, sf_ = sphere()
    bv_, bf_ = sphere(20, 12)
    S["buds_b"] = Inst("budsb", bv_, bf_, 4, blue)
    S["buds_w"] = Inst("budsw", bv_, bf_, 4, clay)
    S["buds_w"].o.color = (.70, .71, .72, 1)
    S["buds_x"] = Inst("budsx", bv_, bf_, 1, clay)
    S["buds_x"].o.color = (.05, .05, .052, 1)
    # blossoms: the tokens. One random order of places, shared by every crown, so the fork's set is a subset of ours
    slots = np.array(slots)
    order = rng.permutation(len(slots))[:SLOTS]
    S["slots"] = slots[order]
    S["bl_w"] = Inst("blw", sv_, sf_, 60, cclay)
    S["bl_b"] = Inst("blb", sv_, sf_, 8, blue)
    S["bl_w"].o.color = (.70, .71, .72, 1)
    # the race: two copies of the tree
    S["race"] = []
    for side_, x in (("fork", -RACE_X), ("ours", RACE_X)):
        objs = []
        for o in [tree["trunk"], tree["pot"], tree["soil"], tree["hero_static"]] + tree["branch"] + tree["pad"]:
            c = o.copy()
            sc.collection.objects.link(c)
            c.location = (x, 0, 0)
            objs.append(c)
        w = Inst(f"{side_}w", sv_, sf_, SLOTS, clay)
        w.o.color = (.70, .71, .72, 1)
        f = Inst(f"{side_}f", sv_, sf_, 460, blue if side_ == "ours" else grey_flash)
        S["race"].append({"x": x, "objs": objs, "w": w, "f": f})
    # light: rims from behind, a little from the top, almost nothing in front
    S["L"] = {
        "back": kit.area("back", (-3.2, 3.6, 3.4), (0, 0, 1.1), 2.2, 0, 2.2),
        "rim": kit.area("rim", (3.4, 3.0, 1.6), (0, 0, 1.0), 1.6, 0, 2.4),
        "top": kit.area("top", (0.4, 0.9, 5.2), (0, 0, 1.0), 2.4, 0),
        "fill": kit.area("fill", (2.0, -6.5, 2.2), (0, 0, 1.0), 5, 0),
        "in": kit.area("in", (0, 0, 0), (0, 0, 0), 0.4, 0),
        "inf": kit.area("inf", (0, 0, 0), (0, 0, 0), 0.6, 0),
        "rback": kit.area("rback", (0, 4.2, 4.2), (0, 0, 1.1), 5.5, 0, 2.5),
        "rl": kit.area("rl", (-5.2, 2.6, 2.0), (-1.4, 0, 1.0), 2.0, 0, 2.4),
        "rr": kit.area("rr", (5.2, 2.6, 2.0), (1.4, 0, 1.0), 2.0, 0, 2.4),
        "rtop": kit.area("rtop", (0, 0.6, 5.8), (0, 0, 1.0), 4.5, 0),
        "rfill": kit.area("rfill", (0.5, -7.5, 2.6), (0, 0, 1.0), 7, 0),
    }
    H, Hd, Hn = Vector(S["H"]), Vector(S["Hd"]), Vector(S["Hn"])
    out = Hd.cross(Vector((0, 0, 1))).normalized()
    if out.y > 0:
        out = -out
    S["out"] = out
    kit.look(S["L"]["in"], H - out * 0.9 + Vector((0, 0, 0.55)), H)
    kit.look(S["L"]["inf"], H + out * 1.2 + Vector((0.3, 0, 0.5)), H)
    return sc


def hpath(t, keys):
    """Camera path through keys [(time, point)], Hermite with each key's velocity taken from the slower of its two
    segments, so a fast move into a slow drift arrives without overshooting."""
    ts = [k[0] for k in keys]
    ps = [np.array(k[1], float) for k in keys]
    if t <= ts[0]:
        return Vector(ps[0])
    if t >= ts[-1]:
        return Vector(ps[-1])
    sl = [(ps[i + 1] - ps[i]) / (ts[i + 1] - ts[i]) for i in range(len(ps) - 1)]
    v = [sl[0]] + [min(sl[i - 1], sl[i], key=np.linalg.norm) for i in range(1, len(ps) - 1)] + [sl[-1]]
    i = max(j for j in range(len(ts) - 1) if ts[j] <= t)
    h = ts[i + 1] - ts[i]
    u = (t - ts[i]) / h
    u2, u3 = u * u, u * u * u
    return Vector((2 * u3 - 3 * u2 + 1) * ps[i] + (u3 - 2 * u2 + u) * h * v[i] + (-2 * u3 + 3 * u2) * ps[i + 1] + (u3 - u2) * h * v[i + 1])


def show(o, on):
    o.hide_render = not on


def blossoms(t):
    """Blossoms on the first tree: (count, newest-first ages)."""
    times = [-1.0 + 0.35 * k for k in range(4)] + [0.18 + 0.27 * k for k in range(16)]
    for p in PASSES:
        times += [p + PASS + 0.02 * k for k in range(DRAFTS)]
    return [t - x for x in times if x <= t]


def frame(t):
    import bpy  # noqa: F401
    cam, L, tree = S["cam"], S["L"], S["tree"]
    anchors = {}
    race = t >= CUT
    for k in L:
        L[k].data.energy = 0
    S["cclay"].node_tree.nodes["CLIP"].outputs[0].default_value = -99
    main = [tree["trunk"], tree["pot"], tree["soil"], tree["hero_static"]] + tree["branch"] + tree["pad"]
    for o in main:
        show(o, not race)
    for cp in tree["blue"]:
        for o in cp:
            show(o, False)
    for r in S["race"]:
        for o in r["objs"]:
            show(o, race)
        show(r["w"].o, race)
        show(r["f"].o, race)
    for k in ("hstem", "hstem_b", "hleaf", "hleaf_b"):
        show(S[k], False)
    for k in ("buds_b", "buds_w", "buds_x", "bl_w", "bl_b"):
        show(S[k].o, not race)
    inside = 4.4 < t < 11.6
    H, out = Vector(S["H"]), S["out"]
    if not race:
        # blossoms on the one tree
        ages = blossoms(t)
        sl = S["slots"]
        w = [(sl[k], 0.034 * kit.ease(a / 0.12)) for k, a in enumerate(ages) if a >= 0.30]
        b = [(sl[k], 0.034 * kit.ease(a / 0.12)) for k, a in enumerate(ages) if a < 0.30]
        S["bl_w"].set([x[0] for x in w], [x[1] for x in w])
        S["bl_b"].set([x[0] for x in b], [x[1] for x in b])
        # the twig being read
        if inside:
            show(tree["hero_static"], False)
            u = kit.clamp((t - SWEEP0) / (SWEEP1 - SWEEP0))
            n = int(round(u * BLOCK)) if t < BUDS + 0.05 and t >= SWEEP0 else 0
            Lv = S["Lv"]
            far = np.full_like(Lv, -400.0)
            lit = np.arange(BLOCK) < n
            S["hleaf"].data.vertices.foreach_set("co", np.where(lit[:, None, None], far, Lv).astype("f4").ravel())
            S["hleaf_b"].data.vertices.foreach_set("co", np.where(lit[:, None, None], Lv, far).astype("f4").ravel())
            S["hleaf"].data.update()
            S["hleaf_b"].data.update()
            show(S["hleaf"], n < BLOCK)
            show(S["hleaf_b"], n > 0)
            stem_lit = SWEEP0 <= t < BUDS + 0.05
            show(S["hstem"], not stem_lit)
            show(S["hstem_b"], stem_lit)
            # buds: open together, then the last draft is pruned and drops; the rest stay
            g = kit.ease((t - BUDS) / 0.35)
            bp, r = S["bud_pos"], S["bud_r"]
            if t < PRUNE:
                S["buds_b"].set(bp, [r * g] * 4)
                S["buds_w"].set([], [])
                S["buds_x"].set([], [])
            else:
                S["buds_b"].set([], [])
                S["buds_w"].set(bp[:3], [r] * 3)
                fall = t - PRUNE - 0.25
                p = np.array(bp[3]) - [0, 0, 0.5 * 9.8 * 0.05 * max(0.0, fall) ** 2]
                S["buds_x"].set([p], [r * (1 - kit.ramp(t, PRUNE + 0.3, 1.0))])
        else:
            for k in ("buds_b", "buds_w", "buds_x"):
                S[k].set([], [])
            if t >= PRUNE:
                S["buds_w"].set(S["bud_pos"][:3], [S["bud_r"]] * 3)
        # a step climbs the layers
        for p in PASSES:
            u = (t - p) / PASS
            if 0 <= u < 1.05:
                head = u * (LAYERS + 6)
                for i in range(LAYERS):
                    on = 0 <= head - i < 6
                    if on:
                        show(tree["branch"][i], False)
                        show(tree["pad"][i], False)
                        if i == S["hero"]:
                            show(tree["hero_static"], False)
                        for o in tree["blue"][i]:
                            show(o, True)
        # light
        cut = kit.ramp(t, 4.5, 1.6) * (1 - kit.ramp(t, 10.3, 1.1))
        S["cclay"].node_tree.nodes["CLIP"].outputs[0].default_value = kit.lerp(-1.7, S["H"][1] - 0.03, cut) if cut > 0 else -99
        for o, c in S["base_col"]:          # while the twig is read, the idle tree goes dark
            o.color = tuple(x * (1 - 0.72 * cut) for x in c[:3]) + (1,)
        near = 1 - kit.ramp(t, 4.6, 1.4) + kit.ramp(t, 10.2, 1.4)
        near = kit.clamp(near)
        cl = 1 - near
        L["back"].data.energy, L["rim"].data.energy, L["top"].data.energy, L["fill"].data.energy = 520 * near, 300 * near, 150 * near, 18 * near
        L["in"].data.energy, L["inf"].data.energy = 34 * cl, 5 * cl
        # camera
        cin = H + out * 0.78 + Vector((0, 0, 0.20))
        cin2 = H + out * 0.66 + Vector(S["Hd"]) * 0.05 + Vector((0, 0, 0.16))
        keys = [(0, (1.35, -5.6, 1.45)), (4.0, (0.95, -5.0, 1.33)), (6.4, tuple(cin)), (10.2, tuple(cin2)), (12.4, (1.9, -5.5, 1.5)), (CUT, (2.5, -6.2, 1.35))]
        ats = [(0, (-0.5, 0, 0.9)), (4.0, (-0.45, 0, 0.92)), (6.4, tuple(H - Vector((0, 0, 0.03)))), (10.2, tuple(H + Vector(S["Hd"]) * 0.04 - Vector((0, 0, 0.03)))), (12.4, (-0.42, 0, 1.12)), (CUT, (-0.5, 0, 1.16))]
        kit.look(cam, hpath(t, keys), hpath(t, ats))
        cam.data.lens = 50
        cam.data.clip_start = 0.01
    else:
        fork, ours = rate(S["data"])
        el = kit.clamp(t - RACE0, 0, RACE_LEN)
        sl = S["slots"]
        for r, tps in zip(S["race"], (fork, ours)):
            n = int(tps * el)
            off = np.array([r["x"], 0, 0])
            # each token's blossom: when it was made, at the measured average rate
            k = np.arange(n)
            age = el - k / tps if n else np.zeros(0)
            if t > RACE0 + RACE_LEN:
                age = age + (t - RACE0 - RACE_LEN)
            pop = np.clip(age / 0.10, 0, 1)
            sz = 0.020 * pop * pop * (3 - 2 * pop)
            fresh = age < 0.22
            if r["x"] > 0 and t > END + 0.6:     # the result: what ours made beyond the fork's count, in the accent
                fresh = fresh | (k >= int(fork * RACE_LEN))
            r["w"].set(sl[k[~fresh]] + off, sz[~fresh])
            r["f"].set(sl[k[fresh]] + off, sz[fresh])
            anchors["fork" if r["x"] < 0 else "ours"] = (r["x"], 0, 2.55)
        on = kit.ramp(t, CUT, 0.3)
        L["rback"].data.energy, L["rl"].data.energy, L["rr"].data.energy = 900 * on, 260 * on, 260 * on
        L["rtop"].data.energy, L["rfill"].data.energy = 380 * on, 60 * on
        loc = kit.path(t, [(CUT, (-0.4, -7.6, 2.2)), (END, (0.1, -7.0, 2.0)), (DURATION, (0.25, -6.7, 2.25))])
        at = kit.path(t, [(CUT, (0, 0, 1.12)), (END, (0, 0, 1.12)), (DURATION, (0.05, 0, 1.02))])
        kit.look(cam, loc, at)
        cam.data.lens = 34
    return anchors


# --- overlay (runs after rendering, without Blender) ---
def _e(x):
    x = max(0.0, min(1.0, x))
    return x * x * (3 - 2 * x)


def _fade(t, a, b, d=0.4):
    return _e((t - a) / d) * (1 - _e((t - b + d) / d))


POSTER = 3.0
CLIPS = {}
MUTED = "#9aa5b6"


def overlay(t, pts, data):
    d = {r["gpu"]: r for r in data["decode"]}["RTX 5090"]
    fork, ours = d["fork_mtp"], d["ours_mtp"]
    out = []
    if t < 10:
        k = _fade(t, 0.6, 4.7, 0.5)
        out.append({"text": "INTRODUCING", "x": 0.5, "y": 0.775, "size": 28, "mono": True, "color": MUTED, "anchor": "c", "a": k})
        out.append({"text": "tritrun", "x": 0.5, "y": 0.855, "size": 128, "anchor": "c", "a": k})
        out.append({"text": f"A Ternary Bonsai engine, {ours / fork:.2f}× faster than PrismML's", "x": 0.5, "y": 0.935, "size": 30,
                    "weight": "Regular", "color": MUTED, "anchor": "c", "a": _fade(t, 1.1, 4.7, 0.5)})
    else:
        k = _e((t - 20.6) / 0.4)
        for key, name in (("fork", "PrismML's llama.cpp fork"), ("ours", "tritrun")):
            if key in pts:
                x, y = pts[key]
                out.append({"text": name, "x": x, "y": y - 0.03, "size": 32, "weight": "Regular", "anchor": "c", "a": k,
                            "color": "#e8edf4" if key == "ours" else "#a3afc0"})
        k = _fade(t, 20.9, 22.3, 0.35)
        out.append({"text": "A BLOSSOM PER TOKEN, 5 SECONDS", "x": 0.06, "y": 0.81, "size": 28, "mono": True, "color": MUTED, "a": k})
        out.append({"text": "The same race, on the trees.", "x": 0.06, "y": 0.88, "size": 64, "a": k})
        a = _e((t - 22.4) / 0.45)
        out.append({"text": "TRITRUN, OPEN SOURCE", "x": 0.06, "y": 0.81, "size": 28, "mono": True, "color": MUTED, "a": a})
        out.append({"text": f"{ours / fork:.2f}× faster.", "x": 0.06, "y": 0.88, "size": 64, "a": a})
        out.append({"text": "github.com/amw720386/tritrun", "x": 0.06, "y": 0.94, "size": 26, "weight": "Regular", "color": MUTED, "a": _e((t - 23.0) / 0.45)})
    return out
