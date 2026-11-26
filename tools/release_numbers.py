"""Write results/release.json, the numbers the README reads, from the raw logs.

    python tools/release_numbers.py
"""
import json
import os
import re

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
RES = os.path.join(ROOT, "results")
# card: (log, decode section, prefill section)
CARDS = {
    "T4": ("paired_graphs.log", "m_paired_t4.log", "m_prefill_t4.log"),
    "L4": ("paired_graphs.log", "m_paired_l4.log", "m_prefill_l4.log"),
    "A10": ("paired_graphs.log", "m_paired_a10.log", "m_pre_ab_a10.log"),
    "RTX 5090": ("rtx5090_graphs.log", "f_paired.log", "f_prefill.log"),
}


def sections(name):
    out, cur = {}, None
    for line in open(os.path.join(RES, name), encoding="utf-8"):
        if line.startswith("# "):
            cur = line[2:].strip()
            out[cur] = []
        elif cur:
            out[cur].append(line.rstrip("\n"))
    return out


def pct(a, b):
    return round(100 * (b / a - 1), 2)


def main():
    decode, prefill = [], []
    for card, (log, dsec, psec) in CARDS.items():
        sec = sections(log)
        summ = json.loads(next(l for l in sec[dsec] if l.startswith("SUMMARY"))[8:])
        fp = max(summ["fork_plain"]["decode_tps"], summ["fork_plain_ub64"]["decode_tps"])
        fm = max(v["decode_tps"] for k, v in summ.items() if k.startswith("fork_mtp"))
        op, om = summ["ours_plain"]["decode_tps"], summ["ours_mtp3"]["decode_tps"]
        reps = [json.loads(l.split(": ", 1)[1]) for l in sec[dsec] if "ours_mtp3:" in l and l.startswith("[")]
        decode.append({
            "gpu": card,
            "fork_plain": round(fp, 1), "ours_plain": round(op, 1), "plain_gain_pct": pct(fp, op),
            "fork_mtp": round(fm, 1), "ours_mtp": round(om, 1), "mtp_gain_pct": pct(fm, om),
            "fork_plain_vram": summ["fork_plain_ub64"]["vram_mib"], "ours_plain_vram": summ["ours_plain"]["vram_mib"],
            "fork_mtp_vram": summ["fork_mtp_p0_ub64"]["vram_mib"], "ours_mtp_vram": summ["ours_mtp3"]["vram_mib"],
            "mtp_vram_saving_pct": -pct(summ["fork_mtp_p0_ub64"]["vram_mib"], summ["ours_mtp3"]["vram_mib"]),
            "tok_per_round": round(reps[0]["tok_per_round"], 2),
        })
        rx = re.compile(r"^\[[^\]]+\] (ours|fork) mtp=0: (.*)$")
        best = {"ours": {}, "fork": {}}
        for l in sec[psec]:
            m = rx.match(l)
            if m:
                for n, v in json.loads(m.group(2)).items():
                    best[m.group(1)][n] = max(best[m.group(1)].get(n, 0), v["tps"])
        row = {"gpu": card}
        for n in ("64", "512", "2048"):
            row[f"fork_{n}"], row[f"ours_{n}"] = round(best["fork"][n]), round(best["ours"][n])
            row[f"gain_{n}_pct"] = pct(best["fork"][n], best["ours"][n])
        prefill.append(row)
    out = {"decode": decode, "prefill": prefill,
           "sources": {c: v for c, v in CARDS.items()}}
    # chart rows, one per engine and card
    out["decode_rows"] = [r for d in decode for r in (
        {"gpu": d["gpu"], "engine": "PrismML llama.cpp fork", "plain": d["fork_plain"], "mtp": d["fork_mtp"]},
        {"gpu": d["gpu"], "engine": "tritrun", "plain": d["ours_plain"], "mtp": d["ours_mtp"]})]
    out["prefill_rows"] = [r for p in prefill for r in (
        {"gpu": p["gpu"], "engine": "PrismML llama.cpp fork", "t512": p["fork_512"], "t2048": p["fork_2048"]},
        {"gpu": p["gpu"], "engine": "tritrun", "t512": p["ours_512"], "t2048": p["ours_2048"]})]
    for k in ("decode_rows", "prefill_rows"):   # ours first, so it takes the accent colour
        out[k].sort(key=lambda r: r["engine"] != "tritrun")
    json.dump(out, open(os.path.join(RES, "release.json"), "w"), indent=1)
    for d in decode:
        print(d)
    for p in prefill:
        print(p)


if __name__ == "__main__":
    main()
