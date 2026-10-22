#!/usr/bin/env python3
"""Error analysis of OUR classifier, on the same axes as the Pl@ntNet analysis
(plantnet_error_analysis.py), plus a per-family side-by-side of the two systems.

Everything is computed from the per-sample predictions that eval_full.py writes,
so no model runs are needed.

    python confusion_analysis.py --run bioclip2_linear_336
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
REPO = ROOT
SPECIES_JSON = ROOT / "data/species_data.json"
OUT = ROOT / "results/analysis"


def norm(s: str) -> str:
    return " ".join((s or "").lower().replace("×", "x").split())


def binom(s: str) -> str:
    p = norm(s).split()
    return " ".join(p[:2]) if len(p) >= 2 else norm(s)


def genus(s: str) -> str:
    p = norm(s).split()
    return p[0] if p else ""


def taxonomy():
    sp = json.loads(SPECIES_JSON.read_text(encoding="utf-8"))["species"]
    fam_cn2lat = {s["family"]: s.get("family_latin", s["family"]) for s in sp}
    return fam_cn2lat


def load_ours(run: str) -> list[dict]:
    p = ROOT / "results" / run / "predictions.jsonl"
    return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines()
            if l.strip()]


def load_plantnet() -> list[dict]:
    p = ROOT / "data/external_baselines/plantnet/raw_all_test.jsonl"
    recs = [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines()
            if l.strip()]
    return [r for r in recs if r.get("ok") and r.get("top")]


def ours_stats(rows: list[dict], fam_cn2lat: dict) -> dict:
    n = len(rows)
    exact = gright = gwrong = 0
    per_fam = defaultdict(lambda: [0, 0])
    confusions = Counter()
    per_sp = defaultdict(lambda: {"n": 0, "hit": 0, "in3": 0})
    per_dom = defaultdict(lambda: [0, 0])
    for r in rows:
        t_sp, p_sp = binom(r["true_species"]), binom(r["pred_species"])
        t_fam = norm(r.get("true_family_latin") or "")
        p_fam = norm(r.get("pred_family_latin") or "")
        if p_sp == t_sp:
            exact += 1
        elif genus(p_sp) == genus(t_sp):
            gright += 1
        else:
            gwrong += 1
        per_fam[t_fam][1] += 1
        per_fam[t_fam][0] += (p_fam == t_fam)
        if t_fam:
            confusions[(t_fam, p_fam or "(empty)")] += 1
        d = per_sp[t_sp]
        d["n"] += 1
        d["hit"] += (p_sp == t_sp)
        d["in3"] += any(binom(x) == t_sp for x in (r.get("top3") or []))
        dom = r.get("image_domain") or "?"
        per_dom[dom][1] += 1
        per_dom[dom][0] += (p_sp == t_sp)
    return {"n": n, "exact": exact, "gright": gright, "gwrong": gwrong,
            "per_fam": per_fam, "confusions": confusions, "per_sp": per_sp,
            "per_dom": per_dom}


def plantnet_stats(recs: list[dict], fam_cn2lat: dict) -> dict:
    n = len(recs)
    exact = gright = gwrong = 0
    per_fam = defaultdict(lambda: [0, 0])
    confusions = Counter()
    per_sp = defaultdict(lambda: {"n": 0, "hit": 0, "in3": 0})
    per_dom = defaultdict(lambda: [0, 0])
    for r in recs:
        top = r["top"]
        t_sp = binom(r["true_species"])
        p_sp = binom(top[0].get("sci") or "")
        t_fam = norm(fam_cn2lat.get(r.get("true_family_cn"), ""))
        p_fam = norm(top[0].get("family") or "")
        if p_sp == t_sp:
            exact += 1
        elif genus(p_sp) == genus(t_sp):
            gright += 1
        else:
            gwrong += 1
        per_fam[t_fam][1] += 1
        per_fam[t_fam][0] += (p_fam == t_fam)
        if t_fam:
            confusions[(t_fam, p_fam or "(empty)")] += 1
        d = per_sp[t_sp]
        d["n"] += 1
        d["hit"] += (p_sp == t_sp)
        d["in3"] += any(binom(c.get("sci") or "") == t_sp for c in top[:10])
        dom = r.get("image_domain") or "?"
        per_dom[dom][1] += 1
        per_dom[dom][0] += (p_sp == t_sp)
    return {"n": n, "exact": exact, "gright": gright, "gwrong": gwrong,
            "per_fam": per_fam, "confusions": confusions, "per_sp": per_sp,
            "per_dom": per_dom}


def report(tag: str, st: dict) -> None:
    n = st["n"]
    print(f"\n=== {tag} (n={n}) ===")
    print(f"  exact species            : {st['exact']/n:.4f} ({st['exact']})")
    print(f"  genus right, species wrong: {st['gright']/n:.4f} ({st['gright']})")
    print(f"  genus wrong              : {st['gwrong']/n:.4f} ({st['gwrong']})")
    tot = len(st["per_sp"])
    nc = sum(1 for d in st["per_sp"].values() if d["hit"] == 0)
    n3 = sum(1 for d in st["per_sp"].values() if d["in3"] == 0)
    print(f"  species never correct    : {nc}/{tot} ({nc/tot:.1%})")
    print(f"  species never in top-3   : {n3}/{tot} ({n3/tot:.1%})")
    for dom, (h, c) in st["per_dom"].items():
        print(f"  domain {dom:9s}: {h/c:.3f} (n={c})")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="bioclip2_linear_336")
    ap.add_argument("--top", type=int, default=12)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    fam_cn2lat = taxonomy()

    ours = ours_stats(load_ours(args.run), fam_cn2lat)
    pn = plantnet_stats(load_plantnet(), fam_cn2lat)
    report(f"ours: {args.run}", ours)
    report("Pl@ntNet (all)", pn)

    fams = sorted({f for f in list(ours["per_fam"]) + list(pn["per_fam"]) if f})
    rows = []
    for f in fams:
        o, p = ours["per_fam"].get(f, [0, 0]), pn["per_fam"].get(f, [0, 0])
        if max(o[1], p[1]) < 5:
            continue
        rows.append({"family": f, "n_test": o[1] or p[1],
                     "ours_acc": round(o[0] / o[1], 3) if o[1] else "",
                     "ours_n": o[1],
                     "plantnet_acc": round(p[0] / p[1], 3) if p[1] else "",
                     "plantnet_n": p[1],
                     "gap": (round(o[0] / o[1] - p[0] / p[1], 3)
                             if o[1] and p[1] else "")})
    rows.sort(key=lambda r: (r["plantnet_acc"] if r["plantnet_acc"] != ""
                             else 2))
    with (OUT / "per_family_ours_vs_plantnet.csv").open(
            "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"\n=== per-family: ours vs Pl@ntNet (n>=5, sorted by Pl@ntNet asc) ===")
    print(f"{'family':18s} {'n':>4s} {'ours':>6s} {'Pl@ntNet':>9s} {'gap':>7s}")
    for r in rows:
        print(f"{r['family']:18s} {r['n_test']:4d} "
              f"{r['ours_acc'] if r['ours_acc'] != '' else '-':>6} "
              f"{r['plantnet_acc'] if r['plantnet_acc'] != '' else '-':>9} "
              f"{r['gap'] if r['gap'] != '' else '-':>7}")

    print(f"\n=== our top family confusions (true -> predicted, cross-family only) ===")
    cross = [((tf, pf), c) for (tf, pf), c in ours["confusions"].most_common()
             if tf != pf and tf and pf != "(empty)"]
    if not cross:
        print("  (none: every cross-family error is unique)")
    for (tf, pf), c in cross[:args.top]:
        print(f"  {tf:18s} -> {pf:18s} {c:4d} / {ours['per_fam'][tf][1]}")
    print(f"  total cross-family errors: {sum(c for _, c in cross)}"
          f" / {ours['n']} ({sum(c for _, c in cross)/ours['n']:.1%})")

    md = ["| family | n_test | ours | Pl@ntNet | gap |", "|---|---:|---:|---:|---:|"]
    for r in rows:
        md.append(f"| {r['family']} | {r['n_test']} | "
                  f"{r['ours_acc'] if r['ours_acc'] != '' else '—'} | "
                  f"{r['plantnet_acc'] if r['plantnet_acc'] != '' else '—'} | "
                  f"{r['gap'] if r['gap'] != '' else '—'} |")
    (OUT / "per_family_ours_vs_plantnet.md").write_text("\n".join(md) + "\n",
                                                        encoding="utf-8")
    print(f"\nwrote {OUT}/per_family_ours_vs_plantnet.csv + .md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
