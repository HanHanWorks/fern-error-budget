#!/usr/bin/env python3
"""Error analysis of the Pl@ntNet baseline from results already collected.

Separates the two failure modes a specialist baseline can have:

  * genus-right, species-wrong  -> the taxon is probably not in the referential
    (a coverage gap, not a perceptual failure)
  * genus-wrong                 -> genuine misidentification

Also reports per-family accuracy and the dominant family confusions, which is
what the paper needs to explain *where* Pl@ntNet stops working.

    python plantnet_error_analysis.py --file plantnet/raw_all_test.jsonl
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


def norm(s: str) -> str:
    return " ".join((s or "").lower().replace("×", "x").split())


def genus(s: str) -> str:
    p = norm(s).split()
    return p[0] if p else ""


def binom(s: str) -> str:
    p = norm(s).split()
    return " ".join(p[:2]) if len(p) >= 2 else norm(s)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", default="plantnet/raw_all_test.jsonl")
    ap.add_argument("--top", type=int, default=15)
    args = ap.parse_args()

    path = Path(args.file)
    recs = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines()
            if l.strip()]
    recs = [r for r in recs if r.get("ok") and r.get("top")]
    if not recs:
        print("no usable records")
        return 1

    fam_cn2lat = {}
    if SPECIES_JSON.exists():
        for s in json.loads(SPECIES_JSON.read_text(encoding="utf-8"))["species"]:
            if s.get("family_latin"):
                fam_cn2lat[s["family"]] = s["family_latin"]

    n = len(recs)
    exact = gright_species_wrong = gwrong = 0
    per_fam = defaultdict(lambda: [0, 0])
    confusions = Counter()
    no_fern_candidate = 0
    for r in recs:
        t_sp = binom(r["true_species"])
        t_gen = genus(r["true_species"])
        t_fam = norm(fam_cn2lat.get(r["true_family_cn"], ""))
        top = r["top"]
        p_sp = binom(top[0].get("sci") or "")
        p_gen = genus(top[0].get("sci") or "")
        p_fam = norm(top[0].get("family") or "")
        if p_sp == t_sp:
            exact += 1
        elif p_gen == t_gen:
            gright_species_wrong += 1
        else:
            gwrong += 1
        per_fam[t_fam][1] += 1
        if p_fam == t_fam:
            per_fam[t_fam][0] += 1
        confusions[(t_fam, p_fam)] += 1
        # does any returned candidate belong to the true family?
        if t_fam and not any(norm(c.get("family") or "") == t_fam for c in top):
            no_fern_candidate += 1

    print(f"file: {path.name}  n = {n}")
    print(f"  exact species      : {exact/n:.4f}  ({exact})")
    print(f"  genus right, sp wrong: {gright_species_wrong/n:.4f}  "
          f"({gright_species_wrong})   <- coverage-gap signature")
    print(f"  genus wrong          : {gwrong/n:.4f}  ({gwrong})")
    print(f"  true family absent from ALL candidates: {no_fern_candidate/n:.4f} "
          f"({no_fern_candidate})")

    print(f"\nper-family accuracy (true family), n>=5:")
    rows = [(f, v[0] / v[1], v[1]) for f, v in per_fam.items() if v[1] >= 5]
    for f, a, c in sorted(rows, key=lambda x: x[1]):
        print(f"  {f:16s} {a:.3f}  (n={c})")

    print(f"\ntop family confusions (true -> predicted), same-family excluded:")
    for (tf, pf), c in confusions.most_common(args.top):
        if tf == pf or not tf or not pf:
            continue
        print(f"  {tf:16s} -> {pf:16s} {c}")

    # ---- confidence calibration -------------------------------------------
    # If the API's top-1 score were calibrated, a threshold could rescue
    # precision. Check before recommending any threshold rule.
    bins = [(i / 10, (i + 1) / 10) for i in range(10)]
    rows_c = []
    for lo, hi in bins:
        sel = [(1 if binom(r["top"][0].get("sci") or "") ==
                binom(r["true_species"]) else 0,
                float(r["top"][0].get("score") or 0.0)) for r in recs]
        sel = [s for s in sel if lo <= s[1] < hi or (hi == 1.0 and s[1] == 1.0)]
        if sel:
            acc = sum(s[0] for s in sel) / len(sel)
            conf = sum(s[1] for s in sel) / len(sel)
            rows_c.append((lo, hi, len(sel), acc, conf))
    ece = 0.0
    for lo, hi, c, acc, conf in rows_c:
        ece += (c / n) * abs(acc - conf)
    print(f"\nconfidence calibration (top-1 score vs correctness):")
    print(f"  {'bin':>10} {'n':>5} {'acc':>7} {'mean_conf':>10} {'gap':>7}")
    for lo, hi, c, acc, conf in rows_c:
        print(f"  {lo:.1f}-{hi:.1f} {c:>5} {acc:>7.3f} {conf:>10.3f} "
              f"{acc-conf:>7.3f}")
    print(f"  ECE(10 bins) = {ece:.4f}")
    # coverage-style: what fraction of correct answers sit in the top bin?
    hi_sel = [(1 if binom(r["top"][0].get("sci") or "") ==
               binom(r["true_species"]) else 0,
               float(r["top"][0].get("score") or 0.0)) for r in recs
              if float(r["top"][0].get("score") or 0.0) >= 0.5]
    if hi_sel:
        print(f"  score>=0.5: n={len(hi_sel)} "
              f"({len(hi_sel)/n:.0%} of samples), "
              f"acc={sum(s[0] for s in hi_sel)/len(hi_sel):.3f}")

    # ---- per-species coverage ---------------------------------------------
    # The headline claim "the referential does not know these species" is only
    # defensible at species level: count how many of OUR species Pl@ntNet never
    # retrieves at all.
    per_sp = defaultdict(lambda: {"n": 0, "hit": 0, "in_cands": 0})
    for r in recs:
        t = binom(r["true_species"])
        cands = [binom(c.get("sci") or "") for c in r["top"]]
        d = per_sp[t]
        d["n"] += 1
        if cands and cands[0] == t:
            d["hit"] += 1
        if t in cands:
            d["in_cands"] += 1
    total_sp = len(per_sp)
    never_correct = [s for s, d in per_sp.items() if d["hit"] == 0]
    never_candidate = [s for s, d in per_sp.items() if d["in_cands"] == 0]
    print(f"\nper-species coverage (species with >=1 test image): {total_sp}")
    print(f"  never identified correctly : {len(never_correct)} "
          f"({len(never_correct)/total_sp:.1%})")
    print(f"  never even in a candidate list: {len(never_candidate)} "
          f"({len(never_candidate)/total_sp:.1%})")
    multi = {s: d for s, d in per_sp.items() if d["n"] >= 5}
    if multi:
        best = sorted(multi.items(), key=lambda kv: -kv[1]["hit"] / kv[1]["n"])
        print(f"  best-covered species (n>=5), top 8:")
        for s, d in best[:8]:
            if d["hit"] == 0:
                break
            print(f"    {s:36s} {d['hit']/d['n']:.2f}  (n={d['n']})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
