#!/usr/bin/env python3
"""Score VLM JSONL output with the same matching rules as the Pl@ntNet scorer,
so the two baselines are directly comparable.

    python vlm_score.py                       # every vlm/*.jsonl
    python vlm_score.py --glob 'open_*'
"""
from __future__ import annotations

import argparse
import glob
import json
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
VLM = ROOT / "data/external_baselines/vlm"


def norm(s: str) -> str:
    return " ".join((s or "").lower().replace("*", "").replace("×", "x").split())


def binom(s: str) -> str:
    p = norm(s).split()
    return " ".join(p[:2]) if len(p) >= 2 else norm(s)


def genus(s: str) -> str:
    p = norm(s).split()
    return p[0] if p else ""


def wilson(k: int, n: int, z: float = 1.96):
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    h = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5)
    return ((c - h) / d, (c + h) / d)


def score_file(path: Path) -> dict | None:
    recs = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines()
            if l.strip()]
    err = Counter(r.get("error") for r in recs if not r.get("ok"))
    recs = [r for r in recs if r.get("ok")]
    if not recs:
        return None
    n = len(recs)
    top1 = top3 = cov3 = fam1 = 0
    per_cls = defaultdict(int)
    per_cls_tot = defaultdict(int)
    dom = defaultdict(lambda: [0, 0])
    src = defaultdict(lambda: [0, 0])
    lat = []
    for r in recs:
        t = binom(r["true_species"])
        cands = [binom(x.get("name") or "") for x in (r.get("top3") or [])]
        h1 = bool(cands) and cands[0] == t
        h3 = t in cands[:3]
        top1 += h1
        top3 += h3
        cov3 += (t in cands)
        tf = norm(r.get("true_family_latin") or "")
        pf = norm((r["top3"][0].get("family") or "") if r.get("top3") else "")
        fam1 += bool(tf) and pf == tf
        per_cls[t] += int(h1)
        per_cls_tot[t] += 1
        d = r.get("image_domain") or "?"
        dom[d][0] += int(h1); dom[d][1] += 1
        s = r.get("source") or "?"
        src[s][0] += int(h1); src[s][1] += 1
        if r.get("latency"):
            lat.append(r["latency"])
    macro = sum(per_cls[c] / per_cls_tot[c] for c in per_cls_tot) / \
        max(len(per_cls_tot), 1)
    return {
        "file": path.name, "n": n, "errors": dict(err),
        "species_top1": top1 / n, "species_top1_ci95": wilson(top1, n),
        "species_top3": top3 / n, "species_coverage_top3": cov3 / n,
        "species_macro_top1": macro, "family_top1": fam1 / n,
        "domain_top1": {k: v[0] / v[1] for k, v in dom.items()},
        "domain_n": {k: v[1] for k, v in dom.items()},
        "source_top1": {k: v[0] / v[1] for k, v in src.items()},
        "latency_mean": sum(lat) / len(lat) if lat else None,
        "latency_p50": sorted(lat)[len(lat) // 2] if lat else None,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--glob", default="*.jsonl")
    args = ap.parse_args()
    files = sorted(Path(p) for p in glob.glob(str(VLM / args.glob)))
    if not files:
        print("no files")
        return 1
    print(f"{'file':48s} {'n':>5} {'top1':>7} {'top3':>7} {'cover@3':>8} "
          f"{'macro':>7} {'fam':>7} {'lat_s':>7}")
    print("-" * 106)
    for p in files:
        s = score_file(p)
        if not s:
            continue
        lo, hi = s["species_top1_ci95"]
        print(f"{s['file']:48s} {s['n']:>5} {s['species_top1']:>7.4f} "
              f"{s['species_top3']:>7.4f} {s['species_coverage_top3']:>8.4f} "
              f"{s['species_macro_top1']:>7.4f} {s['family_top1']:>7.4f} "
              f"{(s['latency_mean'] or 0):>7.2f}")
        print(f"    top-1 95% CI [{lo:.3f}, {hi:.3f}]  errors={s['errors']}")
        for k in ("domain_top1", "source_top1"):
            if s[k]:
                print(f"    {k[:-4]:8s}: " + ", ".join(
                    f"{a}={b:.3f}" for a, b in s[k].items()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
