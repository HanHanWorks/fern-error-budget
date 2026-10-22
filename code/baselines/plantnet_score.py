#!/usr/bin/env python3
"""Score Pl@ntNet JSONL output: species top-1/top-3, family top-1, coverage.

    python plantnet_score.py                       # all raw_*.jsonl found
    python plantnet_score.py --glob 'raw_k-china*'
"""
from __future__ import annotations

import argparse
import glob
import json
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUTDIR = ROOT / "plantnet"


def norm(name: str) -> str:
    return " ".join((name or "").lower().replace("×", "x").split())


def binomial(name: str) -> str:
    p = norm(name).split()
    return " ".join(p[:2]) if len(p) >= 2 else norm(name)


def genus(name: str) -> str:
    p = norm(name).split()
    return p[0] if p else ""


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval — honest CI for a proportion near 0 or 1."""
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    half = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5)
    return ((c - half) / d, (c + half) / d)


def score_file(path: Path) -> dict | None:
    recs = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            recs.append(json.loads(line))
    recs = [r for r in recs if r.get("ok")]
    if not recs:
        return None
    n = len(recs)
    top1 = top3 = fam1 = cover = gen1 = 0
    per_class = defaultdict(int)
    per_class_tot = defaultdict(int)
    dom = defaultdict(lambda: [0, 0])
    src = defaultdict(lambda: [0, 0])
    for r in recs:
        truth = binomial(r["true_species"])
        cands = [binomial(x.get("sci") or "") for x in (r.get("top") or [])]
        hit1 = bool(cands) and cands[0] == truth
        hit3 = truth in cands[:3]
        cover_hit = truth in cands
        gen_truth = genus(r["true_species"])
        gen_hit = bool(cands) and genus(cands[0]) == gen_truth
        fam_truth = norm(r.get("true_family_latin") or "")
        fam_pred = norm((r.get("top") or [{}])[0].get("family") or "") \
            if r.get("top") else ""
        f_hit = bool(fam_truth) and fam_pred == fam_truth
        top1 += hit1; top3 += hit3; cover += cover_hit; fam1 += f_hit
        gen1 += gen_hit
        per_class[truth] += int(hit1); per_class_tot[truth] += 1
        d = r.get("image_domain") or "?"
        dom[d][0] += int(hit1); dom[d][1] += 1
        s = r.get("source") or "?"
        src[s][0] += int(hit1); src[s][1] += 1
    macro = sum(per_class[c] / per_class_tot[c] for c in per_class_tot) / \
        max(len(per_class_tot), 1)
    return {
        "file": path.name, "n": n,
        "species_top1": top1 / n, "species_top3": top3 / n,
        "species_macro_top1": macro,
        "species_top1_ci95": wilson(top1, n),
        "genus_top1": gen1 / n,
        "species_coverage_top10": cover / n,
        "family_top1": fam1 / n,
        "domain_top1": {k: v[0] / v[1] for k, v in dom.items()},
        "domain_n": {k: v[1] for k, v in dom.items()},
        "source_top1": {k: v[0] / v[1] for k, v in src.items()},
        "source_n": {k: v[1] for k, v in src.items()},
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--glob", default="raw_*.jsonl")
    args = ap.parse_args()
    files = sorted(Path(p) for p in glob.glob(str(OUTDIR / args.glob)))
    if not files:
        print(f"no files matching {OUTDIR / args.glob}")
        return 1
    print(f"{'file':34s} {'n':>5} {'sp_top1':>8} {'sp_top3':>8} "
          f"{'genus':>8} {'macro':>8} {'cover@10':>9} {'fam_top1':>8}")
    print("-" * 102)
    for f in files:
        s = score_file(f)
        if not s:
            continue
        print(f"{s['file']:34s} {s['n']:>5} {s['species_top1']:>8.4f} "
              f"{s['species_top3']:>8.4f} {s['genus_top1']:>8.4f} "
              f"{s['species_macro_top1']:>8.4f} "
              f"{s['species_coverage_top10']:>9.4f} {s['family_top1']:>8.4f}")
        lo, hi = s["species_top1_ci95"]
        print(f"    species top-1 95% CI: [{lo:.3f}, {hi:.3f}]")
        print(f"    domain: " + ", ".join(
            f"{k}={v:.3f}(n={s['domain_n'][k]})"
            for k, v in s["domain_top1"].items()))
        print(f"    source: " + ", ".join(
            f"{k}={v:.3f}(n={s['source_n'][k]})"
            for k, v in s["source_top1"].items()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
