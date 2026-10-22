#!/usr/bin/env python3
"""Paired significance testing for every system evaluated on the same 902-image
fern test split.

Why paired: all systems saw the SAME images, so an unpaired test throws away the
information that makes the comparison precise. For each pair we run
  * McNemar's test on the discordant pairs (exact binomial p, plus the
    continuity-corrected chi-square p -- they agree closely at this n)
  * a paired bootstrap of the accuracy difference (10k resamples)

Matching rule is the one used everywhere else in this project (binom(): first
two words, lowercased, "*"/"x" stripped), so numbers line up with the registry.

    python paired_significance.py                     # all systems
    python paired_significance.py --systems bioclip2_linear_336,plantnet_all
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import random
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
REPO = ROOT
OUT = ROOT / "results/analysis"


def norm(s: str) -> str:
    return " ".join((s or "").lower().replace("*", "").replace("×", "x").split())


def binom(s: str) -> str:
    p = norm(s).split()
    return " ".join(p[:2]) if len(p) >= 2 else norm(s)


# ---------------------------------------------------------------- loaders
def _correctness_from_records(recs, pred_getter, true_key="true_species"):
    """-> {sample_id: bool} over records with ok=True (or no ok flag)."""
    out = {}
    for r in recs:
        if "ok" in r and not r.get("ok"):
            continue
        sid = r.get("sample_id")
        if not sid:
            continue
        t = binom(r.get(true_key) or "")
        p = pred_getter(r)
        if t:
            out[sid] = (binom(p) == t)
    return out


def load_specialist(run: str) -> dict:
    p = ROOT / "results" / run / "predictions.jsonl"
    recs = [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines()
            if l.strip()]
    return _correctness_from_records(recs, lambda r: r.get("pred_species"))


def load_plantnet() -> dict:
    p = ROOT / "data/external_baselines/plantnet/raw_all_test.jsonl"
    recs = [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines()
            if l.strip()]
    def top(r):
        t = r.get("top") or []
        return (t[0].get("sci") if t else "") or ""
    return _correctness_from_records(recs, top)


def load_vlm(pattern: str) -> dict:
    p = next((ROOT / "data/external_baselines/vlm").glob(pattern), None)
    if p is None:
        return {}
    recs = [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines()
            if l.strip()]
    def top(r):
        t = r.get("top3") or []
        return (t[0].get("name") if t else "") or ""
    return _correctness_from_records(recs, top)


def load_vlm_leakfree(pattern: str) -> dict:
    """Same as load_vlm but drops images whose true name is OCR-readable."""
    import difflib
    import re as _re

    p = next((ROOT / "data/external_baselines/vlm").glob(pattern), None)
    if p is None:
        return {}
    ocr = {}
    tl = ROOT / "data/external_baselines/vlm/text_ocr.jsonl"
    for line in tl.read_text(encoding="utf-8").splitlines():
        if line.strip():
            r = json.loads(line)
            ocr[r["sample_id"]] = r.get("ocr_text") or ""

    def nm(s):
        s = (s or "").lower().replace("×", "x")
        return " ".join(_re.sub(r"[^a-z0-9\s\-]", " ", s).split())

    def leak(tsp, text):
        q = nm(tsp).split()
        if len(q) < 2:
            return False
        g, e = q[0], q[1]
        t = nm(text)
        if f"{g} {e}" in t:
            return True
        if len(e) >= 6:
            return any(abs(len(k) - len(e)) <= 4 and
                       difflib.SequenceMatcher(None, e, k).ratio() >= 0.8
                       for k in t.split())
        return False

    recs = [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines()
            if l.strip()]
    out = {}
    for r in recs:
        if "ok" in r and not r.get("ok"):
            continue
        sid = r.get("sample_id")
        if not sid or sid in ocr and leak(r.get("true_species") or "", ocr[sid]):
            continue
        t = binom(r.get("true_species") or "")
        t3 = r.get("top3") or []
        if t:
            out[sid] = binom((t3[0].get("name") if t3 else "") or "") == t
    return out


SYSTEMS = {
    # specialists (exact species head)
    "bioclip2_linear_336": lambda: load_specialist("bioclip2_linear_336"),
    "bioclip2_ft_336": lambda: load_specialist("bioclip2_ft_336"),
    "bioclip2_ft_hier": lambda: load_specialist("bioclip2_ft_hier"),
    "bioclip2_linear": lambda: load_specialist("bioclip2_linear"),
    "bioclip2_ft": lambda: load_specialist("bioclip2_ft"),
    "dinov2_linear_336": lambda: load_specialist("dinov2_linear_336"),
    "dinov2_ft": lambda: load_specialist("dinov2_ft"),
    "dinov2_linear": lambda: load_specialist("dinov2_linear"),
    # specialist API
    "plantnet_all": load_plantnet,
    # general VLMs, open-ended
    "qwen3.7-flash_open": lambda: load_vlm("open_qwen3.7-flash-2026-07-15_*.jsonl"),
    "qwen3-vl-8b_open": lambda: load_vlm("open_qwen3-vl-8b-instruct_*.jsonl"),
    # general VLMs, closed-set 152-way (directly comparable to the classifier)
    "qwen3.7-flash_species152": lambda: load_vlm("species152_qwen3.7-flash_*.jsonl"),
    "qwen3-vl-8b_species152": lambda: load_vlm("species152_qwen3-vl-8b-instruct_*.jsonl"),
    # leak-controlled variants of the open-ended task
    "qwen3.7-flash_open_noleak":
        lambda: load_vlm_leakfree("open_qwen3.7-flash-2026-07-15_*.jsonl"),
    "qwen3-vl-8b_open_noleak":
        lambda: load_vlm_leakfree("open_qwen3-vl-8b-instruct_*.jsonl"),
}


# ---------------------------------------------------------------- statistics
def mcnemar(a: dict, b: dict):
    """Exact binomial + continuity-corrected chi-square, on common samples."""
    common = sorted(set(a) & set(b))
    n = len(common)
    b01 = sum(1 for s in common if a[s] and not b[s])   # A right, B wrong
    c10 = sum(1 for s in common if not a[s] and b[s])   # A wrong, B right
    disc = b01 + c10
    if disc == 0:
        p_exact = 1.0
        p_chi2 = 1.0
    else:
        k = min(b01, c10)
        tail = sum(math.comb(disc, i) for i in range(k + 1)) / 2 ** disc
        p_exact = min(1.0, 2 * tail)
        chi2 = (abs(b01 - c10) - 1) ** 2 / disc if disc else 0.0
        p_chi2 = math.erfc(math.sqrt(max(chi2, 0.0) / 2))   # 1 df
    acc_a = sum(a[s] for s in common) / n if n else 0.0
    acc_b = sum(b[s] for s in common) / n if n else 0.0
    return {"n_paired": n, "a_only_right": b01, "b_only_right": c10,
            "acc_a": acc_a, "acc_b": acc_b, "diff": acc_a - acc_b,
            "p_exact": p_exact, "p_chi2": p_chi2}


def bootstrap_ci(hits: list[int], n_boot: int = 10000, seed: int = 20261013):
    """95% percentile CI for a single system's accuracy (i.i.d. resample)."""
    rng = random.Random(seed)
    n = len(hits)
    if n == 0:
        return (0.0, 0.0)
    stats = []
    for _ in range(n_boot):
        k = sum(hits[rng.randrange(n)] for _ in range(n))
        stats.append(k / n)
    stats.sort()
    return (stats[int(0.025 * n_boot)], stats[int(0.975 * n_boot)])


def bootstrap_diff_ci(a: dict, b: dict, n_boot: int = 10000, seed: int = 20261013):
    """95% percentile CI for the paired accuracy difference A - B."""
    rng = random.Random(seed)
    common = sorted(set(a) & set(b))
    n = len(common)
    if n == 0:
        return (0.0, 0.0)
    da = [1 if a[s] else 0 for s in common]
    db = [1 if b[s] else 0 for s in common]
    stats = []
    for _ in range(n_boot):
        idx = [rng.randrange(n) for _ in range(n)]
        stats.append(sum(da[i] - db[i] for i in idx) / n)
    stats.sort()
    return (stats[int(0.025 * n_boot)], stats[int(0.975 * n_boot)])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--systems", default="", help="comma list (default: all)")
    ap.add_argument("--boot", type=int, default=10000)
    args = ap.parse_args()

    names = ([s.strip() for s in args.systems.split(",") if s.strip()]
             or list(SYSTEMS))
    data = {n: SYSTEMS[n]() for n in names}
    for n in names:
        print(f"loaded {n:30s} n_ok={len(data[n])}")

    OUT.mkdir(parents=True, exist_ok=True)

    # 1) per-system accuracy with bootstrap CI
    rows = []
    for n in names:
        hits = [1 if v else 0 for v in data[n].values()]
        acc = sum(hits) / len(hits) if hits else 0.0
        lo, hi = bootstrap_ci(hits, args.boot)
        errs = 902 - len(hits)
        rows.append({"system": n, "n_ok": len(hits), "n_missing": errs,
                     "acc": round(acc, 4), "ci95_lo": round(lo, 4),
                     "ci95_hi": round(hi, 4)})
    with (OUT / "accuracy_summary.csv").open("w", newline="",
                                             encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    print("\n=== accuracy (902-image test split) ===")
    print(f"{'system':30s} {'n_ok':>5s} {'acc':>7s}  95% CI")
    for r in rows:
        print(f"{r['system']:30s} {r['n_ok']:5d} {r['acc']:7.4f}  "
              f"[{r['ci95_lo']:.3f}, {r['ci95_hi']:.3f}]")

    # 2) all pairwise McNemar
    pairs = []
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            if not data[a] or not data[b]:
                continue
            st = mcnemar(data[a], data[b])
            lo, hi = bootstrap_diff_ci(data[a], data[b], args.boot)
            pairs.append({"a": a, "b": b,
                          **{k: (round(v, 6) if isinstance(v, float) else v)
                             for k, v in st.items()},
                          "diff_ci95_lo": round(lo, 4),
                          "diff_ci95_hi": round(hi, 4),
                          "diff_pct": round(100 * st["diff"], 1),
                          "sig": ("***" if st["p_exact"] < 0.001 else
                                  "**" if st["p_exact"] < 0.01 else
                                  "*" if st["p_exact"] < 0.05 else "n.s.")})
    with (OUT / "paired_mcnemar.csv").open("w", newline="",
                                           encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(pairs[0]))
        w.writeheader()
        w.writerows(pairs)

    print("\n=== pairwise McNemar (paired on common samples) ===")
    print(f"{'A':28s} {'B':28s} {'n':>4s} {'A-B':>7s} {'p_exact':>10s} {'sig':>5s}")
    for r in sorted(pairs, key=lambda r: -abs(r["diff"])):
        print(f"{r['a']:28s} {r['b']:28s} {r['n_paired']:4d} "
              f"{r['diff_pct']:+6.1f}pp {r['p_exact']:10.2e} {r['sig']:>5s}")

    # 3) markdown for RESULTS.md
    lines = ["| 系统 A | 系统 B | n(配对) | Δ 准确率 | McNemar p | 95% CI (Δ) |",
             "|---|---|---:|---:|---:|---|"]
    for r in sorted(pairs, key=lambda r: -abs(r["diff"])):
        cen = "0" if r["diff_ci95_lo"] <= 0 <= r["diff_ci95_hi"] else "1"
        lines.append(
            f"| {r['a']} | {r['b']} | {r['n_paired']} | "
            f"{r['diff_pct']:+.1f} pp | {r['p_exact']:.2e} {r['sig']} | "
            f"[{100*r['diff_ci95_lo']:+.1f}, {100*r['diff_ci95_hi']:+.1f}] pp"
            f"{' ← 含 0' if cen == '0' else ''} |")
    (OUT / "paired_mcnemar.md").write_text("\n".join(lines) + "\n",
                                           encoding="utf-8")
    print(f"\nwrote {OUT}/accuracy_summary.csv, paired_mcnemar.csv, "
          f"paired_mcnemar.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
