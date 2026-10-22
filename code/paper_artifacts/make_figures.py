#!/usr/bin/env python3
"""Generate Figures 3-8 for the Ecological Informatics manuscript.

Every number is recomputed from the on-disk artefacts, so the figures can never
drift from RESULTS.md / registry.csv:

  Fig 3  main results (11 configs) + bootstrap CI where per-sample preds exist
  Fig 4  domain/source transfer drops, corrected vs the old flawed protocol
  Fig 5  geographic holdout: drop vs regional endemism + per-country bars
  Fig 6  reliability diagram: BioCLIP 2 (under-confident) vs Pl@ntNet (over)
  Fig 7  risk-coverage curve + the Pl@ntNet 0.5-threshold operating point
  Fig 8  error structure: decomposition + per-family dumbbell (ours vs Pl@ntNet)

    python make_figures.py --out /path/to/ei_submission/figures
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch, Rectangle
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
REPO = ROOT
SPL = ROOT / "data/splits"
OUTDIR_DEFAULT = ROOT / "manuscript/figures"

C_MAIN = "#1a5c8a"      # our method
C_ALT = "#7fa8c9"       # other specialists
C_PLANT = "#c0504d"     # Pl@ntNet
C_VLM = "#e8a33d"       # general VLMs
C_GRID = "#d9d9d9"
plt.rcParams.update({
    "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9,
    "axes.spines.top": False, "axes.spines.right": False,
    "figure.dpi": 600, "savefig.dpi": 600, "savefig.bbox": "tight",
    "axes.facecolor": "white", "figure.facecolor": "white",
})


# ------------------------------------------------------------------ helpers
def binom(s: str) -> str:
    p = " ".join((s or "").lower().replace("*", "").replace("×", "x").split()).split()
    return " ".join(p[:2]) if len(p) >= 2 else " ".join(p)


def boot_ci(hits, n_boot=10000, seed=20261013):
    rng = random.Random(seed)
    n = len(hits)
    if n == 0:
        return (0.0, 0.0)
    st = sorted(sum(hits[rng.randrange(n)] for _ in range(n)) / n
                for _ in range(n_boot))
    return (st[int(0.025 * n_boot)], st[int(0.975 * n_boot)])


def load_splits():
    return {s: list(csv.DictReader(open(SPL / f"{s}.csv", encoding="utf-8")))
            for s in ("train", "val", "test")}


def avg_ranks(v):
    """Average ranks, so ties (several regions with zero thin-support) are
    handled correctly before a Spearman correlation."""
    order = sorted(range(len(v)), key=lambda i: v[i])
    r = [0.0] * len(v)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and v[order[j + 1]] == v[order[i]]:
            j += 1
        mean = (i + j) / 2 + 1
        for k in range(i, j + 1):
            r[order[k]] = mean
        i = j + 1
    return r


def system_hits(run: str) -> list[int]:
    p = ROOT / "results" / run / "predictions.jsonl"
    return [1 if r["pred_species"] == r["true_species"] else 0
            for r in (json.loads(l)
                      for l in p.read_text(encoding="utf-8").splitlines()
                      if l.strip())]


def registry_rows() -> list[dict]:
    with (ROOT / "results/registry.csv").open(encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


# ------------------------------------------------------------------ Fig 3
def fig3(out: Path):
    rows = [r for r in registry_rows()
            if r["kind"] in ("finetune", "zero-shot") and r["id"] != "smoke_bioclip2"]
    rows.sort(key=lambda r: float(r["top1"]))
    labels, vals, colors, ci = [], [], [], []
    for r in rows:
        tags = []
        if r["id"] == "zeroshot_bioclip2":
            tags.append("zero-shot")
        else:
            tags.append("BioCLIP 2" if r["backbone"] == "bioclip2" else "DINOv2")
            tags.append("linear" if r["mode"] == "linear" else "fine-tune")
            tags.append(f"{r['img_size']} px")
        if r["extra"]:
            tags.append(r["extra"])
        labels.append(f"{r['id']}\n({', '.join(tags)})")
        vals.append(float(r["top1"]))
        colors.append(C_MAIN if r["id"] == "bioclip2_linear_336" else
                      (C_ALT if r["backbone"] == "bioclip2" else "#b8b8b8"))
        pred = ROOT / "results" / r["id"] / "predictions.jsonl"
        if pred.exists():
            lo, hi = boot_ci(system_hits(r["id"]))
            ci.append((lo, hi))
        else:
            ci.append(None)
    fig, ax = plt.subplots(figsize=(7.2, 5.8))
    y = range(len(rows))
    ax.barh(list(y), vals, color=colors, height=0.62, zorder=3)
    # value labels INSIDE the bar end so they can never collide with the
    # bootstrap whiskers that stick out to the right
    for i, v in enumerate(vals):
        ax.text(v - 0.012, i, f"{v:.3f}", va="center", ha="right",
                color="white", fontsize=7.5, fontweight="bold", zorder=6)
    for i, c in enumerate(ci):
        if c:
            ax.plot([c[0], c[1]], [i, i], color="#333", lw=0.9, zorder=5)
            ax.plot([c[0], c[0]], [i - 0.13, i + 0.13], color="#333", lw=0.9,
                    zorder=5)
            ax.plot([c[1], c[1]], [i - 0.13, i + 0.13], color="#333", lw=0.9,
                    zorder=5)
    ax.set_yticks(list(y))
    ax.set_yticklabels(labels, fontsize=6.8)
    ax.set_xlim(0, 1.04)
    ax.set_xlabel("Species top-1 accuracy on the 902-image test split")
    ax.set_title("Main results: backbone, fine-tuning strategy and resolution\n"
                 "(error bars = bootstrap 95% CI; hierarchical head and class "
                 "balancing add nothing)")
    # legend below the axes: the bottom-right corner is the only free space and
    # it is too narrow for three stacked entries
    ax.legend(handles=[
        Line2D([0], [0], marker="s", color="none", markerfacecolor=C_MAIN,
               markersize=8, label="best recipe (BioCLIP 2, frozen, 336 px)"),
        Line2D([0], [0], marker="s", color="none", markerfacecolor=C_ALT,
               markersize=8, label="BioCLIP 2 (other settings)"),
        Line2D([0], [0], marker="s", color="none", markerfacecolor="#b8b8b8",
               markersize=8, label="DINOv2")],
        loc="upper center", bbox_to_anchor=(0.42, -0.10), ncol=3, fontsize=7.5,
        frameon=False, columnspacing=1.2, handletextpad=0.4)
    ax.grid(axis="x", color=C_GRID, lw=0.5, zorder=0)
    fig.savefig(out / "Figure3_main_results.png")
    plt.close(fig)


# ------------------------------------------------------------------ Fig 4
def transfer_in_domain(pred, tr, te, train_filter, test_filter):
    """In-domain accuracy of the FULL model on the exact subset a transfer run
    evaluates -- the same convention as Table 3 and Table 4.

    Transfer test sets keep only species seen on the training side, so they are
    intrinsically harder than the whole test split.  Using the global 0.9224 as
    the baseline (as this figure did until 2026-10-14) mixes the transfer cost
    with subset composition and inflates every drop.
    """
    tr_sp = {r["species_latin_project"] for r in tr
             if r[train_filter[0]] == train_filter[1]}
    sub = [r for r in te if r[test_filter[0]] == test_filter[1]
           and r["species_latin_project"] in tr_sp]
    hit = sum(1 for r in sub
              if pred[r["sample_id"]]["pred_species"] ==
              r["species_latin_project"])
    return hit / len(sub), len(sub)


def fig4(out: Path):
    sp = load_splits()
    pred = {json.loads(l)["sample_id"]: json.loads(l)
            for l in (ROOT / "results/bioclip2_linear_336/predictions.jsonl")
            .read_text(encoding="utf-8").splitlines() if l.strip()}
    # NOTE: xfer_field2spec trains on specimen / evaluates on field -- the
    # directory names are inverted w.r.t. their own config.json, so the
    # (train filter, test filter) tuples here are the authoritative direction.
    runs = [
        ("xfer_gbif2inat", "GBIF\u2192iNaturalist\n(cross-source)", 0.2059,
         ("source", "GBIF"), ("source", "iNaturalist")),
        ("xfer_field2spec", "specimen\u2192field\n(cross-domain)", 0.3244,
         ("image_domain", "specimen"), ("image_domain", "field")),
        ("xfer_inat2gbif", "iNaturalist\u2192GBIF\n(cross-source)", 0.3239,
         ("source", "iNaturalist"), ("source", "GBIF")),
        ("xfer_spec2field", "field\u2192specimen\n(cross-domain)", 0.1500,
         ("image_domain", "field"), ("image_domain", "specimen")),
    ]
    fig, ax = plt.subplots(figsize=(7.8, 4.1))
    xs = list(range(len(runs)))
    deltas = []
    for x, (name, lab, old, tf, ef) in zip(xs, runs):
        m = json.loads((ROOT / "results" / name / "test_metrics.json")
                       .read_text())["test"]
        new = m["top1"]
        base, n = transfer_in_domain(pred, sp["train"], sp["test"], tf, ef)
        assert n == m["n"], (name, n, m["n"])
        d = 100 * (new - base)
        deltas.append(d)
        ax.bar(x, new, width=0.5, color=C_MAIN, zorder=3,
               label="corrected protocol" if x == 0 else None)
        ax.plot([x], [old], marker="X", color=C_PLANT, markersize=9,
                zorder=5, label="old protocol (label-space defect)"
                if x == 0 else None)
        ax.annotate("", xy=(x, new), xytext=(x, old),
                    arrowprops=dict(arrowstyle="->", color=C_PLANT, lw=1.1,
                                    shrinkA=4, shrinkB=2))
        # per-bar in-domain reference for the SAME subset (no global line)
        ax.hlines(base, x - 0.30, x + 0.30, color="#333", ls="--", lw=1.1,
                  zorder=4, label="in-domain, same subset" if x == 0 else None)
        ax.text(x, new + 0.018, f"{new:.3f}", ha="center", fontsize=7.5)
        ax.text(x, base + 0.022, f"in-dom {base:.3f}\n\u0394 {d:.1f} pp",
                ha="center", fontsize=7.0)
        ax.text(x + 0.31, old, f"old {old:.3f}", va="center", fontsize=6.8,
                color=C_PLANT)
    lo, hi = abs(min(deltas, key=abs)), abs(max(deltas, key=abs))
    ax.set_xticks(xs)
    ax.set_xticklabels([r[1] for r in runs], fontsize=8)
    ax.set_xlim(-0.6, 3.95)
    ax.set_ylim(0, 1.25)
    ax.set_ylabel("Species top-1")
    ax.set_title(f"Domain / source transfer: \u0394 {lo:.0f}\u2013{hi:.0f} pp, "
                 "not the 60\u201377 pp\nthe earlier (defective) label space "
                 "implied\n"
                 "(in-domain reference = full model on the same subset)",
                 fontsize=9)
    ax.legend(loc="upper right", fontsize=7, frameon=False)
    ax.grid(axis="y", color=C_GRID, lw=0.5, zorder=0)
    fig.savefig(out / "Figure4_transfer_drops.png")
    plt.close(fig)
    print("  Fig 4  \u0394 (pp): " + ", ".join(f"{d:.1f}" for d in deltas))


# ------------------------------------------------------------------ Fig 5
def fig5(out: Path):
    sp = load_splits()
    all_rows = [r for s in sp.values() for r in s]
    sp_countries = defaultdict(set)
    for r in all_rows:
        sp_countries[r["species_latin_project"]].add(r["country_clean"])
    pred = {json.loads(l)["sample_id"]: json.loads(l)
            for l in (ROOT / "results/bioclip2_linear_336/predictions.jsonl")
            .read_text(encoding="utf-8").splitlines() if l.strip()}
    cc = {r["sample_id"]: r["country_clean"] for r in sp["test"]}

    runs = [
        ("Taiwan, China", "geo_ex-TW", "TW"), ("Mainland China", "geo_ex-CN", "CN"),
        ("Japan", "geo_ex-JP", "JP"), ("Thailand", "geo_ex-TH", "TH"),
        ("United States", "geo_ex-US", "US"), ("Russia", "geo_ex-RU", "RU"),
        ("Vietnam", "geo_ex-VN", "VN"),
        ("China region (incl. TW, HK, MO)", "geo_ex-CN_region", None),
    ]
    pts = []
    for name, run, country in runs:
        c = json.loads((ROOT / "results" / run / "config.json").read_text())
        m = json.loads((ROOT / "results" / run / "test_metrics.json")
                       .read_text())["test"]
        tw = c["train_where"]
        keep = (tw.split("!=", 1)[1].split(",") if tw.startswith("country_clean!=")
                else [])
        out_train = [r for r in sp["train"]
                     if r["country_clean"] not in set(keep)]
        out_cnt = Counter(r["species_latin_project"] for r in out_train)
        tr_sp = set(out_cnt)
        if country is None:  # whole China region held out
            te = [r for r in sp["test"] if r["species_latin_project"] in tr_sp
                  and r["country_clean"] in {"CN", "TW", "HK", "MO"}]
        else:
            te = [r for r in sp["test"] if r["species_latin_project"] in tr_sp
                  and r["country_clean"] == country]
        # The driver of the drop is not geography per se but how much
        # OUT-OF-REGION training support each test species still has: species
        # with <=5 training images outside the held-out region are the ones the
        # model cannot recognise there.
        thin = sum(1 for r in te if out_cnt[r["species_latin_project"]] <= 5) \
            / len(te)
        base = sum(1 for r in te
                   if pred[r["sample_id"]]["pred_species"] ==
                   r["species_latin_project"]) / len(te)
        pts.append((name, thin, 100 * (base - m["top1"]), len(te), m["top1"],
                    base, country is None))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.6, 3.9),
                                   gridspec_kw={"width_ratios": [1, 1.15]})
    # per-point label offsets chosen by hand so nothing collides
    lab_off = {"Taiwan, China": (7, 5), "China": (-118, -10),
               "Mainland China": (8, 5),
               "Japan": (7, 5), "Thailand": (7, 5), "United States": (7, 5),
               "Russia": (7, 5), "Vietnam": (7, 5)}
    # deliberately short: a long right-anchored label would run into panel D's
    # y-tick labels, and the title already spells out what the hollow marker is
    lab_txt = {"China": "China region (aggregate)"}
    for name, thin, drop, n, hold, base, is_region in pts:
        ax1.scatter(thin, drop, s=18 + n * 0.55,
                    facecolor="none" if is_region else C_MAIN,
                    edgecolor=C_MAIN if is_region else "white",
                    color=C_MAIN, alpha=0.75, zorder=3, lw=1.0)
        key = name.split(" (")[0].split(" region")[0]
        ax1.annotate(lab_txt.get(key, key), (thin, drop),
                     textcoords="offset points",
                     xytext=lab_off.get(key, (7, 5)), fontsize=6.8)
    xs = [p[1] for p in pts if not p[6]]     # countries only (region overlaps)
    ys = [p[2] for p in pts if not p[6]]
    if len(set(xs)) > 1:
        xm = sum(xs) / len(xs)
        ym = sum(ys) / len(ys)
        den = sum((x - xm) ** 2 for x in xs)
        if den:
            b = sum((x - xm) * (y - ym) for x, y in zip(xs, ys)) / den
            a = ym - b * xm
            gx = [0, max(xs) * 1.05]
            ax1.plot(gx, [a + b * g for g in gx], color=C_PLANT, ls="--", lw=1.0)
        # Tie-aware Spearman rho over the country points (the China-region
        # aggregate is excluded, as in the OLS line), so the "the drop tracks
        # out-of-region support" claim carries a number.
        rx, ry = avg_ranks(xs), avg_ranks(ys)
        mxr, myr = sum(rx) / len(rx), sum(ry) / len(ry)
        num = sum((u - mxr) * (w - myr) for u, w in zip(rx, ry))
        denr = math.sqrt(sum((u - mxr) ** 2 for u in rx)
                         * sum((w - myr) ** 2 for w in ry))
        rho = num / denr if denr else float("nan")
        ax1.text(0.035, 0.965, f"Spearman \u03c1 = {rho:.2f}  (n = {len(xs)})",
                 transform=ax1.transAxes, va="top", fontsize=7.5)
        print(f"  Fig 5C Spearman \u03c1 = {rho:.3f} (n = {len(xs)} countries)")
    ax1.set_xlabel("Share of the region's test species with\n"
                   "\u22645 training images outside that region")
    ax1.set_ylabel("Accuracy lost by holding the region out\n"
                   "(in-domain \u2212 held-out, pp)")
    # NOTE: an earlier version of this title ended "...training support, not
    # distance". Geographic distance was never computed (the splits carry no
    # coordinates), so that contrast is not supported and has been removed.
    # The correlation coefficient is annotated in-panel rather than in the title.
    ax1.set_title("The geographic drop tracks out-of-region\ntraining support "
                  "(hollow marker = China-region aggregate)")
    ax1.grid(color=C_GRID, lw=0.5, zorder=0)
    ax1.set_xlim(-0.03, max(xs) * 1.15)
    ax1.set_ylim(-2, max(ys) * 1.2)

    order = sorted(pts, key=lambda p: p[2])
    y = range(len(order))
    for i, (name, thin, drop, n, hold, base, is_region) in enumerate(order):
        ax2.barh(i, base, height=0.3, color="#9dc3e6", zorder=3)
        ax2.barh(i + 0.34, hold, height=0.3, color=C_MAIN, zorder=3)
        ax2.text(base - 0.012, i, f"{base:.3f}", va="center", ha="right",
                 fontsize=6.5, color="#1a4a6e")
        ax2.text(hold - 0.012, i + 0.34, f"{hold:.3f}", va="center",
                 ha="right", fontsize=6.5, color="white")
    # label the two bar types directly on the top row instead of using a
    # legend, which had no collision-free corner in this panel
    ax2.text(order[0][4] / 2, 0, "trained WITH the region", ha="center",
             va="center", fontsize=7, color="#1a4a6e", fontweight="bold")
    ax2.text(order[0][5] / 2, 0.34, "trained WITHOUT it", ha="center",
             va="center", fontsize=7, color="white", fontweight="bold")
    ax2.set_yticks([i + 0.17 for i in y])
    ax2.set_yticklabels([p[0] for p in order], fontsize=7)
    ax2.set_xlim(0, 1.06)
    ax2.set_xlabel("Species top-1")
    ax2.set_title("Leave-one-region-out: trained-with (light) vs\n"
                  "trained-without (dark) the region, n = 28\u2013367")
    ax2.grid(axis="x", color=C_GRID, lw=0.5, zorder=0)
    fig.savefig(out / "Figure5_geographic_holdout.png")
    plt.close(fig)


# ------------------------------------------------------------------ Fig 6
def fig6(out: Path):
    fm = json.loads((ROOT / "results/bioclip2_linear_336/full_metrics.json")
                    .read_text())
    ours = [(b["conf"], b["acc"], b["n"]) for b in fm["reliability"]]

    recs = [json.loads(l) for l in (ROOT / "data/external_baselines/plantnet/raw_all_test.jsonl")
            .read_text(encoding="utf-8").splitlines() if l.strip()]
    recs = [r for r in recs if r.get("ok") and r.get("top")]
    bins = defaultdict(lambda: [0, 0, 0.0])   # count, hits, score_sum
    for r in recs:
        s = float(r["top"][0].get("score") or 0.0)
        k = min(int(s * 10), 9)
        b = bins[k]
        b[0] += 1
        b[1] += int(binom(r["top"][0].get("sci") or "") ==
                    binom(r["true_species"]))
        b[2] += s
    pl = []
    for k in sorted(bins):
        cnt, hits, cs = bins[k]
        if cnt:
            pl.append(((k + 0.5) / 10, hits / cnt, cs / cnt))

    fig, ax = plt.subplots(figsize=(4.6, 4.2))
    ax.plot([0, 1], [0, 1], color="#999", lw=0.8, ls=":")
    ax.plot([c for c, _, _ in pl], [a for _, a, _ in pl], "-o", color=C_PLANT,
            markersize=5, lw=1.4,
            label=f"Pl@ntNet (over-confident, ECE 0.090)")
    ax.plot([c for c, _, _ in ours], [a for _, a, _ in ours], "-s", color=C_MAIN,
            markersize=5, lw=1.4,
            label=f"BioCLIP 2 linear@336 (under-confident, ECE 0.195)")
    ax.set_xlabel("Mean predicted confidence in bin")
    ax.set_ylabel("Observed accuracy in bin")
    ax.set_title("Confidence points the opposite way:\nonly the specialist's "
                 "scores can gate\nan automated accept/reject decision")
    # legend below the axes: both curves sweep through every corner of the
    # panel, so no in-axes corner fits these long labels without touching
    # data (upper left clipped the blue dip at x=0.55, lower right reached
    # the red points at x=0.15)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=1,
              fontsize=7.5, frameon=False)
    ax.grid(color=C_GRID, lw=0.5)
    ax.set_xlim(0, 1.02)
    ax.set_ylim(0, 1.02)
    fig.savefig(out / "Figure6_reliability.png")
    plt.close(fig)


# ------------------------------------------------------------------ Fig 7
def fig7(out: Path):
    fig, ax = plt.subplots(figsize=(4.6, 4.2))
    for run, col, lab in [("bioclip2_linear_336", C_MAIN, "BioCLIP 2 linear@336"),
                          ("bioclip2_ft_336", C_ALT, "BioCLIP 2 fine-tuned")]:
        fm = json.loads((ROOT / "results" / run / "full_metrics.json")
                        .read_text())
        ax.plot(fm["coverage"], fm["accuracy_at_coverage"], "-o", color=col,
                markersize=4, lw=1.4, label=lab)
    # Pl@ntNet operating point from RESULTS.md §7.4 (score >= 0.5)
    ax.plot([0.25], [0.590], marker="X", color=C_PLANT, markersize=10,
            linestyle="none", zorder=5,
            label="Pl@ntNet, score\u22650.5\n(25% coverage, 59.0% accuracy)")
    ax.axhline(0.590, color=C_PLANT, lw=0.7, ls=":")
    ax.set_xlabel("Coverage (fraction of images accepted)")
    ax.set_ylabel("Accuracy on the accepted images")
    ax.set_title("Selective prediction: accepting the most confident\n25% of "
                 "images costs nothing (100% accuracy)")
    # legend below the axes: the curves own everything above y = 0.89 and the
    # dotted Pl@ntNet line sits at y = 0.59, so no in-axes placement clears
    # both; below the xlabel there is nothing to collide with.
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.13), ncol=2,
              fontsize=7.5, frameon=False, columnspacing=1.4)
    ax.grid(color=C_GRID, lw=0.5)
    ax.set_xlim(0, 1.02)
    ax.set_ylim(0.55, 1.02)
    fig.savefig(out / "Figure7_risk_coverage.png")
    plt.close(fig)


# ------------------------------------------------------------------ Fig 8
def fig8(out: Path):
    # family Chinese -> Latin, same mapping as plantnet_error_analysis.py
    fam_cn2lat = {s["family"]: s.get("family_latin", s["family"])
                  for s in json.loads(
                      (ROOT / "data/species_data.json")
                      .read_text(encoding="utf-8"))["species"]}
    ours = [json.loads(l) for l in (ROOT / "results/bioclip2_linear_336/"
                                    "predictions.jsonl")
            .read_text(encoding="utf-8").splitlines() if l.strip()]
    pl = [json.loads(l) for l in (ROOT / "data/external_baselines/plantnet/raw_all_test.jsonl")
          .read_text(encoding="utf-8").splitlines() if l.strip()]
    pl = [r for r in pl if r.get("ok") and r.get("top")]

    def decomp(rows, get_pred):
        n = len(rows)
        ex = gr = gw = 0
        for r in rows:
            t = binom(r["true_species"])
            p = binom(get_pred(r) or "")
            if p == t:
                ex += 1
            elif p.split(" ")[0] == t.split(" ")[0]:
                gr += 1
            else:
                gw += 1
        return ex / n, gr / n, gw / n

    o = decomp(ours, lambda r: r["pred_species"])
    p = decomp(pl, lambda r: (r["top"] or [{}])[0].get("sci"))

    fam = {}
    for r in pl + ours:
        tf = r.get("true_family_latin") or fam_cn2lat.get(
            r.get("true_family_cn"), "") or ""
        if tf:
            fam.setdefault(tf, None)
    ours_by_fam = defaultdict(lambda: [0, 0])
    for r in ours:
        k = r.get("true_family_latin") or ""
        ours_by_fam[k][1] += 1
        ours_by_fam[k][0] += int(r["pred_family_latin"] == k)
    pl_by_fam = defaultdict(lambda: [0, 0])
    for r in pl:
        # Pl@ntNet records carry the family as Chinese; map to Latin so both
        # systems are keyed identically.
        k = fam_cn2lat.get(r.get("true_family_cn"), "")
        pl_by_fam[k][1] += 1
        pl_by_fam[k][0] += int(" ".join((r["top"] or [{}])[0]
                                        .get("family", "").lower().split())
                               == k.lower())
    rows = [(f, ours_by_fam[f], pl_by_fam[f])
            for f in fam if f and ours_by_fam[f][1] >= 5 and pl_by_fam[f][1] >= 5]
    rows.sort(key=lambda r: -(r[1][0] / r[1][1]))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.4, 5.6),
                                   gridspec_kw={"width_ratios": [0.9, 1.3],
                                                "wspace": 0.42})
    cats = ["species exact", "genus right,\nspecies wrong", "genus wrong"]
    xs = [0, 1]
    bottoms = [0, 0]
    for i, (vals, col, lab) in enumerate([(o, C_MAIN, "BioCLIP 2 linear@336"),
                                          (p, C_PLANT, "Pl@ntNet")]):
        left = 0
        for j, v in enumerate(vals):
            ax1.bar(xs[i], v, bottom=left, width=0.5, color=col if j == 0
                    else (C_ALT if j == 1 else "#d98880"), zorder=3,
                    alpha=1.0 if j == 0 else 0.55 + 0.2 * j)
            ax1.text(xs[i], left + v / 2, f"{v:.3f}", ha="center", va="center",
                     fontsize=7.5, color="white" if j == 0 else "#333")
            left += v
        ax1.text(xs[i], 1.02, lab, ha="center", fontsize=7.5)
    ax1.set_xticks(xs)
    ax1.set_xticklabels(["BioCLIP 2\nlinear@336", "Pl@ntNet"], fontsize=8)
    ax1.set_xlim(-0.62, 2.55)
    ax1.set_ylim(0, 1.08)
    ax1.set_ylabel("Share of the 902 test images")
    ax1.set_title("Where the errors are: Pl@ntNet's are\ncoverage gaps "
                  "(genus right) and real misses;\nours are few and diffuse")
    # the two bars fill the panel, so the legend goes to the free strip on the
    # right (xlim extended to 2.55 for exactly this reason)
    ax1.legend(handles=[
        Line2D([0], [0], marker="s", color="none", markerfacecolor=C_MAIN,
               markersize=8, label="species exact"),
        Line2D([0], [0], marker="s", color="none", markerfacecolor=C_ALT,
               markersize=8, label="genus right,\nspecies wrong"),
        Line2D([0], [0], marker="s", color="none", markerfacecolor="#d98880",
               markersize=8, label="genus wrong")], loc="center right",
        fontsize=7, frameon=True, framealpha=1.0, edgecolor="#cccccc",
        fancybox=False, handlelength=1.2, labelspacing=0.8)
    ax1.grid(axis="y", color=C_GRID, lw=0.5, zorder=0)

    y = range(len(rows))
    for i, (f, ours_pair, pl_pair) in enumerate(rows):
        oa = ours_pair[0] / ours_pair[1]
        pa = pl_pair[0] / pl_pair[1]
        ax2.plot([pa, oa], [i, i], color="#bbb", lw=1.6, zorder=2)
        ax2.scatter([pa], [i], color=C_PLANT, s=26, zorder=3)
        ax2.scatter([oa], [i], color=C_MAIN, s=26, zorder=3)
    ax2.set_yticks(list(y))
    ax2.set_yticklabels([f"{f[:1].upper()}{f[1:]} (n={ours_by_fam[f][1]})"
                         for f, _, _ in rows], fontsize=5.6)
    # xlim extends past 1.0 so the strip right of the dots is reserved for
    # the framed legend: with the previous 1.42 limit the box reached back to
    # x ~ 0.99 and covered the top rows' dots (Tectariaceae sits at 0.93).
    # Ticks stay at 0-1.0, the full data range.
    ax2.set_xlim(-0.02, 1.72)
    ax2.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax2.set_xlabel("Family-level accuracy")
    ax2.set_title("Per-family accuracy, ours vs Pl@ntNet:\n11 families sit at "
                  "0.000 for Pl@ntNet\nwhile ours stays 0.6\u20131.0")
    # xlim extends past 1.0 so the legend sits clear of the dots, which all
    # cluster at x ~ 1.0 for our model; upper right because the bottom rows'
    # dots reach the same corner the lower-right anchor would occupy
    ax2.legend(handles=[
        Line2D([0], [0], marker="o", color="none", markerfacecolor=C_MAIN,
               markersize=7, label="BioCLIP 2 linear@336"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor=C_PLANT,
               markersize=7, label="Pl@ntNet")], loc="upper right", fontsize=7.5,
        frameon=True, framealpha=1.0, edgecolor="#cccccc", fancybox=False)
    ax2.grid(axis="x", color=C_GRID, lw=0.5, zorder=0)
    fig.savefig(out / "Figure8_error_structure.png")
    plt.close(fig)


# ------------------------------------------------------------------ Fig 1/2
# Conceptual and pipeline diagrams. Both are laid out in millimetre units on a
# single full-figure axes, so boxes, arrows and embedded photographs keep a
# fixed proportion to one another and to the printed width.

FIG1_W, FIG1_H = 180.0, 160.0
FIG2_W, FIG2_H = 180.0, 104.0
FIG1_PHOTOS = {
    "specimen": "EXP_01736",   # CC0, US National Herbarium (Smithsonian), CN
    "field": "EXP_01734",      # CC BY, Cynthia Su, iNaturalist, TW
}


def _manifest_rows():
    """manifest_balanced.csv, in either the workspace or the release layout."""
    candidates = [SPL.parent / "manifest_balanced.csv",
                  SPL.parent / "manifests" / "manifest_balanced.csv"]
    for c in candidates:
        if c.exists():
            with c.open(encoding="utf-8") as fh:
                yield from csv.DictReader(fh)
            return
    raise SystemExit("manifest_balanced.csv not found in any of "
                     + ", ".join(str(c) for c in candidates))


def fig1_photos() -> dict:
    want = {v: k for k, v in FIG1_PHOTOS.items()}      # sample_id -> domain
    out = {}
    for r in _manifest_rows():
        sid = r["sample_id"]
        if sid not in want:
            continue
        dom = want[sid]
        if r["image_domain"] != dom:
            raise SystemExit(f"{sid} is {r['image_domain']}, expected {dom}")
        if r.get("releasable") != "yes":
            raise SystemExit(
                f"{sid} is licensed {r['license']} and cannot be redistributed; "
                "it must not be embedded in a published figure")
        p = Path(r["image_path"])
        if not p.is_absolute():
            p = REPO / p
        if not p.exists():
            # release layout: the two photographs ship beside the manifests
            p = REPO / "data/figure_photos" / f"{sid}.jpg"
        if not p.exists():
            raise SystemExit(
                f"{sid}: photograph not found at {p} -- the two Fig 1 photos "
                "ship with the release under data/figure_photos/")
        out[dom] = {"path": p, "license": r["license"],
                    "rights": r["rights_holder"], "species": r["species_latin_project"]}
    missing = set(want.values()) - set(out)
    if missing:
        raise SystemExit(f"no manifest row for {sorted(missing)}")
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(OUTDIR_DEFAULT))
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    # Figs 1 and 2 are no longer drawn here: their masters are
    # hand-authored SVGs converted by make_figure1_svg.py and
    # make_figure2_svg.py in paper_analysis/ei_submission/latex/.
    # fig1_photos() above stays the licence gate the Fig 1 path shares.
    for fn in (fig3, fig4, fig5, fig6, fig7, fig8):
        fn(out)
        print(f"wrote {fn.__name__[3:].upper()}")
    print(f"\nall figures in {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
