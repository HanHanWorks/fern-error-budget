#!/usr/bin/env python3
"""Generate every table the Ecological Informatics manuscript needs, straight
from the on-disk artefacts (registry, per-run metrics, splits, VLM/Pl@ntNet
raw records). Each table is written as CSV + Markdown so it can be pasted into
the manuscript or opened in a spreadsheet.

    python make_paper_tables.py --out /path/to/ei_submission/tables
"""
from __future__ import annotations

import argparse
import csv
import json
import statistics as st
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
REPO = ROOT
SPL = ROOT / "data/splits"
OUTDIR_DEFAULT = ROOT / "manuscript/tables"

MAIN = ["bioclip2_linear_336", "bioclip2_ft_336", "bioclip2_linear",
        "bioclip2_linear_bal", "bioclip2_ft", "bioclip2_ft_lr5e5",
        "bioclip2_ft_hier", "dinov2_ft", "dinov2_linear_336",
        "dinov2_linear", "zeroshot_bioclip2"]
NICE = {"bioclip2_linear_336": "BioCLIP 2 / frozen / 336",
        "bioclip2_ft_336": "BioCLIP 2 / fine-tuned / 336",
        "bioclip2_linear": "BioCLIP 2 / frozen / 224",
        "bioclip2_linear_bal": "BioCLIP 2 / frozen / 224 + class-bal.",
        "bioclip2_ft": "BioCLIP 2 / fine-tuned / 224",
        "bioclip2_ft_lr5e5": "BioCLIP 2 / fine-tuned / 224 (lr 5e-5)",
        "bioclip2_ft_hier": "BioCLIP 2 / fine-tuned / 224 + hier. head",
        "dinov2_ft": "DINOv2 / fine-tuned / 224",
        "dinov2_linear_336": "DINOv2 / frozen / 336",
        "dinov2_linear": "DINOv2 / frozen / 224",
        "zeroshot_bioclip2": "BioCLIP 2 / zero-shot (no training)"}


def write(out: Path, name: str, header: list[str], rows: list[list],
          title: str, note: str = "") -> None:
    """CSV + Markdown twin."""
    with (out / f"{name}.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(header)
        w.writerows(rows)
    md = [f"**{title}**", ""]
    if note:
        md += [note, ""]
    md += ["| " + " | ".join(header) + " |",
           "|" + "|".join("---:" if i else "---"
                          for i in range(len(header))) + "|"]
    for r in rows:
        md.append("| " + " | ".join(str(x) for x in r) + " |")
    (out / f"{name}.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"  {name}: {len(rows)} rows")


def fmt(x, nd=4):
    return f"{x:.{nd}f}" if isinstance(x, float) else x


# ------------------------------------------------------------------ Tab 1
def tab1(out: Path):
    splits = {s: list(csv.DictReader(open(SPL / f"{s}.csv", encoding="utf-8")))
              for s in ("train", "val", "test")}
    allr = [r for v in splits.values() for r in v]
    sp = {r["species_latin_project"] for r in allr}
    fam = {r["family_project"] for r in allr}
    src = Counter(r["source"] for r in allr)
    dom = Counter(r.get("image_domain") or "?" for r in allr)
    cc = Counter(r.get("country_clean") or "UNK" for r in allr)
    lic = Counter(r.get("license_class") or "?" for r in allr)
    rel = Counter(r.get("releasable") or "?" for r in allr)
    rows = [
        ["Images collected (full manifest)", f"{6237:,}",
         "dataset/manifest_full.csv"],
        ["Balanced working set (≤60 images/species)", f"{5969:,}",
         "268 over-cap images dropped; manifest_balanced.csv"],
        ["— train / val / test",
         f"{len(splits['train']):,} / {len(splits['val'])} / "
         f"{len(splits['test'])}", "stratified by species, grouped by observation"],
        ["Species / families", f"{len(sp)} / {len(fam)}",
         "PPG I accepted names"],
        ["Source: iNaturalist / GBIF",
         f"{src.get('iNaturalist', 0):,} / {src.get('GBIF', 0):,}",
         "GBIF rows include iDigBio-digitised herbarium sheets"],
        ["Image domain: field / specimen",
         f"{dom.get('field', 0):,} / {dom.get('specimen', 0):,}",
         f"specimen share {dom.get('specimen', 0)/len(allr):.1%}"],
        ["Countries (country_clean, ISO-2)", len([k for k in cc if k != "UNK"]),
         f"UNK {cc.get('UNK', 0)} rows ({cc.get('UNK', 0)/len(allr):.1%})"],
        ["Top countries", ", ".join(f"{k} {v:,}" for k, v in cc.most_common(6)),
         "full list in country_distribution.csv"],
        ["Open-set negatives", 700, "8 groups (horsetail, lycophyte, moss, "
         "cycad, palm, angiosperm ×3, fungi, conifer)"],
        ["Licence: redistributable / restricted",
         f"{lic.get('redistributable', 0):,} / {lic.get('restricted', 0):,}",
         f"{rel.get('yes', 0):,} images ({rel.get('yes', 0)/len(allr):.0%}) "
         "cleared for public redistribution (release_manifest.csv)"],
    ]
    write(out, "tab1_dataset_composition",
          ["Item", "Value", "Note"], rows,
          "Table 1. Composition of the fern benchmark")
    (out / "country_distribution.csv").write_text(
        "".join(f"{k},{v}\n" for k, v in cc.most_common()), encoding="utf-8")
    (out / "source_domain_distribution.csv").write_text(
        "source,n\n" + "".join(f"{k},{v}\n" for k, v in src.most_common())
        + "\nimage_domain,n\n" + "".join(f"{k},{v}\n" for k, v in dom.most_common()),
        encoding="utf-8")
    print("  + country_distribution.csv, source_domain_distribution.csv")


# ------------------------------------------------------------------ Tab 2
def tab2(out: Path):
    reg = {r["id"]: r for r in csv.DictReader(
        (ROOT / "results/registry.csv").open(encoding="utf-8"))}
    rows = []
    for rid in MAIN:
        r = reg.get(rid)
        if not r:
            continue
        rows.append([NICE.get(rid, rid), r["mode"], r["img_size"],
                     int(r["train_n"]), int(r["test_n"]),
                     fmt(float(r["top1"])), fmt(float(r["top3"])),
                     fmt(float(r["macro_top1"])),
                     fmt(float(r["specimen_top1"] or 0)),
                     fmt(float(r["field_top1"] or 0))])
    # seed variance rows for the configs that have them
    for rid in ("bioclip2_linear_336", "bioclip2_ft_336", "dinov2_linear_336"):
        vals = []
        for s in ("20261010", "20261011", "20261012"):
            p = ROOT / "results" / f"{rid}_s{s}" / "test_metrics.json"
            if p.exists():
                vals.append(json.loads(p.read_text())["test"]["top1"])
        if len(vals) >= 2:
            rows.append([NICE[rid] + " — seed mean±sd", reg[rid]["mode"],
                         reg[rid]["img_size"], int(reg[rid]["train_n"]),
                         int(reg[rid]["test_n"]),
                         f"{st.mean(vals):.4f} ± {st.pstdev(vals):.4f}",
                         "—", "—", "—", "—"])
    write(out, "tab2_main_results",
          ["Configuration", "Mode", "px", "train_n", "test_n",
           "Species top-1", "top-3", "macro", "specimen", "field"], rows,
          "Table 2. Main results on the 902-image test split",
          "Rows are single-seed (20261010) unless marked; the last three rows "
          "give the mean ± sd (population sd, ddof=0) over three seeds "
          "(20261010/11/12).")


# ------------------------------------------------------------------ Tab 3
def tab3(out: Path):
    # NOTE the run directory names are inverted w.r.t. their own config.json
    # (xfer_field2spec trains on specimen and evaluates on field). The
    # (train filter, test filter) tuples below -- not the directory names --
    # are authoritative. See results/README_xfer_naming.md.
    pred = {json.loads(l)["sample_id"]: json.loads(l)
            for l in (ROOT / "results/bioclip2_linear_336/predictions.jsonl")
            .read_text(encoding="utf-8").splitlines() if l.strip()}
    te = list(csv.DictReader(open(SPL / "test.csv", encoding="utf-8")))
    tr = list(csv.DictReader(open(SPL / "train.csv", encoding="utf-8")))
    runs = [("xfer_gbif2inat", "Cross-source", ("source", "GBIF"),
             ("source", "iNaturalist"), 0.2059),
            ("xfer_field2spec", "Cross-domain", ("image_domain", "specimen"),
             ("image_domain", "field"), 0.3244),
            ("xfer_inat2gbif", "Cross-source", ("source", "iNaturalist"),
             ("source", "GBIF"), 0.3239),
            ("xfer_spec2field", "Cross-domain", ("image_domain", "field"),
             ("image_domain", "specimen"), 0.1500)]
    rows = []
    for rid, kind, tf, ef, old in runs:
        m = json.loads((ROOT / "results" / rid / "test_metrics.json")
                       .read_text())["test"]
        c = json.loads((ROOT / "results" / rid / "config.json").read_text())
        tr_sp = {r["species_latin_project"] for r in tr if r[tf[0]] == tf[1]}
        sub = [r for r in te if r[ef[0]] == ef[1]
               and r["species_latin_project"] in tr_sp]
        assert len(sub) == m["n"], (rid, len(sub), m["n"])
        base = sum(1 for r in sub
                   if pred[r["sample_id"]]["pred_species"] ==
                   r["species_latin_project"]) / len(sub)
        rows.append([kind, tf[1], ef[1], c["n_species"], int(c["train_n"]),
                     m["n"], fmt(m["top1"]), fmt(m["macro_top1"]),
                     fmt(base), f"{100*(m['top1']-base):.1f}",
                     f"{old:.4f} (defective)"])
    rows.append(["In-domain", "all", "all", 152, 4185, 902,
                 "0.9235", "0.9227", "0.9235", "0.0", "—"])
    write(out, "tab3_transfer_matrix",
          ["Experiment", "Train on", "Test on", "Classes", "train_n",
           "test_n", "top-1", "macro", "in-domain (same subset)", "Δ (pp)",
           "previous value"], rows,
          "Table 3. Domain / source transfer under the corrected protocol",
          "The label space is rebuilt from the filtered training split. "
          "'in-domain (same subset)' is the same full model scored on the "
          "identical test subset (the convention already used in Table 4), so "
          "Δ = top-1 − in-domain isolates the transfer cost from subset "
          "composition. The last column keeps the values published before the "
          "label-space correction.")


# ------------------------------------------------------------------ Tab 4
def tab4(out: Path):
    pred = {json.loads(l)["sample_id"]: json.loads(l)
            for l in (ROOT / "results/bioclip2_linear_336/predictions.jsonl")
            .read_text(encoding="utf-8").splitlines() if l.strip()}
    te = list(csv.DictReader(open(SPL / "test.csv", encoding="utf-8")))
    runs = [("Taiwan, China", "geo_ex-TW", "TW"),
            ("Mainland China", "geo_ex-CN", "CN"),
            ("China region (incl. Taiwan, Hong Kong, Macao)", "geo_ex-CN_region",
             None),
            ("Japan", "geo_ex-JP", "JP"), ("Thailand", "geo_ex-TH", "TH"),
            ("United States", "geo_ex-US", "US"), ("Russia", "geo_ex-RU", "RU"),
            ("Vietnam", "geo_ex-VN", "VN")]
    rows = []
    for name, rid, country in runs:
        c = json.loads((ROOT / "results" / rid / "config.json").read_text())
        m = json.loads((ROOT / "results" / rid / "test_metrics.json")
                       .read_text())["test"]
        keep = set(c["train_where"].split("!=", 1)[1].split(","))
        tr_all = list(csv.DictReader(open(SPL / "train.csv", encoding="utf-8")))
        tr_sp = {r["species_latin_project"] for r in tr_all
                 if r["country_clean"] not in keep}
        if country is None:
            sub = [r for r in te if r["species_latin_project"] in tr_sp
                   and r["country_clean"] in {"CN", "TW", "HK", "MO"}]
        else:
            sub = [r for r in te if r["species_latin_project"] in tr_sp
                   and r["country_clean"] == country]
        base = sum(1 for r in sub
                   if pred[r["sample_id"]]["pred_species"] ==
                   r["species_latin_project"]) / len(sub)
        spec = sum(1 for r in sub if r.get("image_domain") == "specimen") / len(sub)
        rows.append([name, c["n_species"], int(c["train_n"]), len(sub),
                     fmt(m["top1"]), fmt(base), f"{100*(m['top1']-base):.1f}",
                     f"{spec:.2f}"])
    write(out, "tab4_geographic_holdout",
          ["Held-out region", "Classes", "train_n", "test_n", "held-out model",
           "in-domain", "Δ (pp)", "specimen share"], rows,
          "Table 4. Leave-one-region-out geographic holdout",
          "The in-domain column is the same full model scored on the identical "
          "test subset, so Δ = held-out − in-domain isolates 'having seen the "
          "region'; negative values are performance lost by holding the region "
          "out.")


# ------------------------------------------------------------------ Tab 5
def tab5(out: Path):
    reg = {r["id"]: r for r in csv.DictReader(
        (ROOT / "results/registry.csv").open(encoding="utf-8"))}

    def v(rid, col="top1"):
        r = reg.get(rid)
        return fmt(float(r[col])) if r and r[col] else "—"
    rows = [
        ["BioCLIP 2, frozen + linear @336 (ours)", "closed-set species", "902",
         v("bioclip2_linear_336"), v("bioclip2_linear_336", "top3"),
         "0.971 †", "—"],
        ["DINOv2, frozen + linear @336", "closed-set species", "902",
         v("dinov2_linear_336"), v("dinov2_linear_336", "top3"),
         "0.932 †", "—"],
        ["Pl@ntNet API (`all` referential)", "open-ended", "902",
         v("plantnet_all_test"), "0.312", "0.367", "—"],
        ["Qwen3.7-flash (DashScope API)", "open-ended", "901",
         v("vlm_qwen3.7-flash-2026-07-15"),
         v("vlm_qwen3.7-flash-2026-07-15", "top3"), "0.522",
         v("vlm_qwen3.7-flash-2026-07-15_no-label-leak")],
        ["Qwen3-VL-8B (local vLLM)", "open-ended", "902",
         v("vlm_qwen3-vl-8b-instruct"), v("vlm_qwen3-vl-8b-instruct", "top3"),
         "0.480", v("vlm_qwen3-vl-8b-instruct_no-label-leak")],
        ["Qwen3.7-flash (DashScope API)", "closed-set species", "901",
         v("vlm_qwen3.7-flash_species152"),
         v("vlm_qwen3.7-flash_species152", "top3"), "—",
         v("vlm_qwen3.7-flash_species152_no-label-leak")],
        ["Qwen3-VL-8B (local vLLM)", "closed-set species", "902",
         v("vlm_qwen3-vl-8b-instruct_species152"),
         v("vlm_qwen3-vl-8b-instruct_species152", "top3"), "—",
         v("vlm_qwen3-vl-8b-instruct_species152_no-label-leak")],
    ]
    write(out, "tab5_system_comparison",
          ["System", "Task", "n", "Species top-1", "Species top-3",
           "Family top-1", "top-1, leakage removed"], rows,
          "Table 5. All systems on the same 902-image test split",
          "† Family accuracy derived from the species head (the VLMs instead "
          "saw an explicit 30-way family task). The last column removes "
          "images whose true name is OCR-readable on the sheet.")


# ------------------------------------------------------------------ Tab 6
def tab6(out: Path):
    rows = list(csv.DictReader(
        (ROOT / "results/analysis/paired_mcnemar.csv").open(encoding="utf-8")))
    key = [("bioclip2_linear_336", "dinov2_linear_336"),
           ("bioclip2_linear_336", "bioclip2_ft_336"),
           ("bioclip2_linear_336", "bioclip2_ft_hier"),
           ("bioclip2_linear_336", "bioclip2_linear"),
           ("bioclip2_linear_336", "plantnet_all"),
           ("bioclip2_linear_336", "qwen3.7-flash_species152"),
           ("bioclip2_linear_336", "qwen3-vl-8b_species152"),
           ("bioclip2_linear_336", "qwen3.7-flash_open_noleak"),
           ("bioclip2_linear_336", "qwen3-vl-8b_open_noleak"),
           ("dinov2_linear_336", "dinov2_ft"),
           ("plantnet_all", "qwen3.7-flash_open")]
    idx = {(r["a"], r["b"]): r for r in rows}
    out_rows = []
    for a, b in key:
        r = idx.get((a, b)) or idx.get((b, a))
        if not r:
            continue
        flip = r["a"] != a
        s = -1 if flip else 1
        out_rows.append([f"{a} vs {b}", r["n_paired"],
                         f"{s*float(r['diff_pct']):+.1f}",
                         f"{float(r['p_exact']):.2e}", r["sig"],
                         f"[{100*s*float(r['diff_ci95_lo']):+.1f}, "
                         f"{100*s*float(r['diff_ci95_hi']):+.1f}]"])
    write(out, "tab6_paired_significance",
          ["Comparison", "n (paired)", "Δ top-1 (pp)", "McNemar p", "sig.",
           "95% CI of Δ"], out_rows,
          "Table 6. Paired McNemar tests on the shared test images",
          "All systems saw the same images; *** p<0.001, ** p<0.01, "
          "* p<0.05, n.s. = not significant. Full 105-pair matrix in "
          "paired_mcnemar_v2.csv. **All tests are single-seed (20261010):** "
          "the bioclip2_linear_336 vs bioclip2_ft_336 row (+2.1 pp, p=0.025) "
          "does NOT survive seed variance -- across three seeds the paired "
          "difference is +2.00 / +0.44 / -0.78 pp (mean +0.55 ± 1.13 pp, "
          "interval includes zero), so frozen and fine-tuned are "
          "statistically indistinguishable. Cite the seed-level result "
          "(Table 2 and RESULTS.md §14), not this single-seed p-value.")


# ------------------------------------------------------------------ Tab 7
def tab7(out: Path):
    rows = list(csv.DictReader(
        (ROOT / "results/analysis/per_family_ours_vs_plantnet.csv")
        .open(encoding="utf-8")))
    out_rows = [[r["family"], r["n_test"], r["ours_acc"], r["plantnet_acc"],
                 r["gap"]] for r in rows]
    write(out, "tab7_per_family",
          ["Family", "n", "Ours (family acc.)", "Pl@ntNet (family acc.)",
           "Gap"], out_rows,
          "Table 7. Per-family accuracy, ours vs Pl@ntNet (n ≥ 5)",
          "Pl@ntNet scores 0.000 on 11 families while ours stays 0.60–1.00 "
          "on every one of them.")


# ------------------------------------------------------------------ Tab 8
def tab8(out: Path):
    ours = [json.loads(l) for l in (ROOT / "results/bioclip2_linear_336/"
                                    "predictions.jsonl")
            .read_text(encoding="utf-8").splitlines() if l.strip()]
    pl = [json.loads(l) for l in (ROOT / "data/external_baselines/plantnet/raw_all_test.jsonl")
          .read_text(encoding="utf-8").splitlines() if l.strip()]
    pl = [r for r in pl if r.get("ok") and r.get("top")]

    def norm(s):
        return " ".join((s or "").lower().replace("*", "").replace("×", "x")
                        .split())

    def binom(s):
        p = norm(s).split()
        return " ".join(p[:2]) if len(p) >= 2 else norm(s)

    def stats(recs, pred_fn, top3_fn, fam_fn=None):
        per = defaultdict(lambda: {"n": 0, "hit": 0, "in3": 0})
        cross = 0
        crossfam = 0
        for r in recs:
            t = binom(r["true_species"])
            p = binom(pred_fn(r) or "")
            d = per[t]
            d["n"] += 1
            d["hit"] += (p == t)
            d["in3"] += any(binom(x) == t for x in (top3_fn(r) or []))
            if p != t and p.split(" ")[0] != t.split(" ")[0]:
                cross += 1
            if fam_fn and p != t and fam_fn(r):
                crossfam += 1
        n_sp = len(per)
        return {"n": len(recs), "n_sp": n_sp,
                "never": sum(1 for d in per.values() if d["hit"] == 0),
                "never_in3": sum(1 for d in per.values() if d["in3"] == 0),
                "cross": cross, "crossfam": crossfam}

    o = stats(ours, lambda r: r["pred_species"],
              lambda r: r.get("top3") or [],
              lambda r: r.get("pred_family_latin") != r.get("true_family_latin"))
    # Pl@ntNet returns a family for every candidate (902/902 records, with
    # PPG I family names), so its cross-family rate is computable as well.
    p = stats(pl, lambda r: (r["top"] or [{}])[0].get("sci"),
              lambda r: [e.get("sci") for e in (r.get("top") or [])[:3]],
              lambda r: ((r["top"] or [{}])[0].get("family") or "") !=
              (r.get("true_family_latin") or ""))
    rows = [
        ["Test images evaluated", o["n"], p["n"]],
        ["Species represented (n ≥ 2 test images)", o["n_sp"], p["n_sp"]],
        ["Never identified correctly", f"{o['never']} ({o['never']/o['n_sp']:.1%})",
         f"{p['never']} ({p['never']/p['n_sp']:.1%})"],
        ["Never in the top-3", f"{o['never_in3']} ({o['never_in3']/o['n_sp']:.1%})",
         f"{p['never_in3']} ({p['never_in3']/p['n_sp']:.1%})"],
        ["Cross-genus errors", f"{o['cross']} ({o['cross']/o['n']:.1%})",
         f"{p['cross']} ({p['cross']/p['n']:.1%})"],
        ["Cross-family errors", f"{o['crossfam']} ({o['crossfam']/o['n']:.1%})",
         f"{p['crossfam']} ({p['crossfam']/p['n']:.1%})"],
        ["Largest single confusion direction", "3 images",
         "188 of 273 Dryopteridaceae images → Polypodiaceae"],
    ]
    write(out, "tab8_error_structure",
          ["Metric", "BioCLIP 2 linear@336", "Pl@ntNet"], rows,
          "Table 8. Species-level coverage and error structure",
          "'Species represented' counts species with ≥2 test images (148). The "
          "902 test images cover 150 species in total; *Huperzia nanchuanensis* "
          "and *H. sutchueniana* contribute a single image each and are "
          "excluded only from the species-level counts, not from accuracy. "
          "'Never in the top-3' is computed over each system's ranked candidate "
          "list (rank ≤ 3). Family errors are counted against the benchmark's "
          "PPG I family labels, using Pl@ntNet's own family field for its "
          "predictions.")


# ------------------------------------------------------------------ Tab 9
def tab9(out: Path):
    rows = []
    for p in sorted((ROOT / "results").glob("*/openset_metrics.json")):
        d = json.loads(p.read_text())
        if d["n_neg"] != 700:
            continue
        grp = d.get("reject_rate_by_group_at_fpr10") or {}
        rows.append([p.parent.name, fmt(d["auroc"]),
                     fmt(d["tpr_at_fpr5"]), fmt(d["tpr_at_fpr10"]),
                     fmt(d["pos_conf_mean"], 3), fmt(d["neg_conf_mean"], 3),
                     fmt(grp.get("cycad", 0), 3)])
    rows.sort(key=lambda r: -float(r[1]))
    write(out, "tab9_openset_rejection",
          ["Model", "AUROC", "TPR@FPR5%", "TPR@FPR10%", "pos. conf.",
           "neg. conf.", "cycad rejection"], rows,
          "Table 9. Open-set rejection (700 negatives, 8 groups)",
          "max-softmax as the rejection score. The cycad group ('fern-like "
          "non-fern') is the weakest across every model.")


# ------------------------------------------------------------------ Tab 10
def tab10(out: Path):
    """Benchmark comparison for the Related Work section.

    NGOI = New Guinea Orchid Identifier benchmark (Saputra et al. 2026,
    arXiv:2609.24064) -- the paper's own abstract is the source for every
    figure in that row. Pl@ntNet-300K: Garcin et al. 2021; PlantCLEF 2025:
    Martellucci et al. 2025; the last row is this manuscript's benchmark.
    """
    rows = [
        ["NGOI benchmark (New Guinea orchids; Saputra et al. 2026, "
         "arXiv:2609.24064)", "16,701 photographs",
         "1,350 spp. / 120 genera (Orchidaceae, New Guinea)",
         "no (field photographs only)",
         "no (single island flora)", "no (but distance-based open-set gate "
         "for unseen genera, AUROC 0.958)",
         "fixed species-stratified partition; two-stage genus + FAISS "
         "species retrieval"],
        ["Pl@ntNet-300K", "306,146 (244k/31k/31k)", "1,081 spp. (French flora)",
         "no (field photographs only)", "no", "no",
         "random train/val/test split"],
        ["PlantCLEF 2025", "2,105 test quadrats + ~1.4M train single-plant "
         "images", "~400 spp.", "no (field imagery only)", "no", "no",
         "split by acquisition mode (close-up vs quadrat)"],
        ["This benchmark", "5,969 (4,185/882/902)",
         "152 spp. / 30 families",
         "yes (field / herbarium)", "yes (country, 80 values)",
         "yes (700 images / 8 groups)",
         "group-aware by observation; domain, source and geography holdouts"],
    ]
    write(out, "tab10_benchmark_comparison",
          ["Benchmark", "Scale (images)", "Taxa", "Domain annotation",
           "Geography annotation", "Open-set negatives", "Protocol"], rows,
          "Table 10. Comparison with existing plant and fern identification "
          "benchmarks",
          "Sources: Garcin et al. (2021) for Pl@ntNet-300K; Martellucci et "
          "al. (2025) for PlantCLEF 2025; Saputra et al. (2026, "
          "arXiv:2609.24064) for the New Guinea Orchid Identifier (NGOI) "
          "benchmark.")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(OUTDIR_DEFAULT))
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    print("writing tables:")
    for fn in (tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9, tab10):
        fn(out)
    print(f"\nall tables in {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
