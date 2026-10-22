#!/usr/bin/env python3
"""Full evaluation of a trained checkpoint: writes per-sample predictions and
derives every metric the paper needs that `train_fern.py` did not produce.

Adds, on the same 902-image test split:
  * family-level accuracy derived from the species head (comparable to the
    VLM's closed-set family task and to Pl@ntNet's family number)
  * per-family breakdown
  * confidence calibration (ECE + reliability bins)
  * risk-coverage / selective prediction (what accuracy if we only accept the
    most confident fraction?)
  * inference latency (single-image, warm)

    python eval_full.py --run runs/bioclip2_linear_336
"""
from __future__ import annotations

import argparse
import csv
import json
import time
from collections import defaultdict
from pathlib import Path
import os

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision.transforms import InterpolationMode
from torchvision import transforms as T

import train_fern as tf

ROOT = Path(os.environ.get("FERN_ROOT", "."))
SPECIES_JSON = ROOT / "data/species_data.json"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="runs/bioclip2_linear_336")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    run_dir = ROOT / args.run
    ck = torch.load(run_dir / "best.pt", map_location="cpu")
    cfg = ck["cfg"]
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    trunk, dim, mean, std = tf.build_backbone(cfg["backbone"], cfg["img_size"])
    model = tf.Classifier(trunk, dim, cfg["n_species"], cfg["n_family"],
                          cfg.get("hier", False))
    model.load_state_dict(ck["model"])
    model = model.to(dev).eval()

    ds = tf.FernDataset("test", tf.build_tf(cfg["img_size"], mean, std, False))
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                    num_workers=args.workers)
    sp_lat = {s["latin_name"]: s.get("family_latin", s["family"])
              for s in json.loads(SPECIES_JSON.read_text(encoding="utf-8"))["species"]}

    # Label space must be rebuilt from the train split, exactly as
    # train_fern.py main() does: FernDataset.species only lists the species
    # present in *that* split (test has 150 of the trained 152), so mapping
    # logit indices through the test split's own list raises IndexError.
    tr_ds = tf.FernDataset("train", tf.build_tf(cfg["img_size"], mean, std, False))
    for spec_key in ("train_where", "eval_where"):
        spec = cfg.get(spec_key)
        if not spec:
            continue
        col, _, val = spec.partition("=")
        target = tr_ds if spec_key == "train_where" else ds
        target.rows = [r for r in target.rows if (r.get(col) or "") == val]
    species = sorted({r["species_latin_project"] for r in tr_ds.rows})
    keep = set(species)
    ds.rows = [r for r in ds.rows if r["species_latin_project"] in keep]
    assert len(species) == cfg["n_species"], (len(species), cfg["n_species"])

    rows, lat = [], []
    # warm-up + single-image latency
    warm = torch.zeros(1, 3, cfg["img_size"], cfg["img_size"], device=dev)
    with torch.no_grad():
        for _ in range(5):
            model(warm)
        torch.cuda.synchronize()
        for _ in range(30):
            t0 = time.perf_counter()
            model(warm)
            torch.cuda.synchronize()
            lat.append((time.perf_counter() - t0) * 1000)
    with torch.no_grad():
        i = 0
        for x, y, yf, dom in dl:
            x = x.to(dev)
            with torch.autocast("cuda", dtype=torch.float16, enabled=dev == "cuda"):
                logits, _ = model(x)
            p = torch.softmax(logits.float(), 1)
            conf, idx = p.max(1)
            top3 = p.topk(3, 1).indices
            for b in range(x.shape[0]):
                r = ds.rows[i + b]
                pred = species[idx[b].item()]
                rows.append({
                    "sample_id": r["sample_id"],
                    "true_species": r["species_latin_project"],
                    "true_family_cn": r["family_project"],
                    "true_family_latin": sp_lat.get(r["species_latin_project"], ""),
                    "image_domain": r.get("image_domain", ""),
                    "source": r.get("source", ""),
                    "pred_species": pred,
                    "pred_family_latin": sp_lat.get(pred, ""),
                    "confidence": float(conf[b]),
                    "top3": [species[t] for t in top3[b].tolist()],
                })
            i += x.shape[0]

    out = run_dir / "predictions.jsonl"
    with out.open("w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")

    n = len(rows)
    s1 = sum(r["pred_species"] == r["true_species"] for r in rows) / n
    s3 = sum(r["true_species"] in r["top3"] for r in rows) / n
    f1 = sum(r["pred_family_latin"] == r["true_family_latin"] for r in rows) / n
    per_fam = defaultdict(lambda: [0, 0])
    for r in rows:
        k = r["true_family_latin"]
        per_fam[k][1] += 1
        per_fam[k][0] += (r["pred_family_latin"] == k)
    macro_fam = np.mean([v[0] / v[1] for v in per_fam.values()])

    bins = np.linspace(0, 1, 11)
    ece = 0.0
    rel = []
    confs = np.array([r["confidence"] for r in rows])
    correct = np.array([r["pred_species"] == r["true_species"] for r in rows])
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (confs > lo) & (confs <= hi)
        if m.sum() == 0:
            continue
        acc, cf = correct[m].mean(), confs[m].mean()
        ece += m.mean() * abs(acc - cf)
        rel.append({"bin": [round(float(lo), 1), round(float(hi), 1)],
                    "n": int(m.sum()), "acc": round(float(acc), 4),
                    "conf": round(float(cf), 4)})

    order = np.argsort(-confs)
    cov, risk = [], []
    for frac in np.linspace(0.05, 1.0, 20):
        k = max(1, int(frac * n))
        cov.append(round(float(frac), 3))
        risk.append(round(float(correct[order[:k]].mean()), 4))

    res = {
        "run": args.run, "n": n,
        "species_top1": round(float(s1), 4), "species_top3": round(float(s3), 4),
        "family_top1_derived": round(float(f1), 4),
        "family_macro": round(float(macro_fam), 4),
        "ece_10bin": round(float(ece), 4), "reliability": rel,
        "coverage": cov, "accuracy_at_coverage": risk,
        "latency_ms_mean": round(float(np.mean(lat)), 2),
        "latency_ms_p50": round(float(np.median(lat)), 2),
        "throughput_img_s": round(1000.0 / float(np.mean(lat)), 1),
        "per_family": {k: {"acc": round(v[0] / v[1], 3), "n": v[1]}
                       for k, v in sorted(per_fam.items())},
    }
    (run_dir / "full_metrics.json").write_text(json.dumps(res, indent=2),
                                               encoding="utf-8")
    print(json.dumps({k: v for k, v in res.items()
                      if k not in ("reliability", "per_family", "coverage",
                                   "accuracy_at_coverage")}, indent=2))
    print("risk-coverage (cov -> acc):")
    print("  " + ", ".join(f"{c:.2f}->{a:.3f}" for c, a in
                           zip(cov[::4], risk[::4])))
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
