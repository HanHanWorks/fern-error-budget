#!/usr/bin/env python3
"""Open-set evaluation: can the fern classifier reject non-fern inputs?

Loads a trained checkpoint, scores the fern test split (positives) and the
negative set (horsetails, out-of-scope lycophytes, bryophytes, cycads, palms,
angiosperms), then reports the accept/reject separation.

Metric: max-softmax confidence as the rejection score.
    AUROC          ranking quality of positives over negatives
    TPR@FPR5/10    positive recall at a 5%/10% false-accept rate
    by-group       rejection rate per negative group

    python openset_eval.py --run runs/bioclip2_linear_336
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
import os

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T
from torchvision.transforms import InterpolationMode

import train_fern as tf

ROOT = Path(os.environ.get("FERN_ROOT", "."))
NEG = ROOT / "data/negatives"


class Paths(Dataset):
    """Yield (tensor, path, group) for arbitrary image paths."""

    def __init__(self, items, tfm):
        self.items = items
        self.tfm = tfm

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        path, group = self.items[i]
        img = Image.open(path).convert("RGB")
        return self.tfm(img), group


def roc(pos, neg):
    """AUROC via rank statistic (no sklearn dependency)."""
    allv = sorted([(v, 1) for v in pos] + [(v, 0) for v in neg])
    n_pos, n_neg = len(pos), len(neg)
    rank_sum, i = 0.0, 0
    while i < len(allv):
        j = i
        while j + 1 < len(allv) and allv[j + 1][0] == allv[i][0]:
            j += 1
        avg_rank = (i + j) / 2 + 1
        for k in range(i, j + 1):
            if allv[k][1] == 1:
                rank_sum += avg_rank
        i = j + 1
    return (rank_sum - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


def tpr_at_fpr(pos, neg, fpr):
    thr = sorted(neg)[int((1 - fpr) * len(neg)) - 1]
    return sum(1 for p in pos if p > thr) / len(pos), thr


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

    tfm = T.Compose([
        T.Resize(int(cfg["img_size"] * 1.14), interpolation=InterpolationMode.BICUBIC),
        T.CenterCrop(cfg["img_size"]), T.ToTensor(), T.Normalize(mean, std)])

    with (ROOT / "data/dataset/splits/test.csv").open(encoding="utf-8") as fh:
        test_rows = list(csv.DictReader(fh))
    pos_items = [(ROOT / "data/images" / Path(r["image_path"]).name, "fern")
                 for r in test_rows]
    with (NEG / "manifest_negatives.csv").open(encoding="utf-8") as fh:
        neg_rows = list(csv.DictReader(fh))
    neg_items = [(NEG / "images" / Path(r["image_path"]).name,
                  r["negative_group"]) for r in neg_rows]

    def scores(items):
        dl = DataLoader(Paths(items, tfm), batch_size=args.batch_size,
                        shuffle=False, num_workers=args.workers)
        out, groups = [], []
        with torch.no_grad():
            for x, g in dl:
                x = x.to(dev)
                with torch.autocast("cuda", dtype=torch.float16, enabled=dev == "cuda"):
                    logits, _ = model(x)
                out.extend(torch.softmax(logits.float(), 1).max(1).values.tolist())
                groups.extend(list(g))
        return out, groups

    pos, _ = scores(pos_items)
    neg, neg_groups = scores(neg_items)
    auroc = roc(pos, neg)
    res = {
        "run": args.run, "n_pos": len(pos), "n_neg": len(neg),
        "auroc": auroc,
        "pos_conf_mean": sum(pos) / len(pos),
        "neg_conf_mean": sum(neg) / len(neg),
    }
    for f in (0.05, 0.10):
        t, thr = tpr_at_fpr(pos, neg, f)
        res[f"tpr_at_fpr{int(f*100)}"] = t
        res[f"threshold_fpr{int(f*100)}"] = thr
        res[f"neg_reject_at_fpr{int(f*100)}"] = 1 - f
    thr = res["threshold_fpr10"]
    by = defaultdict(lambda: [0, 0])
    for s, g in zip(neg, neg_groups):
        by[g][0] += int(s <= thr)
        by[g][1] += 1
    res["reject_rate_by_group_at_fpr10"] = {
        g: v[0] / v[1] for g, v in sorted(by.items())}
    print(json.dumps(res, indent=2))
    (run_dir / "openset_metrics.json").write_text(
        json.dumps(res, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
