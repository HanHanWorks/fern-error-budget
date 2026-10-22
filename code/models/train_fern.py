#!/usr/bin/env python3
"""Fine-tune a plant/vision backbone for fern species + family identification.

Two backbones, one identical protocol, so the comparison in the paper is fair:

    --backbone bioclip2   BioCLIP 2 (open_clip ViT-L/14, biology-pretrained)
    --backbone dinov2     DINOv2 ViT-L/14 (timm, self-supervised)

Modes:
    --mode linear     freeze the trunk, train the classifier head only
    --mode finetune   train everything

Reports micro top-1/top-3 plus macro (class-balanced) accuracy, and splits the
test set by image_domain (herbarium specimen vs field photo) so the cross-domain
gap is measured rather than hidden.

Example:
    python train_fern.py --backbone bioclip2 --mode finetune --epochs 20 \
        --img-size 224 --batch-size 32 --out runs/bioclip2_ft224
"""
from __future__ import annotations

import argparse
import csv
import json
import time
from collections import Counter, defaultdict
from pathlib import Path
import os

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T
from torchvision.transforms import InterpolationMode

ROOT = Path(os.environ.get("FERN_ROOT", "."))
IMAGES = ROOT / "data/images"
SPLITS = ROOT / "data/dataset/splits"
BIOCLIP_W = ROOT / "weights/bioclip-2"
DINOV2_W = ROOT / "weights/dinov2/dinov2_vitl14_pretrain.pth"

CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)
IN_MEAN = (0.485, 0.456, 0.406)
IN_STD = (0.229, 0.224, 0.225)


# --------------------------------------------------------------------- data
class FernDataset(Dataset):
    def __init__(self, split: str, tf):
        with (SPLITS / f"{split}.csv").open(encoding="utf-8") as fh:
            self.rows = list(csv.DictReader(fh))
        self.tf = tf
        self.species = sorted({r["species_latin_project"] for r in self.rows})
        self.families = sorted({r["family_project"] for r in self.rows})
        self.sidx = {s: i for i, s in enumerate(self.species)}
        self.fidx = {f: i for i, f in enumerate(self.families)}

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, i: int):
        r = self.rows[i]
        img = Image.open(IMAGES / Path(r["image_path"]).name).convert("RGB")
        return (self.tf(img),
                self.sidx[r["species_latin_project"]],
                self.fidx[r["family_project"]],
                r.get("image_domain", "other"))


def build_tf(img_size: int, mean, std, train: bool):
    if train:
        return T.Compose([
            T.RandomResizedCrop(img_size, scale=(0.55, 1.0),
                                interpolation=InterpolationMode.BICUBIC),
            T.RandomHorizontalFlip(),
            T.ColorJitter(0.2, 0.2, 0.2),
            T.ToTensor(), T.Normalize(mean, std),
        ])
    return T.Compose([
        T.Resize(int(img_size * 1.14), interpolation=InterpolationMode.BICUBIC),
        T.CenterCrop(img_size),
        T.ToTensor(), T.Normalize(mean, std),
    ])


# ------------------------------------------------------------------ models
class Classifier(nn.Module):
    """Trunk + LayerNorm + linear head (optionally a second family head)."""

    def __init__(self, trunk, dim: int, n_species: int, n_family: int, hier: bool):
        super().__init__()
        self.trunk = trunk
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, n_species)
        self.hier = hier
        self.fhead = nn.Linear(dim, n_family) if hier else None

    def forward(self, x):
        f = self.norm(self.trunk(x))
        if self.hier:
            return self.head(f), self.fhead(f)
        return self.head(f), None


class TrunkWrap(nn.Module):
    """Normalise a backbone's output to a single pooled feature tensor."""

    def __init__(self, trunk):
        super().__init__()
        self.trunk = trunk

    def forward(self, x):
        out = self.trunk(x)
        if isinstance(out, (tuple, list)):
            out = out[0]
        return out


def build_backbone(name: str, img_size: int):
    """Return (trunk, feature_dim, mean, std). Dim is inferred by a dry run so
    it stays correct across open_clip / timm versions and return-type changes."""
    if name == "bioclip2":
        import open_clip
        # BioCLIP 2 is a ViT-L/14 whose preprocess stats are the standard CLIP
        # ones (we apply our own transforms, so only the weights are needed).
        # force_image_size rebuilds the positional embedding for a different
        # input resolution (the checkpoint's is 224); open_clip interpolates it.
        model, _, _ = open_clip.create_model_and_transforms(
            "ViT-L-14",
            pretrained=str(BIOCLIP_W / "open_clip_model.safetensors"),
            force_image_size=img_size,
        )
        trunk, mean, std = model.visual, CLIP_MEAN, CLIP_STD
    elif name == "dinov2":
        import timm
        trunk = timm.create_model(
            "vit_large_patch14_dinov2.lvd142m",
            pretrained=False, num_classes=0, dynamic_img_size=True,
        )
        sd = torch.load(DINOV2_W, map_location="cpu")
        trunk.load_state_dict(sd.get("model", sd), strict=False)
        mean, std = IN_MEAN, IN_STD
    else:
        raise ValueError(name)

    wrap = TrunkWrap(trunk).eval()
    with torch.no_grad():
        dim = wrap(torch.zeros(1, 3, img_size, img_size)).shape[-1]
    return wrap, int(dim), mean, std


# ----------------------------------------------------------------- metrics
def evaluate(model, loader, dev, n_species):
    model.eval()
    top1 = top3 = n = 0
    per_class = defaultdict(int)
    per_class_tot = defaultdict(int)
    dom_hit = defaultdict(int)
    dom_tot = defaultdict(int)
    with torch.no_grad():
        for x, y, yf, dom in loader:
            x, y = x.to(dev, non_blocking=True), y.to(dev, non_blocking=True)
            with torch.autocast("cuda", dtype=torch.float16, enabled=dev == "cuda"):
                logits, _ = model(x)
            k = min(3, logits.shape[1])
            _, pred = logits.topk(k, dim=1)
            hit1 = pred[:, 0].eq(y)
            hit3 = pred.eq(y[:, None]).any(1)
            top1 += hit1.sum().item()
            top3 += hit3.sum().item()
            n += y.numel()
            for yi, h, d in zip(y.tolist(), hit1.tolist(), dom):
                per_class[yi] += int(h)
                per_class_tot[yi] += 1
                dom_hit[d] += int(h)
                dom_tot[d] += 1
    macro = float(np.mean([per_class[c] / per_class_tot[c] for c in per_class_tot]))
    return {
        "n": n,
        "top1": top1 / max(n, 1),
        "top3": top3 / max(n, 1),
        "macro_top1": macro,
        "classes_seen": len(per_class_tot),
        "domain_top1": {d: dom_hit[d] / dom_tot[d] for d in dom_tot},
        "domain_n": dict(dom_tot),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone", choices=["bioclip2", "dinov2"], required=True)
    ap.add_argument("--mode", choices=["linear", "finetune"], default="finetune")
    ap.add_argument("--img-size", type=int, default=224)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--wd", type=float, default=0.05)
    ap.add_argument("--warmup", type=int, default=1)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--hier", action="store_true",
                    help="add a family head with an auxiliary loss")
    ap.add_argument("--hier-weight", type=float, default=0.3)
    ap.add_argument("--class-balance", action="store_true",
                    help="weight the loss by inverse class frequency")
    ap.add_argument("--label-smoothing", type=float, default=0.1)
    ap.add_argument("--limit-train", type=int, default=0)
    ap.add_argument("--train-where", type=str, default="",
                    help="keep only train rows matching COL=VAL[,VAL...] or "
                         "COL!=VAL[,VAL...] (e.g. image_domain=field, "
                         "source=GBIF, country_clean!=TW)")
    ap.add_argument("--eval-where", type=str, default="",
                    help="restrict val/test rows with COL=VAL[,VAL...] or "
                         "COL!=VAL[,VAL...]")
    ap.add_argument("--seed", type=int, default=20261010)
    ap.add_argument("--out", type=str, required=True)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    out = Path(args.out) if args.out.startswith("/") else ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)

    trunk, dim, mean, std = build_backbone(args.backbone, args.img_size)
    if args.mode == "linear":
        for p in trunk.parameters():
            p.requires_grad = False

    tr_ds = FernDataset("train", build_tf(args.img_size, mean, std, True))
    va_ds = FernDataset("val", build_tf(args.img_size, mean, std, False))
    te_ds = FernDataset("test", build_tf(args.img_size, mean, std, False))
    if args.limit_train:
        tr_ds.rows = tr_ds.rows[: args.limit_train]
    def apply_where(ds, spec: str):
        # COL=VAL[,VAL...]  keep rows whose COL is in the list
        # COL!=VAL[,VAL...] keep rows whose COL is NOT in the list
        # (e.g. country_clean!=TW trains everywhere except Taiwan)
        if not spec:
            return
        if "!=" in spec:
            col, _, val = spec.partition("!=")
            negate = True
        else:
            col, _, val = spec.partition("=")
            negate = False
        vals = {v.strip() for v in val.split(",") if v.strip()}
        if negate:
            ds.rows = [r for r in ds.rows if (r.get(col) or "") not in vals]
        else:
            ds.rows = [r for r in ds.rows if (r.get(col) or "") in vals]

    apply_where(tr_ds, args.train_where)
    apply_where(va_ds, args.eval_where)
    apply_where(te_ds, args.eval_where)
    if not tr_ds.rows:
        raise SystemExit(f"train-where '{args.train_where}' left no rows")
    # Label space must come from the FILTERED train split: FernDataset builds
    # it in __init__ from the raw file, so after a where-filter species with
    # zero training rows would otherwise keep their class index and compete
    # with random weights (this corrupted the Batch-4 transfer runs; corrected
    # 2026-10-13 -- see EXPERIMENT_LOG.md).
    tr_ds.species = sorted({r["species_latin_project"] for r in tr_ds.rows})
    tr_ds.families = sorted({r["family_project"] for r in tr_ds.rows})
    tr_ds.sidx = {s: i for i, s in enumerate(tr_ds.species)}
    tr_ds.fidx = {f: i for i, f in enumerate(tr_ds.families)}
    # label spaces must come from train; map val/test onto it
    sidx, fidx = tr_ds.sidx, tr_ds.fidx
    for ds in (va_ds, te_ds):
        ds.sidx, ds.fidx = sidx, fidx
        ds.rows = [r for r in ds.rows if r["species_latin_project"] in sidx]

    n_sp, n_fam = len(sidx), len(fidx)
    model = Classifier(trunk, dim, n_sp, n_fam, args.hier).to(dev)
    tl = DataLoader(tr_ds, batch_size=args.batch_size, shuffle=True,
                    num_workers=args.workers, pin_memory=True, drop_last=True)
    vl = DataLoader(va_ds, batch_size=64, shuffle=False,
                    num_workers=args.workers, pin_memory=True)
    tel = DataLoader(te_ds, batch_size=64, shuffle=False,
                     num_workers=args.workers, pin_memory=True)

    w = None
    if args.class_balance:
        cnt = Counter(tr_ds.sidx[r["species_latin_project"]] for r in tr_ds.rows)
        w = torch.tensor([len(tr_ds.rows) / (n_sp * max(cnt.get(i, 1), 1))
                          for i in range(n_sp)], device=dev, dtype=torch.float32)
    crit = nn.CrossEntropyLoss(weight=w, label_smoothing=args.label_smoothing)
    crit_f = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)

    lr = args.lr or (1e-3 if args.mode == "linear" else 2e-5)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=args.wd)
    steps = max(len(tl) * args.epochs, 1)
    warm = max(len(tl) * args.warmup, 1)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: s / warm if s < warm else 0.5 * (1 + np.cos(
            np.pi * (s - warm) / max(steps - warm, 1))))
    scaler = torch.amp.GradScaler("cuda", enabled=dev == "cuda")

    cfg = vars(args) | {"n_species": n_sp, "n_family": n_fam, "dim": dim,
                        "train_n": len(tr_ds.rows), "val_n": len(va_ds.rows),
                        "test_n": len(te_ds.rows), "device": dev}
    (out / "config.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    print(json.dumps({k: cfg[k] for k in
                      ("backbone", "mode", "img_size", "epochs", "batch_size",
                       "lr", "hier", "n_species", "train_n", "test_n")}))

    best, hist = -1.0, []
    for ep in range(1, args.epochs + 1):
        model.train()
        if args.mode == "linear":
            model.trunk.eval()
        t0, tot, run = time.time(), 0, 0.0
        for x, y, yf, _ in tl:
            x = x.to(dev, non_blocking=True)
            y = y.to(dev, non_blocking=True)
            yf = yf.to(dev, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.float16, enabled=dev == "cuda"):
                logits, flogits = model(x)
                loss = crit(logits, y)
                if flogits is not None:
                    loss = loss + args.hier_weight * crit_f(flogits, yf)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            scaler.step(opt)
            scaler.update()
            sched.step()
            run += loss.item() * y.numel()
            tot += y.numel()
        vm = evaluate(model, vl, dev, n_sp)
        rec = {"epoch": ep, "train_loss": run / max(tot, 1),
               "val_top1": vm["top1"], "val_macro": vm["macro_top1"],
               "sec": round(time.time() - t0, 1)}
        hist.append(rec)
        print(f"ep {ep:>2} loss {rec['train_loss']:.4f} "
              f"val_top1 {vm['top1']:.4f} val_macro {vm['macro_top1']:.4f} "
              f"({rec['sec']}s)", flush=True)
        if vm["top1"] > best:
            best = vm["top1"]
            torch.save({"model": model.state_dict(), "cfg": cfg}, out / "best.pt")
            (out / "val_metrics.json").write_text(json.dumps(vm, indent=2),
                                                  encoding="utf-8")
        (out / "history.json").write_text(json.dumps(hist, indent=2),
                                          encoding="utf-8")

    ck = torch.load(out / "best.pt", map_location=dev)
    model.load_state_dict(ck["model"])
    res = {"backbone": args.backbone, "mode": args.mode,
           "img_size": args.img_size, "hier": args.hier,
           "best_val_top1": best, "test": evaluate(model, tel, dev, n_sp)}
    (out / "test_metrics.json").write_text(json.dumps(res, indent=2),
                                           encoding="utf-8")
    print("TEST " + json.dumps(res["test"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
