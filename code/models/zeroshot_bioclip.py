#!/usr/bin/env python3
"""Zero-shot BioCLIP 2 classification on the fern test split.

No training at all: match each test image against text embeddings of the
species (and family) names. This is the honest "specialist model out of the
box" baseline that general-purpose VLMs are compared against.

    python zeroshot_bioclip.py --weights bioclip2
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
import os

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T
from torchvision.transforms import InterpolationMode

ROOT = Path(os.environ.get("FERN_ROOT", "."))
IMAGES = ROOT / "data/images"
SPLITS = ROOT / "data/dataset/splits"
WEIGHTS = {
    "bioclip2": ROOT / "weights/bioclip-2/open_clip_model.safetensors",
}
SPECIES_JSON = ROOT / "data/species_data.json"
CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)


class TestSet(Dataset):
    def __init__(self, split: str, size: int = 224):
        with (SPLITS / f"{split}.csv").open(encoding="utf-8") as fh:
            self.rows = list(csv.DictReader(fh))
        self.tf = T.Compose([
            T.Resize(int(size * 1.14), interpolation=InterpolationMode.BICUBIC),
            T.CenterCrop(size), T.ToTensor(),
            T.Normalize(CLIP_MEAN, CLIP_STD)])

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        r = self.rows[i]
        img = Image.open(IMAGES / Path(r["image_path"]).name).convert("RGB")
        return self.tf(img), r["species_latin_project"], r["family_project"], \
            r.get("image_domain", "other")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", choices=list(WEIGHTS), default="bioclip2")
    ap.add_argument("--split", default="test")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--templates", type=str,
                    default="a photo of {}.|a photograph of a {} plant.|{}")
    args = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    import open_clip
    model, _, _ = open_clip.create_model_and_transforms(
        "ViT-L-14", pretrained=str(WEIGHTS[args.weights]))
    model = model.to(dev).eval()
    tok = open_clip.get_tokenizer("ViT-L-14")

    ds = TestSet(args.split)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                    num_workers=args.workers)
    species = sorted({r["species_latin_project"] for r in ds.rows})
    # The CLIP text encoder is English; Chinese family names embed to noise
    # (family_top1 came out at chance). Use the Latin family names instead.
    fam_latin = {}
    if SPECIES_JSON.exists():
        for s in json.loads(SPECIES_JSON.read_text(encoding="utf-8"))["species"]:
            fam_latin.setdefault(s["family"], s.get("family_latin", s["family"]))
    fam_prompt = {f: fam_latin.get(f, f) for f in
                  {r["family_project"] for r in ds.rows}}
    families = sorted(fam_prompt.values())
    templates = args.templates.split("|")

    def text_emb(names):
        embs = []
        with torch.no_grad():
            for tpl in templates:
                t = tok([tpl.format(n) for n in names]).to(dev)
                e = model.encode_text(t)
                embs.append(e / e.norm(dim=-1, keepdim=True))
        e = torch.stack(embs).mean(0)
        return e / e.norm(dim=-1, keepdim=True)

    tsp = text_emb(species)
    tfam = text_emb(families)
    sidx = {s: i for i, s in enumerate(species)}
    fidx = {f: i for i, f in enumerate(families)}

    hit1 = hit3 = tot = 0
    fam_hit = fam_tot = 0
    dom_hit, dom_tot = defaultdict(int), defaultdict(int)
    per_class, per_class_tot = defaultdict(int), defaultdict(int)
    with torch.no_grad():
        for x, sp, fam, dom in dl:
            x = x.to(dev)
            f = model.encode_image(x)
            f = f / f.norm(dim=-1, keepdim=True)
            ls = f @ tsp.T
            lf = f @ tfam.T
            y = torch.tensor([sidx[s] for s in sp], device=dev)
            yf = torch.tensor([fidx[fam_prompt[s]] for s in fam], device=dev)
            k = min(3, ls.shape[1])
            top = ls.topk(k, dim=1).indices
            h1 = top[:, 0].eq(y)
            h3 = top.eq(y[:, None]).any(1)
            hit1 += h1.sum().item(); hit3 += h3.sum().item(); tot += y.numel()
            fam_hit += lf.argmax(1).eq(yf).sum().item(); fam_tot += yf.numel()
            for yi, h, d in zip(y.tolist(), h1.tolist(), dom):
                per_class[yi] += int(h); per_class_tot[yi] += 1
                dom_hit[d] += int(h); dom_tot[d] += 1

    macro = float(np.mean([per_class[c] / per_class_tot[c]
                           for c in per_class_tot]))
    res = {
        "weights": args.weights, "split": args.split, "n": tot,
        "species_top1": hit1 / max(tot, 1),
        "species_top3": hit3 / max(tot, 1),
        "species_macro_top1": macro,
        "family_top1": fam_hit / max(fam_tot, 1),
        "family_chance": 1.0 / len(families),
        "domain_top1": {d: dom_hit[d] / dom_tot[d] for d in dom_tot},
        "domain_n": dict(dom_tot),
        "n_species": len(species), "n_families": len(families),
    }
    print(json.dumps(res, indent=2))
    (ROOT / f"runs/zeroshot_{args.weights}.json").write_text(
        json.dumps(res, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
