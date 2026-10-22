#!/usr/bin/env python3
"""Verify both backbones load from local weights and run a forward pass on GPU.

Run on the server:
    python3 probe_models.py
"""
from __future__ import annotations

import json
from pathlib import Path
import os

import torch

ROOT = Path(os.environ.get("FERN_ROOT", "."))
BIOCLIP = ROOT / "weights/bioclip-2"
DINOV2 = ROOT / "weights/dinov2/dinov2_vitl14_pretrain.pth"

dev = "cuda" if torch.cuda.is_available() else "cpu"
print(f"device: {dev} | torch {torch.__version__}")


def probe_bioclip() -> None:
    import open_clip

    cfg = json.loads((BIOCLIP / "open_clip_config.json").read_text())
    model, _, preprocess = open_clip.create_model_and_transforms(
        "ViT-L-14",
        pretrained=str(BIOCLIP / "open_clip_model.safetensors"),
    )
    model = model.to(dev).eval()
    n = sum(p.numel() for p in model.parameters())
    print(f"[bioclip2] loaded, params={n/1e6:.1f}M, image_size={model.visual.image_size}")
    print(f"[bioclip2] cfg vision={cfg['model_cfg']['vision_cfg']}")
    x = torch.randn(2, 3, 224, 224, device=dev)
    with torch.no_grad():
        feat = model.encode_image(x)
    print(f"[bioclip2] encode_image -> {tuple(feat.shape)}")
    print(f"[bioclip2] preprocess: {preprocess}")


def probe_dinov2() -> None:
    import timm

    model = timm.create_model(
        "vit_large_patch14_dinov2.lvd142m",
        pretrained=False, num_classes=0, dynamic_img_size=True,
    )
    sd = torch.load(DINOV2, map_location="cpu")
    sd = sd.get("model", sd)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    print(f"[dinov2] missing={len(missing)} unexpected={len(unexpected)}")
    if missing:
        print("   missing sample:", missing[:3])
    model = model.to(dev).eval()
    n = sum(p.numel() for p in model.parameters())
    print(f"[dinov2] loaded, params={n/1e6:.1f}M, num_features={model.num_features}")
    x = torch.randn(2, 3, 224, 224, device=dev)
    with torch.no_grad():
        feat = model(x)
    print(f"[dinov2] forward -> {tuple(feat.shape)}")


if __name__ == "__main__":
    probe_bioclip()
    print("-" * 50)
    probe_dinov2()
