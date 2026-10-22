#!/usr/bin/env python3
"""Collect runs/*/test_metrics.json (plus zero-shot results) into one table.

    python summarize_runs.py [--csv out.csv]
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import os

ROOT = Path(os.environ.get("FERN_ROOT", "."))
RUNS = ROOT / "runs"


def load_all() -> list[dict]:
    rows = []
    for p in sorted(RUNS.glob("*/test_metrics.json")):
        d = json.loads(p.read_text())
        t = d["test"]
        rows.append({
            "run": p.parent.name, "backbone": d["backbone"], "mode": d["mode"],
            "img_size": d["img_size"], "hier": d.get("hier", False),
            "top1": t["top1"], "top3": t["top3"], "macro_top1": t["macro_top1"],
            "specimen_top1": t["domain_top1"].get("specimen"),
            "field_top1": t["domain_top1"].get("field"),
            "n": t["n"],
        })
    for p in sorted(RUNS.glob("zeroshot_*.json")):
        d = json.loads(p.read_text())
        rows.append({
            "run": p.stem, "backbone": d["weights"], "mode": "zero-shot",
            "img_size": 224, "hier": False,
            "top1": d["species_top1"], "top3": d["species_top3"],
            "macro_top1": d["species_macro_top1"],
            "specimen_top1": d["domain_top1"].get("specimen"),
            "field_top1": d["domain_top1"].get("field"),
            "n": d["n"],
        })
    return rows


def fmt(v):
    return "-" if v is None else f"{v:.4f}"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="")
    args = ap.parse_args()
    rows = load_all()
    hdr = ["run", "backbone", "mode", "img_size", "top1", "top3",
           "macro_top1", "specimen_top1", "field_top1"]
    print(" | ".join(f"{h:>14}" for h in hdr))
    print("-" * (17 * len(hdr)))
    for r in sorted(rows, key=lambda r: -r["top1"]):
        print(" | ".join(f"{str(r[h]):>14}" if h not in
                         ("top1", "top3", "macro_top1", "specimen_top1",
                          "field_top1") else f"{fmt(r[h]):>14}" for h in hdr))
    if args.csv:
        Path(args.csv).write_text(
            "\n".join([",".join(hdr)] + [",".join(str(r[h]) for h in hdr)
                                         for r in rows]), encoding="utf-8")
        print(f"wrote {args.csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
