#!/usr/bin/env python3
"""Derive an explicit, reproducible evaluation protocol from the test split.

The transfer experiments showed the single most important result of the study:
a model at 92.4% on the mixed test split collapses to 15-32% once the training
and test domains/sources differ. That number is only reproducible if the
subsets are shipped as explicit files rather than recomputed ad hoc, so this
script materialises them next to the main split.

Outputs (all derived from dataset/splits/test.csv, nothing deleted):

    dataset/testsets/test_indomain.csv     the full test split (baseline)
    dataset/testsets/test_field.csv        field photographs only
    dataset/testsets/test_specimen.csv     herbarium sheets only
    dataset/testsets/test_inaturalist.csv  iNaturalist only
    dataset/testsets/test_gbif.csv         GBIF only
    dataset/testsets/test_core.csv         species with >= MIN_CORE images
    dataset/testsets/test_rare.csv         species with <  MIN_CORE images
    dataset/testsets/protocol.md           what each subset is for

Usage:
    python3 build_testsets.py [--min-core 30]
"""
from __future__ import annotations

import argparse
import collections
import csv
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
SPLITS = Path(__file__).resolve().parents[2] / "data/splits"
BALANCED = Path(__file__).resolve().parents[2] / "data/manifests/manifest_balanced.csv"
OUT = Path(__file__).resolve().parents[2] / "data/testsets"

# The country column mixes ISO codes with free-text localities from iNaturalist
# ("臺灣島, 阿里山鄉, CYQ, TW"). Extract a trailing ISO-3166 alpha-2 code when
# one is present so a geography holdout becomes possible later.
ISO2 = re.compile(r"(?:^|[,\s])([A-Z]{2})\s*$")


def norm_country(raw: str) -> str:
    v = (raw or "").strip()
    if not v:
        return ""
    m = ISO2.search(v)
    if m:
        return m.group(1)
    if len(v) == 2 and v.isalpha():
        return v.upper()
    return ""


def load(p: Path) -> list[dict]:
    with p.open(encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def write(name: str, rows: list[dict], fields: list[str]) -> None:
    p = OUT / name
    with p.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    print(f"  {name:24s} {len(rows):>5} rows")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-core", type=int, default=30,
                    help="species need this many BALANCED images to enter test_core")
    args = ap.parse_args()

    test = load(SPLITS / "test.csv")
    balanced = load(BALANCED)
    for r in test:
        r["country_iso"] = norm_country(r.get("country", ""))
    fields = list(test[0].keys())

    per_sp = collections.Counter(r["species_latin_project"] for r in balanced)
    core_sp = {s for s, n in per_sp.items() if n >= args.min_core}

    OUT.mkdir(parents=True, exist_ok=True)
    print("writing testsets:")
    write("test_indomain.csv", test, fields)
    write("test_field.csv",
          [r for r in test if r.get("image_domain") == "field"], fields)
    write("test_specimen.csv",
          [r for r in test if r.get("image_domain") == "specimen"], fields)
    write("test_inaturalist.csv",
          [r for r in test if r.get("source") == "iNaturalist"], fields)
    write("test_gbif.csv",
          [r for r in test if r.get("source") == "GBIF"], fields)
    write("test_core.csv",
          [r for r in test if r["species_latin_project"] in core_sp], fields)
    write("test_rare.csv",
          [r for r in test if r["species_latin_project"] not in core_sp], fields)

    iso = collections.Counter(r["country_iso"] for r in test if r["country_iso"])
    lines = [
        "# Evaluation protocol",
        "",
        "All subsets are derived from `dataset/splits/test.csv`; no image is",
        "moved or removed. Group-aware splitting guarantees that every image of",
        "an observation stays inside one split.",
        "",
        "| subset | purpose |",
        "|---|---|",
        "| test_indomain | headline number: mixed-domain, mixed-source |",
        "| test_field | field photographs only |",
        "| test_specimen | herbarium sheets only |",
        "| test_inaturalist | citizen-science photographs |",
        "| test_gbif | museum / aggregate records |",
        f"| test_core | species with >= {args.min_core} balanced images |",
        f"| test_rare | species with < {args.min_core} balanced images |",
        "",
        "## Why this protocol",
        "",
        "Training on the full train split and testing on the full test split",
        "reports 92.4% top-1. Training and testing *within* a single domain or",
        "source and crossing them collapses to 15-32%. Reporting only the mixed",
        "number therefore overstates readiness for deployment, where the image",
        "domain of the field stream rarely matches the training corpus.",
        "",
        "## Transfer matrix to report",
        "",
        "| train | test | subset files |",
        "|---|---|---|",
        "| source=GBIF | source=iNaturalist | train.csv / test_inaturalist.csv |",
        "| source=iNaturalist | source=GBIF | train.csv / test_gbif.csv |",
        "| image_domain=specimen | image_domain=field | train.csv / test_field.csv |",
        "| image_domain=field | image_domain=specimen | train.csv / test_specimen.csv |",
        "",
        "## Known limitations",
        "",
        "- Two species (*Neolepisorus fortunei*, *Huperzia nanchuanensis*) have too",
        "  few images to appear in every split.",
        "- `country_iso` was recovered heuristically from a free-text field; a",
        "  geography holdout needs manual verification before it is claimed.",
        "- No open-set negatives are included; rejection cannot be evaluated yet.",
        "",
        f"## Country coverage in the test split ({len(iso)} ISO codes recovered)",
        "",
        "| code | images |",
        "|---|---:|",
    ]
    for c, n in iso.most_common(15):
        lines.append(f"| {c} | {n} |")
    (OUT / "protocol.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"  protocol.md written ({len(iso)} ISO codes recovered)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
