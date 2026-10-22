#!/usr/bin/env python3
"""Turn manifest.csv into a cleaned, balanced, split-ready dataset package.

Reads the raw download manifest and writes (all non-destructive, everything
derived):

    dataset/manifest_full.csv        every downloaded image, annotated
    dataset/manifest_balanced.csv    capped per-species working set
    dataset/release_manifest.csv     redistributable-license subset
    dataset/splits/{train,val,test}.csv
    dataset/report.md                statistics + provenance summary

Design decisions, all documented in the report:
  * image_domain is derived from basis_of_record so the herbarium-vs-field
    confound becomes an explicit, controllable variable instead of a hidden one.
  * Capping keeps a spread of domains (round-robin) and prefers openly licensed
    images, which both reduces imbalance and raises the releasable share.
  * Splits are group-aware: images sharing an observation/occurrence group never
    straddle train/val/test, so the same plant cannot leak across splits.

Usage:
    python3 build_dataset.py --cap 60 --seed 20261010
"""

from __future__ import annotations

import argparse
import collections
import csv
import json
import random
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
MANIFEST = HERE / "manifest.csv"
SPECIES_JSON = ROOT / "data" / "species_data.json"
OUT = HERE / "dataset"

REDISTRIBUTABLE = {"cc0", "cc-by", "cc-by-sa", "publicdomain"}
DOMAIN_OF = {
    "PRESERVED_SPECIMEN": "specimen",
    "LIVING_SPECIMEN": "specimen",
    "MATERIAL_SAMPLE": "specimen",
    "HUMAN_OBSERVATION": "field",
    "MACHINE_OBSERVATION": "field",
}
DOMAIN_ORDER = ("field", "specimen", "other")


def load_rows() -> list[dict]:
    with MANIFEST.open(encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def annotate(rows: list[dict]) -> None:
    for r in rows:
        r["image_domain"] = DOMAIN_OF.get(
            (r.get("basis_of_record") or "").upper(), "other")
        lic = (r.get("license") or "").lower()
        r["license_class"] = "redistributable" if lic in REDISTRIBUTABLE else "restricted"
        r["releasable"] = "yes" if lic in REDISTRIBUTABLE else "no"
        # `inat_quality` was added mid-collection. Every iNat row gathered before
        # that change came from a research-grade-only query, so backfill it
        # rather than leave a misleading blank.
        if r.get("source") == "iNaturalist" and not (r.get("inat_quality") or "").strip():
            r["inat_quality"] = "research(backfilled)"
        # group key: one physical observation/occurrence must not span splits
        r["group_key"] = (r.get("observation_url") or r.get("occurrence_url")
                          or r.get("source_record_id") or r["sample_id"])


def cap_species(rows: list[dict], cap: int) -> list[dict]:
    """Round-robin across domains to keep both specimen and field images."""
    by_sp = collections.defaultdict(list)
    for r in rows:
        by_sp[r["species_id"]].append(r)
    kept: list[dict] = []
    for sid, items in by_sp.items():
        if len(items) <= cap:
            kept.extend(items)
            continue
        buckets = {d: [] for d in DOMAIN_ORDER}
        for r in items:
            buckets[r.get("image_domain", "other")].append(r)
        for d in buckets:
            # open licences first, then a stable id order
            buckets[d].sort(key=lambda r: (r["license_class"] != "redistributable",
                                           r["sample_id"]))
        chosen, idx = [], {d: 0 for d in DOMAIN_ORDER}
        while len(chosen) < cap:
            progressed = False
            for d in DOMAIN_ORDER:
                if len(chosen) >= cap:
                    break
                if idx[d] < len(buckets[d]):
                    chosen.append(buckets[d][idx[d]])
                    idx[d] += 1
                    progressed = True
            if not progressed:
                break
        kept.extend(chosen)
    kept.sort(key=lambda r: r["sample_id"])
    return kept


def make_splits(rows: list[dict], seed: int,
                ratios=(0.70, 0.15, 0.15)) -> dict[str, list[dict]]:
    """Species-stratified, group-aware 3-way split."""
    rng = random.Random(seed)
    by_sp = collections.defaultdict(list)
    for r in rows:
        by_sp[r["species_id"]].append(r)
    out = {"train": [], "val": [], "test": []}
    for sid in sorted(by_sp):
        items = by_sp[sid]
        groups = collections.defaultdict(list)
        for r in items:
            groups[r["group_key"]].append(r)
        gkeys = sorted(groups)
        rng.shuffle(gkeys)
        n = len(items)
        targets = {"train": round(n * ratios[0]),
                   "val": round(n * ratios[1])}
        targets["test"] = n - targets["train"] - targets["val"]
        counts = {k: 0 for k in out}
        for k in gkeys:
            g = groups[k]
            # place the group where it best reduces the remaining gap
            best = max(out, key=lambda s: targets[s] - counts[s])
            out[best].extend(g)
            counts[best] += len(g)
    return out


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cap", type=int, default=60)
    ap.add_argument("--seed", type=int, default=20261010)
    args = ap.parse_args()

    rows = load_rows()
    annotate(rows)
    fields = list(rows[0].keys())
    balanced = cap_species(rows, args.cap)
    splits = make_splits(balanced, args.seed)
    release = [r for r in balanced if r["releasable"] == "yes"]

    OUT.mkdir(parents=True, exist_ok=True)
    write_csv(OUT / "manifest_full.csv", rows, fields)
    write_csv(OUT / "manifest_balanced.csv", balanced, fields)
    write_csv(OUT / "release_manifest.csv", release, fields)
    for name, part in splits.items():
        write_csv(OUT / "splits" / f"{name}.csv", part, fields)

    species = json.loads(SPECIES_JSON.read_text(encoding="utf-8"))["species"]
    fam_of = {s["latin_name"]: s["family"] for s in species}

    def profile(rs: list[dict]) -> dict:
        sp = collections.Counter(r["species_latin_project"] for r in rs)
        fam = collections.Counter(r["family_project"] for r in rs)
        dom = collections.Counter(r["image_domain"] for r in rs)
        lic = collections.Counter(r["license_class"] for r in rs)
        return {"n": len(rs), "species": len(sp), "families": len(fam),
                "sp_counts": sp, "fam_counts": fam, "domains": dom, "lic": lic,
                "min_sp": min(sp.values()) if sp else 0,
                "max_sp": max(sp.values()) if sp else 0}

    full = profile(rows)
    prof = profile(balanced)
    lines = ["# Fern dataset package", "",
             f"Source manifest: `{MANIFEST.name}`", ""]
    lines += ["## Full download", "",
              f"- images: **{full['n']}**",
              f"- species: {full['species']} / {len(species)}",
              f"- families: {full['families']}",
              f"- per-species range: {full['min_sp']}-{full['max_sp']}",
              f"- image domains: {dict(full['domains'])}",
              f"- licence classes: {dict(full['lic'])}", ""]
    lines += [f"## Balanced working set (cap = {args.cap})", "",
              f"- images: **{prof['n']}**",
              f"- per-species range: {prof['min_sp']}-{prof['max_sp']}",
              f"- image domains: {dict(prof['domains'])}",
              f"- licence classes: {dict(prof['lic'])}", "",
              f"## Redistributable subset", "",
              f"- images: **{len(release)}** "
              f"({len(release)/max(prof['n'],1):.0%} of the balanced set)", ""]
    lines += ["## Splits (species-stratified, group-aware)", ""]
    for name in ("train", "val", "test"):
        p = profile(splits[name])
        lines.append(f"- {name}: {p['n']} images, {p['species']} species")
    lines += ["", "## Per-family balance (balanced set)", "",
              "| family | images |", "|---|---:|"]
    for f, c in prof["fam_counts"].most_common():
        lines.append(f"| {f} | {c} |")
    (OUT / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"full={full['n']} balanced={prof['n']} release={len(release)}")
    print(f"species={prof['species']} families={prof['families']} "
          f"per-species {prof['min_sp']}-{prof['max_sp']}")
    print(f"domains={dict(prof['domains'])}")
    print(f"splits: train={len(splits['train'])} val={len(splits['val'])} "
          f"test={len(splits['test'])}")
    print(f"wrote -> {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
