#!/usr/bin/env python3
"""Collect open-set NEGATIVE images: taxa a fern identifier must reject.

The benchmark has no negative class, so rejection (open-set) cannot be
evaluated at all. This collects a stratified negative set from the same public
sources and with the same provenance fields as the positive set, so positives
and negatives can be handled by one pipeline.

Groups (each an iNaturalist taxon lookup, so the label is not hand-assigned):

    horsetail         Equisetum           fern-adjacent, out of scope
    lycophyte_oos     Selaginella         lycophyte, out of the 155-species scope
    bryophyte         Marchantiophyta     liverworts/mosses
    cycad             Cycas               fern-like foliage, non-fern
    palm              Arecaceae           fern-like fronds, angiosperm
    angiosperm        Rosa / Trifolium / Plantago   ordinary flowering plants

Usage:
    python3 download_negatives.py --per-taxon 70
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import os

from PIL import Image

HERE = Path(__file__).resolve().parent
IMG_DIR = HERE / "negative_expansion/images"
MANIFEST = HERE / "negative_expansion/manifest_negatives.csv"

UA = ("CFern-Negatives/1.0 (academic research dataset; "
      "contact: " + os.environ.get("API_CONTACT", "") + ")")
INAT_TAXA = "https://api.inaturalist.org/v1/taxa"
INAT_OBS = "https://api.inaturalist.org/v1/observations"

MIN_SIDE = 240
MAX_SIDE = 2048
QUALITY = 88
IMG_TIMEOUT = 25
ACCEPT = ("cc0", "cc-by", "cc-by-sa", "cc-by-nc", "cc-by-nc-sa", "publicdomain")

FIELDS = [
    "sample_id", "negative_group", "taxon_query", "taxon_name",
    "source", "source_record_id", "source_photo_id", "observation_url",
    "basis_of_record", "country", "event_date", "media_type", "license",
    "rights_holder", "inat_quality", "image_url", "image_path",
    "width", "height", "sha1", "download_status",
]

GROUPS: list[tuple[str, list[str]]] = [
    ("horsetail", ["Equisetum"]),
    ("lycophyte_oos", ["Selaginella"]),
    ("bryophyte", ["Marchantiophyta"]),
    ("cycad", ["Cycas"]),
    ("palm", ["Arecaceae"]),
    ("angiosperm", ["Rosa", "Trifolium", "Plantago"]),
    # Two extra groups: taxa photographed in the same habitats as ferns, so a
    # deployment pipeline will actually encounter them.
    ("fungi", ["Agaricales"]),
    ("conifer", ["Pinaceae"]),
]


def norm_lic(t: str) -> str:
    t = (t or "").strip().lower()
    if "zero/1.0" in t or "/cc0" in t:
        return "cc0"
    if "publicdomain" in t:
        return "publicdomain"
    for frag, code in (("/by-nc-sa/", "cc-by-nc-sa"), ("/by-nc/", "cc-by-nc"),
                       ("/by-sa/", "cc-by-sa"), ("/by/", "cc-by")):
        if frag in t:
            return code
    return t[:40]


_SIZE_RE = re.compile(r"/(square|thumbnail|small|medium|large|original)\.(jpg|jpeg)$")


def size_url(url: str, size: str = "medium") -> str:
    """Point at a bigger rendition, KEEPING the original extension (.jpeg on
    the iNat open-data S3 mirror, .jpg elsewhere)."""
    return _SIZE_RE.sub(lambda m: f"/{size}.{m.group(2)}", url)


def get_json(url: str, timeout: int = 30) -> dict:
    for attempt in range(3):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": UA})
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return json.loads(r.read().decode("utf-8"))
        except Exception:  # noqa: BLE001
            if attempt < 2:
                time.sleep(3 * (attempt + 1))
    return {}


def fetch_bytes(url: str, timeout: int = IMG_TIMEOUT):
    try:
        req = urllib.request.Request(url, headers={"User-Agent": UA})
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.read(12_000_000)
    except Exception:  # noqa: BLE001
        return None


def save_image(raw: bytes, dest: Path, url: str):
    try:
        with Image.open(io.BytesIO(raw)) as im:
            im = im.convert("RGB")
            w, h = im.size
            if min(w, h) < MIN_SIDE:
                big = size_url(url, "large")
                if big != url:
                    raw2 = fetch_bytes(big)
                    if raw2:
                        try:
                            with Image.open(io.BytesIO(raw2)) as im2:
                                im = im2.convert("RGB")
                                w, h = im.size
                        except Exception:  # noqa: BLE001
                            pass
                if min(w, h) < MIN_SIDE:
                    return None
            if max(w, h) > MAX_SIDE:
                r = MAX_SIDE / max(w, h)
                im = im.resize((int(w * r), int(h * r)), Image.LANCZOS)
            im.save(dest, "JPEG", quality=QUALITY)
            return im.size
    except Exception:  # noqa: BLE001
        return None


def taxon_id(query: str):
    d = get_json(f"{INAT_TAXA}?" + urllib.parse.urlencode(
        {"q": query, "per_page": 5, "locale": "en"}))
    res = d.get("results") or []
    for r in res:
        if (r.get("name") or "").lower() == query.lower():
            return r.get("id"), r.get("name"), r.get("rank")
    if res:
        return res[0].get("id"), res[0].get("name"), res[0].get("rank")
    return None, None, None


def candidates(query: str, want: int, seen: set[str]) -> list[dict]:
    tid, name, rank = taxon_id(query)
    if not tid:
        print(f"    taxon not found: {query}")
        return []
    print(f"    taxon {query} -> {name} (id={tid}, rank={rank})")
    out: list[dict] = []
    for quality in ("research", None):
        page_size = 50
        for page in range(1, 9):
            if len(out) >= want + 10:
                break
            params = {"taxon_id": tid, "photos": "true", "order_by": "votes",
                      "per_page": page_size, "page": page}
            if quality:
                params["quality_grade"] = quality
            d = get_json(f"{INAT_OBS}?{urllib.parse.urlencode(params)}")
            res = d.get("results") or []
            if not res:
                break
            for obs in res:
                okey = str(obs.get("id") or "")
                if not okey or okey in seen:
                    continue
                photos = obs.get("photos") or []
                if not photos:
                    continue
                ph = photos[0]
                lic = norm_lic(ph.get("license_code") or "")
                if lic not in ACCEPT:
                    continue
                url = size_url(ph.get("url") or "")
                if not url:
                    continue
                seen.add(okey)
                out.append({
                    "taxon_query": query, "taxon_name": name or query,
                    "source": "iNaturalist", "source_record_id": okey,
                    "source_photo_id": str(ph.get("id") or ""),
                    "observation_url":
                        f"https://www.inaturalist.org/observations/{okey}",
                    "basis_of_record": "HUMAN_OBSERVATION",
                    "country": str(obs.get("place_guess") or "")[:60],
                    "event_date": str(obs.get("observed_on_string") or "")[:40],
                    "media_type": "StillImage", "license": lic,
                    "rights_holder":
                        str(ph.get("attribution") or "").replace('"', "")[:180],
                    "inat_quality": str(obs.get("quality_grade") or ""),
                    "image_url": url,
                })
            if len(res) < page_size:
                break
        if len(out) >= want + 10:
            break
    return out


def flush(rows: list[dict]) -> None:
    tmp = MANIFEST.with_suffix(".csv.tmp")
    with tmp.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    tmp.replace(MANIFEST)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-taxon", type=int, default=70)
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    IMG_DIR.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    if MANIFEST.exists():
        with MANIFEST.open(encoding="utf-8") as fh:
            rows = list(csv.DictReader(fh))
    sha_seen = {r["sha1"] for r in rows if r.get("sha1")}
    nxt = 1
    if rows:
        nxt = max(int(r["sample_id"].split("_")[1]) for r in rows) + 1
    seen: dict[str, set[str]] = {}
    for r in rows:
        seen.setdefault(r["taxon_query"], set()).add(r["source_record_id"])

    print(f"existing negatives: {len(rows)}")
    for group, queries in GROUPS:
        for q in queries:
            have = seen.setdefault(q, set())
            want = args.per_taxon
            # count existing for this query
            already = sum(1 for r in rows if r["taxon_query"] == q)
            need = want - already
            if need <= 0:
                print(f"[{group}/{q}] already {already}, skip")
                continue
            print(f"[{group}/{q}] need {need}")
            cands = candidates(q, need, have)[: need + 8]
            got = 0
            if cands:
                with ThreadPoolExecutor(max_workers=args.workers) as pool:
                    for cand, raw in zip(cands, pool.map(
                            lambda c: fetch_bytes(c["image_url"]), cands)):
                        if got >= need or raw is None:
                            continue
                        sha = hashlib.sha1(raw).hexdigest()
                        if sha in sha_seen:
                            continue
                        sid = f"NEG_{nxt:05d}"
                        dest = IMG_DIR / f"{sid}.jpg"
                        dims = save_image(raw, dest, cand["image_url"])
                        if dims is None:
                            dest.unlink(missing_ok=True)
                            continue
                        row = {"sample_id": sid, "negative_group": group,
                               "image_path": str(dest.relative_to(HERE)),
                               "width": dims[0], "height": dims[1],
                               "sha1": sha, "download_status": "ok"}
                        row.update(cand)
                        rows.append(row)
                        sha_seen.add(sha)
                        got += 1
                        nxt += 1
                        flush(rows)
            print(f"   => +{got} (total {len(rows)})")
    by_group: dict[str, int] = {}
    for r in rows:
        by_group[r["negative_group"]] = by_group.get(r["negative_group"], 0) + 1
    print("DONE", len(rows), by_group)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
