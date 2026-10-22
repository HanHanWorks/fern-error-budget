#!/usr/bin/env python3
"""Download a first expansion batch of fern images from iNaturalist (primary)
and GBIF (fallback) into a working folder.

Design constraints:
  - Respect API throttling: strictly serial requests with long sleeps and
    exponential backoff on 429/5xx. The earlier concurrent probe produced
    false negatives because of throttling, so this script never parallelises.
  - Hard per-species cap and global cap so the batch stays bounded and fits
    comfortably on disk.
  - Records full provenance for every image: source, record id, photo id,
    URL, licence, rights holder, SHA-1, dimensions. Mirrors the schema of the
    existing external_validation manifest so downstream code can reuse it.
  - Skips duplicates by SHA-1 and by (taxon, photo id).
  - Resumable: existing files are recorded and not re-downloaded.
  - Taxonomy is recorded verbatim as returned by each source. Reconciling the
    project (PPG I) family assignment against GBIF's taxonomy is deliberately
    left for a later pass, as requested.

Usage:
    python3 download_expansion_batch.py --per-species-cap 50 --global-cap 3000
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import os

from PIL import Image

ROOT = Path(__file__).resolve().parents[2]  # repository root
SPECIES_JSON = ROOT / "data" / "species_data.json"
WORK = Path(__file__).resolve().parent  # finetune_expansion/
IMG_DIR = WORK / "images"
MANIFEST = WORK / "manifest.csv"
STATE = WORK / "download_state.json"
SUMMARY = WORK / "download_summary.json"

USER_AGENT = (
    "CFern-Finetune-Expansion/1.0 (research dataset expansion; "
    "contact: " + os.environ.get("API_CONTACT", "") + ")"
)
INAT_OBS = "https://api.inaturalist.org/v1/observations"
GBIF_SEARCH = "https://api.gbif.org/v1/occurrence/search"

SLEEP_BETWEEN = 1.0
BACKOFF_429 = 25.0
BACKOFF_5XX = 8.0
MAX_BACKOFF_ROUNDS = 5
# iNaturalist's observations endpoint stalls on large result sets. Measured on
# Huperzia selago (6874 records): per_page=10 + order_by=votes returns in 2.5s,
# per_page=25 without order_by took 56s. Keeping order_by=votes is the key.
NET_TIMEOUT = 30
OBS_PER_PAGE = 25
MIN_SIDE = 320  # iNat "medium" is only 375px; "large" is the right tier
MAX_SIDE = 2048  # downscale before saving to bound disk use
JPEG_QUALITY = 88
# iNat size tiers: square(75) < thumbnail(140) < small(240) < medium(375) < large(1024) < original
INAT_SIZE_TIER = "large"

# Licence families we accept. Others are recorded but not downloaded unless
# --allow-restricted is passed.
PERMISSIVE = ("cc0", "cc-by", "cc-by-sa", "cc-by-nc", "cc-by-nc-sa", "publicdomain")

FIELDS = [
    "sample_id",
    "dataset",
    "species_id",
    "family_project",
    "species_latin_project",
    "source",
    "source_taxon_name",
    "source_family_reported",
    "source_rank",
    "source_record_id",
    "source_photo_id",
    "observation_url",
    "image_url",
    "image_path",
    "license",
    "rights_holder",
    "width",
    "height",
    "sha1",
    "download_status",
]


def log(msg: str) -> None:
    print(msg, flush=True)


def fetch_json(url: str, label: str) -> dict:
    """Serial GET with 429/5xx-aware exponential backoff."""
    delay = BACKOFF_429
    soft_delay = BACKOFF_5XX
    for _ in range(MAX_BACKOFF_ROUNDS):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
            with urllib.request.urlopen(req, timeout=NET_TIMEOUT) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            time.sleep(SLEEP_BETWEEN)
            return data
        except urllib.error.HTTPError as exc:
            if exc.code == 429:
                log(f"      429 on {label}; sleeping {delay:.0f}s")
                time.sleep(delay)
                delay = min(delay * 2, 300)
                continue
            if 500 <= exc.code < 600:
                log(f"      HTTP {exc.code} on {label}; sleeping {soft_delay:.0f}s")
                time.sleep(soft_delay)
                soft_delay = min(soft_delay * 2, 120)
                continue
            log(f"      HTTP {exc.code} on {label}; giving up on this record")
            time.sleep(SLEEP_BETWEEN)
            return {}
        except Exception as exc:  # noqa: BLE001
            log(f"      ! {label}: {exc}")
            time.sleep(SLEEP_BETWEEN)
            return {}
    return {}


def load_state() -> dict:
    if STATE.exists():
        return json.loads(STATE.read_text(encoding="utf-8"))
    return {"per_species": {}, "sha1_seen": [], "next_id": 1}


def save_state(state: dict) -> None:
    WORK.mkdir(parents=True, exist_ok=True)
    STATE.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")


def load_manifest() -> list[dict]:
    if not MANIFEST.exists():
        return []
    with MANIFEST.open(encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def norm_license(text: str) -> str:
    t = (text or "").strip().lower()
    for p in PERMISSIVE:
        if t.startswith(p):
            return p
    return t or "unknown"


def is_acceptable_license(code: str, allow_restricted: bool) -> bool:
    if allow_restricted:
        return True
    return code.startswith(("cc0", "cc-by", "cc-by-sa", "publicdomain"))


def resolve_taxon_id(species_latin: str) -> tuple[int | None, str, str | None]:
    """Resolve a species name to an iNaturalist taxon id.

    Returns (taxon_id, matched_name, matched_family).
    """
    params = urllib.parse.urlencode(
        {"q": species_latin, "rank": "species", "per_page": 8, "locale": "en"}
    )
    data = fetch_json(f"https://api.inaturalist.org/v1/taxa?{params}", f"taxa:{species_latin}")
    results = data.get("results") or []
    if not results:
        return None, "", None
    target = species_latin.strip().lower()
    for row in results:
        if (row.get("name") or "").strip().lower() == target:
            return row.get("id"), row.get("name") or "", row.get("family")
    row = results[0]
    return row.get("id"), row.get("name") or "", row.get("family")


def save_image(raw: bytes, dest: Path) -> tuple[int, int] | None:
    """Validate, downscale and save. Returns (width, height) or None if unusable."""
    import io

    try:
        with Image.open(io.BytesIO(raw)) as im:
            im = im.convert("RGB")
            w, h = im.size
            if min(w, h) < MIN_SIDE:
                return None
            if max(w, h) > MAX_SIDE:
                ratio = MAX_SIDE / max(w, h)
                im = im.resize((int(w * ratio), int(h * ratio)), Image.LANCZOS)
            im.save(dest, "JPEG", quality=JPEG_QUALITY)
            return im.size
    except Exception:  # noqa: BLE001
        return None


def download_bytes(url: str, retries: int = 3) -> bytes | None:
    """Fetch image bytes with retries.

    Long edge downloads frequently end in http.client.IncompleteRead on this
    network, so a single failure must not silently drop an image.
    """
    delay = 2.0
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
            with urllib.request.urlopen(req, timeout=NET_TIMEOUT) as resp:
                data = resp.read()
            time.sleep(0.4)
            return data
        except Exception as exc:  # noqa: BLE001
            log(f"      ! download attempt {attempt + 1}/{retries} failed: "
                f"{type(exc).__name__}")
            if attempt == retries - 1:
                return None
            time.sleep(delay)
            delay *= 2
    return None


def inat_observations(taxon_id: int, page: int) -> list[dict]:
    params = urllib.parse.urlencode(
        {
            "taxon_id": taxon_id,
            "quality_grade": "research",
            "photos": "true",
            "order_by": "votes",
            "per_page": OBS_PER_PAGE,
            "page": page,
        }
    )
    data = fetch_json(f"{INAT_OBS}?{params}", f"inat_obs p{page}")
    return data.get("results") or []


def process_inat_species(rec: dict, args: argparse.Namespace, state: dict,
                         sha1_seen: set[str], rows: list[dict], counter: dict) -> None:
    sp_id = rec["species_id"]
    sp_latin = rec["latin_name"]
    family = rec["family"]

    taxon_id, matched_name, matched_family = resolve_taxon_id(sp_latin)
    if not taxon_id:
        log(f"   [skip] no iNat taxon for {sp_latin}")
        return
    log(f"   iNat taxon {taxon_id}: {matched_name} (family={matched_family})")

    got = 0
    attempts = 0
    # Each page holds OBS_PER_PAGE observations and may contribute several
    # photos, so cap page reads rather than loop count.
    max_pages = max(1, args.per_species_cap // 2 + 1)
    while got < args.per_species_cap and attempts < max_pages:
        attempts += 1
        obs_list = inat_observations(taxon_id, page=attempts)
        if not obs_list:
            break
        for obs in obs_list:
            if got >= args.per_species_cap:
                break
            for photo in obs.get("photos") or []:
                if got >= args.per_species_cap:
                    break
                url = photo.get("url") or ""
                if not url:
                    continue
                # iNat size tiers: square/thumbnail/small/medium/large/original.
                # medium is only ~375px, too small for diagnostic morphology.
                url = re.sub(
                    r"/(square|thumbnail|small|medium|original)\.jpg$",
                    f"/{INAT_SIZE_TIER}.jpg",
                    url,
                )
                pid = str(photo.get("id") or "")
                license_code = norm_license(photo.get("license_code") or "")
                if not is_acceptable_license(license_code, args.allow_restricted):
                    continue
                raw = download_bytes(url)
                if not raw:
                    continue
                digest = hashlib.sha1(raw).hexdigest()
                if digest in sha1_seen:
                    continue
                counter["n"] += 1
                sid = f"EXP_{counter['n']:05d}"
                dest = IMG_DIR / f"{sid}.jpg"
                dims = save_image(raw, dest)
                if dims is None:
                    dest.unlink(missing_ok=True)
                    counter["n"] -= 1
                    continue
                sha1_seen.add(digest)
                rows.append(
                    {
                        "sample_id": sid,
                        "dataset": "finetune_expansion_batch1",
                        "species_id": sp_id,
                        "family_project": family,
                        "species_latin_project": sp_latin,
                        "source": "iNaturalist",
                        "source_taxon_name": matched_name,
                        "source_family_reported": matched_family or "",
                        "source_rank": "species",
                        "source_record_id": str(obs.get("id") or ""),
                        "source_photo_id": pid,
                        "observation_url": f"https://www.inaturalist.org/observations/{obs.get('id')}",
                        "image_url": url,
                        "image_path": str(dest.relative_to(ROOT)),
                        "license": license_code,
                        "rights_holder": ((photo.get("attribution") or "").replace('"', "")[:180]),
                        "width": dims[0],
                        "height": dims[1],
                        "sha1": digest,
                        "download_status": "ok",
                    }
                )
                flush_manifest(rows)
                got += 1
    log(f"      -> downloaded {got} images for {sp_latin}")


def process_gbif_species(rec: dict, args: argparse.Namespace, sha1_seen: set[str],
                         rows: list[dict], counter: dict) -> None:
    sp_latin = rec["latin_name"]
    params = urllib.parse.urlencode(
        {
            "scientific_name": sp_latin,
            "media_type": "StillImage",
            "limit": args.per_species_cap,
        }
    )
    data = fetch_json(f"{GBIF_SEARCH}?{params}", f"gbif:{sp_latin}")
    results = data.get("results") or []
    if not results:
        return
    log(f"   GBIF returned {len(results)} media-bearing occurrences for {sp_latin}")

    got = 0
    for occ in results:
        if got >= args.per_species_cap:
            break
        media = occ.get("media") or []
        if not media:
            continue
        m = media[0]
        url = m.get("identifier") or ""
        if not url:
            continue
        identifier = occ.get("identifier") or ""
        license_code = norm_license(m.get("license") or "")
        if not is_acceptable_license(license_code, args.allow_restricted):
            continue
        raw = download_bytes(url)
        if not raw:
            continue
        digest = hashlib.sha1(raw).hexdigest()
        if digest in sha1_seen:
            continue
        counter["n"] += 1
        sid = f"EXP_{counter['n']:05d}"
        dest = IMG_DIR / f"{sid}.jpg"
        dims = save_image(raw, dest)
        if dims is None:
            dest.unlink(missing_ok=True)
            counter["n"] -= 1
            continue
        sha1_seen.add(digest)
        fam = occ.get("family") or ""
        rows.append(
            {
                "sample_id": sid,
                "dataset": "finetune_expansion_batch1",
                "species_id": rec["species_id"],
                "family_project": rec["family"],
                "species_latin_project": sp_latin,
                "source": "GBIF",
                "source_taxon_name": (occ.get("scientificName") or sp_latin)[:120],
                "source_family_reported": str(fam),
                "source_rank": occ.get("rank") or "species",
                "source_record_id": str(occ.get("key") or ""),
                "source_photo_id": str(m.get("id") or ""),
                "observation_url": identifier[:220],
                "image_url": url[:220],
                "image_path": str(dest.relative_to(ROOT)),
                "license": license_code,
                "rights_holder": str(m.get("rightsHolder") or "")[:180],
                "width": dims[0],
                "height": dims[1],
                "sha1": digest,
                "download_status": "ok",
            }
        )
        flush_manifest(rows)
        got += 1
    log(f"      -> downloaded {got} GBIF images for {sp_latin}")


# Shared lock so concurrent GBIF workers append to rows/counter atomically.
LOCK = threading.Lock()


def drain_gbif_queue(queue: list, args: argparse.Namespace, sha1_seen: set[str],
                     rows: list[dict], counter: dict, wait: bool = True) -> None:
    """Run queued GBIF fallbacks in a small thread pool.

    GBIF's occurrence/search endpoint frequently stalls past the socket timeout,
    so each task is bounded and the main loop only waits a short grace period
    before carrying on. iNat is the primary source; GBIF only fills gaps.
    """
    if not queue:
        return
    with ThreadPoolExecutor(max_workers=args.gbif_workers) as pool:
        futs = [
            pool.submit(_gbif_worker, rec, args, sha1_seen, rows, counter, LOCK)
            for rec, _, _ in queue
        ]
        deadline = time.time() + (args.gbif_grace if not wait else 999)
        for fut in futs:
            remaining = max(1.0, deadline - time.time())
            try:
                fut.result(timeout=remaining)
            except Exception as exc:  # noqa: BLE001
                log(f"      ! gbif worker: {exc}")
            if not wait and time.time() > deadline:
                log(f"      gbif grace expired, leaving {len(futs)} in flight")
                break


def _gbif_worker(rec: dict, args: argparse.Namespace, sha1_seen: set[str],
                 rows: list[dict], counter: dict, lock) -> None:
    with lock:
        process_gbif_species(rec, args, sha1_seen, rows, counter)


def flush_manifest(rows: list[dict]) -> None:
    """Atomically rewrite the manifest so image files never become orphans.

    A crash between saving an image and recording it would otherwise leave
    provenance-less files on disk, which cannot be used in the manuscript.
    """
    tmp = MANIFEST.with_suffix(".csv.tmp")
    with tmp.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    tmp.replace(MANIFEST)


def main() -> int:
    global MANIFEST, STATE, SUMMARY
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-species-cap", type=int, default=40)
    ap.add_argument("--global-cap", type=int, default=2500)
    ap.add_argument("--allow-restricted", action="store_true")
    ap.add_argument("--fallback-gbif-below", type=int, default=15,
                    help="if iNat yields fewer than this, try GBIF")
    ap.add_argument("--max-species", type=int, default=0,
                    help="process at most N species this run (0 = all); "
                         "use to slice the job into chunks")
    ap.add_argument("--start-species", type=int, default=0,
                    help="skip the first N species (resume by chunk)")
    ap.add_argument("--gbif-workers", type=int, default=4)
    ap.add_argument("--gbif-grace", type=float, default=25.0,
                    help="seconds the main loop waits for queued GBIF jobs")
    ap.add_argument("--manifest-out", type=str, default="",
                    help="write manifest to this path instead of ./manifest.csv")
    ap.add_argument("--state-out", type=str, default="",
                    help="write resumable state to this path")
    args = ap.parse_args()
    lock = threading.Lock()

    # Allow the parallel driver to redirect output paths so that concurrent
    # workers never write the same manifest/state file.
    if args.manifest_out:
        MANIFEST = Path(args.manifest_out)
        MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    if args.state_out:
        STATE = Path(args.state_out)
        STATE.parent.mkdir(parents=True, exist_ok=True)

    IMG_DIR.mkdir(parents=True, exist_ok=True)
    WORK.mkdir(parents=True, exist_ok=True)

    rows = load_manifest()
    state = load_state()
    sha1_seen = set(state.get("sha1_seen", []))
    for r in rows:
        if r.get("sha1"):
            sha1_seen.add(r["sha1"])
    counter = {"n": state.get("next_id", 1)}
    gbif_target: list = []

    payload = json.loads(SPECIES_JSON.read_text(encoding="utf-8"))
    species = payload["species"]
    if args.start_species:
        species = species[args.start_species:]
        log(f"Skipping first {args.start_species} species this run")
    if args.max_species:
        species = species[: args.max_species]
        log(f"Processing at most {args.max_species} species this run")
    total_spec = len(species)
    log(f"Target batch: {total_spec} species, per-species cap {args.per_species_cap}, "
        f"global cap {args.global_cap}, existing rows {len(rows)}")
    log("Serial mode: no parallel requests (avoids the earlier 429 false negatives)")
    log("=" * 70)

    for idx, rec in enumerate(species, 1):
        if len(rows) >= args.global_cap:
            log(f"\nGLOBAL CAP {args.global_cap} reached, stopping.")
            break
        if state["per_species"].get(rec["species_id"], 0) >= args.per_species_cap:
            continue
        log(f"\n[{idx}/{total_spec}] {rec['latin_name']}  ({rec['family']})")

        before = len(rows)
        process_inat_species(rec, args, state, sha1_seen, rows, counter)
        gained = len(rows) - before
        log(f"      iNat gained {gained}")

        if gained < args.fallback_gbif_below:
            # GBIF's occurrence endpoint is slow (often 30s timeout per query).
            # Run it in a thread so the main loop never blocks on it, and cap
            # the wait; iNat already supplied the bulk of the images.
            gbif_target.append((rec, len(rows), counter["n"]))
            log(f"      queued GBIF fallback (non-blocking)")
            if len(gbif_target) >= args.gbif_workers:
                drain_gbif_queue(gbif_target, args, sha1_seen, rows, counter, wait=False)
                gbif_target.clear()

        state["per_species"][rec["species_id"]] = state["per_species"].get(rec["species_id"], 0) + gained
        state["sha1_seen"] = list(sha1_seen)
        state["next_id"] = counter["n"]
        save_state(state)
        flush_manifest(rows)

        if idx % 10 == 0:
            log(f"  --- checkpoint: {len(rows)} images total")

    log("=" * 70)
    if gbif_target:
        log(f"Draining {len(gbif_target)} queued GBIF fallback jobs...")
        drain_gbif_queue(gbif_target, args, sha1_seen, rows, counter, wait=True)
        flush_manifest(rows)

    by_sp: dict[str, int] = {}
    by_fam: dict[str, int] = {}
    by_src: dict[str, int] = {}
    for r in rows:
        by_sp[r["species_latin_project"]] = by_sp.get(r["species_latin_project"], 0) + 1
        by_fam[r["family_project"]] = by_fam.get(r["family_project"], 0) + 1
        by_src[r["source"]] = by_src.get(r["source"], 0) + 1

    summary = {
        "generated": time.strftime("%Y-%m-%d %H:%M:%S"),
        "per_species_cap": args.per_species_cap,
        "global_cap": args.global_cap,
        "total_images": len(rows),
        "species_covered": len(by_sp),
        "species_in_benchmark": total_spec,
        "families_covered": len(by_fam),
        "by_source": by_src,
        "top_families": sorted(by_fam.items(), key=lambda kv: -kv[1])[:15],
        "species_zero_images": [
            r["latin_name"] for r in species if by_sp.get(r["latin_name"], 0) == 0
        ],
    }
    SUMMARY.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    log(f"IMAGES: {len(rows)}")
    log(f"SPECIES COVERED: {len(by_sp)} / {total_spec}")
    log(f"FAMILIES COVERED: {len(by_fam)}")
    log(f"BY SOURCE: {by_src}")
    zero = summary["species_zero_images"]
    log(f"SPECIES WITH ZERO IMAGES: {len(zero)}")
    if zero:
        log(f"  {zero}")
    log(f"manifest: {MANIFEST}")
    log(f"summary: {SUMMARY}")
    return 0


if __name__ == "__main__":
    sys.exit(main())