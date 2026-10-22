#!/usr/bin/env python3
"""Bulk GBIF downloader for the fern expansion batch (v3).

Root cause of the previous failures: a single GBIF occurrence/search call with a
large `limit` grows very slow (measured 4s at limit=100, 64s for a heavily
populated species). With a 45s socket timeout those calls raised URLError and
147/155 species yielded nothing.

This version pages with limit=30 and accumulates pages until the per-species
target is met, with a generous socket timeout. It is strictly serial (GBIF and
iNaturalist both throttle aggressively on concurrency), resumable, and records
full provenance for every image.

Outputs (all under finetune_expansion/):
  images/EXP_XXXXX.jpg   image files
  manifest.csv           one row per image, full provenance
  download_state.json    resumable per-species counters
  bulk_download.log      run log
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import os

from PIL import Image

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
SPECIES_JSON = ROOT / "data" / "species_data.json"
IMG_DIR = HERE / "images"
MANIFEST = HERE / "manifest.csv"
STATE = HERE / "download_state.json"

USER_AGENT = (
    "CFern-Finetune-Expansion/3.0 (research dataset build; "
    "contact: " + os.environ.get("API_CONTACT", "") + ")"
)
GBIF_SEARCH = "https://api.gbif.org/v1/occurrence/search"

PAGE_SIZE = 30          # small pages stay fast; large ones stall for a minute
# Cap pages so one slow species cannot monopolise a burst. Measured page cost
# ranges 2-60s. 8 pages x 30 = 240 records is enough to reach fresh material
# for a species whose head pages are already downloaded.
MAX_PAGES = 8
NET_TIMEOUT = 45        # per-request socket timeout for GBIF API calls
# Media hosts are mostly AWS S3 / iNaturalist / museum boxes. Measured: a dead
# host costs the full timeout per image, so 10s keeps a bad host from eating
# the species budget while still allowing slow-but-real transfers to finish.
IMAGE_TIMEOUT = 10
MAX_IMAGE_BYTES = 12_000_000  # refuse absurd payloads
SLEEP_PAGE = 0.3
SLEEP_IMAGE = 0.1
MIN_SIDE = 300
MAX_SIDE = 2048
JPEG_QUALITY = 88
DOWNLOAD_RETRIES = 2
SPECIES_TIME_BUDGET = 150  # seconds per species before we move on
ACCEPT = ("cc0", "cc-by", "cc-by-sa", "cc-by-nc", "cc-by-nc-sa", "publicdomain")

FIELDS = [
    "sample_id", "dataset", "species_id", "family_project",
    "species_latin_project", "source", "source_taxon_name",
    "source_family_reported", "gbif_taxon_key", "gbif_dataset_key",
    "gbif_occurrence_key", "occurrence_status", "basis_of_record",
    "country", "state_province", "event_date", "institution_code",
    "occurrence_url", "media_type", "license", "rights_holder",
    "image_url", "image_path", "width", "height", "sha1", "download_status",
]


def log(msg: str) -> None:
    print(msg, flush=True)


def fetch_json(url: str, label: str, timeout: int = NET_TIMEOUT) -> dict:
    """GET with retry.

    A refused or reset connection usually means we pushed too many concurrent
    requests at GBIF, so back off progressively rather than immediately
    retrying and compounding the throttling.
    """
    delay = 5.0
    for attempt in range(3):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            time.sleep(SLEEP_PAGE)
            return data
        except urllib.error.HTTPError as exc:
            log(f"      HTTP {exc.code} on {label}; sleep {delay:.0f}s")
            time.sleep(delay)
            delay *= 2
        except Exception as exc:  # noqa: BLE001
            log(f"      ! {label}: {type(exc).__name__} {str(exc)[:60]}")
            if attempt < 2:
                time.sleep(delay)
                delay *= 2
                continue
            return {}
    return {}


def download_bytes(url: str, retries: int = DOWNLOAD_RETRIES, timeout: int = IMAGE_TIMEOUT) -> bytes | None:
    """Fetch image bytes.

    Image hosts (notably AWS S3, which serves most GBIF media) can accept a
    connection and then never send a body. The socket timeout only covers
    connect+idle gaps, so an explicit wall-clock cap is needed here, otherwise
    a single stalled host freezes the whole run. IMAGE_TIMEOUT is deliberately
    much shorter than NET_TIMEOUT: a large JPEG that cannot arrive in 20s is
    not worth waiting for, and there are usually many candidate records.
    """
    delay = 1.5
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                # Bounded read so a slow trickle cannot hang the process.
                data = resp.read(MAX_IMAGE_BYTES + 1)
                if len(data) > MAX_IMAGE_BYTES:
                    return None
            time.sleep(SLEEP_IMAGE)
            return data
        except Exception:  # noqa: BLE001
            if attempt == retries - 1:
                return None
            time.sleep(delay)
            delay *= 2
    return None


def save_image(raw: bytes, dest: Path) -> tuple[int, int] | None:
    try:
        with Image.open(io.BytesIO(raw)) as im:
            im = im.convert("RGB")
            w, h = im.size
            if min(w, h) < MIN_SIDE:
                return None
            if max(w, h) > MAX_SIDE:
                r = MAX_SIDE / max(w, h)
                im = im.resize((int(w * r), int(h * r)), Image.LANCZOS)
            im.save(dest, "JPEG", quality=JPEG_QUALITY)
            return im.size
    except Exception:  # noqa: BLE001
        return None


def norm_lic(text: str) -> str:
    """GBIF returns full licence URLs; extract the short code."""
    t = (text or "").strip().lower()
    if not t:
        return "unknown"
    if "zero/1.0" in t or "/cc0" in t:
        return "cc0"
    if "publicdomain" in t:
        return "publicdomain"
    for frag, code in (
        ("/by-nc-sa/", "cc-by-nc-sa"),
        ("/by-nc/", "cc-by-nc"),
        ("/by-sa/", "cc-by-sa"),
        ("/by/", "cc-by"),
    ):
        if frag in t:
            return code
    return t[:40]


def flush_manifest(rows: list[dict]) -> None:
    tmp = MANIFEST.with_suffix(".csv.tmp")
    with tmp.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    tmp.replace(MANIFEST)


def save_state(st: dict) -> None:
    tmp = STATE.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(st, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(STATE)


def gather_occurrences(name: str, want: int, skip_keys: set[str] | None = None) -> list[dict]:
    """Page through occurrences until we plausibly have enough candidates.

    `skip_keys` are occurrence ids already stored for this species; pages that
    contain nothing new keep paging, which is what lets a species that was
    partially downloaded pick up fresh records instead of re-reading the same
    first page every round.
    """
    skip_keys = skip_keys or set()
    out: list[dict] = []
    seen: set[str] = set()
    fresh = 0
    for page in range(MAX_PAGES):
        if fresh >= want * 2:
            break
        p = urllib.parse.urlencode(
            {
                "scientific_name": name,
                "media_type": "StillImage",
                "limit": PAGE_SIZE,
                "offset": page * PAGE_SIZE,
            }
        )
        d = fetch_json(f"{GBIF_SEARCH}?{p}", f"occ:{name} p{page}")
        res = d.get("results") or []
        if not res:
            break
        for occ in res:
            k = str(occ.get("key") or "")
            if not k or k in seen:
                continue
            seen.add(k)
            out.append(occ)
            if k not in skip_keys:
                fresh += 1
        if len(res) < PAGE_SIZE:
            break
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-species-cap", type=int, default=30)
    ap.add_argument("--global-cap", type=int, default=3000)
    ap.add_argument("--start-species", type=int, default=0)
    ap.add_argument("--max-species", type=int, default=0)
    ap.add_argument("--species-indices", type=str, default="",
                    help="comma-separated species indices, e.g. 12,40,77; "
                         "overrides start/max and allows sparse selection")
    ap.add_argument("--allow-restricted", action="store_true")
    ap.add_argument("--image-workers", type=int, default=6,
                    help="parallel image fetches per species (hosts are independent)")
    ap.add_argument("--pool-timeout", type=float, default=120.0,
                    help="hard cap on waiting for the fetch pool")
    ap.add_argument("--image-timeout", type=int, default=10,
                    help="per-image socket timeout in seconds")
    ap.add_argument("--spare-fetch", type=int, default=8,
                    help="extra candidates to fetch beyond the remaining need")
    args = ap.parse_args()

    IMG_DIR.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    if MANIFEST.exists():
        with MANIFEST.open(encoding="utf-8") as fh:
            rows = list(csv.DictReader(fh))
    st = json.loads(STATE.read_text(encoding="utf-8")) if STATE.exists() else {
        "per_species": {}, "next_id": 1
    }
    sha1_seen = {r["sha1"] for r in rows if r.get("sha1")}
    # occurrence key per species, so already-downloaded records are skipped
    # before spending a network round trip on them.
    seen_occ_by_species: dict[str, set[str]] = {}
    for r in rows:
        ok = r.get("gbif_occurrence_key")
        if ok:
            seen_occ_by_species.setdefault(r["species_id"], set()).add(ok)
    nxt = st.get("next_id", 1)
    # Derive the allocator high-water mark from the manifest itself so a stale
    # or concurrently rewritten state file can never hand out an id that is
    # already taken (which would silently overwrite an image on disk).
    if rows:
        max_id = max(int(r["sample_id"].split("_")[1]) for r in rows)
        if max_id + 1 > nxt:
            log(f"state next_id {nxt} < manifest max {max_id}; bumping to {max_id + 1}")
            nxt = max_id + 1
            st["next_id"] = nxt
            save_state(st)

    payload = json.loads(SPECIES_JSON.read_text(encoding="utf-8"))
    all_species = payload["species"]
    if args.species_indices:
        # Explicit sparse selection, used by the daemon. Slicing
        # start/max would silently cover the wrong species whenever the
        # requested indices wrap around the end of the list.
        idxs = [int(x) for x in args.species_indices.split(",") if x.strip()]
        species = [all_species[i] for i in idxs]
    else:
        species = all_species
        if args.start_species:
            species = species[args.start_species:]
        if args.max_species:
            species = species[: args.max_species]

    log(f"v3 GBIF paged | species={len(species)} per_cap={args.per_species_cap} "
        f"global_cap={args.global_cap} existing={len(rows)}")
    log(f"page_size={PAGE_SIZE} timeout={NET_TIMEOUT}s max_pages={MAX_PAGES}")
    log("=" * 70)

    t_start = time.time()
    for idx, rec in enumerate(species, 1):
        if len(rows) >= args.global_cap:
            log("GLOBAL CAP reached.")
            break
        sp = rec["latin_name"]
        already = st["per_species"].get(rec["species_id"], 0)
        need = args.per_species_cap - already
        if need <= 0:
            log(f"[{idx}/{len(species)}] {sp} ({rec['family']}) "
                f"already {already}/{args.per_species_cap}, skip")
            continue

        t_species = time.time()
        # Records already stored for this species; they are skipped before any
        # network round trip so later rounds can page deeper for fresh ones.
        have_occ = seen_occ_by_species.setdefault(rec["species_id"], set())
        occs = gather_occurrences(sp, need, have_occ)
        got = 0
        dead = 0
        small = 0
        skipped = 0
        already_have = 0

        # Collect downloadable candidates first, then fetch them with a small
        # thread pool. GBIF media is spread over ~10 independent hosts (AWS S3,
        # iNaturalist, museum servers, ...). Fetching serially means one slow
        # host stalls the whole species; in parallel the fast hosts finish while
        # the slow ones hit their timeout independently.
        candidates: list[tuple[dict, str]] = []
        # Occurrence keys already stored for this species. Re-fetching them
        # wastes a full network round trip only for the SHA-1 check to throw
        # the bytes away, and on a slow host that costs a timeout slot each.
        have_occ = seen_occ_by_species.setdefault(rec["species_id"], set())
        for occ in occs:
            media = occ.get("media") or []
            if not media:
                continue
            m = media[0]
            mtype = (m.get("type") or "").lower()
            if mtype and "image" not in mtype:
                continue
            url = m.get("identifier") or ""
            if not url:
                continue
            occ_key = str(occ.get("key") or "")
            if occ_key and occ_key in have_occ:
                already_have += 1
                continue
            lic = norm_lic(m.get("license") or "")
            if not args.allow_restricted and lic not in ACCEPT:
                skipped += 1
                continue
            candidates.append((occ, url))
            # Only over-fetch when we still need a lot. Fetching 3x the need
            # wastes the whole time budget on images we would discard, and
            # every dead host costs a full timeout slot.
            if len(candidates) >= need + args.spare_fetch:
                break

        # Fetch in waves so a host that is timing out cannot consume the whole
        # budget: take a small batch, keep what succeeded, only then extend.
        log(f"      candidates={len(candidates)} (need {need}), "
            f"fetching with {args.image_workers} workers")

        fetched: list[tuple[dict, bytes]] = []
        # shutdown(wait=False) is essential: the context-manager form blocks on
        # exit until every queued future settles, which reintroduces the exact
        # hang this parallelism was meant to avoid.
        pool = ThreadPoolExecutor(max_workers=args.image_workers)
        try:
            fut_map = {pool.submit(download_bytes, url, DOWNLOAD_RETRIES, args.image_timeout): occ
                       for occ, url in candidates}

            def collect(fut) -> None:
                """Harvest one finished future into `fetched`."""
                nonlocal dead
                occ = fut_map[fut]
                try:
                    raw = fut.result()
                except Exception:  # noqa: BLE001
                    raw = None
                if raw is None:
                    dead += 1
                else:
                    fetched.append((occ, raw))

            # Poll in a bounded loop rather than relying on as_completed's
            # timeout: when as_completed raises, the futures that had already
            # completed are discarded along with it, which loses finished
            # downloads and reports "fetched 0" even though the bytes arrived.
            deadline = min(
                time.time() + args.pool_timeout,
                t_species + SPECIES_TIME_BUDGET,
            )
            pending_futs = dict(fut_map)
            while pending_futs:
                if time.time() > deadline:
                    break
                done_now = [f for f in pending_futs if f.done()]
                if not done_now:
                    time.sleep(0.2)
                    continue
                for f in done_now:
                    collect(f)
                    del pending_futs[f]
                if time.time() - t_species > SPECIES_TIME_BUDGET:
                    log(f"      time budget {SPECIES_TIME_BUDGET}s hit, "
                        f"stopping fetches")
                    break
            if pending_futs:
                log(f"      fetch deadline reached with {len(pending_futs)} "
                    f"in flight; kept {len(fetched)}")
            dead += len(pending_futs)
        finally:
            pool.shutdown(wait=False, cancel_futures=True)

        for occ, raw in fetched:
            if got >= need:
                break
            if time.time() - t_species > SPECIES_TIME_BUDGET:
                break
            digest = hashlib.sha1(raw).hexdigest()
            if digest in sha1_seen:
                # Same bytes under a different record; remember the record so
                # the next round does not fetch it again.
                ok = str(occ.get("key") or "")
                if ok:
                    have_occ.add(ok)
                continue
            media = occ.get("media") or [{}]
            m = media[0]
            mtype = (m.get("type") or "").lower()
            sid = f"EXP_{nxt:05d}"
            dest = IMG_DIR / f"{sid}.jpg"
            if dest.exists():
                # Never clobber an existing image; skip to the next free id.
                log(f"      id collision on {sid}, advancing")
                nxt += 1
                continue
            dims = save_image(raw, dest)
            if dims is None:
                dest.unlink(missing_ok=True)
                small += 1
                continue
            rows.append(
                {
                    "sample_id": sid,
                    "dataset": "finetune_expansion_batch1",
                    "species_id": rec["species_id"],
                    "family_project": rec["family"],
                    "species_latin_project": sp,
                    "source": "GBIF",
                    "source_taxon_name": str(
                        occ.get("scientificName") or occ.get("verbatimScientificName") or ""
                    )[:150],
                    "source_family_reported": str(occ.get("family") or ""),
                    "gbif_taxon_key": occ.get("taxonKey") or "",
                    "gbif_dataset_key": occ.get("datasetKey") or "",
                    "gbif_occurrence_key": occ.get("key") or "",
                    "occurrence_status": occ.get("status") or "",
                    "basis_of_record": occ.get("basisOfRecord") or "",
                    "country": str(occ.get("countryCode") or occ.get("country") or "")[:60],
                    "state_province": str(occ.get("stateProvince") or "")[:80],
                    "event_date": str(occ.get("eventDate") or occ.get("year") or "")[:40],
                    "institution_code": str(occ.get("institutionCode") or "")[:80],
                    "occurrence_url": str(occ.get("references") or occ.get("identifier") or "")[:240],
                    "media_type": mtype[:60],
                    "license": lic,
                    "rights_holder": str(m.get("rightsHolder") or occ.get("rightsHolder") or "")[:180],
                    "image_url": url[:240],
                    "image_path": str(dest.relative_to(ROOT)),
                    "width": dims[0],
                    "height": dims[1],
                    "sha1": digest,
                    "download_status": "ok",
                }
            )
            sha1_seen.add(digest)
            ok = str(occ.get("key") or "")
            if ok:
                have_occ.add(ok)
            nxt += 1
            flush_manifest(rows)
            got += 1

        # Only persist a positive counter. A zero entry would mark the species
        # as "attempted" and make the driver skip it forever on later rounds.
        if got > 0:
            st["per_species"][rec["species_id"]] = already + got
        st["next_id"] = nxt
        save_state(st)

        el = time.time() - t_start
        log(f"[{idx}/{len(species)}] {sp} ({rec['family']}) "
            f"cand={len(occs)} +{got} -> {len(rows)} total | "
            f"dup-skip={already_have} dead={dead} small={small} "
            f"lic-skip={skipped} | {el:.0f}s")

    by_sp: dict[str, int] = {}
    by_fam: dict[str, int] = {}
    for r in rows:
        by_sp[r["species_latin_project"]] = by_sp.get(r["species_latin_project"], 0) + 1
        by_fam[r["family_project"]] = by_fam.get(r["family_project"], 0) + 1

    log("=" * 70)
    log(f"IMAGES {len(rows)} | SPECIES {len(by_sp)} | FAMILIES {len(by_fam)} | "
        f"{time.time()-t_start:.0f}s")
    zero = [r["latin_name"] for r in species if by_sp.get(r["latin_name"], 0) == 0]
    log(f"ZERO-IMAGE SPECIES: {len(zero)}")
    if zero:
        log(f"  {zero}")
    summary = {
        "generated": time.strftime("%Y-%m-%d %H:%M:%S"),
        "total_images": len(rows),
        "species_covered": len(by_sp),
        "families_covered": len(by_fam),
        "per_family": sorted(by_fam.items(), key=lambda kv: -kv[1]),
        "per_species": sorted(by_sp.items(), key=lambda kv: -kv[1]),
        "zero_image_species": zero,
        "elapsed_sec": round(time.time() - t_start),
    }
    (HERE / "download_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    log(f"summary: {HERE / 'download_summary.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())