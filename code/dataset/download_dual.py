#!/usr/bin/env python3
"""Dual-source fern image downloader (iNaturalist + GBIF with failover).

Why this exists: during the previous run both public APIs throttled this host
independently — GBIF returned `Errno 61 Connection refused` for three hours
while iNaturalist was also closing connections. Neither source is reliable
alone, so this downloader tries both per species and keeps whatever arrives.

Design:
  - iNaturalist primary (research grade, votes-ordered, `large` photos).
  - GBIF fallback, paged at30 records because large limits take 10-60s.
  - Per-source occurrence memo so already-downloaded records are skipped
    before spending a request on them.
  - Thread-pool image fetch with self-harvesting poll loop; `as_completed`
    with a timeout silently discards already-finished futures, which showed up
    as "fetched 0" while the bytes were actually arriving.
  - Atomic manifest rewrite after every image; durable resume state.
  - Short budgets everywhere so a slow host can never freeze a species.

Usage:
    python3 download_dual.py --species-indices 12,40 --per-species-cap 30
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
import re
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from PIL import Image

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
SPECIES_JSON = ROOT / "data" / "species_data.json"
IMG_DIR = HERE / "images"
MANIFEST = HERE / "manifest.csv"
STATE = HERE / "download_state.json"

UA = ("CFern-Finetune-Expansion/4.1 (academic research dataset; "
      "contact: " + os.environ.get("API_CONTACT", "") + ")")

# This host reaches the public internet through a local proxy whose port
# changes between sessions, so the environment variables are re-read at every
# start and no port is ever hardcoded. An env-provided proxy is still verified
# before use: it can be up yet unable to reach the target host.
PROXY_PORTS_FALLBACK = (7890, 7897, 1087, 8080, 8118)


def proxy_candidates() -> list[str]:
    cands = [
        os.environ.get("HTTPS_PROXY") or os.environ.get("https_proxy"),
        os.environ.get("HTTP_PROXY") or os.environ.get("http_proxy"),
    ]
    cands += [f"http://127.0.0.1:{p}" for p in PROXY_PORTS_FALLBACK]
    out: list[str] = []
    for c in cands:
        if c and c not in out:
            out.append(c)
    return out


def _probe(opener) -> bool:
    """Verify the path reaches both the JSON API and the image CDN.

    A route can pass the API probe and still fail on the S3/media hosts, which
    looks exactly like "downloaded 0 images" even though candidates were found.
    Requiring both keeps the daemon from settling on a half-working route.
    """
    for url in (
        "https://api.inaturalist.org/v1/taxa?q=probe&per_page=1",
        "https://inaturalist-open-data.s3.amazonaws.com/photos/140915980/square.jpg",
    ):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": UA})
            with opener.open(req, timeout=12) as r:
                r.read(2048)
        except Exception:  # noqa: BLE001
            return False
    return True


def install_opener() -> str:
    """Pick a working network path: an explicit proxy, or a direct connection.

    Order matters. The local proxy port changes between sessions and its
    process may already be dead, so an env-provided proxy is tried first but
    verified. Direct connection is probed independently and often works even
    when the proxy does not, so it must be a real candidate rather than an
    assumed fallback.
    """
    for proxy in proxy_candidates():
        opener = urllib.request.build_opener(
            urllib.request.ProxyHandler({"http": proxy, "https": proxy}))
        if _probe(opener):
            urllib.request.install_opener(opener)
            log(f"network: proxy {proxy}")
            return proxy
    direct = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    if _probe(direct):
        urllib.request.install_opener(direct)
        log("network: direct (no proxy needed)")
        return "direct"
    # Nothing verified; install a plain opener and let retries handle it.
    urllib.request.install_opener(urllib.request.build_opener())
    log("network: unverified, will retry")
    return "unverified"


def reopen_best() -> None:
    """Re-probe and reinstall the best available opener.

    Called when a burst produced nothing but the network looked broken: the
    proxy process may have died or come back on a different port.
    """
    install_opener()

INAT_TAXA = "https://api.inaturalist.org/v1/taxa"
INAT_OBS = "https://api.inaturalist.org/v1/observations"
GBIF_OCC = "https://api.gbif.org/v1/occurrence/search"
GBIF_MATCH = "https://api.gbif.org/v1/species/match"
IDIGBIO_MEDIA = "https://search.idigbio.org/v2/search/media/"

PAGE = 25                # iNat page size; large values stall
MAX_PAGES = 6
API_TIMEOUT = 25
# Dead museum CDNs burn the whole per-species budget on 45s hangs; the images
# we can actually reach (iNat S3) answer in 1-3s, so fail fast instead.
IMG_TIMEOUT = 20
MAX_BYTES = 12_000_000
SLEEP_API = 0.8
SLEEP_IMG = 0.0
# Worst-case wall clock per species. 240s still allows ~40 image fetches with
# 8 workers while stopping a genuinely exhausted taxon from monopolising a run.
SPECIES_BUDGET = 240
# iNat serves fixed-size derivatives; `medium` is 375x500 (>= MIN_SIDE=300) at
# ~100KB, while `large` is ~325KB and takes 3x longer. On a throttled proxy the
# timeout, not the network, is what kills the batch, so use the small derivative.
INAT_SIZE = "medium"
# Multiple photos of one observation are not independent samples, so this stays
# at 1 by default. Scarce taxa (some Chinese ferns have <10 observations but 3-5
# photos each) can be topped up with >1; build_dataset.py keeps every photo of
# an observation in the same split, so the extra views cannot leak across sets.
PHOTOS_PER_OBS = 1
# 240 keeps portrait field photos (e.g. 281x500) that carry valid morphology,
# while staying above the 224-px short side used by ViT training inputs.
MIN_SIDE = 240
MAX_SIDE = 2048
QUALITY = 88
ACCEPT = ("cc0", "cc-by", "cc-by-sa", "cc-by-nc", "cc-by-nc-sa", "publicdomain")

LOCK = threading.Lock()

FIELDS = [
    "sample_id", "dataset", "species_id", "family_project",
    "species_latin_project", "source", "source_taxon_name",
    "source_family_reported", "source_record_id", "source_photo_id",
    "observation_url", "occurrence_url", "basis_of_record", "country",
    "event_date", "institution_code", "media_type", "license",
    "rights_holder", "image_url", "image_path", "width", "height", "sha1",
    "download_status", "inat_quality",
]


def log(m: str) -> None:
    print(m, flush=True)


def get_json(url: str, label: str, timeout: int = API_TIMEOUT) -> dict:
    """GET JSON with backoff. Returns {} on failure."""
    delay = 4.0
    for attempt in range(3):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": UA})
            with urllib.request.urlopen(req, timeout=timeout) as r:
                data = json.loads(r.read().decode("utf-8"))
            time.sleep(SLEEP_API)
            return data
        except Exception as exc:  # noqa: BLE001
            log(f"      !{label}: {type(exc).__name__}")
            if attempt < 2:
                time.sleep(delay)
                delay *= 2
    return {}


def norm_lic(t: str) -> str:
    t = (t or "").strip().lower()
    if not t:
        return "unknown"
    if "zero/1.0" in t or "/cc0" in t:
        return "cc0"
    if "publicdomain" in t:
        return "publicdomain"
    for frag, code in (("/by-nc-sa/", "cc-by-nc-sa"), ("/by-nc/", "cc-by-nc"),
                       ("/by-sa/", "cc-by-sa"), ("/by/", "cc-by")):
        if frag in t:
            return code
    return t[:40]


# iNat serves the same photo at several fixed sizes. `static.inaturalist.org`
# uses `.jpg`, but the open-data S3 mirror uses `.jpeg`; rewriting the size must
# keep whatever extension the record came with, otherwise every S3 URL 404s.
_SIZE_RE = re.compile(r"/(square|thumbnail|small|medium|large|original)\.(jpg|jpeg)$")


def size_url(url: str, size: str = INAT_SIZE) -> str:
    """Point an iNat photo URL at `size`, preserving its original extension."""
    return _SIZE_RE.sub(lambda m: f"/{size}.{m.group(2)}", url)


IMG_FAIL: dict = {}


def fetch_bytes(url: str, timeout: int = IMG_TIMEOUT) -> bytes | None:
    try:
        req = urllib.request.Request(url, headers={"User-Agent": UA})
        with urllib.request.urlopen(req, timeout=timeout) as r:
            data = r.read(MAX_BYTES + 1)
        if len(data) > MAX_BYTES:
            return None
        time.sleep(SLEEP_IMG)
        return data
    except Exception as e:  # noqa: BLE001
        # Silent `return None` made a timeout look identical to a duplicate and
        # hid the real bottleneck, so tally the reason per class.
        with LOCK:
            k = type(e).__name__
            IMG_FAIL[k] = IMG_FAIL.get(k, 0) + 1
        return None


REJECT_STATS: dict = {}


def save_image(raw: bytes, dest: Path, url: str = "") -> tuple[int, int] | None:
    try:
        with Image.open(io.BytesIO(raw)) as im:
            im = im.convert("RGB")
            w, h = im.size
            if min(w, h) < MIN_SIDE:
                # A thumbnail slipped through (iNat serves `.jpeg` renditions that
                # the extension rewrite missed). Retry once at a larger size.
                big = size_url(url)
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
                    _note_reject(f"small {w}x{h}")
                    return None
            if max(w, h) > MAX_SIDE:
                r = MAX_SIDE / max(w, h)
                im = im.resize((int(w * r), int(h * r)), Image.LANCZOS)
            im.save(dest, "JPEG", quality=QUALITY)
            return im.size
    except Exception as e:  # noqa: BLE001
        _note_reject(f"decode {type(e).__name__}")
        return None


def _note_reject(reason: str) -> None:
    """Keep a size histogram of rejects; a high reject count with no visibility
    is indistinguishable from a network failure."""
    with LOCK:
        REJECT_STATS[reason] = REJECT_STATS.get(reason, 0) + 1


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


# ---------------------------------------------------------------- iNaturalist

def inat_taxon(name: str) -> tuple[int | None, str]:
    p = urllib.parse.urlencode({"q": name, "rank": "species",
                               "per_page": 5, "locale": "en"})
    d = get_json(f"{INAT_TAXA}?{p}", f"inat/taxa")
    res = d.get("results") or []
    if not res:
        return None, ""
    tgt = name.strip().lower()
    for row in res:
        if (row.get("name") or "").strip().lower() == tgt:
            return row.get("id"), row.get("family") or ""
    return res[0].get("id"), res[0].get("family") or ""


def _inat_sweep(tid: int, family: str, name: str, quality: str | None,
                cap: int, seen_photos: set[str], out: list[dict]) -> None:
    """Page one quality tier of a taxon's observations into `out`."""
    # The iNat observations endpoint occasionally takes 100s+ on a taxon with
    # thousands of records. Page size adapts: shrink on slow responses instead
    # of burning the whole species budget on one request.
    page_size = PAGE
    slow = 0
    for page in range(1, MAX_PAGES + 1):
        if len(out) >= cap or slow >= 2:
            break
        params = {"taxon_id": tid, "photos": "true", "order_by": "votes",
                  "per_page": page_size, "page": page}
        if quality:
            params["quality_grade"] = quality
        t_req = time.time()
        d = get_json(f"{INAT_OBS}?{urllib.parse.urlencode(params)}",
                     f"inat/obs p{page}")
        dt = time.time() - t_req
        if dt > 30:
            slow += 1
            page_size = max(5, page_size // 2)
            log(f"      inat/obs slow {dt:.0f}s -> per_page={page_size}")
        res = d.get("results") or []
        if not res:
            break
        for obs in res:
            okey = str(obs.get("id") or "")
            if not okey:
                continue
            photos = obs.get("photos") or []
            if not photos:
                continue
            taken = 0
            for ph in photos:
                if taken >= PHOTOS_PER_OBS or len(out) >= cap:
                    break
                pid = str(ph.get("id") or "")
                # Dedupe at the photo level: an observation already mined for
                # its first photo can still contribute its other photos.
                if pid and pid in seen_photos:
                    continue
                lic = norm_lic(ph.get("license_code") or "")
                if lic not in ACCEPT:
                    continue
                url = size_url(ph.get("url") or "")
                if not url:
                    continue
                out.append({
                    "source": "iNaturalist",
                    "source_taxon_name": name,
                    "source_family_reported": family,
                    "source_record_id": okey,
                    "source_photo_id": pid,
                    "observation_url": f"https://www.inaturalist.org/observations/{okey}",
                    "occurrence_url": "",
                    "basis_of_record": "HUMAN_OBSERVATION",
                    "country": str(obs.get("place_guess") or "")[:60],
                    "event_date": str(obs.get("observed_on_string") or "")[:40],
                    "institution_code": "",
                    "media_type": "StillImage",
                    "license": lic,
                    "rights_holder": str(ph.get("attribution") or "").replace('"', "")[:180],
                    "image_url": url,
                    "inat_quality": str(obs.get("quality_grade") or ""),
                })
                if pid:
                    seen_photos.add(pid)
                taken += 1
        if len(res) < page_size:
            break


def inat_candidates(name: str, want: int,
                    seen: set[str], seen_photos: set[str] | None = None) -> list[dict]:
    """Resolve taxon then page observations, returning candidate dicts.

    Research-grade first, then a second sweep over all grades. Scarce species
    (e.g. Drynaria baronii: 1 research-grade vs 15 total) are otherwise left
    far short of the cap; the grade is recorded per image so downstream use can
    filter back to research-grade only.
    """
    tid, family = inat_taxon(name)
    if not tid:
        log("      inat: taxon not found")
        return []
    log(f"      inat taxon {tid} ({family})")
    out: list[dict] = []
    local_seen = set(seen_photos or ())
    cap = want + 10
    _inat_sweep(tid, family, name, "research", cap, local_seen, out)
    if len(out) < cap:
        log(f"      inat: {len(out)} research-grade; extending to needs_id")
        _inat_sweep(tid, family, name, None, cap, local_seen, out)
    return out


# --------------------------------------------------------------------- iDigBio

def _rights_to_license(rights: str) -> str:
    """Map an iDigBio rights string (e.g. 'BY-NC-SA') to a manifest code."""
    r = (rights or "").strip().upper().replace("CC ", "").replace("CC-", "")
    if r in {"CC0", "PUBLIC DOMAIN", "PUBLICDOMAIN", "PDM"}:
        return "cc0"
    if "ND" in r:            # No-Derivatives is outside the accepted set
        return ""
    if r in {"BY-NC-SA", "BY-NC-ND"}:
        return "cc-by-nc-sa" if r == "BY-NC-SA" else ""
    if r == "BY-NC":
        return "cc-by-nc"
    if r == "BY-SA":
        return "cc-by-sa"
    if r == "BY":
        return "cc-by"
    return ""


def idigbio_candidates(name: str, want: int, seen: set[str]) -> list[dict]:
    """iDigBio media search: mostly US herbarium sheets, CC-licensed."""
    out: list[dict] = []
    for page in range(MAX_PAGES):
        if len(out) >= want + 10:
            break
        q = urllib.parse.quote(json.dumps({"scientificname": name}))
        url = (f"{IDIGBIO_MEDIA}?rq={q}&limit=100&offset={page * 100}")
        d = get_json(url, f"idigbio p{page}")
        items = d.get("items") or []
        if not items:
            break
        for it in items:
            t = it.get("indexTerms") or {}
            img = t.get("accessuri") or ""
            if not img:
                continue
            rid = str(it.get("uuid") or (t.get("recordids") or [""])[0])
            if not rid or rid in seen:
                continue
            lic = _rights_to_license(t.get("rights") or "")
            if lic not in ACCEPT:
                continue
            out.append({
                "source": "iDigBio",
                "source_taxon_name": name,
                "source_family_reported": "",
                "source_record_id": rid,
                "source_photo_id": str(it.get("uuid") or ""),
                "observation_url": "",
                "occurrence_url": (f"https://portal.idigbio.org/portal/records/{rid}"[:240]),
                "basis_of_record": "PRESERVED_SPECIMEN",
                "country": "",
                "event_date": "",
                "institution_code": str(t.get("recordset") or "")[:80],
                "media_type": str(t.get("format") or "StillImage")[:60],
                "license": lic,
                "rights_holder": str(t.get("rights") or "")[:180],
                "image_url": img[:240],
                "inat_quality": "",
            })
        if len(items) < 100:
            break
    return out


# ---------------------------------------------------------------------- GBIF

def gbif_taxon_keys(name: str) -> list[int]:
    """Resolve accepted/species keys so records filed under synonyms are found.

    A plain `scientific_name` search misses occurrences catalogued under a
    synonym (e.g. Lemmaphyllum squamatum is a synonym whose accepted taxon holds
    its own media). Matching first and searching by key recovers those.
    """
    d = get_json(f"{GBIF_MATCH}?" + urllib.parse.urlencode({"name": name}),
                 "gbif/match")
    keys: list[int] = []
    for k in (d.get("usageKey"), d.get("acceptedUsageKey"), d.get("speciesKey")):
        if k and k not in keys:
            keys.append(k)
    return keys


def _gbif_occ_to_cand(occ: dict, name: str) -> dict | None:
    """Convert one GBIF occurrence into a candidate, or None if unusable."""
    media = occ.get("media") or []
    if not media:
        return None
    # A single occurrence often lists several renditions; picking media[0]
    # blindly lands on `square.jpg` thumbnails ~90% of the time, which then
    # fail MIN_SIDE. Prefer the first non-thumbnail image regardless of order.
    m = None
    for cand_m in media:
        ct = (cand_m.get("type") or "").lower()
        if ct and "image" not in ct:
            continue
        cid = cand_m.get("identifier") or ""
        if not cid:
            continue
        low = cid.lower()
        if low.endswith("square.jpg") or "square." in low:
            continue
        m = cand_m
        break
    if m is None:
        return None
    lic = norm_lic(m.get("license") or "")
    if lic not in ACCEPT:
        return None
    url = m.get("identifier") or ""
    if not url:
        return None
    return {
        "source": "GBIF",
        "source_taxon_name": str(occ.get("scientificName") or name)[:150],
        "source_family_reported": str(occ.get("family") or ""),
        "source_record_id": str(occ.get("key") or ""),
        "source_photo_id": str(m.get("id") or ""),
        "observation_url": "",
        "occurrence_url": str(occ.get("references") or occ.get("identifier") or "")[:240],
        "basis_of_record": str(occ.get("basisOfRecord") or ""),
        "country": str(occ.get("countryCode") or "")[:60],
        "event_date": str(occ.get("eventDate") or occ.get("year") or "")[:40],
        "institution_code": str(occ.get("institutionCode") or "")[:80],
        "media_type": str(m.get("type") or "StillImage")[:60],
        "license": lic,
        "rights_holder": str(m.get("rightsHolder") or "")[:180],
        "image_url": url[:240],
    }


def gbif_candidates(name: str, want: int, seen: set[str]) -> list[dict]:
    out: list[dict] = []
    seen_keys = set(seen)
    queries: list[dict] = [{"scientific_name": name}]
    for k in gbif_taxon_keys(name):
        queries.append({"taxon_key": k})
    for q in queries:
        if len(out) >= want + 10:
            break
        for page in range(MAX_PAGES):
            if len(out) >= want + 10:
                break
            params = dict(q)
            params.update({"media_type": "StillImage", "limit": 30,
                           "offset": page * 30})
            p = urllib.parse.urlencode(params)
            d = get_json(f"{GBIF_OCC}?{p}", f"gbif p{page}")
            res = d.get("results") or []
            if not res:
                break
            for occ in res:
                key = str(occ.get("key") or "")
                if not key or key in seen_keys:
                    continue
                cand = _gbif_occ_to_cand(occ, name)
                if cand is None:
                    continue
                seen_keys.add(key)
                out.append(cand)
            if len(res) < 30:
                break
    return out


def harvest(fut_map: dict, deadline: float, fetched: list) -> int:
    """Poll futures and keep every completed result.

    `as_completed(timeout=...)` raises and drops the futures that already
    finished, which loses completed downloads. Polling `done()` ourselves keeps
    everything that landed before the deadline.
    """
    dead = 0
    pending = dict(fut_map)
    while pending and time.time() < deadline:
        ready = [f for f in pending if f.done()]
        if not ready:
            time.sleep(0.15)
            continue
        for f in ready:
            cand = pending.pop(f)
            try:
                raw = f.result()
            except Exception:  # noqa: BLE001
                raw = None
            if raw is None:
                dead += 1
            else:
                fetched.append((cand, raw))
    return dead + len(pending)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-species-cap", type=int, default=30)
    ap.add_argument("--species-indices", type=str, default="")
    ap.add_argument("--max-species", type=int, default=0)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--sources", type=str, default="inat,gbif",
                    help="comma list of sources, tried in order")
    ap.add_argument("--max-pages", type=int, default=0,
                    help="override page depth per source (deep pass)")
    ap.add_argument("--budget", type=int, default=0,
                    help="override per-species wall-clock budget in seconds")
    ap.add_argument("--photos-per-obs", type=int, default=0,
                    help="iNat photos to take per observation (deep pass)")
    args = ap.parse_args()
    global MAX_PAGES, SPECIES_BUDGET, PHOTOS_PER_OBS
    if args.max_pages:
        MAX_PAGES = args.max_pages
    if args.budget:
        SPECIES_BUDGET = args.budget
    if args.photos_per_obs:
        PHOTOS_PER_OBS = args.photos_per_obs

    IMG_DIR.mkdir(parents=True, exist_ok=True)
    proxy = install_opener()
    rows: list[dict] = []
    if MANIFEST.exists():
        with MANIFEST.open(encoding="utf-8") as fh:
            rows = list(csv.DictReader(fh))
    st = json.loads(STATE.read_text(encoding="utf-8")) if STATE.exists() else {
        "per_species": {}, "next_id": 1}
    sha1_seen = {r["sha1"] for r in rows if r.get("sha1")}
    seen_rec: dict[str, set[str]] = {}
    for r in rows:
        k = r.get("source_record_id")
        if k:
            seen_rec.setdefault(r["species_id"], set()).add(k)
    # iNat photo ids are a separate namespace, so a species already mined for
    # one photo of an observation can still contribute that observation's other
    # photos without being blocked by the record-level set above.
    seen_photo: dict[str, set[str]] = {}
    for r in rows:
        if r.get("source") != "iNaturalist":
            continue
        p = r.get("source_photo_id")
        if p:
            seen_photo.setdefault(r["species_id"], set()).add(p)
    nxt = st.get("next_id", 1)
    if rows:
        mx = max(int(r["sample_id"].split("_")[1]) for r in rows)
        nxt = max(nxt, mx + 1)

    allsp = json.loads(SPECIES_JSON.read_text(encoding="utf-8"))["species"]
    if args.species_indices:
        idxs = [int(x) for x in args.species_indices.split(",") if x.strip()]
        species = [allsp[i] for i in idxs]
    else:
        species = allsp[: args.max_species] if args.max_species else allsp

    src_order = [s.strip() for s in args.sources.split(",") if s.strip()]
    log(f"dual-source | species={len(species)} cap={args.per_species_cap} "
        f"sources={src_order} existing={len(rows)}")
    log("=" * 66)

    t_all = time.time()
    for n, rec in enumerate(species, 1):
        sid = rec["species_id"]
        name = rec["latin_name"]
        already = st["per_species"].get(sid, 0)
        need = args.per_species_cap - already
        if need <= 0:
            log(f"[{n}/{len(species)}] {name} already {already}, skip")
            continue
        t0 = time.time()
        have = seen_rec.setdefault(sid, set())
        log(f"[{n}/{len(species)}] {name} ({rec['family']}) need {need}")

        cands: list[dict] = []
        for src in src_order:
            if len(cands) >= need + 8 or time.time() - t0 > SPECIES_BUDGET:
                break
            if src == "inat":
                got = inat_candidates(name, need, have,
                                      seen_photo.setdefault(sid, set()))
            elif src == "gbif":
                got = gbif_candidates(name, need, have)
            elif src == "idigbio":
                got = idigbio_candidates(name, need, have)
            else:
                got = []
            log(f"      {src}: {len(got)} candidates")
            if got:
                cands.extend(got)
        cands = cands[: need + 10]

        got = 0
        fetched: list = []
        if cands:
            deadline = time.time() + SPECIES_BUDGET
            pool = ThreadPoolExecutor(max_workers=args.workers)
            try:
                fm = {pool.submit(fetch_bytes, c["image_url"]): c for c in cands}
                dead = harvest(fm, deadline, fetched)
            finally:
                # cancel_futures needs 3.9+; the 3.7 interpreter lacks it.
                try:
                    pool.shutdown(wait=False, cancel_futures=True)
                except TypeError:
                    pool.shutdown(wait=False)
        else:
            dead = 0
        n_dup = 0
        n_reject = 0

        for cand, raw in fetched:
            if got >= need:
                break
            digest = hashlib.sha1(raw).hexdigest()
            if digest in sha1_seen:
                n_dup += 1
                continue
            with LOCK:
                sample = f"EXP_{nxt:05d}"
            dest = IMG_DIR / f"{sample}.jpg"
            if dest.exists():
                nxt += 1
                continue
            dims = save_image(raw, dest, cand.get("image_url", ""))
            if dims is None:
                dest.unlink(missing_ok=True)
                n_reject += 1
                continue
            row = {"sample_id": sample,
                   "dataset": "finetune_expansion_batch1",
                   "species_id": sid, "family_project": rec["family"],
                   "species_latin_project": name,
                   "image_path": str(dest.relative_to(ROOT)),
                   "width": dims[0], "height": dims[1], "sha1": digest,
                   "download_status": "ok"}
            row.update(cand)
            with LOCK:
                rows.append(row)
                sha1_seen.add(digest)
                nxt += 1
                if cand.get("source_record_id"):
                    have.add(cand["source_record_id"])
                flush_manifest(rows)
            got += 1

        if got:
            st["per_species"][sid] = already + got
        st["next_id"] = nxt
        save_state(st)
        log(f"      => +{got} (total {len(rows)}) "
            f"fetched={len(fetched)}/{len(cands)} nodata={dead} "
            f"dup={n_dup} reject={n_reject} | {time.time()-t0:.0f}s")

    log("=" * 66)
    if REJECT_STATS:
        top = sorted(REJECT_STATS.items(), key=lambda kv: -kv[1])[:8]
        log("reject breakdown: " + ", ".join(f"{k}x{v}" for k, v in top))
    if IMG_FAIL:
        log("image fetch errors: " + ", ".join(
            f"{k}x{v}" for k, v in sorted(IMG_FAIL.items(), key=lambda kv: -kv[1])))
    sp_covered = len({r["species_latin_project"] for r in rows})
    fams = len({r["family_project"] for r in rows})
    log(f"DONE images={len(rows)} species={sp_covered} families={fams} "
        f"elapsed={time.time()-t_all:.0f}s")
    zero = [r["latin_name"] for r in species
            if r["latin_name"] not in {x["species_latin_project"] for x in rows}]
    if zero:
        log(f"ZERO in this run: {zero}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
