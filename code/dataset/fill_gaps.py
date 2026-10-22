#!/usr/bin/env python3
"""Drive download_dual.py until every reachable species reaches a target count.

Why not dual_daemon.py: that supervisor keeps `per_species` in memory and
writes it back after each round, clobbering the fresher state file that
download_dual.py just wrote. It also never increments `real_fail` (the guard
is a no-op compare), so exhausted species are retried forever. Both bugs let
over-represented species inflate past the cap.

This driver instead treats manifest.csv as the single source of truth:
  * recomputes per-species counts from the manifest every round;
  * passes only genuinely-under-target species to download_dual.py;
  * lets download_dual.py own download_state.json (never rewrites it);
  * retires a species after N zero-gain rounds, but never counts a strike when
    the whole chunk produced nothing (that is a network outage, not exhaustion).

Usage:
    python3 fill_gaps.py --target 30 --chunk 8 --workers 8
"""

from __future__ import annotations

import argparse
import collections
import csv
import json
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
SPECIES_JSON = ROOT / "data" / "species_data.json"
MANIFEST = HERE / "manifest.csv"
STRIKES = HERE / "fill_gaps_strikes.json"
LOG = HERE / "fill_gaps.log"

# Species with no image anywhere on iNat/GBIF; the user agreed to skip them.
DEFAULT_SKIP = {"FERNS_007", "FERNS_015", "FERNS_076"}


def api_healthy() -> bool:
    """True when the metadata APIs answer, so a zero-gain round is real."""
    checks = ("https://api.gbif.org/v1/occurrence/search?limit=1",
              "https://api.inaturalist.org/v1/observations?per_page=1")
    for url in checks:
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "fern-fill/1.0"})
            with urllib.request.urlopen(req, timeout=15) as r:
                r.read(64)
        except Exception:  # noqa: BLE001
            return False
    return True


def log(msg: str) -> None:
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    try:
        with LOG.open("a", encoding="utf-8") as fh:
            fh.write(line + "\n")
    except Exception:  # noqa: BLE001
        pass


def counts() -> collections.Counter:
    if not MANIFEST.exists():
        return collections.Counter()
    with MANIFEST.open(encoding="utf-8") as fh:
        return collections.Counter(r["species_id"] for r in csv.DictReader(fh))


def load_json(path: Path, default):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return default


def save_json(path: Path, obj) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2),
                   encoding="utf-8")
    tmp.replace(path)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", type=int, default=30)
    ap.add_argument("--chunk", type=int, default=8)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--sources", type=str, default="inat,gbif,idigbio")
    ap.add_argument("--max-strikes", type=int, default=4)
    ap.add_argument("--max-rounds", type=int, default=200)
    ap.add_argument("--pause", type=float, default=5.0)
    ap.add_argument("--skip", type=str, default="")
    ap.add_argument("--max-pages", type=int, default=12)
    ap.add_argument("--budget", type=int, default=300)
    ap.add_argument("--photos-per-obs", type=int, default=3)
    args = ap.parse_args()

    species = json.loads(SPECIES_JSON.read_text(encoding="utf-8"))["species"]
    idx_by_id = {s["species_id"]: i for i, s in enumerate(species)}
    skip = set(DEFAULT_SKIP)
    skip |= {x.strip() for x in args.skip.split(",") if x.strip()}
    strikes = load_json(STRIKES, {})

    log("=" * 68)
    log(f"fill_gaps target={args.target} chunk={args.chunk} "
        f"workers={args.workers} skip={sorted(skip)}")
    c0 = counts()
    log(f"start: {sum(c0.values())} images, {len(c0)} species covered")
    log("=" * 68)

    t0 = time.time()
    for rnd in range(1, args.max_rounds + 1):
        cur = counts()
        todo = []
        retired = []
        for s in species:
            sid = s["species_id"]
            if sid in skip:
                continue
            n = cur.get(sid, 0)
            if n >= args.target:
                continue
            if strikes.get(sid, 0) >= args.max_strikes:
                retired.append(s["latin_name"])
                continue
            # Ordering: fewest strikes first, then species nearest the target.
            # Data-rich species reach the cap in seconds, so banking them first
            # is far faster than starting with near-exhausted taxa.
            todo.append((strikes.get(sid, 0), -n, idx_by_id[sid], sid,
                         s["latin_name"]))
        todo.sort()

        if not todo:
            log(f"ALL REACHABLE SPECIES AT TARGET (retired: {len(retired)})")
            break

        picked = todo[: args.chunk]
        idxs = [p[2] for p in picked]
        names = ", ".join(p[4] for p in picked)
        log(f"round {rnd}: {len(todo)} pending | picking {len(picked)}: {names}")

        before = sum(cur.values())
        cmd = [sys.executable, "-u", str(HERE / "download_dual.py"),
               "--per-species-cap", str(args.target),
               "--species-indices", ",".join(str(i) for i in idxs),
               "--workers", str(args.workers),
               "--sources", args.sources,
               "--max-pages", str(args.max_pages),
               "--budget", str(args.budget),
               "--photos-per-obs", str(args.photos_per_obs)]
        try:
            proc = subprocess.Popen(cmd, cwd=str(HERE), stdout=subprocess.PIPE,
                                    stderr=subprocess.STDOUT, text=True,
                                    bufsize=1)
            assert proc.stdout is not None
            for ln in proc.stdout:          # stream so progress is visible live
                ln = ln.rstrip()
                if ln:
                    log("  " + ln)
            proc.wait(timeout=120)
        except subprocess.TimeoutExpired:
            proc.kill()
            log("  runner timeout; continuing")

        after = counts()
        gained = sum(after.values()) - before
        healthy = api_healthy()
        if gained == 0 and not healthy:
            # Genuine API outage: do not punish individual species.
            log("  no gain and metadata APIs unreachable; backing off")
            time.sleep(max(30.0, args.pause * 6))
            continue
        if gained == 0:
            log("  no gain, but APIs are healthy: counting strikes")
        for *_rest, sid, name in picked:
            if after.get(sid, 0) > cur.get(sid, 0):
                strikes[sid] = 0
            else:
                strikes[sid] = strikes.get(sid, 0) + 1
                log(f"  {name}: no gain (strike {strikes[sid]}/{args.max_strikes})")
        save_json(STRIKES, strikes)
        log(f"  round gain +{gained} (total {sum(after.values())}); "
            f"elapsed {time.time()-t0:.0f}s")
        time.sleep(args.pause)

    cur = counts()
    log("=" * 68)
    log(f"fill_gaps finished: {sum(cur.values())} images, "
        f"{len(cur)} species; elapsed {time.time()-t0:.0f}s")
    below = [(s['latin_name'], cur.get(s['species_id'], 0)) for s in species
             if s['species_id'] not in skip and cur.get(s['species_id'], 0) < args.target]
    log(f"still below target: {len(below)}")
    for n, c in sorted(below, key=lambda x: x[1]):
        log(f"   {c:>3}  {n}")
    log("=" * 68)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
