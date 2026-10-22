#!/usr/bin/env python3
"""Self-resuming bulk driver.

This environment kills long-running processes, so instead of one long job this
driver runs the downloader in short bursts, each burst processing a slice of
species and checkpointing state after every image. On restart it reads the
resumable state and continues with the species that are still short of target.

Usage:
    python3 run_bulk_resilient.py --burst 8 --rounds 40
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
PY = sys.executable
SPECIES = ROOT / "data" / "species_data.json"
STATE = HERE / "download_state.json"
MANIFEST = HERE / "manifest.csv"
LOG = HERE / "bulk_download.log"


def pending_species(cap: int, after: int = -1) -> tuple[list[int], int]:
    """Indices of species still needing images, searched from `after` onward.

    Returns (pending, cursor). The cursor lets the caller resume where it left
    off instead of rescanning the head of the list every round: species with
    very few available images stay near the top and would otherwise be
    re-examined on every single burst, wasting most of the run on no-ops.
    """
    payload = json.loads(SPECIES.read_text(encoding="utf-8"))
    species = payload["species"]
    st = json.loads(STATE.read_text(encoding="utf-8")) if STATE.exists() else {}
    per = st.get("per_species", {})
    n = len(species)
    start = 0 if after < 0 else (after + 1) % n
    out: list[int] = []
    idx = start
    for _ in range(n):
        rec = species[idx]
        if per.get(rec["species_id"], 0) < cap:
            out.append(idx)
        idx = (idx + 1) % n
    return out, start


def manifest_rows() -> int:
    if not MANIFEST.exists():
        return 0
    with MANIFEST.open(encoding="utf-8") as fh:
        return sum(1 for _ in csv_rows(fh))


def csv_rows(fh):
    import csv as _csv

    return _csv.DictReader(fh)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cap", type=int, default=30, help="per-species target")
    ap.add_argument("--burst", type=int, default=8, help="species per burst")
    ap.add_argument("--rounds", type=int, default=40, help="max bursts")
    ap.add_argument("--pause", type=float, default=2.0, help="seconds between bursts")
    args = ap.parse_args()

    t0 = time.time()
    logf = LOG.open("a", encoding="utf-8")
    logf.write(f"\n===== run start {time.strftime('%Y-%m-%d %H:%M:%S')} "
               f"cap={args.cap} burst={args.burst} =====\n")
    logf.flush()

    cursor = -1
    for rnd in range(args.rounds):
        todo, cursor = pending_species(args.cap, cursor)
        if not todo:
            print("ALL SPECIES REACHED TARGET", flush=True)
            logf.write("ALL SPECIES REACHED TARGET\n")
            break
        start = todo[0]
        count = min(args.burst, len(todo))
        n_rows = manifest_rows()
        # Advance the cursor past this burst so the next round resumes after it.
        cursor = start + count - 1
        print(f"[round {rnd+1}/{args.rounds}] {len(todo)} species pending, "
              f"burst {start}:{start+count}, manifest={n_rows}", flush=True)
        logf.write(f"[round {rnd+1}] start={start} count={count} manifest={n_rows}\n")
        logf.flush()

        cmd = [
            PY, "-u", str(HERE / "download_gbif_bulk.py"),
            "--per-species-cap", str(args.cap),
            "--global-cap", "4000",
            "--start-species", str(start),
            "--max-species", str(count),
        ]
        # Stream child output line by line so progress is visible immediately.
        try:
            proc = subprocess.Popen(
                cmd, cwd=str(HERE), stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT, text=True, bufsize=1,
            )
            assert proc.stdout is not None
            for line in proc.stdout:
                line = line.rstrip()
                if line:
                    print("   ", line, flush=True)
                    logf.write("    " + line + "\n")
                    logf.flush()
            proc.wait(timeout=60)
        except subprocess.TimeoutExpired:
            print("    [burst timed out]", flush=True)

        now = manifest_rows()
        print(f"    -> manifest {n_rows} => {now} (+{now - n_rows})", flush=True)
        if now == n_rows:
            # No progress this burst: back off so the server can recover.
            time.sleep(20)
        else:
            time.sleep(args.pause)

    logf.write(f"===== run end {time.strftime('%Y-%m-%d %H:%M:%S')} "
               f"elapsed={time.time()-t0:.0f}s =====\n")
    logf.close()
    print(f"done. elapsed={time.time()-t0:.0f}s manifest={manifest_rows()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())