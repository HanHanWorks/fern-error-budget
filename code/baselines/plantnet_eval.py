#!/usr/bin/env python3
"""Evaluate Pl@ntNet as a specialist baseline on the fern test split.

Runs on the LOCAL machine: the GPU server cannot reach my-api.plantnet.org
(egress is filtered there), but the image copies are available locally under
`../finetune_expansion/images/`.

Design notes:
  * One request identifies one plant individual, so one request per image.
  * Results are appended to JSONL and re-read on restart, so a run can be
    stopped at the 500/day free-plan quota and resumed the next day.
  * The quota returned by the API (`remainingIdentificationRequests`) is
    tracked and the run stops before it is exhausted.

Usage:
    python plantnet_eval.py --project all --limit 60
    python plantnet_eval.py --project k-china
"""
from __future__ import annotations

import argparse
import csv
import json
import mimetypes
import os
import time
import urllib.error
import urllib.request
import uuid
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
REPO = ROOT
IMAGES = ROOT / "data/open_images"
SPLITS = ROOT / "data/splits"
TESTSETS = ROOT / "data/testsets"
SPECIES_JSON = ROOT / "data/species_data.json"
KEYFILE = ROOT / ".plantnet_key"
OUTDIR = ROOT / "plantnet"
API = "https://my-api.plantnet.org/v2/identify/"


def api_key() -> str:
    key = os.environ.get("PLANTNET_KEY", "").strip()
    if not key and KEYFILE.exists():
        key = KEYFILE.read_text().strip()
    if not key:
        raise SystemExit("no Pl@ntNet key (set PLANTNET_KEY or write .plantnet_key)")
    return key


def norm(name: str) -> str:
    return " ".join((name or "").lower().replace("×", "x").split())


def binomial(name: str) -> str:
    parts = norm(name).split()
    return " ".join(parts[:2]) if len(parts) >= 2 else norm(name)


def post_image(key: str, project: str, path: Path, timeout: int = 60) -> dict:
    return post_bytes(key, project, path.read_bytes(), path.name, timeout)


def post_bytes(key: str, project: str, data: bytes, fname: str,
               timeout: int = 60) -> dict:
    boundary = uuid.uuid4().hex
    ctype = mimetypes.guess_type(fname)[0] or "image/jpeg"
    body = b"".join([
        f"--{boundary}\r\n".encode(),
        b'Content-Disposition: form-data; name="images"; filename="'
        + fname.encode() + b'"\r\n',
        f"Content-Type: {ctype}\r\n\r\n".encode(),
        data, b"\r\n",
        f"--{boundary}--\r\n".encode(),
    ])
    url = f"{API}{project}?api-key={key}&lang=en"
    req = urllib.request.Request(
        url, data=body, method="POST",
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}",
                 "User-Agent": "fern-benchmark/1.0 (academic research)"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8"))


def downscale(path: Path, max_side: int) -> bytes:
    """Re-encode at a smaller long side.

    A few source images are rejected by the API with HTTP 404 even though an
    identically-sized, identically-encoded sibling is accepted -- the trigger
    is content, not format or dimensions. Re-encoding at a smaller size is a
    reproducible workaround, and the record is flagged so the deviation can be
    disclosed rather than silently mixed in.
    """
    import io
    with Image.open(path) as im:
        im = im.convert("RGB")
        w, h = im.size
        r = max_side / max(w, h)
        if r < 1:
            im = im.resize((max(1, int(w * r)), max(1, int(h * r))),
                           Image.LANCZOS)
        buf = io.BytesIO()
        im.save(buf, "JPEG", quality=92)
        return buf.getvalue()


def load_rows(split: str) -> list[dict]:
    p = SPLITS / f"{split}.csv"
    with p.open(encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def stratified_sample(rows: list[dict], n: int, seed: int) -> list[dict]:
    """Sample spread evenly over species, then keep the original order."""
    if n >= len(rows):
        return rows
    import random
    rng = random.Random(seed)
    by_sp: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_sp[r["species_latin_project"]].append(r)
    species = sorted(by_sp)
    rng.shuffle(species)
    picked: list[dict] = []
    i = 0
    while len(picked) < n and any(by_sp[s] for s in species):
        s = species[i % len(species)]
        if by_sp[s]:
            picked.append(by_sp[s].pop(0))
        i += 1
    keep = {r["sample_id"] for r in picked}
    return [r for r in rows if r["sample_id"] in keep]


def family_map() -> dict[str, str]:
    """Chinese project family -> Latin family, for comparing against Pl@ntNet."""
    out = {}
    if SPECIES_JSON.exists():
        for s in json.loads(SPECIES_JSON.read_text(encoding="utf-8"))["species"]:
            if s.get("family_latin"):
                out[s["family"]] = s["family_latin"]
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--project", default="all",
                    help="referential: all | k-china | k-world-flora | ...")
    ap.add_argument("--split", default="test")
    ap.add_argument("--split-file", default="",
                    help="explicit CSV of rows to evaluate (overrides --split)")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--sample", type=int, default=0,
                    help="stratified-by-species subsample size")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--min-remaining", type=int, default=5,
                    help="stop when the API reports fewer than this left")
    ap.add_argument("--max-attempts", type=int, default=3,
                    help="stop retrying an image after this many DETERMINISTIC "
                         "client errors (429/5xx still retry forever)")
    ap.add_argument("--fallback-max", type=int, default=1024,
                    help="on HTTP 404, retry once with the long side capped at "
                         "this many pixels and flag the record as downscaled")
    ap.add_argument("--sleep", type=float, default=0.4)
    args = ap.parse_args()

    key = api_key()
    OUTDIR.mkdir(parents=True, exist_ok=True)
    out = OUTDIR / f"raw_{args.project}_{args.split}.jsonl"
    done: set[str] = set()
    attempts: Counter = Counter()
    exhausted: set[str] = set()
    # A deterministic 4xx means the API will never accept this image. Without
    # giving up on it, `todo` could never reach zero and an unattended daily
    # job would run forever. Transient codes (429, 5xx) are always retried.
    PERMANENT = {"HTTP400", "HTTP404", "HTTP415", "HTTP422"}
    last_remaining = None
    if out.exists():
        for line in out.read_text(encoding="utf-8").splitlines():
            if line.strip():
                try:
                    rec = json.loads(line)
                    sid = rec["sample_id"]
                    attempts[sid] += 1
                    # Only successful records count as done; a failed request
                    # must be retried, otherwise a transient API error would
                    # permanently drop that image from the evaluation.
                    if rec.get("ok"):
                        done.add(sid)
                        exhausted.discard(sid)
                    elif (rec.get("error") in PERMANENT
                          and attempts[sid] >= args.max_attempts):
                        exhausted.add(sid)
                    if rec.get("remaining") is not None:
                        last_remaining = rec["remaining"]
                except Exception:  # noqa: BLE001
                    pass
    # A sample that eventually succeeded must never be reported as given up.
    exhausted -= done

    if args.split_file:
        with open(args.split_file, encoding="utf-8") as fh:
            rows = list(csv.DictReader(fh))
    else:
        rows = load_rows(args.split)
    if args.sample:
        rows = stratified_sample(rows, args.sample, args.seed)
    if args.limit:
        rows = rows[: args.limit]
    todo = [r for r in rows
            if r["sample_id"] not in done and r["sample_id"] not in exhausted]
    print(f"project={args.project} split={args.split} "
          f"total={len(rows)} done={len(done)} todo={len(todo)} "
          f"exhausted={len(exhausted)}")
    if exhausted:
        print(f"  gave up on {len(exhausted)} image(s) the API rejects "
              f"deterministically: {sorted(exhausted)}")

    fam_lat = family_map()
    # Do NOT trust the quota recorded by the previous session: the free plan
    # resets on its own day boundary, so a stale "2 left" would make an
    # unattended daily job exit forever without ever resuming. Probe live
    # instead -- an exhausted plan answers HTTP 429, which costs no quota.
    remaining = None
    if last_remaining is not None:
        print(f"(last session ended with {last_remaining} left; "
              f"probing live quota)")
    t0 = time.time()
    with out.open("a", encoding="utf-8") as fh:
        for i, r in enumerate(todo, 1):
            if remaining is not None and remaining <= args.min_remaining:
                print(f"quota guard: {remaining} left, stopping. "
                      f"Re-run tomorrow to continue.")
                break
            img = IMAGES / Path(r["image_path"]).name
            rec = {"sample_id": r["sample_id"],
                   "true_species": r["species_latin_project"],
                   "true_family_cn": r["family_project"],
                   "true_family_latin": fam_lat.get(r["family_project"], ""),
                   "image_domain": r.get("image_domain", ""),
                   "source": r.get("source", ""),
                   "project": args.project}
            try:
                d = post_image(key, args.project, img)
                res = d.get("results") or []
                rec["ok"] = True
                rec["best_match"] = d.get("bestMatch")
                rec["top"] = [
                    {"sci": (x.get("species") or {}).get("scientificNameWithoutAuthor"),
                     "family": ((x.get("species") or {}).get("family") or
                                {}).get("scientificNameWithoutAuthor"),
                     "score": x.get("score")}
                    for x in res
                ]
                remaining = d.get("remainingIdentificationRequests")
                rec["remaining"] = remaining
            except urllib.error.HTTPError as e:
                if e.code in (429, 403):
                    rec["ok"] = False
                    rec["error"] = f"HTTP{e.code}"
                    fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    print(f"  rate/quota error {e.code}; stopping.")
                    break
                if e.code == 404 and args.fallback_max:
                    # Documented fallback for content-specific rejections.
                    try:
                        d = post_bytes(key, args.project,
                                       downscale(img, args.fallback_max),
                                       img.name)
                        res = d.get("results") or []
                        rec["ok"] = True
                        rec["downscaled"] = args.fallback_max
                        rec["first_error"] = "HTTP404"
                        rec["best_match"] = d.get("bestMatch")
                        rec["top"] = [
                            {"sci": (x.get("species") or {}).get(
                                "scientificNameWithoutAuthor"),
                             "family": ((x.get("species") or {}).get("family")
                                        or {}).get("scientificNameWithoutAuthor"),
                             "score": x.get("score")} for x in res]
                        remaining = d.get("remainingIdentificationRequests")
                        rec["remaining"] = remaining
                        print(f"  {rec['sample_id']}: recovered via "
                              f"{args.fallback_max}px downscale")
                    except urllib.error.HTTPError as e2:
                        rec["ok"] = False
                        rec["error"] = f"HTTP{e2.code}(after-downscale)"
                    except Exception as e2:  # noqa: BLE001
                        rec["ok"] = False
                        rec["error"] = type(e2).__name__
                else:
                    rec["ok"] = False
                    rec["error"] = f"HTTP{e.code}"
            except Exception as e:  # noqa: BLE001
                rec["ok"] = False
                rec["error"] = type(e).__name__
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            fh.flush()
            if i % 25 == 0 or i == len(todo):
                rate = i / max(time.time() - t0, 1e-9)
                print(f"  {i}/{len(todo)} remaining={remaining} "
                      f"({rate:.1f}/s)", flush=True)
            time.sleep(args.sleep)
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
