#!/usr/bin/env python3
"""Evaluate a general-purpose VLM on the SAME fern test split as the specialist
models, so the comparison is apples-to-apples.

Earlier VLMA numbers in the manuscript came from a different, smaller
155-image benchmark; this script produces numbers on the 902-image split that
BioCLIP 2 / DINOv2 / Pl@ntNet were scored on.

Design notes
  * `enable_thinking=False`. With thinking on, one image took 91.7 s and burned
    ~7.3k reasoning tokens; disabling it gives ~1-7 s and identical answers on
    spot checks. Thinking is not part of how a user would run this at scale.
  * Full resolution is sent (images are already <=2048 px), i.e. the VLM gets
    more information than Pl@ntNet's typical upload -- deliberately generous,
    so the specialist-vs-general comparison is conservative.
  * Three tasks: open-ended naming (comparable to Pl@ntNet) and closed-set
    selection from an explicit candidate list (comparable to a classifier).

    python vlm_eval.py --task open
    python vlm_eval.py --task family30
    python vlm_eval.py --task species152
"""
from __future__ import annotations

import argparse
import base64
import csv
import json
import re
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
REPO = ROOT
IMAGES = ROOT / "data/open_images"
SPLITS = ROOT / "data/splits"
SPECIES_JSON = ROOT / "data/species_data.json"
KEYFILE = ROOT / ".dashscope_key"
OUTDIR = ROOT / "data/external_baselines/vlm"
URL = "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions"

PROMPT_OPEN = (
    "You are a botanist identifying ferns (Pteridophyta). Look at the plant in "
    "this image and identify the most likely species.\n"
    "Reply with ONLY a JSON object, no markdown:\n"
    '{"top3": [{"scientific_name": "Genus species", "family": "...", '
    '"confidence": 0.00}, {"scientific_name": "...", "family": "...", '
    '"confidence": 0.00}, {"scientific_name": "...", "family": "...", '
    '"confidence": 0.00}]}\n'
    "The first entry must be your single best guess. Use standard botanical "
    "scientific names (binomials). Confidence is 0-1."
)

# Probe for label leakage: herbarium sheets often carry the species name as
# printed/handwritten text, and a VLM can simply read it. Transcribing the text
# lets us quantify how often the answer was available as text rather than
# inferred from morphology.
PROMPT_OCR = (
    "Transcribe verbatim ALL readable text visible in this image "
    "(labels, handwriting, stamps, watermarks, signage). "
    "Do not describe the plant. Do not translate. If there is no readable "
    "text, reply exactly: NONE"
)


def load_rows(split: str) -> list[dict]:
    with (SPLITS / f"{split}.csv").open(encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def taxonomy() -> tuple[list[str], list[str], dict[str, str]]:
    sp = json.loads(SPECIES_JSON.read_text(encoding="utf-8"))["species"]
    species = sorted({s["latin_name"] for s in sp})
    fam_lat = {s["family"]: s.get("family_latin", s["family"]) for s in sp}
    families = sorted(set(fam_lat.values()))
    return species, families, fam_lat


def build_prompt(task: str, species: list[str], families: list[str]) -> str:
    if task == "open":
        return PROMPT_OPEN
    if task == "ocr":
        return PROMPT_OCR
    if task == "family30":
        return (
            "You are a botanist identifying ferns. Look at the image and choose "
            "the most likely family from this candidate list ONLY:\n"
            + ", ".join(families)
            + "\nReply with ONLY a JSON object, no markdown:\n"
            '{"top3": [{"family": "...", "confidence": 0.00}, '
            '{"family": "...", "confidence": 0.00}, '
            '{"family": "...", "confidence": 0.00}]}\n'
            "Every family must be copied exactly from the candidate list."
        )
    if task == "species152":
        return (
            "You are a botanist identifying ferns. Look at the image and choose "
            "the most likely species from this candidate list ONLY:\n"
            + ", ".join(species)
            + "\nReply with ONLY a JSON object, no markdown:\n"
            '{"top3": [{"scientific_name": "...", "confidence": 0.00}, '
            '{"scientific_name": "...", "confidence": 0.00}, '
            '{"scientific_name": "...", "confidence": 0.00}]}\n'
            "Every name must be copied exactly from the candidate list."
        )
    raise ValueError(task)


def call(key: str, model: str, prompt: str, data: bytes,
         timeout: int = 180) -> tuple[str, float, dict]:
    b64 = base64.b64encode(data).decode()
    payload = {
        "model": model, "enable_thinking": False, "temperature": 0.0,
        "max_tokens": 400,
        "messages": [{"role": "user", "content": [
            {"type": "text", "text": prompt},
            {"type": "image_url",
             "image_url": {"url": f"data:image/jpeg;base64,{b64}"}}]}],
    }
    req = urllib.request.Request(
        URL, data=json.dumps(payload).encode(), method="POST",
        headers={"Authorization": f"Bearer {key}",
                 "Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        d = json.loads(r.read().decode())
    return (d["choices"][0]["message"]["content"], time.time() - t0,
            d.get("usage") or {})


_BINOM = re.compile(r"\b([A-Z][a-z]{3,})\s+([a-z][a-z\-]{2,})\b")


def parse(text: str) -> dict:
    """Extract top-3 names from JSON, falling back to binomial regex."""
    t = (text or "").replace("*", "").replace("```json", "").replace("```", "")
    out = {"top3": [], "raw": (text or "")[:4000]}
    m = re.search(r"\{.*\}", t, re.S)
    if m:
        try:
            d = json.loads(m.group(0))
            for x in (d.get("top3") or [])[:3]:
                if not isinstance(x, dict):
                    continue
                nm = x.get("scientific_name") or x.get("name") or ""
                fam = x.get("family") or ""
                try:
                    conf = float(x.get("confidence"))
                except (TypeError, ValueError):
                    conf = None
                out["top3"].append({"name": nm.strip(), "family": fam.strip(),
                                    "confidence": conf})
            return out
        except Exception:  # noqa: BLE001
            pass
    for g, s in _BINOM.findall(t)[:3]:
        out["top3"].append({"name": f"{g} {s}", "family": None,
                            "confidence": None})
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=["open", "ocr", "family30", "species152"],
                    default="open")
    ap.add_argument("--split", default="test")
    ap.add_argument("--model", default="qwen3.7-flash-2026-07-15")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--sample", type=int, default=0)
    ap.add_argument("--where", type=str, default="",
                    help="keep only rows matching COL=VAL (e.g. "
                         "image_domain=specimen)")
    ap.add_argument("--token-budget", type=int, default=0,
                    help="stop submitting once cumulative total_tokens exceeds "
                         "this (0 = unlimited). Checked before each request.")
    ap.add_argument("--stratify", action="store_true",
                    help="reorder rows round-robin across species so that any "
                         "prefix is a species-stratified sample")
    args = ap.parse_args()

    key = KEYFILE.read_text().strip()
    OUTDIR.mkdir(parents=True, exist_ok=True)
    out = OUTDIR / f"{args.task}_{args.model}_{args.split}.jsonl"
    done: set[str] = set()
    if out.exists():
        for line in out.read_text(encoding="utf-8").splitlines():
            if line.strip():
                try:
                    r = json.loads(line)
                    if r.get("ok"):
                        done.add(r["sample_id"])
                except Exception:  # noqa: BLE001
                    pass
    rows = load_rows(args.split)
    if args.where:
        col, _, val = args.where.partition("=")
        rows = [r for r in rows if (r.get(col) or "") == val]
        print(f"filter {args.where} -> {len(rows)} rows")
    if args.sample:
        import random
        rng = random.Random(7)
        by_sp: dict[str, list[dict]] = {}
        for r in rows:
            by_sp.setdefault(r["species_latin_project"], []).append(r)
        keys = sorted(by_sp); rng.shuffle(keys)
        picked = []
        i = 0
        while len(picked) < args.sample and any(by_sp[k] for k in keys):
            k = keys[i % len(keys)]
            if by_sp[k]:
                picked.append(by_sp[k].pop(0))
            i += 1
        keep = {r["sample_id"] for r in picked}
        rows = [r for r in rows if r["sample_id"] in keep]
    if args.limit:
        rows = rows[: args.limit]
    if args.stratify:
        # Round-robin across species buckets: any prefix then covers every
        # species proportionally, which keeps a budget-truncated run
        # representative instead of biased toward the file order.
        import random
        rng = random.Random(7)
        buckets: dict[str, list[dict]] = {}
        for r in rows:
            buckets.setdefault(r["species_latin_project"], []).append(r)
        for v in buckets.values():
            rng.shuffle(v)
        keys = sorted(buckets, key=lambda k: (-len(buckets[k]), k))
        ordered, alive = [], True
        while alive:
            alive = False
            for k in keys:
                if buckets[k]:
                    ordered.append(buckets[k].pop())
                    alive = True
        rows = ordered
    todo = [r for r in rows if r["sample_id"] not in done]
    species, families, fam_lat = taxonomy()
    prompt = build_prompt(args.task, species, families)
    print(f"task={args.task} model={args.model} total={len(rows)} "
          f"done={len(done)} todo={len(todo)} workers={args.workers}")

    lock = threading.Lock()
    t0 = time.time()
    n_done = 0
    spent = 0
    budget_hit = False
    with out.open("a", encoding="utf-8") as fh:
        def work(r):
            img = IMAGES / Path(r["image_path"]).name
            rec = {"sample_id": r["sample_id"],
                   "true_species": r["species_latin_project"],
                   "true_family_cn": r["family_project"],
                   "true_family_latin": fam_lat.get(r["family_project"], ""),
                   "image_domain": r.get("image_domain", ""),
                   "source": r.get("source", ""), "task": args.task,
                   "model": args.model}
            for attempt in range(3):
                try:
                    txt, dt, usage = call(key, args.model, prompt,
                                          img.read_bytes())
                    rec.update(parse(txt))
                    rec["ok"] = True
                    rec["latency"] = round(dt, 2)
                    rec["usage"] = usage
                    return rec
                except urllib.error.HTTPError as e:
                    if e.code in (429, 500, 502, 503):
                        time.sleep(2 * (attempt + 1))
                        continue
                    rec["ok"] = False
                    rec["error"] = f"HTTP{e.code}"
                    return rec
                except Exception as e:  # noqa: BLE001
                    if attempt < 2:
                        time.sleep(2)
                        continue
                    rec["ok"] = False
                    rec["error"] = type(e).__name__
                    return rec
            rec["ok"] = False
            rec["error"] = "retries-exhausted"
            return rec

        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            # Submit in waves so the token budget can stop the run early
            # without leaving a tail of in-flight requests.
            wave = max(args.workers * 4, 8)
            for start in range(0, len(todo), wave):
                if args.token_budget and spent >= args.token_budget:
                    budget_hit = True
                    break
                chunk = todo[start:start + wave]
                futs = {pool.submit(work, r): r for r in chunk}
                for fut in as_completed(futs):
                    rec = fut.result()
                    with lock:
                        fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
                        fh.flush()
                        n_done += 1
                        spent += (rec.get("usage") or {}).get("total_tokens", 0)
                        if n_done % 25 == 0 or n_done == len(todo):
                            el = time.time() - t0
                            print(f"  {n_done}/{len(todo)}  "
                                  f"{n_done/max(el,1e-9):.2f}/s  "
                                  f"tokens={spent:,}", flush=True)
    if budget_hit:
        print(f"token budget reached: {spent:,} tokens used")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
