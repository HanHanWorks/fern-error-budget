#!/usr/bin/env python3
"""Measure how often the answer is readable as TEXT in the image.

Independent of the VLM: runs the local `tesseract` binary and checks whether
the ground-truth name is recoverable from the OCR output. This tests the
"the model read the label instead of identifying the plant" hypothesis without
spending API quota (the VLM's own transcription probe needs a paid key).

Outputs vlm/text_ocr.jsonl plus a per-domain leakage table cross-tabulated
against the VLM's identification results.

    python text_leakage.py
"""
from __future__ import annotations

import argparse
import csv
import difflib
import json
import re
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
REPO = ROOT
IMAGES = ROOT / "data/open_images"
SPLITS = ROOT / "data/splits"
OUT = ROOT / "data/external_baselines/vlm/text_ocr.jsonl"
TESS = "/opt/homebrew/bin/tesseract"


def norm(s: str) -> str:
    s = (s or "").lower().replace("×", "x")
    s = re.sub(r"[^a-z0-9\s\-]", " ", s)
    return " ".join(s.split())


def binom(s: str) -> tuple[str, str]:
    p = norm(s).split()
    return (p[0], p[1]) if len(p) >= 2 else (norm(s), "")


def readable_in(true_sp: str, text: str, ratio: float = 0.80) -> bool:
    """True if the binomial, or a close OCR variant of the epithet, is present."""
    g, e = binom(true_sp)
    if not e:
        return False
    t = norm(text)
    if f"{g} {e}" in t:
        return True
    if len(e) >= 6:
        for tok in t.split():
            if abs(len(tok) - len(e)) > 4:
                continue
            if difflib.SequenceMatcher(None, e, tok).ratio() >= ratio:
                return True
    return False


def ocr(path: Path) -> str:
    try:
        out = subprocess.run([TESS, str(path), "-"],
                             capture_output=True, timeout=60)
        return out.stdout.decode("utf-8", "ignore")
    except Exception:  # noqa: BLE001
        return ""


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen3.7-flash-2026-07-15")
    ap.add_argument("--split", default="test")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    with (SPLITS / f"{args.split}.csv").open(encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))

    done: dict[str, dict] = {}
    if OUT.exists():
        for line in OUT.read_text(encoding="utf-8").splitlines():
            if line.strip():
                r = json.loads(line)
                done[r["sample_id"]] = r
    todo = [r for r in rows if r["sample_id"] not in done]
    print(f"tesseract OCR: total={len(rows)} done={len(done)} todo={len(todo)}")
    lock = __import__("threading").Lock()
    with OUT.open("a", encoding="utf-8") as fh:
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futs = {pool.submit(ocr, IMAGES / Path(r["image_path"]).name): r
                    for r in todo}
            n = 0
            for fut in as_completed(futs):
                r = futs[fut]
                txt = fut.result()
                rec = {"sample_id": r["sample_id"],
                       "true_species": r["species_latin_project"],
                       "image_domain": r.get("image_domain", ""),
                       "source": r.get("source", ""),
                       "ocr_chars": len(txt.strip()),
                       "ocr_text": txt[:4000]}
                with lock:
                    fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    fh.flush()
                    done[r["sample_id"]] = rec
                    n += 1
                    if n % 200 == 0:
                        print(f"  {n}/{len(todo)}", flush=True)
    print(f"wrote {OUT.name}\n")

    ident_p = ROOT / "data/external_baselines/vlm" / f"open_{args.model}_{args.split}.jsonl"
    ident = {}
    if ident_p.exists():
        for line in ident_p.read_text(encoding="utf-8").splitlines():
            if line.strip():
                r = json.loads(line)
                if r.get("ok"):
                    ident[r["sample_id"]] = r

    from collections import Counter, defaultdict
    agg = defaultdict(Counter)
    for sid, o in done.items():
        dom = o["image_domain"] or "?"
        lk = readable_in(o["true_species"], o["ocr_text"])
        agg[dom]["n"] += 1
        agg[dom]["leak"] += lk
        agg[dom]["has_text"] += (o["ocr_chars"] >= 40)
        if sid in ident:
            a = ident[sid]
            cands = [binom((x.get("name") or "")) for x in (a.get("top3") or [])]
            hit = bool(cands) and cands[0] == binom(a["true_species"])
            agg[dom]["vlm_n"] += 1
            agg[dom]["vlm_hit"] += hit
            agg[dom]["vlm_leak"] += lk
            agg[dom]["vlm_hit&leak"] += (hit and lk)
            agg[dom]["vlm_hit&clean"] += (hit and not lk)
            agg[dom]["vlm_clean"] += (not lk)

    print(f"{'domain':10s} {'n':>5} {'name as text':>13} {'>=40 chars':>11} "
          f"{'VLM top1':>9} {'|leak':>7} {'|clean':>8}")
    print("-" * 66)
    tot = Counter()
    for dom, c in sorted(agg.items()):
        tot.update(c)
        print(f"{dom:10s} {c['n']:>5} {c['leak']/c['n']:>13.3f} "
              f"{c['has_text']/c['n']:>11.3f} "
              f"{c['vlm_hit']/max(c['vlm_n'],1):>9.3f} "
              f"{(c['vlm_hit&leak']/c['vlm_leak'] if c['vlm_leak'] else float('nan')):>7.3f} "
              f"{c['vlm_hit&clean']/max(c['vlm_clean'],1):>8.3f}")
    n = tot["n"]
    print(f"\nALL n={n}  answer-readable-as-text={tot['leak']/n:.3f}  "
          f"VLM top1={tot['vlm_hit']/max(tot['vlm_n'],1):.3f}  "
          f"VLM top1|leak={tot['vlm_hit&leak']/max(tot['vlm_leak'],1):.3f}  "
          f"VLM top1|clean={tot['vlm_hit&clean']/max(tot['vlm_clean'],1):.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
