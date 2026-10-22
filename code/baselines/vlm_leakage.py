#!/usr/bin/env python3
"""Quantify label-text leakage in the VLM's fern identification.

Herbarium sheets carry the species name as printed or handwritten text, so a
VLM can answer by reading rather than by identifying morphology. Joining the
OCR probe with the identification run separates the two:

  * leaked     -- the true binomial (or genus+epithet) is present in the text
                  the model itself transcribed from the image
  * clean      -- no such text; any correct answer must come from morphology

    python vlm_leakage.py
"""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
VLM = ROOT / "data/external_baselines/vlm"


def norm(s: str) -> str:
    s = (s or "").lower().replace("×", "x")
    s = re.sub(r"[^a-z0-9\s\-]", " ", s)
    return " ".join(s.split())


def binom(s: str) -> tuple[str, str]:
    p = norm(s).split()
    return (p[0], p[1]) if len(p) >= 2 else (norm(s), "")


def leaked(true_sp: str, text: str) -> bool:
    """True if the answer is readable in the transcribed text."""
    g, e = binom(true_sp)
    if not g:
        return False
    t = norm(text)
    if e and f"{g} {e}" in t:
        return True
    # genus and epithet both present, possibly separated by other words
    return e and (g in t) and (e in t)


def load(p: Path) -> dict:
    out = {}
    if not p.exists():
        return out
    for line in p.read_text(encoding="utf-8").splitlines():
        if line.strip():
            r = json.loads(line)
            if r.get("ok"):
                out[r["sample_id"]] = r
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen3.7-flash-2026-07-15")
    ap.add_argument("--split", default="test")
    args = ap.parse_args()

    ident = load(VLM / f"open_{args.model}_{args.split}.jsonl")
    ocr = load(VLM / f"ocr_{args.model}_{args.split}.jsonl")
    both = [s for s in ident if s in ocr]
    print(f"images with both ID and OCR probe: {len(both)}")

    stats = defaultdict(lambda: Counter())
    for s in both:
        a, b = ident[s], ocr[s]
        dom = a.get("image_domain") or "?"
        t = binom(a["true_species"])
        cands = [binom((x.get("name") or "")) for x in (a.get("top3") or [])]
        hit = bool(cands) and cands[0] == t
        lk = leaked(a["true_species"], b.get("raw") or "")
        stats[dom]["n"] += 1
        stats[dom]["hit"] += hit
        stats[dom]["leak"] += lk
        stats[dom]["hit&leak"] += (hit and lk)
        stats[dom]["hit&clean"] += (hit and not lk)
        stats[dom]["clean_n"] += (not lk)
        stats[dom]["none_text"] += norm(b.get("raw") or "") in ("none", "")

    print(f"\n{'domain':10s} {'n':>5} {'top1':>7} {'text has name':>14} "
          f"{'top1|leak':>10} {'top1|clean':>11} {'no text at all':>15}")
    print("-" * 78)
    for dom, c in sorted(stats.items()):
        n = c["n"]
        clean = c["clean_n"] or 1
        print(f"{dom:10s} {n:>5} {c['hit']/n:>7.3f} {c['leak']/n:>14.3f} "
              f"{(c['hit&leak']/c['leak'] if c['leak'] else float('nan')):>10.3f} "
              f"{c['hit&clean']/clean:>11.3f} {c['none_text']/n:>15.3f}")

    tot = Counter()
    for c in stats.values():
        tot.update(c)
    n = tot["n"]
    print(f"\nALL {n}: top1={tot['hit']/n:.3f} | "
          f"text-contains-answer={tot['leak']/n:.3f} | "
          f"top1 on leaked subset="
          f"{(tot['hit&leak']/tot['leak'] if tot['leak'] else float('nan')):.3f} | "
          f"top1 on clean subset="
          f"{tot['hit&clean']/(tot['clean_n'] or 1):.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
