#!/usr/bin/env python3
"""Build the submission Figures 1-2 from the user-authored PPTX masters.

The masters are hand-drawn in PowerPoint and live in figures/*.pptx. This
script is the single reproducible path from master to submission artefact:

1. bump font sizes inside a copy of the slide XML (shapes stay put), because
   the masters are drawn on oversized canvases and the text would otherwise
   land below the 6 pt journal floor once scaled to the 180 mm text width;
2. convert with LibreOffice (headless) to vector PDF;
3. crop each page to its content bounding box plus padding;
4. render 600 dpi PNGs for the release tree.

Outputs (next to this script's parent figures/ dir):
  latex/figures/Figure1_workflow.pdf / Figure2_pipeline.pdf
  figures/Figure1_workflow.png       / Figure2_pipeline.png
"""
from __future__ import annotations

import copy
import html
import re
import shutil
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

import pymupdf

_script = Path(__file__).resolve().parent
if (_script.parent / "figures" / "fig1_fixed.pptx").exists():
    # workspace layout: ei_submission/latex/*.py, masters in ei_submission/figures
    SUB = _script.parent
    MASTERS_DIR = SUB / "figures"
    PDF_OUT = _script / "figures"
    PNG_OUT = SUB / "figures"
else:
    # release layout: code/latex/*.py, masters in figure_masters/
    ROOT = _script.parents[1]
    SUB = ROOT
    MASTERS_DIR = ROOT / "figure_masters"
    PDF_OUT = ROOT / "manuscript" / "latex" / "figures"
    PNG_OUT = ROOT / "manuscript" / "figures"
SOFFICE = "/opt/homebrew/bin/soffice"

# master filename -> (submission name, font rules)
# Font rules: piecewise (threshold_pt, factor) plus per-shape overrides.
# A rule entry (substring, factor) matches a <p:sp> whose text contains the
# substring; factor None keeps that shape's original sizes (it was drawn
# edge-to-edge and any growth would collide), a number overrides the
# piecewise factor for that shape only.
MASTERS = {
    "fig1_fixed.pptx": ("Figure1_workflow", {
        "piecewise": [(11.9, 1.30), (15.9, 1.12), (999, 1.06)],
        "keep": [
            "Can we trust this number",      # pill already full width
            "when the two regimes",          # red card note, one shape per line
            "separated between",
            "and test",
            "Remove metadata",               # T3 note, edge to edge
            "Smithsonian Institu",           # credit line, edge to edge
            "High confidence",               # ACCEPT chip: text would hit icon
            "Uncertain",                     # ABSTAIN chip
            "Out-of-domain",                 # REJECT chip
            "Expert verification",           # HUMAN REVIEW chip
        ],
    }),
    "Fig2_benchmark_construction.pptx": ("Figure2_pipeline", {
        "piecewise": [(10.9, 1.40), (13.9, 1.30), (15.9, 1.18), (999, 1.10)],
        "keep": [
            "Data side (1",                  # legend row, tightly spaced
            "Annotation & split (6",
            "Evaluation only (8",
            "Source Retrieval",              # card titles were tight already
            "Taxonomic Backbone",
            "Collected Pool",
            "Licence Stratification",
            "Balancing",
            "Three-axis Annotation",
            "Group Split",
            "Open-set Negatives",
            "(field / specimen)",            # inline stat rows, no slack
            "(data source)",
            "(normalised country)",
            "family treatment",              # card body lines at full width
            "Cap at 60",
            "268",
            "Specimens",
            "1,432",
            "4,537",
            "(24%",
            "(76%",
        ],
    }),
}
CROP_PAD_PT = 6          # breathing room around the content bbox, in pt
PNG_DPI = 600


def _rule_factor(rules: dict, shape_text: str, old_pt: float) -> float | None:
    for entry in rules.get("keep", []):
        key, f = entry if isinstance(entry, tuple) else (entry, None)
        if key in shape_text:
            return f
    for threshold, f in rules["piecewise"]:
        if old_pt <= threshold:
            return f
    return 1.0


def bump_fonts(src: Path, dst: Path, rules: dict) -> None:
    """Per-shape font bump inside the slide XML: shapes stay put, sizes grow.

    Each <p:sp> is inspected separately: shapes matching a `keep` rule are
    left untouched (or get the rule's factor), everything else takes the
    piecewise factor for its current size. Sizes are stored in hundredths of
    a point and rounded to 0.5 pt steps.
    """
    with zipfile.ZipFile(src) as zin:
        entries = {i.filename: zin.read(i.filename) for i in zin.infolist()}

    xml = entries["ppt/slides/slide1.xml"]
    out = []
    pos = 0
    for m in re.finditer(rb"<p:sp>.*?</p:sp>", xml, re.S):
        out.append(xml[pos:m.start()])
        shape = m.group(0)
        text = b" ".join(re.findall(rb"<a:t>(.*?)</a:t>", shape, re.S))
        try:
            text_s = html.unescape(text.decode("utf-8", "replace"))
        except Exception:
            text_s = ""

        def sub(mm: re.Match) -> bytes:
            old = int(mm.group(1))
            f = _rule_factor(rules, text_s, old / 100.0)
            if f is None or f == 1.0:
                new = old
            else:
                new = old * f
                new = int(round(new / 50.0) * 50)
            return f'sz="{new}"'.encode()

        out.append(re.sub(rb'sz="(\d+)"', sub, shape))
        pos = m.end()
    out.append(xml[pos:])
    entries["ppt/slides/slide1.xml"] = b"".join(out)

    with zipfile.ZipFile(dst, "w", zipfile.ZIP_DEFLATED) as zout:
        for name, data in entries.items():
            zout.writestr(name, data)


def convert(pptx: Path, work: Path) -> Path:
    subprocess.run(
        [SOFFICE, "--headless", "--convert-to", "pdf",
         "--outdir", str(work), str(pptx)],
        check=True, capture_output=True)
    return work / (pptx.stem + ".pdf")


def crop_and_export(pdf: Path, base: str) -> tuple[Path, Path]:
    doc = pymupdf.open(pdf)
    page = doc[0]
    bbox = pymupdf.Rect()
    for block in page.get_text("blocks"):
        bbox |= pymupdf.Rect(block[:4])
    for img in page.get_image_info():
        bbox |= pymupdf.Rect(img["bbox"])
    page_area = page.rect.get_area()
    for d in page.get_drawings():
        # slide-background rects cover (almost) the whole page; counting them
        # would make the crop a no-op and leave the master's empty margins in
        if d["rect"].get_area() < 0.85 * page_area:
            bbox |= d["rect"]
    if bbox.is_empty:
        bbox = page.rect
    bbox = (bbox + (-CROP_PAD_PT, -CROP_PAD_PT, CROP_PAD_PT, CROP_PAD_PT))
    bbox &= page.rect
    page.set_cropbox(bbox)

    out_pdf = PDF_OUT / f"{base}.pdf"
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    doc.save(out_pdf, garbage=4, deflate=True)

    png = PNG_OUT / f"{base}.png"
    page.get_pixmap(dpi=PNG_DPI).save(png)
    doc.close()
    return out_pdf, png


def main() -> int:
    with tempfile.TemporaryDirectory() as tmp:
        work = Path(tmp)
        for master, (base, rules) in MASTERS.items():
            src = MASTERS_DIR / master
            if not src.exists():
                print(f"master missing: {src}", file=sys.stderr)
                return 1
            bumped = work / (src.stem + "_bumped.pptx")
            bump_fonts(src, bumped, rules)
            pdf = convert(bumped, work)
            out_pdf, png = crop_and_export(pdf, base)
            d = pymupdf.open(out_pdf)
            p = d[0]
            print(f"{master} -> {out_pdf.name}: "
                  f"{p.rect.width * 25.4 / 72:.0f} x {p.rect.height * 25.4 / 72:.0f} mm, "
                  f"PNG {png.name} ({png.stat().st_size // 1024} KB)")
            d.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
