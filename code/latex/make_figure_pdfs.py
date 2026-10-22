#!/usr/bin/env python3
"""Render vector PDF copies of Figures 3-8 for the LaTeX submission.

Figures 1 and 2 are not drawn here: their masters are hand-authored SVGs
converted by make_figure1_svg.py and make_figure2_svg.py.

Elsevier's artwork rules set a minimum of 1000 dpi for bitmapped line drawings,
and the released PNGs are 600 dpi. For a LaTeX submission the better answer is
to supply the figures as vector PDF, which has no resolution limit at all and
matches the journal's own recommendation for vector drawings.

This does not touch the released PNGs under ``../figures/``; the PDFs are
written into ``figures/`` next to ``main.tex``.

The script imports the project's own figure generator and redirects its
``savefig`` calls from PNG to PDF, so the figures cannot drift from the ones
the paper reports.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.figure as mfig

HERE = Path(__file__).resolve().parent
WORKSPACE = HERE.parents[2]          # latex -> ei_submission -> paper_analysis -> splite
CODE = WORKSPACE / "bioclip_finetune" / "code"
OUT = HERE / "figures"

sys.path.insert(0, str(CODE))

_original = mfig.Figure.savefig


def _patched(self, fname, *args, **kwargs):
    path = Path(fname)
    if path.suffix == ".png":
        path = path.with_suffix(".pdf")
    return _original(self, path, *args, **kwargs)


def main() -> int:
    mfig.Figure.savefig = _patched
    import make_figures as mf

    OUT.mkdir(parents=True, exist_ok=True)
    sys.argv = ["make_figures.py", "--out", str(OUT)]
    mf.main()

    produced = sorted(OUT.glob("*.pdf"))
    print(f"\n  {len(produced)} vector figure(s) in {OUT}")
    for p in produced:
        print(f"    {p.name}  {p.stat().st_size / 1024:.0f} KB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
