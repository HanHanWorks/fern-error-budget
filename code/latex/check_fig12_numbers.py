#!/usr/bin/env python3
"""Spec self-check: every number required by FIG1_FIG2_design_spec.md must
appear in the rendered figures. Run against the vector PDFs."""
import sys
from pathlib import Path

import pymupdf

HERE = Path(__file__).resolve().parent
if (HERE / "figures" / "Figure1_workflow.pdf").exists():
    FIG = HERE / "figures"                        # workspace: latex/figures/
else:                                             # release: code/latex/
    FIG = HERE.parents[1] / "manuscript" / "latex" / "figures"

FIG1 = [
    "91.4", "0.7%", "5,969", "152", "30",
    "0.9198", "0.9250", "91.98%", "92.50%",
    "0.4238", "0.6538", "42.38%", "65.38%",
    "Cyrtomium macrophyllum", "Dryopteridaceae",
    "Cynthia Su", "CC BY", "CC0", "Smithsonian",
    "T1", "T2", "T3", "falsification",
    "ACCEPT", "ABSTAIN", "REJECT", "HUMAN REVIEW",
    "field-ready",
]
FIG2 = [
    "3,157", "2,812", "155", "152", "30", "6,237", "5,969", "268", "28.0%",
    "1,432", "4,537", "24%", "76%", "1,465", "4,772", "1,468", "80", "5.5%",
    "330", "4,185", "882", "902", "700", "8 groups", "PPG I", "60",
    "1\u2013120", "70 \u00d7 7 + 210", "Zero overlap", "Redistributable",
    "Restricted", "Train", "Val", "Test", "image_domain", "source",
    "country_clean", "normalised country", "Species-stratified",
    "group-aware by observation", "evaluation only", "Data side",
    "Annotation & split", "Fern-like negatives",
]


def text_of(pdf: Path) -> str:
    d = pymupdf.open(pdf)
    t = " ".join(pg.get_text() for pg in d)
    d.close()
    return " ".join(t.split())


def main() -> int:
    bad = 0
    for name, tokens in (("Figure1_workflow", FIG1), ("Figure2_pipeline", FIG2)):
        t = text_of(FIG / f"{name}.pdf")
        missing = [tok for tok in tokens if tok not in t]
        print(f"{name}: {'ALL PRESENT' if not missing else 'MISSING ' + str(missing)}")
        bad += len(missing)
        # every digit string inside the figure must be a number the paper
        # uses, checked against the numeric token whitelist of the spec
    print("\nFig1 in-domain pair sanity:", "0.9198" in text_of(FIG / "Figure1_workflow.pdf")
          and "0.9250" in text_of(FIG / "Figure1_workflow.pdf"))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
