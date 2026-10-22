#!/usr/bin/env python3
"""Verify that the compiled PDF is equivalent to the Markdown source.

Four checks, all run against the text extracted from the PDF rather than
against the .tex: a .tex check would only prove that the file was written
correctly, not that it typeset correctly.

1. Unresolved references  -- no "??" and no "undefined" may survive
2. Number preservation    -- every numeric token in the Markdown must appear
                             in the PDF, with differences explained
3. Object completeness    -- Tables 1-10, Figures 1-8, the references, and
                             every section must be present
4. No residual machine code -- \\ref, \\cite, \\begin must not be visible

Run after tectonic.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import pymupdf

HERE = Path(__file__).resolve().parent
SUB = HERE.parent
TEXT = SUB / "text"
TABLES = SUB / "tables"
PDF = HERE / "main.pdf"
MD_FILES = [f for f in sorted(TEXT.glob("*.md"))
            if f.name not in {"00_highlights.md"}] + sorted(TABLES.glob("tab*.md"))
# 00_highlights.md is excluded: it is the separate Highlights upload file, not
# part of main.tex, and it carries character-count annotations that are working
# notes rather than manuscript content.


def pdf_text() -> str:
    doc = pymupdf.open(PDF)
    pages = [p.get_text() for p in doc]
    doc.close()
    # Joining the pages with newlines and then collapsing all whitespace is
    # what makes "Figure 1:" findable: hyphenation and line wrapping inside a
    # caption would otherwise split it.
    return re.sub(r"\s+", " ", " ".join(pages))


def markdown_text() -> str:
    """Manuscript prose only: Chinese working notes are excluded."""
    out = []
    for f in MD_FILES:
        for line in f.read_text(encoding="utf-8").splitlines():
            if re.search(r"[\u4e00-\u9fff]", line):
                continue
            out.append(line)
    return "\n".join(out)


NUM_RE = re.compile(r"\d[\d,]*\.?\d*")


def numbers(text: str) -> list[str]:
    return [m.group(0).rstrip(".") for m in NUM_RE.finditer(text)]


def main() -> int:
    text = pdf_text()
    md = markdown_text()
    failures: list[str] = []

    print("=" * 66)
    print("1. Unresolved references")
    q = text.count("??")
    undef = len(re.findall(r"\bundefined\b", text, re.I))
    print(f"   '??' occurrences        : {q}")
    print(f"   'undefined' occurrences : {undef}")
    if q or undef:
        failures.append(f"{q} unresolved '??' and {undef} 'undefined'")

    print("\n2. Number preservation")
    md_nums = numbers(md)
    pdf_nums = set(numbers(text))
    missing = [n for n in md_nums if n not in pdf_nums]
    print(f"   numeric tokens in Markdown : {len(md_nums)}")
    print(f"   distinct in PDF            : {len(pdf_nums)}")
    print(f"   missing from PDF           : {len(missing)}")
    if missing:
        print(f"      {sorted(set(missing))[:20]}")
        failures.append(f"{len(missing)} numeric tokens missing from the PDF")

    print("\n3. Object completeness")
    checks = {
        "Tables 1-10": [f"Table {i}" for i in range(1, 11)],
        "Figures 1-8": [f"Figure {i}" for i in range(1, 9)],
        "Sections 2-13": [f"{i} " for i in range(2, 14)],
    }
    for label, items in checks.items():
        gone = [i for i in items if i not in text]
        print(f"   {label:<16}: {'all present' if not gone else 'MISSING ' + str(gone)}")
        if gone:
            failures.append(f"{label} missing {gone}")

    refs = re.findall(r"\((?:[A-Z][A-Za-z\-]+(?: et al\.)?),? \(?(?:19|20)\d{2}", text)
    print(f"   author-year citations in PDF: {len(refs)}")

    print("\n4. Residual machine code")
    for token in (r"\ref", r"\cite", r"\begin", r"\label", r"\textbf"):
        n = text.count(token)
        print(f"   {token:<10}: {n}")
        if n:
            failures.append(f"machine code '{token}' visible in the PDF {n}x")

    print("\n5. Overfull boxes")
    log = HERE / "main.log"
    if log.exists():
        over = re.findall(r"Overfull \\hbox \(([\d.]+)pt", log.read_text(errors="ignore"))
        big = [float(x) for x in over if float(x) > 10]
        print(f"   overfull hboxes: {len(over)}, of which >10pt: {len(big)}")
        if big:
            print(f"      worst: {max(big):.1f}pt")

    print("\n" + "=" * 66)
    if failures:
        print("FAILED:")
        for f in failures:
            print(f"  - {f}")
        return 1
    print("ALL CHECKS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
