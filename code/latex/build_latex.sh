#!/usr/bin/env bash
# Build and verify the LaTeX submission from the Markdown source of truth.
#
#   ./build_latex.sh            full build
#   ./build_latex.sh --check    convert only, do not compile
#
# The Markdown under ../text/ is the source of truth. main.tex is generated
# and must never be edited by hand.
set -euo pipefail
cd "$(dirname "$0")"

# Pick an interpreter that can actually import matplotlib. Honouring $PYTHON
# first keeps the script usable elsewhere; the loop only exists so the build
# does not silently fall back to an interpreter without the dependency.
# Candidates are probed, never hard-coded: an absolute path from the machine
# that wrote this script would fail on any other one.
PY=${PYTHON:-}
if [ -z "$PY" ]; then
  for cand in python3 python \
              "$HOME/.workbuddy/binaries/python/envs/default/bin/python"; do
    if command -v "$cand" >/dev/null 2>&1 \
       && "$cand" -c "import matplotlib" 2>/dev/null; then
      PY="$cand"
      break
    fi
  done
fi
[ -n "$PY" ] || PY=python3

CHECK_ONLY=0
[ "${1:-}" = "--check" ] && CHECK_ONLY=1

echo "=== 1/4 convert Markdown -> main.tex ==="
"$PY" md_to_latex.py

if [ "$CHECK_ONLY" = "1" ]; then
    echo "check only: stopping before compilation"
    exit 0
fi

echo
echo "=== 2/5 render vector figures ==="
# Fig 3-8 come from matplotlib; Figs 1-2 are the user-authored PPTX masters,
# converted to vector PDFs (with a per-shape font bump so text survives the
# scale-down to the 180 mm text width) plus 600 dpi release PNGs.
"$PY" make_figure_pdfs.py
"$PY" make_fig12_pptx.py

echo
echo "=== 3/5 compile with tectonic ==="
# tectonic fetches its own packages, so the first run needs network access.
tectonic -X compile main.tex --keep-logs

echo
echo "=== 4/5 verify the compiled PDF ==="
"$PY" verify_latex.py

echo
echo "=== 5/5 figure number self-check (Fig 1/2) ==="
"$PY" check_fig12_numbers.py

echo
echo "done: $(pwd)/main.pdf"
