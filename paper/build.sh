#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
pdflatex -interaction=nonstopmode -halt-on-error main.tex
if command -v bibtex >/dev/null 2>&1; then
  bibtex main
elif command -v bibtex.original >/dev/null 2>&1; then
  bibtex.original main
else
  echo "A working BibTeX executable is required." >&2; exit 1
fi
pdflatex -interaction=nonstopmode -halt-on-error main.tex
pdflatex -interaction=nonstopmode -halt-on-error main.tex
