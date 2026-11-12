# CVPR 2027 submission: First, Do No Harm

Built on the official CVPR 2027 author kit (`cvpr.sty`, `ieeenat_fullname.bst`,
`preamble.tex` copied unchanged from github.com/cvpr-org/author-kit).

## Build

Overleaf: upload this folder and compile `main.tex` with pdfLaTeX.
Locally: `latexmk -pdf main.tex`.

`main.tex` builds the paper with the supplement appended, as the author kit
recommends. Before submitting:
- for the **main paper PDF**, set `\withsupplfalse` in `main.tex`, or delete
  the supplementary pages from the PDF;
- for the **supplementary PDF**, keep `\withsuppltrue` and keep only the pages
  after the references.

## Figures

`python paper/make_figures.py` (petct venv) regenerates every figure in `fig/`
from the locked result files in `D:\data\reports`. No plotted value is
retyped.

## Before submission

1. Resolve every red `[CHECK]` marker in `main.bib` and anywhere in the text
   (`grep -rn CHECK`).
2. Check the page limit: 8 pages, excluding references.
3. Confirm anonymity: no names, no repository links, no commit hashes.
