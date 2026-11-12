# CVPR 2027 submission: First, Do No Harm

Built on the official CVPR 2027 author kit (`cvpr.sty`, `ieeenat_fullname.bst`,
`preamble.tex` copied unchanged from github.com/cvpr-org/author-kit).

## Build

Three entry points, all pdfLaTeX, all sharing `sec/`, `fig/` and `main.bib`:

| file | builds |
|---|---|
| `main_paper.tex` | **main paper only**: upload as the paper PDF |
| `supplementary.tex` | **supplement only**: upload as the supplementary PDF |
| `main.tex` | paper and supplement in one PDF (`\withsupplfalse` drops the supplement) |

Overleaf: upload this folder. Then, under Menu → Main document, choose
`main_paper.tex` and compile; switch to `supplementary.tex` and compile again.
Locally: `latexmk -pdf main_paper.tex` and `latexmk -pdf supplementary.tex`.

Cross-references between the two PDFs ("Sec. S1", "Tab. 1") are hard-coded
with `\extlabel` in each preamble. If you add, remove or reorder a
supplementary section, or a figure, table or theorem that the supplement cites,
update those lines.

## Figures

`python paper/make_figures.py` (petct venv) regenerates every figure in `fig/`
from the locked result files in `D:\data\reports`. No plotted value is
retyped.

## Before submission

1. Resolve every red `[CHECK]` marker in `main.bib` and anywhere in the text
   (`grep -rn CHECK`).
2. Check the page limit: 8 pages, excluding references.
3. Confirm anonymity: no names, no repository links, no commit hashes.
