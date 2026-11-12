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

`python paper/make_figures.py` (petct venv) regenerates fig_overview, fig_qual,
fig_dial and fig_validity. `python paper/make_posthoc.py` regenerates
fig_patients, fig_power, fig_dial_all and fig_l2_B, and writes
`posthoc_stats.json`, the source of every per-patient and subgroup number.
That script first checks that the per-patient tables reproduce the locked
aggregates. Both scripts read only the locked result files in `D:\data\reports`.
No plotted value is retyped. The post-hoc analyses are labelled as such in the
paper (Sec. 4 and deviation 10 in the supplement).

## Page budget (main_paper.tex)

Target: 8 pages of content, then the LLM-usage statement and the references
(pages 9–10). If the compiled content runs past page 8, trim in this order:
1. Move `fig:power` (Sec. 5.5) to supplement S1 and cite it as Fig. S3.
2. Shorten the "Using the protocol at a new site" paragraph (Sec. 6).
3. Move `alg:csi` to the supplement.
4. Reduce `fig_patients` to panels (a) and (b).
If it ends well short of page 8, the first candidate to bring back is the
subgroup table: copy a compact version of Table S10.

## Before submission

1. Resolve every red `[CHECK]` marker in `main.bib` and anywhere in the text
   (`grep -rn CHECK`).
2. Check the page limit: 8 pages, excluding references.
3. Confirm anonymity: no names, no repository links, no commit hashes.
