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
Locally: `latexmk -pdf main_paper.tex` and `latexmk -pdf supplementary.tex`,
or with Tectonic (conda env `texenv`): `tectonic -X compile main_paper.tex`.
Tectonic uses XeTeX, where `times` is not available, so it falls back to a
wider font: its page count and overfull-box warnings overstate pdfLaTeX's.

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
aggregates; it also recounts harm under six alternative definitions
(Table S14). Both scripts read only the locked result files in `D:\data\reports`.
No plotted value is retyped. The post-hoc analyses are labelled as such in the
paper (Sec. 4 and deviations 10 and 11 in the supplement).

`python paper/make_phase21.py` writes the table bodies `sec/gen_*.tex` from the
Phase XXI result files (selection rules, shift test, safeguarded switch; plan
and result in `reports/phase21_*.md`). The paper loads them with `\tabinput`,
an expandable `\input`, so no number in those tables is retyped.

## Page budget (main_paper.tex)

Target: 8 pages of content, then the LLM-usage statement and the references
(pages 9–10). A `\clearpage` before `sec/7_llm` starts the statement on page 9.

Measured with Tectonic (XeTeX) and TeX Gyre Termes/Heros, which have the
metrics of Times/Helvetica, the content ends at about 93% of the last column of
page 8 (roughly 4 lines spare), the references end on page 10, and nothing is
overfull. pdfLaTeX on Overleaf should match to within a few lines. `fig:power`
and the harm-definition table already moved to the supplement (Fig. S1,
Table S14). If the content runs past page 8, trim in this order:
1. Drop the "Under a known shift" paragraph (Sec. 5.5); Table S8 has it.
2. Move `tab:subgroups` to the supplement (Table S13 has the full version).
3. Move `alg:csi` to the supplement.
If the content ends before the last column of page 8, remove the `\clearpage`.

Main-paper figure numbers are hard-coded in `supplementary.tex` (`\extlabel`):
Fig. 2 qual, 3 l2B, 4 patients, 5 validity; Table 3 rules; Sec. 5.4 harmdef.
Supplement numbers are hard-coded in `main_paper.tex`: Fig. S1 power,
Tables S2, S13 and S14.

## Before submission

1. Resolve every red `[CHECK]` marker in `main.bib` and anywhere in the text
   (`grep -rn CHECK`).
2. Check the page limit: 8 pages, excluding references.
3. Confirm anonymity: no names, no repository links, no commit hashes.
