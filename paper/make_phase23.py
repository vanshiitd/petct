#!/usr/bin/env python3
"""Table bodies for the Phase XXIII follow-up confirmation (no number is retyped).
Reads phase23_eval.json and the per-study result tables; writes
paper/sec/gen_fu_{tests,arms,harmed}.tex.   Run: python paper/make_phase23.py (petct venv)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))
from phase5_eval import read
from phase21_safeswitch import harmed

R = Path(r"D:\data\reports")
OUT = Path(__file__).resolve().parent / "sec"
E = json.loads((R / "phase23_eval.json").read_text())


def w(name, rows):
    (OUT / name).write_text("\n".join(rows) + "\n", encoding="utf-8")
    print(f"written {name} ({len(rows)} rows)")


def pv(p):
    return f"{p:.4f}" if p >= 1e-4 else r"$<10^{-4}$"


def test_row(label, t, bold=False):
    verdict = r"\textbf{met}" if t["met"] else "not met"
    km = t["k_max_to_pass"]
    kc = "--" if km < 0 else rf"$\le{km}$"
    return (rf"{label} & {t['k_patients']} / {t['n_patients']} ({t['k_studies']} / {t['n_studies']}) & "
            rf"{pv(t['p'])} & {100 * t['ub95']:.1f}\% & {kc} & {verdict} \\")


def tests():
    s = E["subsets"]
    rows = [test_row(r"H1, Level 2, $\delta=0.10$", E["H1"]),
            test_row(r"H2, Level 1, $\delta=0.05$", E["H2"]),
            r"\midrule",
            test_row(r"Level 2, $\delta=0.05$", E["FT_vs_P19_delta05"]),
            test_row(r"H1, $\B$ patients", s["B"]["H1"]),
            test_row(r"H1, $\C$ patients", s["C"]["H1"]),
            test_row(r"H2, $\B$ patients", s["B"]["H2"]),
            test_row(r"H2, $\C$ patients", s["C"]["H2"])]
    w("gen_fu_tests.tex", rows)


def main_table():
    """Compact version for the main paper: the two pre-registered tests and the 5% secondary."""
    rows = []
    for lab, d, t in ((r"Level 2, $\delta=0.10$", "0.10", E["H1"]),
                      (r"Level 1, $\delta=0.05$", "0.05", E["H2"]),
                      (r"Level 2, $\delta=0.05$", "0.05", E["FT_vs_P19_delta05"])):
        km = t["k_max_to_pass"]
        verdict = r"\textbf{met}" if t["met"] else "not met"
        rows.append(rf"{lab} & {t['k_patients']} / {t['n_patients']} & "
                    rf"{100 * t['ub95']:.1f}\% & {pv(t['p'])} & $\le{km}$ & {verdict} \\")
    w("gen_fu_main.tex", rows)


def arms():
    names = {"M0": "zero-shot segmenter", "HB": r"deployed agent $f_0$",
             "P19": r"\ours, Level 1", "FT": r"\ours, fine-tuned"}
    rows = []
    for k in ("M0", "HB", "P19", "FT"):
        a = E["arms"][k]
        b = r"\textbf{%s}" if k == "FT" else "%s"
        rows.append(rf"{names[k]} & {b % format(a['dice_pos_study'], '.3f')} & "
                    rf"{b % format(a['fp_ml'], '.1f')} & {b % format(a['hits'], ',')} & "
                    rf"{a['neg_cleared']} / {a['n_neg']} \\")
    w("gen_fu_arms.tex", rows)
    return E["arms"]["M0"]["gt_lesions"]


def harmed_patients():
    p19 = read(R / "p23_psmafu_p19", "final")
    ft = read(R / "phase23_ft", "fu")
    # was the same patient harmed by FT at certification (B, locked Phase XX tables)?
    b_ft, b_p19 = read(R / "p20B_ft_hb", "base"), read(R / "p20B_psma_p19", "final")
    cert = {c.rsplit("_", 2)[0] for c in b_ft if c in b_p19 and harmed(b_ft[c], b_p19[c])}
    grp = json.loads((REPO / "reports" / "phase19_split.json").read_text())
    side = {c.rsplit("_", 2)[0]: s for s in "BC" for c in grp[s]}
    cases = sorted(p19)
    pats = sorted({c.rsplit("_", 2)[0] for c in E["H1"]["harmed_studies"]})
    rows = []
    for p in pats:
        mine = [c for c in cases if c.rsplit("_", 2)[0] == p and c in ft]
        hs = [c for c in mine if c in E["H1"]["harmed_studies"]]
        h0 = sum(int(p19[c]["n_hit"]) for c in hs)
        h1 = sum(int(ft[c]["n_hit"]) for c in hs)
        d0 = sum(p19[c]["dice"] for c in hs) / len(hs)
        d1 = sum(ft[c]["dice"] for c in hs) / len(hs)
        rows.append(rf"{side[p]} & {len(hs)} / {len(mine)} & {h0}$\to${h1} & "
                    rf"{d0:.2f}$\to${d1:.2f} & {'yes' if p in cert else 'no'} \\")
    w("gen_fu_harmed.tex", rows)
    print("harmed at certification (B):", sorted(cert))


if __name__ == "__main__":
    tests(); main_table(); arms(); harmed_patients()
