#!/usr/bin/env python3
"""Table bodies for the Phase XXI analyses, written as LaTeX so that no number
is retyped. Reads phase21_rules.json and, when present, the safeguarded-switch
files. Writes paper/sec/gen_*.tex, which the paper \\input's.

Run: python paper/make_phase21.py  (petct venv)
"""
from __future__ import annotations

import json
from pathlib import Path

R = Path(r"D:\data\reports")
OUT = Path(__file__).resolve().parent / "sec"
NAMES = {"NAIVE": "Naive (largest benefit)", "PLUGIN": r"Plug-in ($\hat R\le\delta$)",
         "CRC": "Conformal risk control", "UNCORR": "Uncorrected test",
         "ZERO": "Zero observed harms", "BONF-88": "Bonferroni over 88", "CSI": r"\ours"}
CONTROL = {"NAIVE": "none", "PLUGIN": "none", "CRC": r"$\mathbb{E}[R]$", "UNCORR": "one test",
           "ZERO": "none", "BONF-88": "all tests", "CSI": "all tests"}
ORDER = ["NAIVE", "PLUGIN", "CRC", "UNCORR", "ZERO", "BONF-88", "CSI"]


def pct(x, d=1):
    return f"{100 * x:.{d}f}"


def write(name, lines):
    (OUT / name).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"written {name} ({len(lines)} rows)")


def rules():
    rep = json.loads((R / "phase21_rules.json").read_text())
    a = {r["delta"]: r["rules"] for r in rep["part_a"]}

    def cell(s):
        if s["rollback_freq"] == 1.0:
            return r"\multicolumn{2}{c}{rolls back}"
        return f"{pct(s['violation_rate'])} & ${s['mean_benefit']:+.2f}$"

    main = []
    for r in ORDER:
        bold = r == "CSI"
        row = f"{NAMES[r]} & {CONTROL[r]} & {cell(a[0.05][r])} & {cell(a[0.10][r])} \\\\"
        main.append((r"\midrule" + "\n" if bold else "") + row)
    write("gen_rules_main.tex", main)

    supp = []
    for d in (0.03, 0.05, 0.10, 0.15, 0.20):
        supp.append(rf"\multicolumn{{6}}{{@{{}}l}}{{\emph{{$\delta={d:.2f}$}}}} \\")
        for r in ORDER:
            s = a[d][r]
            lo, hi = s["violation_ci95"]
            supp.append(f"{NAMES[r]} & {pct(s['violation_rate'])} [{pct(lo)}, {pct(hi)}] & "
                        f"{pct(s['mean_risk'])} & {pct(s['p95_risk'])} & "
                        f"${s['mean_benefit']:+.2f}$ & {pct(s['rollback_freq'], 0)} \\\\")
        if d != 0.20:
            supp.append(r"\addlinespace[2pt]")
    write("gen_rules_supp.tex", supp)

    shift = []
    for b in rep["part_b"]:
        row = next(x for x in b["delta"] if x["delta"] == 0.10)["rules"]
        arrow = f"{b['source']} $\\to$ {b['target']}"
        cells = []
        for r in ("NAIVE", "CRC", "UNCORR", "CSI"):
            x = row[r]
            cells.append(f"{pct(x['control']['violation_rate'])} & "
                         f"{pct(x['shifted']['violation_rate'])}")
        shift.append(f"{arrow} & {b['n_select']} / {b['n_shifted']} & " + " & ".join(cells)
                     + r" \\")
    write("gen_shift_supp.tex", shift)
    return rep


def safeswitch():
    fb, fc = R / "phase21_safeswitch_B.json", R / "phase21_safeswitch_C.json"
    if not (fb.exists() and fc.exists()):
        print("safeguarded-switch files not present; skipped")
        return
    B, C = json.loads(fb.read_text()), json.loads(fc.read_text())
    cb = {r["tau"]: r for r in B["taus"]}
    cc = {r["tau"]: r for r in C["taus"]}
    star = {s["tau_star"] for s in B["selection"] if s["tau_star"] is not None}
    rows = []
    for t in sorted(cb):
        b, c = cb[t], cc[t]
        lab = {0.0: r"0 (FT $\cup$ deployed)", 0.5: "0.5 (FT)"}.get(t, f"{t:g}")
        mark = r"$^\star$" if t in star else ""
        rows.append(f"{lab}{mark} & {b['k']} & {b['dice_pos']:.3f} & {b['fp_ml']:.0f} & "
                    f"{c['k']} ({pct(c['ub95'])}\\%) & {c['dice_pos']:.3f} & {c['fp_ml']:.0f} & "
                    f"{c['hits']} \\\\")
    write("gen_safeswitch_supp.tex", rows)


if __name__ == "__main__":
    rules()
    safeswitch()
