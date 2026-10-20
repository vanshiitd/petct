#!/usr/bin/env python3
"""Phase VI, Experiment 1 for the second backbone: Swin against single nnU-Net.

Scores a directory of probability maps through the same `CaseContext` the agent
uses, so a Swin row and an nnU-Net row are produced by one metric
implementation. The comparator is the **single** nnU-Net fold-0 model, not the
5-fold ensemble: Phase IIIb priced the ensemble at +0.0213 Dice over a single
model, so comparing a single Swin against five nnU-Nets would measure the
ensemble, not the backbone.

Writes one CSV per maps directory, then reports the paired difference over
patients.
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from phase4_strategies import CaseContext
from evaluate_predictions import hd95_mm
from phase5_eval import paired_dice, mcnemar_fp, paired_mean, lesion_sens

COLS = ("dice", "fp_ml", "fn_ml", "hd95", "n_fp_comp", "n_gt", "n_hit",
        "pred_ml")


def score_dir(maps: Path, out: Path) -> dict:
    cases = sorted(f.stem for f in maps.glob("*.npz"))
    rows = []
    for i, case in enumerate(cases, 1):
        ctx = CaseContext(case, maps)
        s = ctx.score()
        m = ctx.mask()
        s["hd95"] = (hd95_mm(m, ctx.gt, ctx.spacing)
                     if m.any() and ctx.gt.any() else float("nan"))
        rows.append({"case": case, **{k: s[k] for k in COLS}})
        if i % 50 == 0 or i == len(cases):
            print(f"  {i}/{len(cases)}", flush=True)
    with open(out, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["case", *COLS])
        w.writeheader()
        w.writerows(rows)
    return {r["case"]: r for r in rows}


def read_csv(p: Path) -> dict:
    out = {}
    for r in csv.DictReader(open(p, newline="", encoding="utf-8")):
        for k, v in list(r.items()):
            if k != "case":
                r[k] = float(v) if v not in ("", "nan") else float("nan")
        out[r["case"]] = r
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", type=Path, required=True, help="maps dir A (Swin)")
    ap.add_argument("--b", type=Path, required=True, help="maps dir B (nnU-Net)")
    ap.add_argument("--labels", nargs=2, default=["swin", "nnunet_f0"])
    ap.add_argument("--tag", required=True)
    ap.add_argument("--outdir", type=Path, default=Path(r"D:\data\reports"))
    args = ap.parse_args()

    res = {}
    for lab, d in zip(args.labels, (args.a, args.b)):
        p = args.outdir / f"p6_{args.tag}_{lab}.csv"
        if p.exists():
            print(f"{lab}: reusing {p}")
            res[lab] = read_csv(p)
        else:
            print(f"scoring {lab} from {d}")
            res[lab] = score_dir(d, p)

    A, B = res[args.labels[0]], res[args.labels[1]]
    common = sorted(set(A) & set(B))
    A = {c: A[c] for c in common}
    B = {c: B[c] for c in common}
    pos = [c for c in common if B[c]["n_gt"] > 0]
    neg = [c for c in common if B[c]["n_gt"] == 0]

    print(f"\n{'='*78}\n{args.tag}: {args.labels[0]} vs {args.labels[1]}  "
          f"(n={len(common)}, {len(pos)} tumour-positive, {len(neg)} tumour-free)")
    print(f"  {'metric':28s}{args.labels[1]:>14s}{args.labels[0]:>14s}{'delta':>22s}")
    out = {}
    for key, only, lab in (("dice", "pos", "Dice (tumour-positive)"),
                           ("fp_ml", None, "FP volume mL (all)"),
                           ("fn_ml", None, "FN volume mL (all)"),
                           ("hd95", "pos", "HD95 (tumour-positive)"),
                           ("pred_ml", None, "predicted volume mL")):
        m = paired_mean(A, B, key, only)
        print(f"  {lab:28s}{m['b']:14.4f}{m['a']:14.4f}"
              f"  {m['delta']:+9.4f} [{m['lo']:+.4f}, {m['hi']:+.4f}]")
        out[key] = m
    sa, sb = lesion_sens(A), lesion_sens(B)
    print(f"  {'lesion sensitivity':28s}{sb:14.4f}{sa:14.4f}  {sa-sb:+9.4f}")
    na = sum(1 for c in neg if A[c]["n_fp_comp"] > 0)
    nb = sum(1 for c in neg if B[c]["n_fp_comp"] > 0)
    print(f"  {'tumour-free with any FP':28s}{nb:11d}/{len(neg):<2d}"
          f"{na:11d}/{len(neg):<2d}")
    out["lesion_sens"] = {"a": sa, "b": sb, "delta": sa - sb}
    out["neg_with_fp"] = {"a": na, "b": nb, "n": len(neg)}

    import json
    p = args.outdir / f"p6_compare_{args.tag}.json"
    p.write_text(json.dumps(out, indent=1, default=float), encoding="utf-8")
    print(f"\nwritten {p}")


if __name__ == "__main__":
    main()
