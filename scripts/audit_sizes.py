#!/usr/bin/env python3
"""Audit: the per-experience-size endpoints of final_eval_plan.md section 5.

The plan (l.103-104) asked for "the endpoints (Dice, tumour-free with FP) for
the agent built on each experience size, on out-of-fold and validation". The
final evaluation never produced them; `final_selfimprove.py` stopped at R(a) and
selection precision. These are agent runs with the four saved models
`fe_av_{010,025,050,100}.pkl`, out-of-fold cross-fitted and on validation, so
the comparison is against honest gating.

Test is not touched.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from phase5_eval import read, paired_dice, mcnemar_fp, lesion_sens

R = Path(r"D:\data\reports")
SIZES = [("010", "10%", 58), ("025", "25%", 144),
         ("050", "50%", 289), ("100", "100%", 578)]


def row(agent_pre: Path, base: dict, label: str) -> dict:
    fin = read(agent_pre, "final")
    tr = read(agent_pre, "traces")
    common = sorted(set(fin) & set(base))
    a = {c: fin[c] for c in common}
    b = {c: base[c] for c in common}
    dz = paired_dice(a, b)
    fp = mcnemar_fp(a, b)
    ds = lesion_sens(a) - lesion_sens(b)
    acts = float(np.mean([tr[c]["accepted"] for c in tr]))
    print(f"  {label:22s}{dz['delta']:+9.4f} [{dz['lo']:+.4f}, {dz['hi']:+.4f}]"
          f"{fp['b']:6d} -> {fp['a']:<4d}{fp['cleared']:7d}{ds:+10.4f}"
          f"{acts:9.2f}")
    return {"label": label, "dice": dz, "fp_count": fp,
            "d_lesion_sens": ds, "actions_per_case": acts}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=R / "audit_sizes.json")
    args = ap.parse_args()
    out = {}
    for split, base_pre, prefix in (
            ("out-of-fold (cross-fitted)", R / "au_oof_reason", "af_oof_av"),
            ("validation", R / "au_val_finalagent", "af_val_av")):
        base = read(base_pre, "base")
        print(f"\n{'='*86}\n{split}  (n={len(base)})")
        print(f"  {'experience':22s}{'dDice':>9s}{'  95% CI':>20s}"
              f"{'negFP':>12s}{'cleared':>7s}{'dSens':>10s}{'acts/case':>9s}")
        out[split] = {}
        for tag, lab, npat in SIZES:
            p = R / f"{prefix}{tag}"
            try:
                out[split][lab] = row(p, base, f"{lab} ({npat} patients)")
            except FileNotFoundError:
                print(f"  {lab:22s} not executed")
        # the reference points already measured
        for name, pre in (("full learned selector", "au_oof_learned"),
                          ("FINAL AGENT (fixed)", "au_oof_finalagent")):
            if split.startswith("out-of-fold"):
                try:
                    out[split][name] = row(R / pre, base, name)
                except FileNotFoundError:
                    pass
    args.out.write_text(json.dumps(out, indent=1, default=float),
                        encoding="utf-8")
    print(f"\nwritten {args.out}")


if __name__ == "__main__":
    main()
