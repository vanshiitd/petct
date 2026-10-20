#!/usr/bin/env python3
"""Per-case difficulty groups for work plan section 22.

Three inconsistency groups, taken from the Phase IV diagnosis, plus the mean
stability uncertainty of a case's predicted components:

  anatomically inconsistent    any predicted component sitting mostly in
                               physiological uptake, liver or spleen
  metabolically inconsistent   any predicted component whose SUVmax does not
                               clear the liver reference (suv_max_over_liver < 1)
  topologically inconsistent   the case carries fragmentation or merging

These group cases, they do not select anything. The agent never sees them, and
no threshold anywhere in the project is tuned against them -- so using the
ground-truth diagnosis to define "difficult" is description, not leakage. It is
the same reason a radiology paper may stratify results by tumour stage.

`u_stab` comes from the candidate CSV, which is observable.
"""
from __future__ import annotations

import argparse
import csv
import json
import pickle
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

PHYSIO = 0.5


def groups_from_diag(diag_path: Path) -> dict:
    out = {}
    for r in pickle.load(open(diag_path, "rb")):
        anat = metab = topo = False
        liver = r.get("liver_suv") or float("nan")
        for c in r["components"]:
            if c.get("spurious_class") in ("spurious_physio",
                                           "spurious_liver_spleen"):
                anat = True
            # The diagnosis stores absolute SUVmax per component and the case's
            # liver reference separately, so the ratio is formed here rather
            # than read off a `suv_max_over_liver` field, which these records do
            # not carry.
            if np.isfinite(liver) and liver > 0 and c.get("suv_max") is not None:
                if c["suv_max"] / liver < 1.0:
                    metab = True
            if c.get("merging"):
                topo = True
        for l in r["lesions"]:
            if l.get("fragmentation"):
                topo = True
        out[r["case"]] = {"anat": bool(anat), "metab": bool(metab),
                          "topo": bool(topo)}
    return out


def ustab_from_cand(cand: Path) -> dict:
    acc = {}
    for r in csv.DictReader(open(cand, newline="", encoding="utf-8")):
        if r["kind"] != "pred":
            continue
        v = r.get("ustab_mean", "")
        if v in ("", "nan"):
            continue
        acc.setdefault(r["case"], []).append(float(v))
    return {c: float(np.mean(v)) for c, v in acc.items()}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path,
                    default=Path(r"D:\data\reports\final_case_groups.json"))
    args = ap.parse_args()
    R = Path(r"D:\data\reports")
    res = {}
    for sp, diag in (("oof", "diag_oof.pkl"), ("val", "diag_val.pkl"),
                     ("test", "diag_test.pkl")):
        g = groups_from_diag(R / diag)
        u = ustab_from_cand(R / f"cand_cv_{sp}_pos.csv")
        merged = {}
        for case, d in g.items():
            merged[case] = {**d, "ustab": u.get(case, float("nan"))}
        res[sp] = merged
        n = len(merged)
        print(f"{sp:5s} {n:4d} cases | anat {sum(v['anat'] for v in merged.values()):4d}"
              f" | metab {sum(v['metab'] for v in merged.values()):4d}"
              f" | topo {sum(v['topo'] for v in merged.values()):4d}"
              f" | u_stab finite {sum(np.isfinite(v['ustab']) for v in merged.values()):4d}")
    args.out.write_text(json.dumps(res, indent=1), encoding="utf-8")
    print(f"written {args.out}")


if __name__ == "__main__":
    main()
