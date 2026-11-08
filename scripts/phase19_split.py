#!/usr/bin/env python3
"""Phase XIX: the A / B / C patient split of the 377 PSMA patients.

See `reports/phase19_psma_selfimprove_plan.md` section 1. Patients are
stratified by radionuclide x tumour status, shuffled with seed 2027 and dealt
to A, B, C in turn within each stratum. Writes reports/phase19_split.json.
"""
from __future__ import annotations

import collections
import csv
import json
from pathlib import Path

import numpy as np

R = Path(r"D:\data\reports")
TSV = Path(r"D:\data\psma_nifti\PSMA-PET-CT-Lesions-CLINICAL_v01_20251118.tsv")
CASES = Path(__file__).resolve().parent.parent / "reports" / "cases_psma.txt"
OUT = Path(__file__).resolve().parent.parent / "reports" / "phase19_split.json"
SEED = 2027


def main() -> None:
    cases = CASES.read_text().split()
    pos = {r["case"]: r["case_gt_positive"] == "1"
           for r in csv.DictReader(open(R / "cand_psma.csv", newline="", encoding="utf-8"))}
    tracer = {(r["case_identifier_number"], r["study_date"].replace("-", "")): r["pet_radionuclide"]
              for r in csv.DictReader(open(TSV, encoding="utf-8"), delimiter="\t")}
    strata = collections.defaultdict(list)
    for c in cases:
        pid, date, _ = c.rsplit("_", 2)
        strata[(tracer[(pid, date)], pos[c])].append(c)
    rng = np.random.default_rng(SEED)
    parts = {"A": [], "B": [], "C": []}
    turn = 0
    for key in sorted(strata):
        for c in rng.permutation(sorted(strata[key])):
            parts["ABC"[turn % 3]].append(str(c))
            turn += 1
    for p in parts.values():
        p.sort()
    summary = {p: {"n": len(cs),
                   "tumour_free": sum(not pos[c] for c in cs),
                   "68Ga": sum(tracer[tuple(c.rsplit("_", 2)[:2])] == "68Ga" for c in cs)}
               for p, cs in parts.items()}
    OUT.write_text(json.dumps({"seed": SEED, "plan": "reports/phase19_psma_selfimprove_plan.md",
                               "summary": summary, **parts}, indent=1))
    print(summary)


if __name__ == "__main__":
    main()
