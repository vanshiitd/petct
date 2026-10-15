#!/usr/bin/env python3
"""Replace one case's candidate rows in a CSV with freshly built ones.

Five training scans lost their TotalSegmentator run to an out-of-memory kill, so
their rows in the out-of-fold candidate CSV carry empty anatomy features. Once
the organ masks exist, only those cases need rebuilding -- 5 of 738, where a
full rebuild is ~48 minutes of CPU for no change to the other 733.

The original CSV is copied to a backup before anything is written, the header is
required to match field for field (a column order change would otherwise shift
every value silently), and the script refuses to run if the replacement CSV
covers a case the target does not, or leaves a case with no rows at all.

    python scripts/splice_candidates.py --target reports/cand_cv_oof.csv \
        --patch reports/cand_patch5.csv --backup-dir reports/backup
"""
from __future__ import annotations

import argparse
import csv
import shutil
from pathlib import Path


def read(path: Path) -> tuple[list[str], list[dict]]:
    with open(path, newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        return list(r.fieldnames or []), list(r)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--target", type=Path, required=True)
    ap.add_argument("--patch", type=Path, required=True)
    ap.add_argument("--backup-dir", type=Path, required=True)
    args = ap.parse_args()

    fld_t, rows_t = read(args.target)
    fld_p, rows_p = read(args.patch)
    if fld_t != fld_p:
        raise SystemExit(f"header mismatch\n target {fld_t}\n patch  {fld_p}")

    cases_p = {r["case"] for r in rows_p}
    cases_t = {r["case"] for r in rows_t}
    unknown = sorted(cases_p - cases_t)
    if unknown:
        raise SystemExit(f"patch covers case(s) absent from target: {unknown}")
    for c in sorted(cases_p):
        n_old = sum(1 for r in rows_t if r["case"] == c)
        n_new = sum(1 for r in rows_p if r["case"] == c)
        print(f"  {c}: {n_old} row(s) -> {n_new}")
        if n_new == 0:
            raise SystemExit(f"{c} would be left with no rows")

    args.backup_dir.mkdir(parents=True, exist_ok=True)
    bak = args.backup_dir / (args.target.stem + "_before_splice.csv")
    shutil.copy2(args.target, bak)
    print(f"backup -> {bak}")

    kept = [r for r in rows_t if r["case"] not in cases_p]
    out = kept + rows_p
    out.sort(key=lambda r: (r["case"], r["kind"], int(r["component_id"])))
    with open(args.target, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fld_t)
        w.writeheader()
        w.writerows(out)
    print(f"{len(rows_t)} rows -> {len(out)} "
          f"({len(rows_t) - len(kept)} replaced by {len(rows_p)}) in {args.target}")

    # the anatomy columns are the reason for the splice; say whether they filled
    anat = [c for c in fld_t if c.startswith("frac_") or c.startswith("dist_")
            or c == "organ_dominant"]
    for c in sorted(cases_p):
        rows = [r for r in out if r["case"] == c]
        empty = sum(1 for r in rows for a in anat if r[a] == "")
        print(f"  {c}: {empty} empty anatomy cell(s) across {len(rows)} row(s)")


if __name__ == "__main__":
    main()
