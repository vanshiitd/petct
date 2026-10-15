#!/usr/bin/env python3
r"""Replace scan-relative height with anatomy-anchored position.

`z_rel` is a candidate's height as a fraction of the body extent in its own
scan. The probe in `zrel_probe.py` shows why that is unsafe: AutoPET scan
coverage runs from 600 to 1983 mm, a 3.3x spread, so the same organ lands at a
different z_rel depending on the protocol, and nearly all of z_rel's apparent
power sits in its extreme deciles -- trimming them drops its test AUROC on the
removal task from 0.5425 to 0.4665, below chance. The tree was using it as an
"is this at the edge of the scan" indicator, not as an anatomical prior.

So z_rel is decomposed into the two things it was conflating:

  anatomy   signed z distance in mm from the candidate centroid to fixed
            anatomical landmarks -- liver, heart, urinary bladder, brain,
            kidneys. These are patient-anchored: the liver dome is the liver
            dome whatever the scan covers. This is the plan's spatial
            plausibility prior done properly.

  truncation  distance in mm to the nearest edge of the body extent, and to the
            nearest edge of the scan. Blobs at a truncation boundary really are
            usually artefacts, so this is genuine signal -- it just has nothing
            to do with anatomy and should be named for what it is rather than
            smuggled in through a normalised coordinate.

TotalSegmentator was run with a seven-organ subset that does not include lungs,
so the thoracic anchor here is the heart rather than the lung apices. Landmarks
outside a scan's coverage (commonly the brain, sometimes the bladder) are left
empty and imputed downstream like any other missing feature; per-landmark
coverage is reported so a feature that is absent half the time is visible.

Sign convention: positive means the candidate sits at a HIGHER z index than the
landmark. The script prints the brain-minus-bladder offset so the reader can
see which direction that is in these volumes rather than trusting a comment.

    python scripts/add_position_features.py --cand reports/cand_cv_oof.csv \
        --maps D:\data\cv\oof --out reports/cand_cv_oof_pos.csv
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import SimpleITK as sitk

RAW = Path(r"C:\nnunet_raw\Dataset505_AutoPET_MAEprep_v2")
INV = RAW / "inverse"
TOTALSEG = Path(r"D:\data\totalseg_ml_v2")
LANDMARKS = {"liver": 5, "heart": 51, "bladder": 21, "brain": 90, "kidney": (2, 3)}
NEW = [f"dz_{n}_mm" for n in LANDMARKS] + ["edge_body_mm", "edge_scan_mm"]


def crop_like(path: Path, bbox, shape):
    a = sitk.GetArrayFromImage(sitk.ReadImage(str(path)))
    (z0, z1), (y0, y1), (x0, x1) = bbox
    a = a[z0:z1, y0:y1, x0:x1]
    if a.shape != shape:
        raise SystemExit(f"cropped {a.shape} != prepared {shape} for {path}")
    return a


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cand", type=Path, required=True)
    ap.add_argument("--maps", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    with open(args.cand, newline="", encoding="utf-8") as f:
        rdr = csv.DictReader(f)
        fields = list(rdr.fieldnames or [])
        rows = list(rdr)
    dup = [c for c in NEW if c in fields]
    if dup:
        raise SystemExit(f"{args.cand} already has {dup}")

    by_case: dict[str, list[dict]] = {}
    for r in rows:
        by_case.setdefault(r["case"], []).append(r)
    cases = sorted(by_case)
    print(f"{len(rows)} rows over {len(cases)} cases")

    cover = {n: 0 for n in LANDMARKS}
    brain_minus_bladder = []
    for i, case in enumerate(cases, 1):
        z = np.load(args.maps / f"{case}.npz")
        shape = tuple(int(v) for v in z["shape"])
        sz = float(z["spacing"][0])
        meta = json.loads((INV / f"{case}.json").read_text())
        bbox = meta["crop_bbox_zyx"]
        nz = int(bbox[0][1]) - int(bbox[0][0])
        cshape = (nz, int(bbox[1][1]) - int(bbox[1][0]), int(bbox[2][1]) - int(bbox[2][0]))

        ct = crop_like(Path(meta["source_pet"]).with_name("CT_resample.nii.gz"),
                       bbox, cshape).astype(np.float32)
        body = ct > -500
        zs = np.flatnonzero(body.any(axis=(1, 2)))
        z_lo, z_hi = (int(zs[0]), int(zs[-1])) if zs.size else (0, cshape[0] - 1)
        del ct, body

        ts = TOTALSEG / f"{case}.nii.gz"
        lz: dict[str, float] = {}
        if ts.exists():
            organ = crop_like(ts, bbox, cshape)
            for name, lab in LANDMARKS.items():
                m = np.isin(organ, list(lab)) if isinstance(lab, tuple) else organ == lab
                if m.any():
                    lz[name] = float(np.flatnonzero(m.any(axis=(1, 2))).mean())
                    cover[name] += 1
            del organ
        if "brain" in lz and "bladder" in lz:
            brain_minus_bladder.append(lz["brain"] - lz["bladder"])

        for r in by_case[case]:
            cz = float(r["centroid_z"])
            for name in LANDMARKS:
                r[f"dz_{name}_mm"] = (f"{(cz - lz[name]) * sz:.2f}"
                                      if name in lz else "")
            r["edge_body_mm"] = f"{min(cz - z_lo, z_hi - cz) * sz:.2f}"
            r["edge_scan_mm"] = f"{min(cz, cshape[0] - 1 - cz) * sz:.2f}"
        if i % 100 == 0 or i == len(cases):
            print(f"  {i}/{len(cases)}", flush=True)

    print(f"\nlandmark coverage over {len(cases)} cases:")
    for n in LANDMARKS:
        print(f"  {n:9s} {cover[n]:4d} ({cover[n]/len(cases):.1%})")
    if brain_minus_bladder:
        m = float(np.mean(brain_minus_bladder))
        print(f"\nbrain z index minus bladder z index: {m:+.1f} on average "
              f"({len(brain_minus_bladder)} cases with both)")
        print(f"  so a HIGHER z index means more "
              f"{'superior (toward the head)' if m > 0 else 'inferior (toward the feet)'}")

    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields + NEW)
        w.writeheader()
        w.writerows(rows)
    print(f"\n{len(rows)} rows + {len(NEW)} columns -> {args.out}")


if __name__ == "__main__":
    main()
