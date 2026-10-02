"""Run the autoPET organisers' own evaluation on our predictions.

Imports lab-midas/autoPET val_script.py unchanged and applies it case by case,
then reports its aggregates beside ours so any gap is attributable.

Known differences to look for:
  * connectivity: theirs is 18 (cc3d), ours is scipy's default 6
  * Dice on a case where both masks are empty: theirs is 0/0
  * the official metric set has no HD95 at all
"""
import sys, json, pathlib, warnings
from concurrent.futures import ProcessPoolExecutor
import numpy as np

sys.path.insert(0, r"D:\data\external\autoPET")
from val_script import compute_metrics            # organisers' code, unmodified

PRED = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else r"D:\data\nnunet\predictions\test_503_mapped")
GT = pathlib.Path(sys.argv[2] if len(sys.argv) > 2 else r"D:\data\nnunet\raw\Dataset501_AutoPET\labelsTs")
OUT = sys.argv[3] if len(sys.argv) > 3 else r"D:\data\reports\metric_parity_503_test.json"


def one(name):
    gt = GT / name
    pred = PRED / name
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")           # 0/0 on empty-empty cases
        dice, fp_vol, fn_vol = compute_metrics(gt, pred)
    import nibabel as nib
    g = nib.load(str(gt)).get_fdata()
    p = nib.load(str(pred)).get_fdata()
    return {"case": name.replace(".nii.gz", ""),
            "official_dice": float(dice), "official_fp_ml": float(fp_vol),
            "official_fn_ml": float(fn_vol),
            "gt_positive": bool(g.sum() > 0), "pred_positive": bool(p.sum() > 0)}


if __name__ == "__main__":
    names = sorted(f.name for f in PRED.glob("*.nii.gz"))
    print(f"running the organisers' val_script on {len(names)} cases")
    rows = []
    with ProcessPoolExecutor(max_workers=6) as pool:
        for i, r in enumerate(pool.map(one, names, chunksize=2), 1):
            rows.append(r)
            if i % 50 == 0:
                print(f"  {i}/{len(names)}", flush=True)

    d = np.array([r["official_dice"] for r in rows], float)
    fp = np.array([r["official_fp_ml"] for r in rows], float)
    fn = np.array([r["official_fn_ml"] for r in rows], float)
    pos = np.array([r["gt_positive"] for r in rows])
    predpos = np.array([r["pred_positive"] for r in rows])

    finite = np.isfinite(d)
    print(f"\ncases {len(rows)} | tumour-positive {int(pos.sum())} | tumour-free {int((~pos).sum())}")
    print(f"official Dice is NaN on {int((~finite).sum())} case(s) "
          f"(both masks empty -> 0/0)")

    print("\n=== organisers' metric ===")
    print(f"{'mean Dice over ALL cases (NaN dropped)':48s} {np.nanmean(d):.4f}")
    print(f"{'mean Dice over tumour-POSITIVE cases':48s} {np.nanmean(d[pos]):.4f}")
    print(f"{'median Dice over tumour-positive':48s} {np.nanmedian(d[pos]):.4f}")
    print(f"{'mean Dice, tumour-free cases':48s} {np.nanmean(d[~pos]):.4f}")
    print(f"{'mean FP volume, mL (all cases)':48s} {fp.mean():.4f}")
    print(f"{'mean FN volume, mL (all cases)':48s} {fn.mean():.4f}")
    print(f"{'tumour-free with any prediction':48s} {int((predpos & ~pos).sum())} of {int((~pos).sum())}")

    ours = json.load(open(r"D:\data\reports\ours_test_503.json")) if pathlib.Path(
        r"D:\data\reports\ours_test_503.json").exists() else None
    json.dump(rows, open(OUT, "w"), indent=1)
    print(f"\nper-case written to {OUT}")
