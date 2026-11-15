#!/usr/bin/env python3
"""Phase XXIII: HB and P19 action tables for the follow-up studies.

See `reports/phase23_followup_confirmation_plan.md`. Re-creates the frozen
Phase XIX Level-1 detectors and calibrators with the Phase XIX code path (same
data, same seed), checks that they reproduce the locked C tables exactly, and
then applies them to the follow-up candidates. The certified configuration is
read from the locked `phase19_psma.json`, not re-selected. No outcome is read.

Writes phase23_psmafu_{hb,p19}.npz.
"""
from __future__ import annotations

import collections
import json
import pickle
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from exp3_detect import load, matrix
from phase11_experience_gate import R, features, fit_stage1, predict_stage1, write_table
from phase4_diag_model import PRED_CLASSES
from phase13_context_features import NEW
from phase13_detector import labels
from phase14_calibrated import cal_fit, cal_apply
from phase15_ood import fence
from phase16_combined import union
from phase19_selfimprove import SEED, NFOLD, SPLIT, rows_of, ab_labels

T = "psma"
FU = "psmafu"


def main() -> None:
    sp = json.loads(SPLIT.read_text())
    A, C = set(sp["A"]), set(sp["C"])
    fu_cases = set(Path(SPLIT.parent / "cases_psma_fu.txt").read_text().split())
    old, new = features(), features() + NEW
    frows, _, _, _ = load(R / "cand_cv_oof_ctx.csv", "pred")
    _, Yf = labels(frows, R / "diag_oof.pkl")
    X1f, med1, _ = matrix(frows, new, None)
    D = pickle.load(open(R / "phase4_diag_model.pkl", "rb"))
    Dp, Dr = D["pred"], D["recover"]
    lock = json.loads((R / f"phase19_{T}.json").read_text())
    cfg = tuple(lock["p19"])

    def part(ctx: Path, pos: Path, cases: set):
        rs = rows_of(ctx, "pred", cases)
        ks = [(r["case"], int(r["component_id"])) for r in rs]
        nc = collections.Counter(r["case"] for r in rs)
        nr = np.array([nc[r["case"]] for r in rs])
        Xd = matrix(rs, Dp["feats"], Dp["median"])[0]
        Po = np.column_stack([Dp["models"][c].predict_proba(Xd)[:, 1] for c in PRED_CLASSES])
        rr = rows_of(pos, "recover", cases)
        rk = [(r["case"], int(r["component_id"])) for r in rr]
        Xr = matrix(rr, Dr["feats"], Dr["median"])[0]
        prs = Dr["models"]["missed"].predict_proba(Xr)[:, 1] if rr else np.zeros(0)
        return {"rows": rs, "keys": ks, "nr": nr, "Po": Po, "X1": matrix(rs, new, med1)[0],
                "rk": rk, "prs": prs}

    pa = part(R / f"cand_{T}_ctx.csv", R / f"cand_{T}_pos.csv", A)
    pc = part(R / f"cand_{T}_ctx.csv", R / f"cand_{T}_pos.csv", C)
    check_only = "--check-only" in sys.argv
    pf = None if check_only else part(R / f"cand_{FU}_ctx.csv", R / f"cand_{FU}_pos.csv", fu_cases)
    Ya = ab_labels(R / f"diag_{T}.pkl", pa["keys"], A)

    # ---- adapt on A, exactly as phase19_selfimprove.main ----------------
    fz = fence(list(collections.Counter(r["case"] for r in pa["rows"]).values()))
    assert fz == lock["fence"], (fz, lock["fence"])
    indA = pa["nr"] <= fz
    Co = cal_fit(pa["Po"][indA], Ya[indA])
    acases = sorted(A)
    rng = np.random.default_rng(SEED)
    fold_of = {c: i % NFOLD for i, c in enumerate(rng.permutation(acases))}
    fa = np.array([fold_of[k[0]] for k in pa["keys"]])
    PnA = np.full_like(Ya, np.nan)
    for j in range(NFOLD):
        tr, te = fa != j, fa == j
        M = fit_stage1(np.vstack([X1f, pa["X1"][tr]]), np.vstack([Yf, Ya[tr]]),
                       np.arange(len(X1f) + int(tr.sum())))
        PnA[te] = predict_stage1(M, pa["X1"][te])
    Cn = cal_fit(PnA[indA], Ya[indA])
    Mn = fit_stage1(np.vstack([X1f, pa["X1"]]), np.vstack([Yf, Ya]),
                    np.arange(len(X1f) + len(Ya)))

    def tables(p):
        ind, solo = p["nr"] <= fz, p["nr"] > 1
        Po_cal = cal_apply(Co, p["Po"])
        Pn_cal = cal_apply(Cn, predict_stage1(Mn, p["X1"]))
        return {"hb": union(None, p["Po"], Po_cal, Pn_cal, solo, ind),
                "p19": union(cfg, p["Po"], Po_cal, Pn_cal, solo, ind)}

    # ---- reproduction check on C against the locked tables ---------------
    tc = tables(pc)
    for a in ("hb", "p19"):
        P, act, _ = tc[a]
        z = np.load(R / f"phase19_{T}_{a}_C.npz", allow_pickle=True)
        assert [(str(c), int(i)) for c, i in zip(z["keys_case"], z["keys_cid"])] == pc["keys"]
        dev = float(np.abs(np.where(act[:, None], P, 0.0) - z["P_crossfit"]).max())
        rec = np.where(pc["prs"] >= 0.9, pc["prs"], 0.0)
        drec = float(np.abs(rec - z["rec_crossfit"]).max()) if len(rec) else 0.0
        print(f"check C {a}: max |P - locked| = {dev:.2e}, recovery {drec:.2e}")
        assert dev < 1e-12 and drec < 1e-12, a
    print("reproduction check passed: re-created detectors equal the locked C tables")
    if check_only:
        return

    # ---- follow-ups -------------------------------------------------------
    tf = tables(pf)
    n_case = len({k[0] for k in pf["keys"]})
    print(f"follow-ups: {len(pf['keys'])} components in {n_case} studies; fence {fz}; "
          f"studies over the fence {int(sum(v > fz for v in collections.Counter(k[0] for k in pf['keys']).values()))}")
    for a in ("hb", "p19"):
        P, act, added = tf[a]
        write_table(R / f"phase23_{FU}_{a}.npz", pf["keys"], P, act, pf["rk"], pf["prs"])
        print(f"  {a}: acted {int(act.sum())}, added over HB {int(added.sum())}")


if __name__ == "__main__":
    main()
