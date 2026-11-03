#!/usr/bin/env python3
"""Phase XI: the experience-trained action gate (EAG).

See `reports/phase11_experience_gate_plan.md`. The agent learns *when* to act
from the measured outcomes of its own past actions: a harm model, trained on
whether the hand-built pick deleted a lesion, with case context the
per-component detector never had. The strategy choice stays the hand-built
policy's. A component is acted on iff its max-class probability >= g and its
predicted harm <= h; (g, h) is selected on training folds only.

Deployment is a new `--pfail-table` npz -- acted rows keep their stage-1
probabilities, every other row is zeroed -- run with `--min-pfail 1e-6`.
`phase5_agent.py` is not modified.

Writes, all new files:
  phase11_eag_oof.npz        honest out-of-fold decisions (outer-fold nesting)
  phase11_eag_{val,test}.npz production decisions
  phase11_eag_hb_oof.npz     the hand-built point (g=0.90, h=1.00), for the
                             smoke equivalence check
  phase11_eag.json           selection, held-out numbers, calibration
"""
from __future__ import annotations

import argparse
import collections
import csv
import json
import pickle
import sys
from pathlib import Path

import numpy as np
from scipy.stats import beta
from sklearn.ensemble import HistGradientBoostingClassifier

sys.path.insert(0, str(Path(__file__).resolve().parent))
from exp3_detect import load, matrix, GROUPS
from phase4_diag_model import PRED_CLASSES, DROP, ADD, labels_from_diag
from q_model import patient_of

R = Path(r"D:\data\reports")
SPLITS = Path(r"C:\nnunet_preprocessed\Dataset505_AutoPET_MAEprep_v2\splits_final.json")
FOLDS = (3, 4, 5, 6, 7)
# unchanged from phase4_diag_model.fit_report and audit_crossfit
CLF = dict(random_state=0, max_depth=3, learning_rate=0.08, max_iter=200,
           early_stopping=False)
BASE_GATE = 0.90
G_GRID = [0.90, 0.80, 0.70, 0.60, 0.50]
H_GRID = [0.02, 0.05, 0.10, 0.20, 1.00]
FIT_FLOOR = 0.50          # harm model is fitted where some gate could act
LOST_FRACTION = 0.5       # constraint (a)
CTX = ["ctx_min_pmax", "ctx_n_conf", "ctx_n_comp", "ctx_vol_rank",
       "ctx_suv_rel", "ctx_dz_conf"]


# ---------------------------------------------------------------- data ----
def features() -> list[str]:
    feats = [f for f in GROUPS["d all"] if f not in DROP]
    return feats + [f for f in ADD if f not in feats]


def measured(tables: list[Path]) -> dict:
    """(case, cid) -> {strategy: row} for predicted components."""
    meas = collections.defaultdict(dict)
    for t in tables:
        for r in csv.DictReader(open(t, newline="", encoding="utf-8")):
            if r["strategy"] == "recover":
                continue
            try:
                cid = int(r["component_id"])
            except ValueError:
                continue
            meas[(r["case"], abs(cid))][r["strategy"]] = r
    return meas


def outcome(meas: dict, key, cls: str, policy: dict):
    """Measured outcome of the hand-built pick for class `cls`, as in
    `phase9_class_gates.load_components`; None if there is none."""
    strats = meas.get(key)
    if not strats:
        return None
    pick = next((s for s in policy.get(cls, []) if s in strats), None)
    if pick is None:
        return None
    r = strats[pick]
    try:
        yq = float(r["d_qdice"])
    except ValueError:
        return None
    if not np.isfinite(yq):
        return None
    dd = float(r["d_dice"]) if r["d_dice"] not in ("", "nan") else np.nan
    pos = str(r["gt_positive"]).strip().lower() in ("1", "true", "1.0")
    lost = float(r["lesions_lost"] or 0)
    harm = lost > 0 or (pos and np.isfinite(dd) and dd < -0.01)
    return {"y": yq, "harm": bool(harm), "pos": pos, "lost": lost}


# ------------------------------------------------------------- stage 1 ----
def fit_stage1(X, Y, idx):
    models = []
    for j in range(Y.shape[1]):
        y = Y[idx, j]
        models.append(HistGradientBoostingClassifier(**CLF).fit(X[idx], y)
                      if y.sum() >= 10 else None)
    return models


def predict_stage1(models, X):
    return np.column_stack([m.predict_proba(X)[:, 1] if m is not None
                            else np.zeros(len(X)) for m in models])


def stage1_crossfit(X, Y, fold, within):
    """Class probabilities for rows whose fold is in `within`, each scored by
    models fitted on the other folds of `within`."""
    P = np.full((len(X), Y.shape[1]), np.nan)
    for j in within:
        tr = np.isin(fold, [k for k in within if k != j])
        te = fold == j
        P[te] = predict_stage1(fit_stage1(X, Y, np.where(tr)[0]), X[te])
    return P


# ------------------------------------------------------------- context ----
def context(rows: list[dict], P: np.ndarray) -> np.ndarray:
    """Six case-context features from the *other* components' stage-1 P."""
    pmax = P.max(1)
    by_case = collections.defaultdict(list)
    for i, r in enumerate(rows):
        by_case[r["case"]].append(i)

    def f(r, k):
        v = r.get(k, "")
        return float(v) if v not in ("", "nan") else np.nan

    C = np.full((len(rows), len(CTX)), np.nan)
    for idx in by_case.values():
        vol = np.array([f(rows[i], "volume_ml") for i in idx])
        suv = np.array([f(rows[i], "suv_max") for i in idx])
        cz = np.array([f(rows[i], "centroid_z") for i in idx])
        pm = pmax[idx]
        order = np.argsort(np.argsort(vol))
        for a, i in enumerate(idx):
            oth = [b for b in range(len(idx)) if b != a]
            C[i, 2] = len(idx)
            C[i, 3] = order[a] / max(len(idx) - 1, 1)
            if not oth:
                C[i, 0], C[i, 1] = 1.0, 0
                continue
            po = pm[oth]
            C[i, 0] = po.min()
            C[i, 1] = int((po < 0.3).sum())
            conf = [oth[t] for t in range(len(oth)) if po[t] < 0.5]
            if conf:
                smax = np.nanmax(suv[conf])
                C[i, 4] = suv[a] / smax if smax and np.isfinite(smax) else np.nan
                C[i, 5] = np.nanmin(np.abs(cz[conf] - cz[a]))
    return C


# ---------------------------------------------------------- harm model ----
def harm_design(X, P, C):
    return np.hstack([X, P, C])


def labels_for(keys, P, meas, policy):
    """Measured outcome of the hand-built pick for each row's argmax class."""
    out = []
    for k, p in zip(keys, P):
        out.append(outcome(meas, k, PRED_CLASSES[int(np.argmax(p))], policy)
                   if np.all(np.isfinite(p)) else None)
    return out


def fit_harm(Z, outs, pmax, idx):
    tr = [i for i in idx if outs[i] is not None and pmax[i] >= FIT_FLOOR]
    y = np.array([outs[i]["harm"] for i in tr], float)
    return HistGradientBoostingClassifier(**CLF).fit(Z[tr], y)


def harm_crossfit(Z, outs, pmax, fold, within):
    ph = np.full(len(Z), np.nan)
    for j in within:
        tr = np.where(np.isin(fold, [k for k in within if k != j]))[0]
        te = fold == j
        ph[te] = fit_harm(Z, outs, pmax, tr).predict_proba(Z[te])[:, 1]
    return ph


# ----------------------------------------------------------- selection ----
def evaluate(idx, outs, pmax, ph, g, h):
    acted = [i for i in idx if outs[i] is not None and pmax[i] >= g and ph[i] <= h]
    return {"n": len(acted),
            "y": float(sum(outs[i]["y"] for i in acted)),
            "lost": float(sum(outs[i]["lost"] for i in acted)),
            "y_neg": float(sum(outs[i]["y"] for i in acted if not outs[i]["pos"])),
            "harm": int(sum(outs[i]["harm"] for i in acted))}


def select(idx, outs, pmax, ph):
    hb = evaluate(idx, outs, pmax, np.zeros(len(pmax)), BASE_GATE, 1.0)
    best, table = None, []
    for g in G_GRID:
        for h in H_GRID:
            e = evaluate(idx, outs, pmax, ph, g, h)
            ok = (e["lost"] <= LOST_FRACTION * hb["lost"] and e["y_neg"] >= hb["y_neg"])
            table.append({"g": g, "h": h, **e, "ok": ok})
            if ok:
                key = (e["y"], g, -h)
                if best is None or key > best[0]:
                    best = (key, g, h)
    g, h = (best[1], best[2]) if best else (BASE_GATE, 1.0)
    return g, h, hb, table


# --------------------------------------------------------- calibration ----
def calibration(p, y, bins=10):
    ok = np.isfinite(p)
    p, y = p[ok], y[ok].astype(float)
    if len(p) == 0 or y.sum() == 0:
        return None
    edges = np.quantile(p, np.linspace(0, 1, bins + 1))
    b = np.clip(np.searchsorted(edges, p, side="right") - 1, 0, bins - 1)
    rel, ece = [], 0.0
    for k in range(bins):
        m = b == k
        if m.any():
            rel.append({"p_mean": float(p[m].mean()), "obs": float(y[m].mean()),
                        "n": int(m.sum())})
            ece += m.mean() * abs(p[m].mean() - y[m].mean())
    return {"ece": float(ece), "brier": float(np.mean((p - y) ** 2)),
            "n": int(len(p)), "prevalence": float(y.mean()), "reliability": rel}


def cp_ub(k, n):
    return 1.0 if n == 0 or k >= n else float(beta.ppf(0.95, k + 1, n - k))


# ---------------------------------------------------------------- write ----
def write_table(path, keys, P, act, rec_keys, rec_p):
    Pw = np.where(act[:, None], P, 0.0)
    rec = np.where(rec_p >= BASE_GATE, rec_p, 0.0)
    np.savez_compressed(
        path, keys_case=np.array([k[0] for k in keys]),
        keys_cid=np.array([k[1] for k in keys], dtype=np.int64),
        P_crossfit=Pw, classes=np.array(PRED_CLASSES),
        rec_keys_case=np.array([k[0] for k in rec_keys]),
        rec_keys_cid=np.array([k[1] for k in rec_keys], dtype=np.int64),
        rec_crossfit=rec)
    print(f"  written {path}: {int(act.sum())} acted components, "
          f"{int((rec > 0).sum())} recovery candidates")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cand", type=Path, default=R / "cand_cv_oof_pos.csv")
    ap.add_argument("--diag", type=Path, default=R / "diag_oof.pkl")
    ap.add_argument("--crossfit", type=Path, default=R / "audit_crossfit.npz")
    ap.add_argument("--diagpkl", type=Path, default=R / "phase4_diag_model.pkl")
    ap.add_argument("--table", type=Path, nargs="+",
                    default=[R / "strategy_table_oof.csv", R / "strategy6_oof.csv"])
    args = ap.parse_args()

    policy = json.loads((R / "phase4_policy_noban.json").read_text())["policy"]
    sp = json.loads(SPLITS.read_text())
    owner = {s: k for k in FOLDS for s in sp[k]["val"]}
    feats = features()

    rows, _, _, _ = load(args.cand, "pred")
    X, med, _ = matrix(rows, feats, None)
    keys = [(r["case"], int(r["component_id"])) for r in rows]
    fold = np.array([owner.get(r["case"], -1) for r in rows])
    lab, _ = labels_from_diag(args.diag)
    Y = np.array([[lab.get(k, {}).get(c, 0) for c in PRED_CLASSES] for k in keys], float)
    meas = measured(args.table)

    z = np.load(args.crossfit, allow_pickle=True)
    Pcf_map = {(str(c), int(i)): p for c, i, p in
               zip(z["keys_case"], z["keys_cid"], z["P_crossfit"])}
    P_cf = np.array([Pcf_map[k] for k in keys])
    rec_keys = [(str(c), int(i)) for c, i in zip(z["rec_keys_case"], z["rec_keys_cid"])]
    rec_cf = np.asarray(z["rec_crossfit"], float)
    print(f"{len(rows)} components, {len(feats)} features, "
          f"{int((P_cf.max(1) >= BASE_GATE).sum())} at the 0.90 gate")

    report = {"plan": "reports/phase11_experience_gate_plan.md",
              "grid": {"g": G_GRID, "h": H_GRID}, "outer": []}

    # ---- honest out-of-fold: outer-fold nesting ---------------------------
    act_oof = np.zeros(len(rows), bool)
    ph_oof = np.full(len(rows), np.nan)
    outs_oof = labels_for(keys, P_cf, meas, policy)
    for k in FOLDS:
        T = [f for f in FOLDS if f != k]
        tr_idx = np.where(np.isin(fold, T))[0]
        te = fold == k
        # 1. stage 1: nested inside T for T's rows; all of T for fold k
        P_in = stage1_crossfit(X, Y, fold, T)
        P_out = predict_stage1(fit_stage1(X, Y, tr_idx), X[te])
        dev = float(np.nanmax(np.abs(P_out - P_cf[te])))
        P_in[te] = P_out
        C = context(rows, P_in)
        Z = harm_design(X, P_in, C)
        pmax = P_in.max(1)
        outs = labels_for(keys, P_in, meas, policy)
        # 2. harm model: inner CV on T; fitted on all of T for fold k
        ph = harm_crossfit(Z, outs, pmax, fold, T)
        ph[te] = fit_harm(Z, outs, pmax, tr_idx).predict_proba(Z[te])[:, 1]
        # 3. selection on T only
        g, h, hb_tr, table = select(list(tr_idx), outs, pmax, ph)
        # 4. apply to fold k
        te_idx = list(np.where(te)[0])
        act_oof[te] = (pmax[te] >= g) & (ph[te] <= h)
        ph_oof[te] = ph[te]
        held = evaluate(te_idx, outs, pmax, ph, g, h)
        held_hb = evaluate(te_idx, outs, pmax, np.zeros(len(pmax)), BASE_GATE, 1.0)
        print(f"outer fold {k}: stage-1 max |P - P_crossfit| = {dev:.2e}; "
              f"selected g={g}, h={h}")
        print(f"   training folds: hand-built y={hb_tr['y']:+.2f} lost={hb_tr['lost']:.0f}"
              f"  -> chosen {next(t for t in table if t['g'] == g and t['h'] == h)}")
        print(f"   held-out:  EAG n={held['n']} y={held['y']:+.2f} lost={held['lost']:.0f} "
              f"y_neg={held['y_neg']:+.2f}   |  hand-built n={held_hb['n']} "
              f"y={held_hb['y']:+.2f} lost={held_hb['lost']:.0f} y_neg={held_hb['y_neg']:+.2f}")
        report["outer"].append({"fold": k, "g": g, "h": h, "stage1_dev": dev,
                                "train_handbuilt": hb_tr, "train_table": table,
                                "heldout": held, "heldout_handbuilt": held_hb})

    agg = {s: {m: sum(o[s][m] for o in report["outer"])
               for m in ("n", "y", "lost", "y_neg", "harm")}
           for s in ("heldout", "heldout_handbuilt")}
    report["heldout_total"] = agg
    print(f"\nheld-out total, component level: EAG {agg['heldout']}  "
          f"hand-built {agg['heldout_handbuilt']}")
    a, b = agg["heldout"], agg["heldout_handbuilt"]
    report["certified_harm"] = {
        "eag": {"harm": a["harm"], "n": a["n"], "rate": a["harm"] / max(a["n"], 1),
                "ub95": cp_ub(a["harm"], a["n"])},
        "handbuilt": {"harm": b["harm"], "n": b["n"], "rate": b["harm"] / max(b["n"], 1),
                      "ub95": cp_ub(b["harm"], b["n"])}}
    print(f"certified harm rate: EAG {report['certified_harm']['eag']}  "
          f"hand-built {report['certified_harm']['handbuilt']}")

    write_table(R / "phase11_eag_oof.npz", keys, P_cf, act_oof, rec_keys, rec_cf)
    write_table(R / "phase11_eag_hb_oof.npz", keys, P_cf,
                P_cf.max(1) >= BASE_GATE, rec_keys, rec_cf)

    # ---- calibration (cross-fitted) ----------------------------------------
    cal = {c: calibration(P_cf[:, j], Y[:, j]) for j, c in enumerate(PRED_CLASSES)}
    m = np.array([o is not None and pm >= FIT_FLOOR
                  for o, pm in zip(outs_oof, P_cf.max(1))])
    yh = np.array([o["harm"] if o is not None else 0 for o in outs_oof], float)
    cal["harm_model"] = calibration(ph_oof[m], yh[m])
    report["calibration"] = cal
    for c, v in cal.items():
        if v:
            print(f"  calibration {c:22s} ECE {v['ece']:.4f}  Brier {v['brier']:.4f}  "
                  f"prev {v['prevalence']:.4f}")

    # ---- production model (validation, test) -------------------------------
    pmax_cf = P_cf.max(1)
    C_cf = context(rows, P_cf)
    Z_cf = harm_design(X, P_cf, C_cf)
    ph_prod_cv = harm_crossfit(Z_cf, outs_oof, pmax_cf, fold, list(FOLDS))
    all_idx = list(np.where(fold >= 0)[0])
    g, h, hb_all, table = select(all_idx, outs_oof, pmax_cf, ph_prod_cv)
    harm_model = fit_harm(Z_cf, outs_oof, pmax_cf, all_idx)
    print(f"\nproduction: g={g}, h={h}  (all-OOF selection; hand-built {hb_all})")
    report["production"] = {"g": g, "h": h, "handbuilt": hb_all, "table": table}

    D = pickle.load(open(args.diagpkl, "rb"))
    Dp, Dr = D["pred"], D["recover"]
    assert Dp["feats"] == feats, "stage-1 features differ from the detector's"
    for split in ("val", "test"):
        cand = R / f"cand_cv_{split}_pos.csv"
        rs, _, _, _ = load(cand, "pred")
        Xs, _, _ = matrix(rs, Dp["feats"], Dp["median"])
        Ps = np.column_stack([Dp["models"][c].predict_proba(Xs)[:, 1] for c in PRED_CLASSES])
        Cs = context(rs, Ps)
        phs = harm_model.predict_proba(harm_design(Xs, Ps, Cs))[:, 1]
        acts = (Ps.max(1) >= g) & (phs <= h)
        rr, _, _, _ = load(cand, "recover")
        Xr, _, _ = matrix(rr, Dr["feats"], Dr["median"])
        prs = (Dr["models"]["missed"].predict_proba(Xr)[:, 1] if rr else np.zeros(0))
        write_table(R / f"phase11_eag_{split}.npz",
                    [(r["case"], int(r["component_id"])) for r in rs], Ps, acts,
                    [(r["case"], int(r["component_id"])) for r in rr], prs)
        report[f"production_{split}"] = {"n_comp": len(rs), "acted": int(acts.sum()),
                                         "handbuilt_gated": int((Ps.max(1) >= BASE_GATE).sum())}
        # the hand-built point for this split too, for the reference test run
        write_table(R / f"phase11_eag_hb_{split}.npz",
                    [(r["case"], int(r["component_id"])) for r in rs], Ps,
                    Ps.max(1) >= BASE_GATE,
                    [(r["case"], int(r["component_id"])) for r in rr], prs)

    (R / "phase11_eag.json").write_text(json.dumps(report, indent=1, default=float),
                                        encoding="utf-8")
    print(f"written {R / 'phase11_eag.json'}")


if __name__ == "__main__":
    main()
