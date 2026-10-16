#!/usr/bin/env python3
"""Phase V agent: learned action selection, no Q gate, roi_repredict in the loop.

The Phase IV loop picked a component by p_failure, then walked a fixed
strategy list for that component's predicted class, accepting an action iff
Delta-Q > 0. Phase IV falsified the last step: Q adjudicates single actions at
37.1% precision against a 60% pre-registered threshold.

Phase V replaces it per `reports/phase5_analysis_plan.md`:

    for t in 1..T:
        build every (component, strategy) pair whose p_failure >= min_pfail
        score each with the learned action-value model
        take the best; stop if its predicted value is below eps_hat
        apply it -- no Q, no revert

Q is still computed and logged, as a descriptive quantity only. It decides
nothing.

One action per component per case, deliberately. The 26 observable features come
from the candidate CSV, computed on M0, and recomputing them mid-loop means
re-running build_candidates on a modified mask. Acting once per component means
no action is ever scored from features its own predecessor invalidated. The cost
is that a component needing two edits gets one; that limitation is real and is
reported rather than hidden by stale features.
"""
from __future__ import annotations

import argparse
import csv
import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from phase4_strategies import CaseContext, REGISTRY
from phase4_strategy_table import q_vector
from phase4_agent import observable_index, observable_rows
from phase5_action_value import STRATEGIES
from q_model import read_cand, patient_of
from evaluate_predictions import hd95_mm

CPU_ONLY = [s for s in STRATEGIES if s != "roi_repredict"]


class Phase5Agent:
    def __init__(self, av, qmodel, diagmodel, T, eps_hat, min_pfail,
                 split="oof", policy=None, with_roi=False):
        self.av = av                      # None -> hand-built policy ordering
        self.Q = qmodel
        self.D = diagmodel
        self.T = T
        self.eps_hat = eps_hat
        self.min_pfail = min_pfail
        self.split = split
        self.policy = policy or {}
        self.with_roi = with_roi

    # ---- scoring ---------------------------------------------------------
    def value(self, x_obs, pfail, pcls, strat) -> float:
        """Predicted d_qdice for one (component, strategy) pair."""
        x = list(x_obs) + [pfail]
        x += [1.0 if s == strat else 0.0 for s in STRATEGIES]
        x += [1.0 if c == pcls else 0.0 for c in self.av["pred_classes"]]
        return float(self.av["model"].predict(np.array(x)[None])[0])

    def q_of(self, case, feats, cinfo) -> float:
        vec = q_vector(list(feats.values()), cinfo)
        if self.split == "oof":
            p = patient_of(case)
            for grp, m in self.Q["folds"]:
                if p in grp:
                    return float(m.predict(vec[None])[0])
        return float(self.Q["full"].predict(vec[None])[0])

    # ---- one case --------------------------------------------------------
    def run(self, case, maps, obs_pred, obs_rec, ctxmap, repredictor=None):
        ctx = CaseContext(case, maps)
        Dp, Dr = self.D["pred"], self.D["recover"]
        ids, X = observable_rows(obs_pred, case, Dp["feats"], Dp["median"])
        rids, Xr = observable_rows(obs_rec, case, Dr["feats"], Dr["median"])

        classes = list(Dp["models"].keys())
        P = (np.column_stack([Dp["models"][c].predict_proba(X)[:, 1]
                              for c in classes]) if ids else np.zeros((0, len(classes))))
        Pr = (Dr["models"]["missed"].predict_proba(Xr)[:, 1]
              if rids else np.zeros(0))

        pool = ctx.recovery_candidates() if rids else {}
        base_feats = {c: ctx.features(ctx.comp_slice(c),
                                      ctx.labels[ctx.comp_slice(c)] == c)
                      for c in ctx.comp_ids}
        cinfo = {"c_npred": ctxmap.get(case, {}).get("c_npred", 0),
                 "c_nrec": ctxmap.get(case, {}).get("c_nrec", 0),
                 "c_liver_suv": ctx.liver_suv,
                 "c_vol_m0_ml": sum(f["vol_ml"] for f in base_feats.values() if f)}

        strategies = list(CPU_ONLY)
        if self.with_roi and repredictor is not None:
            strategies.append("roi_repredict")

        # candidate pairs, fixed up front: one action per component
        cands = []
        for i, cid in enumerate(ids):
            j = int(np.argmax(P[i]))
            pf, pcls = float(P[i, j]), classes[j]
            if pf < self.min_pfail:
                continue
            for s in strategies:
                if s == "recover":
                    continue
                cands.append({"key": cid, "rec": False, "obs": X[i],
                              "pfail": pf, "cls": pcls, "strat": s})
        for i, rid in enumerate(rids):
            if rid not in pool or float(Pr[i]) < self.min_pfail:
                continue
            cands.append({"key": ("rec", rid), "rec": True, "obs": Xr[i],
                          "pfail": float(Pr[i]), "cls": "missed",
                          "strat": "recover"})

        if self.av is not None:
            for c in cands:
                c["v"] = self.value(c["obs"], c["pfail"], c["cls"], c["strat"])
        else:
            # hand-built fallback: the class list's rank order, best first
            for c in cands:
                order = self.policy.get(c["cls"], [])
                c["v"] = (len(order) - order.index(c["strat"])
                          if c["strat"] in order else -1.0)

        base = self._truth(ctx)
        q0 = self.q_of(case, base_feats, cinfo)
        trace = {"case": case, "q0": q0, "attempts": 0, "accepted": 0,
                 "iterations": 0, "stopped": ""}
        rows, used, cur, feats = [], set(), base, dict(base_feats)
        t0 = time.time()

        for t in range(1, self.T + 1):
            live = [c for c in cands
                    if c["key"] not in used and c["v"] > self.eps_hat
                    and (c["rec"] or c["key"] in ctx.comp_ids)]
            if not live:
                trace["stopped"] = trace["stopped"] or "no candidate"
                break
            best = max(live, key=lambda c: (c["v"], str(c["key"]), c["strat"]))
            trace["attempts"] += 1

            if best["rec"]:
                sl, new_local = pool[best["key"][1]]
                before = ctx.labels[sl].copy()
                key = ctx.add_component(sl, new_local)
            elif best["strat"] == "roi_repredict":
                sl, new_local = repredictor(ctx, best["key"])
                if sl is None:
                    used.add(best["key"])
                    continue
                before = ctx.labels[sl].copy()
                ctx.set_component(best["key"], sl, new_local)
                key = best["key"]
            else:
                sl, new_local = REGISTRY[best["strat"]](ctx, best["key"])
                if sl is None:
                    used.add(best["key"])
                    continue
                before = ctx.labels[sl].copy()
                if bool((new_local == (before == best["key"])).all()):
                    used.add(best["key"])
                    continue
                ctx.set_component(best["key"], sl, new_local)
                key = best["key"]

            nf = ctx.features(sl, ctx.labels[sl] == key)
            f2 = dict(feats)
            if nf is None:
                f2.pop(key, None)
            else:
                f2[key] = nf
            q2 = self.q_of(case, f2, cinfo)
            tr = self._truth(ctx)
            rows.append({
                "case": case, "patient": patient_of(case), "iteration": t,
                "component_id": best["key"][1] if best["rec"] else best["key"],
                "failure": best["cls"], "p_failure": best["pfail"],
                "action": best["strat"], "v_pred": best["v"],
                "d_dice": tr["dice"] - cur["dice"],
                "d_fp_ml": tr["fp_ml"] - cur["fp_ml"],
                "lesions_lost": cur["n_hit"] - tr["n_hit"],
                "d_q": q2 - self.q_of(case, feats, cinfo),
                "vol_ml": float((nf or {}).get("vol_ml", np.nan)),
                "accepted": 1,
            })
            trace["accepted"] += 1
            feats, cur = f2, tr
            used.add(best["key"])
            trace["iterations"] = t

        trace["seconds"] = time.time() - t0
        trace["q_final"] = self.q_of(case, feats, cinfo)
        return trace, rows, base, cur

    def _truth(self, ctx: CaseContext) -> dict:
        s = ctx.score()
        m = ctx.mask()
        s["hd95"] = (hd95_mm(m, ctx.gt, ctx.spacing)
                     if m.any() and ctx.gt.any() else float("nan"))
        return s


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--maps", type=Path, required=True)
    ap.add_argument("--cand", type=Path, required=True)
    ap.add_argument("--avpkl", type=Path, default=None,
                    help="learned action-value model; omit for the hand-built "
                         "policy ordering")
    ap.add_argument("--policy", type=Path, default=None)
    ap.add_argument("--qpkl", type=Path,
                    default=Path(r"D:\data\reports\q_model.pkl"))
    ap.add_argument("--diagpkl", type=Path,
                    default=Path(r"D:\data\reports\phase4_diag_model.pkl"))
    ap.add_argument("--split", default="oof")
    ap.add_argument("--T", type=int, default=10)
    ap.add_argument("--eps-hat", type=float, default=0.0)
    ap.add_argument("--min-pfail", type=float, default=0.90)
    ap.add_argument("--with-roi", action="store_true")
    ap.add_argument("--out-prefix", type=Path, required=True)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    av = pickle.load(open(args.avpkl, "rb")) if args.avpkl else None
    Q = pickle.load(open(args.qpkl, "rb"))
    D = pickle.load(open(args.diagpkl, "rb"))
    policy = (json.load(open(args.policy))["policy"] if args.policy else None)
    obs_pred = observable_index(args.cand, "pred")
    obs_rec = observable_index(args.cand, "recover")
    _, ctxmap = read_cand(args.cand)

    agent = Phase5Agent(av, Q, D, args.T, args.eps_hat, args.min_pfail,
                        split=args.split, policy=policy,
                        with_roi=args.with_roi)

    pre = get_predictors = None
    if args.with_roi:
        import os
        os.environ.setdefault("nnUNet_raw", r"C:\nnunet_raw")
        os.environ.setdefault("nnUNet_preprocessed", r"C:\nnunet_preprocessed")
        from nnunetv2.preprocessing.preprocessors.default_preprocessor import (
            DefaultPreprocessor)
        from phase4_strategy6 import build_repredictor
        from cv_predict import build_predictor
        pre = DefaultPreprocessor(verbose=False)
        cache = {}

        def get_predictors(folds):
            key = tuple(sorted(folds))
            if key not in cache:
                cache[key] = [build_predictor(f) for f in key]
            return cache[key]

    cases = sorted(f.stem for f in args.maps.glob("*.npz"))
    if args.limit:
        cases = cases[: args.limit]
    print(f"{len(cases)} cases | T {args.T} eps_hat {args.eps_hat} "
          f"min_pfail {args.min_pfail} | selector "
          f"{'learned' if av else 'hand-built'} | roi {args.with_roi}",
          flush=True)

    base_rows, final_rows, traces, exp = [], [], [], []
    t0 = time.time()
    for n, case in enumerate(cases, 1):
        rp = None
        if args.with_roi:
            from phase4_strategy6 import build_repredictor
            rp = build_repredictor(case, args.maps, args.split, pre,
                                   get_predictors)
        trace, rows, base, final = agent.run(case, args.maps, obs_pred,
                                             obs_rec, ctxmap, rp)
        base_rows.append({"case": case, **{k: base[k] for k in
                          ("dice", "fp_ml", "fn_ml", "hd95", "n_fp_comp",
                           "n_gt", "n_hit", "pred_ml")}})
        final_rows.append({"case": case, **{k: final[k] for k in
                           ("dice", "fp_ml", "fn_ml", "hd95", "n_fp_comp",
                            "n_gt", "n_hit", "pred_ml")}})
        traces.append(trace)
        exp.extend(rows)
        if n % 25 == 0 or n == len(cases):
            print(f"  {n}/{len(cases)}  {(time.time()-t0)/n:.1f} s/case  "
                  f"{len(exp)} experience rows", flush=True)

    def write(name, rows):
        if not rows:
            return
        f = args.out_prefix.with_name(args.out_prefix.name + f"_{name}.csv")
        with open(f, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=sorted(rows[0]))
            w.writeheader()
            w.writerows(rows)

    write("base", base_rows)
    write("final", final_rows)
    write("traces", traces)
    write("experience", exp)
    print(f"\nwritten {args.out_prefix}_*.csv")

    import numpy as _np
    def agg(rs, k):
        v = _np.array([r[k] for r in rs], float)
        return _np.nanmean(v)
    print(f"\n{'':12s}{'Dice':>9s}{'FP mL':>9s}{'FN mL':>9s}{'HD95':>9s}{'hits':>8s}")
    for lab, rs in (("M0", base_rows), ("agent", final_rows)):
        print(f"{lab:12s}{agg(rs,'dice'):9.4f}{agg(rs,'fp_ml'):9.3f}"
              f"{agg(rs,'fn_ml'):9.3f}{agg(rs,'hd95'):9.2f}"
              f"{sum(r['n_hit'] for r in rs):8.0f}")
    print(f"\nattempts {sum(t['attempts'] for t in traces)}, "
          f"accepted {sum(t['accepted'] for t in traces)}, "
          f"iterations/case mean "
          f"{_np.mean([t['iterations'] for t in traces]):.2f}")
    print(f"runtime {(time.time()-t0)/len(cases):.1f} s/case")


if __name__ == "__main__":
    main()
