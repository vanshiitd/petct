#!/usr/bin/env python3
"""Phase IV step 3 and 4: the rule-based agent (Algorithm 1) and its experience log.

Per case, with the policy of eq. 30 fixed from the out-of-fold strategy table:

    compute Q.  if Q > tau_Q, accept M0 untouched.
    otherwise, up to T times:
        score every component with P(F_k | O) -- no ground truth
        take the component with the strongest predicted failure
        walk that failure's strategy list in out-of-fold rank order:
            apply the strategy, recompute U-derived features and Q
            accept iff Delta-Q > 0, else revert and try the next strategy
        stop when nothing is accepted, or when |Delta-Q| < eps
    return the best mask seen, by Q.

Every attempt -- accepted or not -- becomes one experience row for Phase V
(plan section 13): patient, region, diagnosed failure, uncertainty context,
other context, action, ground-truth reward r = Delta-Dice - lambda * Delta-HD95
(eq. 29), internal reward Delta-Q (eq. 34), whether it was accepted, and the
iteration. Nothing is fitted on that log here.

The ground-truth reward is computed but never seen by the agent: it decides on
Q alone. Keeping both in the same row is the whole point -- Phase V needs to
learn where the two disagree, and step 3's "is Q a trustworthy judge" check
needs it too.
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
from scipy import ndimage as ndi

sys.path.insert(0, str(Path(__file__).resolve().parent))
from phase4_strategies import CaseContext, REGISTRY
from phase4_strategy_table import q_vector, CPU_STRATEGIES
from phase4_diag_model import PRED_CLASSES, DROP, ADD
from q_model import read_cand, patient_of
from exp3_detect import GROUPS, matrix, load
from evaluate_predictions import hd95_mm


def observable_index(path: Path, kind: str) -> dict:
    """{case: [raw candidate row, ...]} straight from the candidate CSV.

    The diagnosis model was fitted on `exp3_detect.load`, which keeps the whole
    CSV row. `q_model.read_cand` keeps only the handful of columns Q needs, so
    feeding Q's dict to the diagnosis model silently loses features. The two
    readers are therefore kept apart rather than shared.
    """
    rows, _, _, _ = load(path, kind)
    idx = {}
    for r in rows:
        idx.setdefault(r["case"], []).append(r)
    return idx


def observable_rows(obs: dict, case: str, feats: list[str], median: dict):
    raw = obs.get(case, [])
    if not raw:
        return [], None
    ids = [int(r["component_id"]) for r in raw]
    # built by exp3_detect.matrix, not by hand: it is the same function the
    # diagnosis model was fitted with, so log_volume and the median imputation
    # (median is a dict keyed by feature name) cannot drift apart from training.
    X, _, _ = matrix(raw, feats, median)
    return ids, X


class Agent:
    def __init__(self, policy, qmodel, diagmodel, tau_q, T, eps, lam,
                 split="oof", gate=True, min_pfail=0.0):
        self.policy = policy
        self.Q = qmodel
        self.D = diagmodel
        self.tau_q = tau_q
        self.T = T
        self.eps = eps
        self.lam = lam
        self.split = split
        self.gate = gate
        self.min_pfail = min_pfail

    def q_of(self, case, feats, cinfo):
        vec = q_vector(list(feats.values()), cinfo)
        if self.split == "oof":
            p = patient_of(case)
            for grp, m in self.Q["folds"]:
                if p in grp:
                    return float(m.predict(vec[None])[0])
        return float(self.Q["full"].predict(vec[None])[0])

    def run(self, ctx: CaseContext, cand, obs, ctxmap, with_truth=True):
        case = ctx.case
        feats = {cid: ctx.features(ctx.comp_slice(cid), ctx.labels[ctx.comp_slice(cid)] == cid)
                 for cid in ctx.comp_ids}
        cinfo = {"c_npred": ctxmap.get(case, {}).get("c_npred", 0),
                 "c_nrec": ctxmap.get(case, {}).get("c_nrec", 0),
                 "c_liver_suv": ctx.liver_suv,
                 "c_vol_m0_ml": sum(f["vol_ml"] for f in feats.values() if f)}
        q = self.q_of(case, feats, cinfo)
        trace = {"case": case, "q0": q, "iterations": 0, "accepted": 0,
                 "attempts": 0, "stopped": "tau_Q" if q > self.tau_q else ""}
        rows = []
        if q > self.tau_q:
            trace["q_final"] = q
            return q, trace, rows, self._truth(ctx) if with_truth else None

        md = self.D["pred"]
        ids, X = observable_rows(obs["pred"], case, md["feats"], md["median"])
        prob = {}
        if ids:
            for k in PRED_CLASSES:
                m = md["models"].get(k)
                prob[k] = (m.predict_proba(X)[:, 1] if m is not None
                           else np.zeros(len(ids)))

        # Recovery acts on the sub-threshold pool rather than on an existing
        # component, so it is scored by the separate "recover" head and keyed
        # ("rec", rid) to keep it apart from predicted component ids. Only ids
        # present in BOTH the candidate CSV and this context are used: the two
        # were built by different scripts and an id that exists in one but not
        # the other would silently pair a score with the wrong blob.
        pool = ctx.recovery_candidates()
        mr = self.D["recover"]
        rec_all, Xr = observable_rows(obs["recover"], case,
                                      mr["feats"], mr["median"])
        rec_ids, rprob = [], []
        if rec_all:
            pr = mr["models"]["missed"].predict_proba(Xr)[:, 1]
            for rid, s_ in zip(rec_all, pr):
                if rid in pool:
                    rec_ids.append(rid)
                    rprob.append(float(s_))

        truth0 = self._truth(ctx) if with_truth else None
        cur = truth0
        exhausted, best_q, best_state = set(), q, ctx.labels.copy()
        for t in range(1, self.T + 1):
            order = []
            for i, cid in enumerate(ids):
                if cid in exhausted or cid not in ctx.comp_ids:
                    continue
                pk = max(((prob[k][i], k) for k in PRED_CLASSES), default=(0.0, None))
                order.append((pk[0], pk[1], cid))
            for rid, pr in zip(rec_ids, rprob):
                if ("rec", rid) in exhausted:
                    continue
                order.append((pr, "missed", ("rec", rid)))
            order.sort(key=lambda t: (t[0], str(t[2])), reverse=True)
            # post-hoc guard (section 10): act only where the diagnostic head is
            # confident the component really is that failure. Chosen on OOF.
            if self.min_pfail > 0.0:
                order = [o for o in order if o[0] >= self.min_pfail]
            if not order:
                trace["stopped"] = trace["stopped"] or "no candidate"
                break
            pmax, fclass, cid = order[0]
            accepted_any = False
            is_rec = isinstance(cid, tuple)
            for strat in self.policy.get(fclass, []):
                if (strat == "recover") != is_rec:
                    continue
                if not is_rec and strat not in REGISTRY:
                    continue
                if is_rec:
                    sl, new_local = pool[cid[1]]
                    before = ctx.labels[sl].copy()
                    key = ctx.add_component(sl, new_local)
                    nf = ctx.features(sl, ctx.labels[sl] == key)
                    f2 = dict(feats)
                    if nf is not None:
                        f2[key] = nf
                else:
                    sl, new_local = REGISTRY[strat](ctx, cid)
                    if sl is None:
                        continue
                    before = ctx.labels[sl].copy()
                    old_local = before == cid
                    if bool((new_local == old_local).all()):
                        continue
                    ctx.set_component(cid, sl, new_local)
                    nf = ctx.features(sl, ctx.labels[sl] == cid)
                    f2 = dict(feats)
                    if nf is None:
                        f2.pop(cid, None)
                    else:
                        f2[cid] = nf
                    key = cid
                q2 = self.q_of(case, f2, cinfo)
                dq = q2 - q
                trace["attempts"] += 1
                row = {"case": case, "patient": patient_of(case),
                       "iteration": t, "component_id": cid,
                       "organ": int(nf.get("frac_physio", 0) > 0.5) if nf else 0,
                       "failure": fclass, "p_failure": float(pmax),
                       "action": strat,
                       "u_stab": float((nf or {}).get("ustab_mean", np.nan)) if is_rec else (float(feats[cid]["ustab_mean"]) if feats.get(cid) else np.nan),
                       "u_ent": float((nf or {}).get("uent_mean", np.nan)) if is_rec else (float(feats[cid]["uent_mean"]) if feats.get(cid) else np.nan),
                       "suv_max": float((nf or {}).get("suv_max", np.nan)) if is_rec else (float(feats[cid]["suv_max"]) if feats.get(cid) else np.nan),
                       "vol_ml": float((nf or {}).get("vol_ml", np.nan)) if is_rec else (float(feats[cid]["vol_ml"]) if feats.get(cid) else np.nan),
                       "q_before": q, "d_q": dq}
                if with_truth:
                    # measured against the state this action started from, not
                    # against M0: otherwise every row after the first accepted
                    # action reports the running total and the experience log
                    # cannot attribute an effect to the action that caused it.
                    tr = self._truth(ctx)
                    row["d_dice"] = tr["dice"] - cur["dice"]
                    row["d_hd95"] = tr["hd95"] - cur["hd95"]
                    row["reward"] = row["d_dice"] - self.lam * (
                        row["d_hd95"] if np.isfinite(row["d_hd95"]) else 0.0)
                    row["lesions_lost"] = cur["n_hit"] - tr["n_hit"]
                    row["dice_from_m0"] = tr["dice"] - truth0["dice"]
                keep = dq > 0 if self.gate else True
                row["accepted"] = int(keep)
                rows.append(row)
                if keep:
                    q, accepted_any = q2, True
                    feats = f2
                    if with_truth:
                        cur = tr
                    trace["accepted"] += 1
                    if q > best_q:
                        best_q, best_state = q, ctx.labels.copy()
                    break
                ctx.labels[sl] = before      # revert
            trace["iterations"] = t
            if with_truth:
                snap = self._truth(ctx)
                for k in ("dice", "fp_ml", "fn_ml", "hd95", "n_hit", "n_fp_comp"):
                    trace[f"m{t}_{k}"] = snap[k]
                trace[f"m{t}_q"] = q
            if not accepted_any:
                exhausted.add(("rec", cid[1]) if is_rec else cid)
                if len(exhausted) >= len(ids) + len(rprob):
                    trace["stopped"] = "all exhausted"
                    break
            elif abs(dq) < self.eps:
                trace["stopped"] = "eps"
                break
        ctx.labels = best_state
        trace["q_final"] = best_q
        return best_q, trace, rows, (self._truth(ctx) if with_truth else None)

    def _truth(self, ctx: CaseContext) -> dict:
        s = ctx.score()
        m = ctx.mask()
        s["hd95"] = (hd95_mm(m, ctx.gt, ctx.spacing)
                     if m.any() and ctx.gt.any() else float("nan"))
        return s


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--maps", type=Path, required=True)
    ap.add_argument("--cand", type=Path, required=True)
    ap.add_argument("--policy", type=Path, required=True)
    ap.add_argument("--qpkl", type=Path, default=Path(r"D:\data\reports\q_model.pkl"))
    ap.add_argument("--diagpkl", type=Path,
                    default=Path(r"D:\data\reports\phase4_diag_model.pkl"))
    ap.add_argument("--split", default="oof")
    ap.add_argument("--tau-q", type=float, required=True)
    ap.add_argument("--T", type=int, default=3)
    ap.add_argument("--eps", type=float, default=0.005)
    ap.add_argument("--lam", type=float, default=0.001)
    ap.add_argument("--no-gate", action="store_true",
                    help="accept every proposed action (the section 21 ablation)")
    ap.add_argument("--out-prefix", type=Path, required=True)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--min-pfail", type=float, default=0.0,
                    help="act only on proposals whose diagnostic probability is "
                         "at least this (post-hoc guard, section 10)")
    args = ap.parse_args()

    policy = json.loads(args.policy.read_text())["policy"]
    Qm = pickle.load(open(args.qpkl, "rb"))
    Dm = pickle.load(open(args.diagpkl, "rb"))
    cand, ctxmap = read_cand(args.cand)
    obs = {k: observable_index(args.cand, k) for k in ("pred", "recover")}
    agent = Agent(policy, Qm, Dm, args.tau_q, args.T, args.eps, args.lam,
                  split=args.split, gate=not args.no_gate,
                  min_pfail=args.min_pfail)

    cases = sorted(f.stem for f in args.maps.glob("*.npz"))
    if args.limit:
        cases = cases[: args.limit]
    print(f"{len(cases)} cases | tau_Q {args.tau_q} T {args.T} eps {args.eps} "
          f"| gate {'off' if args.no_gate else 'on'} "
          f"| min_pfail {args.min_pfail}", flush=True)

    traces, exp, base_rows, final_rows = [], [], [], []
    t0 = time.time()
    for i, case in enumerate(cases, 1):
        ctx = CaseContext(case, args.maps)
        b = agent._truth(ctx)
        b["case"] = case
        base_rows.append(b)
        _, tr, rows, fin = agent.run(ctx, cand, obs, ctxmap)
        tr["seconds"] = time.time() - t0
        traces.append(tr)
        exp.extend(rows)
        fin["case"] = case
        final_rows.append(fin)
        if i % 25 == 0 or i == len(cases):
            el = time.time() - t0
            print(f"  {i}/{len(cases)}  {el/i:.1f} s/case  "
                  f"{len(exp)} experience rows", flush=True)

    pre = args.out_prefix
    pre.parent.mkdir(parents=True, exist_ok=True)
    for nm, data in (("experience", exp), ("traces", traces),
                     ("base", base_rows), ("final", final_rows)):
        if not data:
            continue
        keys = sorted({k for r in data for k in r})
        with open(pre.with_name(pre.name + f"_{nm}.csv"), "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=keys)
            w.writeheader()
            w.writerows(data)
    print(f"\nwritten {pre}_*.csv")

    def mean(rows, k):
        v = [r[k] for r in rows if r.get(k) == r.get(k)]
        return float(np.mean(v)) if v else float("nan")
    pos0 = [r for r in base_rows if r["n_gt"] > 0]
    pos1 = [r for r in final_rows if r["n_gt"] > 0]
    print(f"\n{'':10s}{'Dice':>9s}{'FP mL':>9s}{'FN mL':>9s}{'HD95':>9s}{'hits':>8s}")
    for nm, rs in (("M0", pos0), ("agent", pos1)):
        print(f"{nm:10s}{mean(rs,'dice'):9.4f}{mean(rs,'fp_ml'):9.3f}"
              f"{mean(rs,'fn_ml'):9.3f}{mean(rs,'hd95'):9.2f}"
              f"{sum(r['n_hit'] for r in rs):8d}")
    it = [t["iterations"] for t in traces]
    print(f"\niterations/case mean {np.mean(it):.2f}, "
          f"attempts {sum(t['attempts'] for t in traces)}, "
          f"accepted {sum(t['accepted'] for t in traces)}, "
          f"gated out at tau_Q {sum(1 for t in traces if t['stopped']=='tau_Q')}")
    print(f"runtime {(time.time()-t0)/max(len(cases),1):.1f} s/case")


if __name__ == "__main__":
    main()
