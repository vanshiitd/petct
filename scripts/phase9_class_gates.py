#!/usr/bin/env python3
"""Phase IX: certified class-specific gates over the unchanged hand-built policy.

See `reports/phase9_class_gates_plan.md`. The hand-built policy's strategy
choice is kept exactly; only the p_failure gate changes, per failure class,
one slice at a time below 0.90. A slice is admitted only if, on training
folds and with cross-fitted p_failure, the hand-built pick's mean `d_qdice`
has a patient-clustered bootstrap LCB above zero AND the harm rate (lesion
lost, or Dice drop on a tumour-positive case) has a 95% upper bound <= 2%.
The gate stops at the first slice that fails.

The fitted `GatedHandBuilt` decodes the same 45-column X as every earlier
`--avpkl` policy and drops into `phase5_agent.py` unchanged.
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

sys.path.insert(0, str(Path(__file__).resolve().parent))
from phase5_action_value import STRATEGIES
from phase4_diag_model import PRED_CLASSES
from q_model import patient_of

R = Path(r"D:\data\reports")
FOLDS = (3, 4, 5, 6, 7)
BASE_GATE = 0.90
SLICES = [(0.85, 0.90), (0.80, 0.85), (0.75, 0.80), (0.70, 0.75),
          (0.65, 0.70), (0.60, 0.65), (0.50, 0.60)]
MIN_N = 30
MAX_HARM_UB = 0.02
NBOOT = 2000
N_OBS = 26  # observable features ahead of the p_failure column in X


class GatedHandBuilt:
    """Hand-built scoring (`len(order) - order.index(strat)`), but -1 for a
    candidate whose p_failure is below its class's gate."""

    def __init__(self, policy: dict, gates: dict, cls_names: list[str]):
        self.policy = policy
        self.gates = gates
        self.cls_names = cls_names
        self.strategies = list(STRATEGIES)
        self.i_pf = N_OBS
        self.i_s0 = N_OBS + 1
        self.i_c0 = self.i_s0 + len(self.strategies)

    def predict(self, X: np.ndarray) -> np.ndarray:
        X = np.atleast_2d(X)
        out = np.empty(len(X))
        for i, row in enumerate(X):
            strat = self.strategies[int(np.argmax(row[self.i_s0:self.i_c0]))]
            c_oh = row[self.i_c0:]
            cls = "missed" if not c_oh.any() else self.cls_names[int(np.argmax(c_oh))]
            order = self.policy.get(cls, [])
            if strat not in order or row[self.i_pf] < self.gates.get(cls, BASE_GATE):
                out[i] = -1.0
            else:
                out[i] = len(order) - order.index(strat)
        return out


def load_components(policy: dict, tables: list[Path], crossfit: Path, splits: Path):
    """One row per component that has a hand-built pick: cross-fitted class
    and p_failure, and the measured outcome of that pick."""
    z = np.load(crossfit, allow_pickle=True)
    classes = list(z["classes"])
    pred = {(c, int(k)): p for c, k, p in zip(z["keys_case"], z["keys_cid"], z["P_crossfit"])}
    rec = {(c, int(k)): float(p) for c, k, p in
           zip(z["rec_keys_case"], z["rec_keys_cid"], z["rec_crossfit"])}

    meas = collections.defaultdict(dict)
    for t in tables:
        for r in csv.DictReader(open(t, newline="", encoding="utf-8")):
            try:
                cid = int(r["component_id"])
            except ValueError:
                continue
            is_rec = r["strategy"] == "recover"
            meas[(r["case"], abs(cid), is_rec)][r["strategy"]] = r

    sp = json.loads(splits.read_text())
    owner = {s: k for k in FOLDS for s in sp[k]["val"]}

    rows = []
    for (case, cid, is_rec), strats in meas.items():
        if is_rec:
            if (case, cid) not in rec:
                continue
            cls, pf = "missed", rec[(case, cid)]
        else:
            if (case, cid) not in pred:
                continue
            p = pred[(case, cid)]
            j = int(np.argmax(p))
            cls, pf = classes[j], float(p[j])
        order = policy.get(cls, [])
        pick = next((s for s in order if s in strats), None)
        if pick is None:
            continue
        r = strats[pick]
        try:
            yq = float(r["d_qdice"])
        except ValueError:
            continue
        if not np.isfinite(yq):
            continue
        dd = float(r["d_dice"]) if r["d_dice"] not in ("", "nan") else np.nan
        pos = str(r["gt_positive"]).strip().lower() in ("1", "true", "1.0")
        lost = float(r["lesions_lost"] or 0)
        harm = lost > 0 or (pos and np.isfinite(dd) and dd < -0.01)
        rows.append({"case": case, "patient": patient_of(case), "cls": cls,
                     "pf": pf, "pick": pick, "y": yq, "harm": bool(harm),
                     "fold": owner.get(case, -1)})
    return rows


def slice_stats(rows: list[dict], seed: int = 2026) -> dict:
    n = len(rows)
    if n == 0:
        return {"n": 0}
    by_p = collections.defaultdict(lambda: [0.0, 0])
    for r in rows:
        by_p[r["patient"]][0] += r["y"]
        by_p[r["patient"]][1] += 1
    s = np.array([v[0] for v in by_p.values()])
    c = np.array([v[1] for v in by_p.values()])
    rng = np.random.default_rng(seed)
    m = len(s)
    boot = np.empty(NBOOT)
    for i in range(NBOOT):
        ix = rng.integers(0, m, m)
        boot[i] = s[ix].sum() / c[ix].sum()
    k = sum(r["harm"] for r in rows)
    ub = 1.0 if k >= n else float(beta.ppf(0.95, k + 1, n - k))
    return {"n": n, "mean": float(s.sum() / c.sum()), "sum": float(s.sum()),
            "lcb": float(np.percentile(boot, 2.5)), "harm": int(k), "harm_ub": ub,
            "patients": m}


def select_gates(rows: list[dict]) -> tuple[dict, list]:
    gates, log = {}, []
    for cls in sorted({r["cls"] for r in rows}):
        g = BASE_GATE
        for lo, hi in SLICES:
            st = slice_stats([r for r in rows if r["cls"] == cls and lo <= r["pf"] < hi])
            ok = (st["n"] >= MIN_N and st["lcb"] > 0 and st["harm_ub"] <= MAX_HARM_UB)
            log.append({"cls": cls, "slice": [lo, hi], **st, "admitted": ok})
            if not ok:
                break
            g = lo
        gates[cls] = g
    return gates, log


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--table", type=Path, nargs="+",
                    default=[R / "strategy_table_oof.csv", R / "strategy6_oof.csv"])
    ap.add_argument("--crossfit", type=Path, default=R / "audit_crossfit.npz")
    ap.add_argument("--splits", type=Path,
                    default=Path(r"C:\nnunet_preprocessed\Dataset505_AutoPET_MAEprep_v2\splits_final.json"))
    ap.add_argument("--out", type=Path, default=R / "phase9_class_gates.pkl")
    ap.add_argument("--report", type=Path, default=R / "phase9_class_gates.json")
    args = ap.parse_args()

    policy = json.loads((R / "phase4_policy_noban.json").read_text())["policy"]
    rows = load_components(policy, args.table, args.crossfit, args.splits)
    print(f"{len(rows)} components with a hand-built pick, "
          f"{sum(r['fold'] < 0 for r in rows)} without a Phase IIIb fold")

    print("\ncross-fitted headroom of the hand-built pick, by band (all OOF):")
    for cls in ["spurious_other", "spurious_physio", "spurious_liver_spleen", "missed"]:
        for lo, hi in [(0.90, 1.01)] + SLICES:
            st = slice_stats([r for r in rows if r["cls"] == cls and lo <= r["pf"] < hi])
            if st["n"]:
                print(f"  {cls:22s} [{lo:.2f},{hi:.2f})  n={st['n']:4d}  "
                      f"mean={st['mean']:+.4f}  LCB={st['lcb']:+.4f}  "
                      f"harm={st['harm']:3d}  harm_ub={st['harm_ub']:.3f}")

    outer = []
    for f in FOLDS:
        tr = [r for r in rows if r["fold"] >= 0 and r["fold"] != f]
        te = [r for r in rows if r["fold"] == f]
        gates, _ = select_gates(tr)
        relaxed = {c: g for c, g in gates.items() if g < BASE_GATE}
        added = [r for r in te if r["pf"] < BASE_GATE and r["pf"] >= gates.get(r["cls"], BASE_GATE)]
        st = slice_stats(added) if added else {"n": 0}
        print(f"\nouter fold {f}: gates {relaxed or 'none relaxed'}")
        if st["n"]:
            print(f"  held-out newly gated: n={st['n']}  mean d_qdice={st['mean']:+.4f}  "
                  f"sum={st['sum']:+.2f}  harm={st['harm']}")
        outer.append({"fold": f, "gates": gates, "heldout_added": st})

    gates, log = select_gates([r for r in rows if r["fold"] >= 0])
    print(f"\nfinal gates (all OOF): { {c: g for c, g in gates.items() if g < BASE_GATE} }")
    for e in log:
        print(f"  {e['cls']:22s} [{e['slice'][0]:.2f},{e['slice'][1]:.2f})  n={e['n']:4d}  "
              f"LCB={e.get('lcb', float('nan')):+.4f}  harm={e.get('harm', 0)}  "
              f"harm_ub={e.get('harm_ub', float('nan')):.3f}  "
              f"{'ADMIT' if e['admitted'] else 'stop'}")

    # build from the imported module so the class pickles as
    # `phase9_class_gates.GatedHandBuilt`, never `__main__`
    import phase9_class_gates as mod
    model = mod.GatedHandBuilt(policy, gates, list(PRED_CLASSES))
    with open(args.out, "wb") as fh:
        pickle.dump({"model": model, "pred_classes": list(PRED_CLASSES)}, fh)
    min_gate = min(gates.values()) if gates else BASE_GATE
    args.report.write_text(json.dumps(
        {"gates": gates, "min_gate": min_gate, "selection_log": log,
         "outer_folds": outer}, indent=1, default=float), encoding="utf-8")
    print(f"\nwritten {args.out} and {args.report}; deploy with --min-pfail {min_gate}")


if __name__ == "__main__":
    main()
