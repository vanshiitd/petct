#!/usr/bin/env python3
"""Phase X: nested selection of the lesion-safety veto's thresholds.

See `reports/phase10_lesion_veto_plan.md`. For every gated (p_failure >= 0.90,
cross-fitted) component with a hand-built pick, the measured outcome of that
pick is known from the strategy tables. A veto configuration (t, c, v) blocks
the pick on a `spurious_*` component when it is interior (edge_body_mm > t),
small (volume < v mL), and the case has another component with p_failure < c.

Selection on training folds maximises lesions saved subject to giving up at
most 0.001 Dice per tumour-positive case and no tumour-free clearance.
"""
from __future__ import annotations

import collections
import csv
import itertools
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
R = Path(r"D:\data\reports")
FOLDS = (3, 4, 5, 6, 7)
GATE = 0.90
SPURIOUS = {"spurious_other", "spurious_physio", "spurious_liver_spleen"}
GRID = list(itertools.product([10, 20, 30, 50], [0.3, 0.5], [1, 5, 20, float("inf")]))
MAX_DICE_COST = 0.001


def load_rows():
    z = np.load(R / "audit_crossfit.npz", allow_pickle=True)
    classes = list(z["classes"])
    P = {(c, int(k)): p for c, k, p in zip(z["keys_case"], z["keys_cid"], z["P_crossfit"])}
    case_min = collections.defaultdict(lambda: 1.0)
    for (c, _), p in P.items():
        case_min[c] = min(case_min[c], float(p.max()))
    policy = json.loads((R / "phase4_policy_noban.json").read_text())["policy"]
    meas = collections.defaultdict(dict)
    for t in ("strategy_table_oof.csv", "strategy6_oof.csv"):
        for r in csv.DictReader(open(R / t, newline="", encoding="utf-8")):
            if r["strategy"] == "recover":
                continue
            try:
                meas[(r["case"], int(r["component_id"]))][r["strategy"]] = r
            except ValueError:
                pass
    cand = {(r["case"], int(r["component_id"])): r
            for r in csv.DictReader(open(R / "cand_cv_oof_pos.csv", encoding="utf-8"))
            if r["kind"] == "pred"}
    sp = json.loads(Path(
        r"C:\nnunet_preprocessed\Dataset505_AutoPET_MAEprep_v2\splits_final.json").read_text())
    owner = {s: k for k in FOLDS for s in sp[k]["val"]}
    base = {r["case"]: r for r in csv.DictReader(
        open(R / "au_oof_attr_policy_base.csv", encoding="utf-8"))}

    rows = []
    for k, p in P.items():
        j = int(np.argmax(p))
        cls, pf = classes[j], float(p[j])
        if pf < GATE or k not in meas or k not in cand:
            continue
        pick = next((s for s in policy.get(cls, []) if s in meas[k]), None)
        if pick is None:
            continue
        r = meas[k][pick]
        dd = float(r["d_dice"]) if r["d_dice"] not in ("", "nan") else np.nan
        pos = float(base[k[0]]["n_gt"]) > 0
        rows.append({"case": k[0], "fold": owner.get(k[0], -1), "cls": cls,
                     "edge": float(cand[k]["edge_body_mm"]),
                     "vol": float(cand[k]["volume_ml"]),
                     "case_min": case_min[k[0]], "pos": pos,
                     "y": float(r["d_qdice"]), "dd": dd,
                     "lost": float(r["lesions_lost"] or 0)})
    return rows


def vetoed(r, cfg):
    t, c, v = cfg
    return (r["cls"] in SPURIOUS and r["edge"] > t and r["vol"] < v
            and r["case_min"] < c)


def evaluate(rows, cfg):
    n_pos = len({r["case"] for r in rows if r["pos"]}) or 1
    V = [r for r in rows if vetoed(r, cfg)]
    saved = sum(r["lost"] for r in V)
    dice_cost = sum(r["y"] for r in V if r["pos"]) / n_pos
    tf_lost = sum(1 for r in V if not r["pos"] and r["y"] > 0)
    return {"vetoed": len(V), "lesions_saved": saved,
            "dice_cost_per_pos_case": dice_cost, "tumour_free_lost": tf_lost}


def select(rows):
    best = None
    for cfg in GRID:
        e = evaluate(rows, cfg)
        if e["dice_cost_per_pos_case"] > MAX_DICE_COST or e["tumour_free_lost"] > 0:
            continue
        key = (e["lesions_saved"], -e["vetoed"])
        if best is None or key > best[0]:
            best = (key, cfg, e)
    return (best[1], best[2]) if best else (None, None)


def main() -> None:
    rows = load_rows()
    print(f"{len(rows)} gated components with a hand-built pick "
          f"({sum(r['lost'] > 0 for r in rows)} delete a lesion)")
    outer = []
    for f in FOLDS:
        tr = [r for r in rows if r["fold"] >= 0 and r["fold"] != f]
        te = [r for r in rows if r["fold"] == f]
        cfg, etr = select(tr)
        lost_te = sum(r["lost"] for r in te)
        if cfg is None:
            # nothing certifiable on these training folds -> no veto
            ete = {"vetoed": 0, "lesions_saved": 0.0,
                   "dice_cost_per_pos_case": 0.0, "tumour_free_lost": 0}
            best_tf = min(GRID, key=lambda g: evaluate(tr, g)["tumour_free_lost"])
            print(f"outer fold {f}: no configuration passes on training folds "
                  f"(least tumour-free loss: {best_tf} -> {evaluate(tr, best_tf)}); no veto")
        else:
            ete = evaluate(te, cfg)
            print(f"outer fold {f}: cfg t={cfg[0]} c={cfg[1]} v={cfg[2]}  "
                  f"held-out: saved {ete['lesions_saved']:.0f}/{lost_te:.0f} lesions  "
                  f"dice cost/pos case {ete['dice_cost_per_pos_case']:+.5f}  "
                  f"tumour-free lost {ete['tumour_free_lost']}  vetoed {ete['vetoed']}")
        outer.append({"fold": f, "cfg": cfg, "train": etr, "heldout": ete,
                      "heldout_lesions_lost_total": lost_te})
    tot_saved = sum(o["heldout"]["lesions_saved"] for o in outer)
    tot_lost = sum(o["heldout_lesions_lost_total"] for o in outer)
    tot_tf = sum(o["heldout"]["tumour_free_lost"] for o in outer)
    print(f"\ncross-fitted: {tot_saved:.0f} of {tot_lost:.0f} lesion deletions prevented, "
          f"tumour-free clearances given up {tot_tf}")

    cfg, e = select([r for r in rows if r["fold"] >= 0])
    if cfg is None:
        print("\nproduction (all OOF): no configuration passes -> no veto certified")
    else:
        print(f"\nproduction cfg (all OOF): edge > {cfg[0]} mm, case min p_failure < "
              f"{cfg[1]}, volume < {cfg[2]} mL  ->  {e}")

    # LABELLED POST-HOC SENSITIVITY, not pre-registered and not confirmatory:
    # the same nested procedure with the cost ceiling the current Q-gated final
    # agent already accepts (0.0015 Dice per tumour-positive case).
    global MAX_DICE_COST
    pre = MAX_DICE_COST
    MAX_DICE_COST = 0.0015
    sens = []
    for f in FOLDS:
        tr = [r for r in rows if r["fold"] >= 0 and r["fold"] != f]
        te = [r for r in rows if r["fold"] == f]
        c2, _ = select(tr)
        ete = evaluate(te, c2) if c2 else {"vetoed": 0, "lesions_saved": 0.0,
                                            "dice_cost_per_pos_case": 0.0,
                                            "tumour_free_lost": 0}
        sens.append({"fold": f, "cfg": c2, "heldout": ete,
                     "lost": sum(r["lost"] for r in te),
                     "n_pos": len({r["case"] for r in te if r["pos"]})})
    MAX_DICE_COST = pre
    saved = sum(s["heldout"]["lesions_saved"] for s in sens)
    lost = sum(s["lost"] for s in sens)
    cost = (sum(s["heldout"]["dice_cost_per_pos_case"] * s["n_pos"] for s in sens)
            / sum(s["n_pos"] for s in sens))
    print(f"\n[post-hoc sensitivity, ceiling 0.0015] cross-fitted: {saved:.0f} of "
          f"{lost:.0f} lesion deletions prevented, Dice cost per tumour-positive case "
          f"{cost:+.5f}, tumour-free lost "
          f"{sum(s['heldout']['tumour_free_lost'] for s in sens)}  "
          f"cfgs {[s['cfg'] for s in sens]}")

    (R / "phase10_veto.json").write_text(json.dumps(
        {"certified": cfg is not None,
         "edge_mm": cfg[0] if cfg else None, "case_min_pfail": cfg[1] if cfg else None,
         "vol_ml": cfg[2] if cfg else None, "classes": sorted(SPURIOUS),
         "all_oof": e, "outer_folds": outer,
         "posthoc_sensitivity_0p0015": {"folds": sens, "saved": saved, "lost": lost,
                                         "dice_cost_per_pos_case": cost}},
        indent=1, default=float), encoding="utf-8")
    print(f"written {R / 'phase10_veto.json'}")


if __name__ == "__main__":
    main()
