#!/usr/bin/env python3
"""Phase XXII development: the deployed agent's action table for split A.

Development only. A is the adaptation split; it certifies and tests nothing.
The reference on A is the hand-built agent on the zero-shot maps (frozen
Phase IV policy, FDG stage-1 detector at 0.90), i.e. Phase XIX's `hb` arm.
The Level-1 certified update differs from it by a few added actions and its
detectors were fitted on A itself, so the hand-built agent is the reference
that is free of A's labels.

Writes phase22_psma_hb_A.npz (new file) and hard-links A's maps into
D:\\data\\cv\\psma_A and C:\\psma_maps\\ft_A.
"""
from __future__ import annotations

import collections
import json
import os
import pickle
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from exp3_detect import matrix
from phase11_experience_gate import R, write_table
from phase4_diag_model import PRED_CLASSES
from phase16_combined import union
from phase19_selfimprove import SPLIT, rows_of

T = "psma"


def main() -> None:
    A = set(json.loads(SPLIT.read_text())["A"])
    D = pickle.load(open(R / "phase4_diag_model.pkl", "rb"))
    Dp, Dr = D["pred"], D["recover"]
    rs = rows_of(R / f"cand_{T}_ctx.csv", "pred", A)
    ks = [(r["case"], int(r["component_id"])) for r in rs]
    nc = collections.Counter(r["case"] for r in rs)
    nr = np.array([nc[r["case"]] for r in rs])
    Xd = matrix(rs, Dp["feats"], Dp["median"])[0]
    Po = np.column_stack([Dp["models"][c].predict_proba(Xd)[:, 1] for c in PRED_CLASSES])
    rr = rows_of(R / f"cand_{T}_pos.csv", "recover", A)
    rk = [(r["case"], int(r["component_id"])) for r in rr]
    Xr = matrix(rr, Dr["feats"], Dr["median"])[0]
    prs = Dr["models"]["missed"].predict_proba(Xr)[:, 1] if rr else np.zeros(0)
    allA = np.ones(len(ks), bool)
    P, act, added = union(None, Po, Po, Po, nr > 1, allA)
    write_table(R / f"phase22_{T}_hb_A.npz", ks, P, act, rk, prs)
    print(f"A: {len(A)} cases, {len(ks)} components, acted {int(act.sum())}, "
          f"added {int(added.sum())}, recovery candidates {len(rk)}")
    for src, dst in ((Path(r"D:\data\cv\psma"), Path(r"D:\data\cv\psma_A")),
                     (Path(r"C:\psma_maps\ft"), Path(r"C:\psma_maps\ft_A"))):
        dst.mkdir(exist_ok=True)
        for c in sorted(A):
            if not (dst / f"{c}.npz").exists():
                os.link(src / f"{c}.npz", dst / f"{c}.npz")
        print(dst, len(list(dst.glob("*.npz"))))


if __name__ == "__main__":
    main()
