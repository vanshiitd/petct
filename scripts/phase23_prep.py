#!/usr/bin/env python3
"""Phase XXIII: prepare the follow-up studies of B and C patients.

See `reports/phase23_followup_confirmation_plan.md`. Same steps as
`phase18_psma_prep.py` (resample onto the FDG grid, export through
`export_case`), written to new directories on D: so nothing earlier changes.

Writes reports/cases_psma_fu.txt and D:\\data\\reports\\phase23_prep.json.
"""
from __future__ import annotations

import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import phase18_psma_prep as p18
from export_nnunet_maeprep import export_case

REPO = Path(__file__).resolve().parent.parent
p18.DST = Path(r"D:\data\psma_fdggrid_fu")
p18.DS = Path(r"D:\data\nnunet_raw_fu\Dataset505_AutoPET_MAEprep_v2")


def followups() -> list[str]:
    sp = json.loads((REPO / "reports" / "phase19_split.json").read_text())
    led = [json.loads(l) for l in open(p18.SRC / "ledger.jsonl", encoding="utf-8")]
    ok = sorted({r["case"] for r in led if r["status"] == "ok"})
    earliest = set(p18.earliest_per_patient())
    bc = {c.rsplit("_", 2)[0] for c in sp["B"] + sp["C"]}
    assert earliest >= set(sp["B"] + sp["C"])
    return [c for c in ok if c not in earliest and c.rsplit("_", 2)[0] in bc]


def prep(case: str) -> dict:
    try:
        return p18.prep(case) or {"case": case, "status": "ok"}
    except Exception as e:                           # noqa: BLE001 - reported, not hidden
        return {"case": case, "status": "error", "detail": repr(e)[:300]}


def main() -> None:
    cases = followups()
    pats = {c.rsplit("_", 2)[0] for c in cases}
    print(f"{len(cases)} follow-up studies of {len(pats)} patients", flush=True)
    assert (len(cases), len(pats)) == (143, 111), "cohort differs from the plan"
    (REPO / "reports" / "cases_psma_fu.txt").write_text("\n".join(cases) + "\n")
    for sub in ("imagesTs", "labelsTs_prep", "inverse"):
        (p18.DS / sub).mkdir(parents=True, exist_ok=True)
    rows = []
    with ProcessPoolExecutor(max_workers=6, initializer=_init) as pool:
        for i, r in enumerate(pool.map(prep, cases, chunksize=2), 1):
            rows.append(r)
            if r.get("status") not in ("ok", "already"):
                print(f"  {r['case']}: {r.get('status')} {r.get('detail', '')}", flush=True)
            if i % 25 == 0:
                print(f"  {i}/{len(cases)}", flush=True)
    st = {}
    for r in rows:
        st[r.get("status")] = st.get(r.get("status"), 0) + 1
    print(f"done {st}")
    Path(r"D:\data\reports\phase23_prep.json").write_text(json.dumps(rows, indent=1))


def _init() -> None:
    # worker processes re-import phase18_psma_prep; point them at the new dirs too
    p18.DST = Path(r"D:\data\psma_fdggrid_fu")
    p18.DS = Path(r"D:\data\nnunet_raw_fu\Dataset505_AutoPET_MAEprep_v2")


if __name__ == "__main__":
    main()
