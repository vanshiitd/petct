#!/usr/bin/env python3
"""Append a training subset to an nnU-Net `splits_final.json` as an extra fold.

nnU-Net names its output directory after the fold index, so running a
label-efficiency experiment as fold 1 keeps it from overwriting fold 0's
checkpoints, logs and validation predictions. The subset trains on fewer cases
but validates on *the same* 76 cases as fold 0, so the two are directly
comparable.

Fold 0 is rewritten byte-for-byte: the script compares its serialised form
before and after and refuses to write if anything moved. nnU-Net's own writer
uses `json.dump(..., indent=False, sort_keys=True)` via `save_json`, so the same
formatting is reproduced here rather than guessed.

    python scripts/add_subset_fold.py \\
        --splits-final C:\\nnunet_preprocessed\\Dataset505_.../splits_final.json \\
        --split splits/autopet_v1.json --subset 10pct
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--splits-final", type=Path, required=True)
    p.add_argument("--split", type=Path, required=True)
    p.add_argument("--subset", required=True, help="key under splits.train.subsets")
    p.add_argument("--base-fold", type=int, default=0,
                   help="fold whose validation list the new fold reuses")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    folds = json.loads(args.splits_final.read_text())
    before = [json.dumps(f, sort_keys=True) for f in folds]
    base = folds[args.base_fold]

    split = json.loads(args.split.read_text())
    subsets = split["splits"]["train"].get("subsets", {})
    if args.subset not in subsets:
        raise SystemExit(f"no subset '{args.subset}' in {args.split}; "
                         f"have {[k for k in subsets if k not in ('method', 'seed')]}")
    sub = subsets[args.subset]
    train = sorted(s for ss in sub["cases"].values() for s in ss)

    stray = set(train) - set(base["train"])
    if stray:
        raise SystemExit(f"{len(stray)} subset case(s) are not in fold "
                         f"{args.base_fold}'s training list: {sorted(stray)[:5]}")
    overlap = set(train) & set(base["val"])
    if overlap:
        raise SystemExit(f"{len(overlap)} subset case(s) are in the validation list")

    # Append. An earlier version spliced at `base_fold + 1`, which silently
    # dropped every fold after the base: adding a 30% subset with --base-fold 0
    # replaced the already-trained 10% fold 1, so the file no longer described
    # what fold 1 had trained on. `base_fold` selects the validation list to
    # reuse and nothing else.
    new_fold = {"train": train, "val": list(base["val"])}
    folds = folds + [new_fold]

    print(f"fold {args.base_fold}: {len(base['train'])} train / {len(base['val'])} val")
    print(f"fold {len(folds) - 1}: {len(new_fold['train'])} train / {len(new_fold['val'])} val"
          f"  (subset '{args.subset}', {sub['n_patients']} patients, "
          f"{sub['n_tumour_positive_scans']} tumour-positive scans)")
    print(f"validation identical to fold {args.base_fold}: "
          f"{new_fold['val'] == base['val']}")

    after = [json.dumps(f, sort_keys=True) for f in folds[: len(before)]]
    if after != before:
        changed = [i for i, (x, y) in enumerate(zip(before, after)) if x != y]
        raise SystemExit(f"existing fold(s) {changed} changed; refusing to write")
    if len(folds) != len(before) + 1:
        raise SystemExit(f"expected {len(before) + 1} folds, built {len(folds)}")
    print(f"all {len(before)} existing fold(s) unchanged: OK")

    if args.dry_run:
        print("\n--dry-run: splits_final.json not modified")
        return
    # match nnU-Net's own save_json formatting
    args.splits_final.write_text(json.dumps(folds, sort_keys=True, indent=False))
    print(f"\nwritten to {args.splits_final}")


if __name__ == "__main__":
    main()
