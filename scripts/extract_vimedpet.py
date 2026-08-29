#!/usr/bin/env python3
"""Batch-extract the ViMedPET multi-part zip archives.

Replaces: preprocessing/ViMedPETproc.ipynb

ViMedPET ships as split archives (.zip + .z01, .z02, ...). Only the .zip is
passed to 7-Zip; it pulls in the companion parts automatically. Python's
zipfile cannot read split archives, which is why this shells out to 7z.

Example:
    python scripts/extract_vimedpet.py --source /data/ViMedPET --target /data/unzipViMedPET
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--source", type=Path, required=True, help="directory containing the .zip archives")
    p.add_argument("--target", type=Path, required=True, help="extraction destination")
    p.add_argument("--sevenzip", default="7z", help="path to the 7z executable")
    args = p.parse_args()

    if not args.source.exists():
        raise SystemExit(f"Source directory does not exist: {args.source}")
    if shutil.which(args.sevenzip) is None:
        raise SystemExit(
            f"'{args.sevenzip}' not found on PATH. Install p7zip "
            f"(Linux: apt install p7zip-full | macOS: brew install p7zip)."
        )

    args.target.mkdir(parents=True, exist_ok=True)
    zip_files = sorted(args.source.rglob("*.zip"))
    if not zip_files:
        raise SystemExit(f"No .zip archives found under {args.source}")

    print(f"Found {len(zip_files)} archives. Extracting to {args.target}\n")

    n_ok = n_err = 0
    for zf in zip_files:
        print(f"Extracting {zf.name}...")
        cmd = [args.sevenzip, "x", str(zf), f"-o{args.target}", "-y"]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0:
            print(f"  -> ok")
            n_ok += 1
        else:
            print(f"  -> FAILED (exit {result.returncode})")
            if result.stderr.strip():
                print(f"     {result.stderr.strip().splitlines()[-1]}")
            n_err += 1

    print(f"\nDone. extracted={n_ok} failed={n_err}")
    if n_err:
        sys.exit(1)


if __name__ == "__main__":
    main()
