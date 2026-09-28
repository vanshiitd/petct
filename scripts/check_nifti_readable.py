#!/usr/bin/env python3
"""Verify that SimpleITK can read every NIfTI in a dataset, in isolated processes.

Both nnU-Net and this repo's training pipeline read through SimpleITK, and ITK
can fail by *segfaulting* (0xC0000005) rather than raising -- which takes the
whole sweep down with it and names no file. Observed here mid-run while the disk
was busy, on files nibabel reads without complaint.

So the reads happen in subprocesses. The fast path checks a chunk of files per
process; if a chunk dies, its files are re-checked one per process to name the
offender exactly. A file that cannot be read must be re-converted before
training, not discovered halfway through it.

    python scripts/check_nifti_readable.py --root D:\\data\\autopet_nifti
    python scripts/check_nifti_readable.py --root ... --jobs 8 --chunk 40
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

# Child mode: read each path given on stdin, printing it BEFORE the read so a
# native crash leaves the offending filename as the last line of output.
CHILD_FLAG = "--_read_files"


def child_main(paths: list[str]) -> int:
    import SimpleITK as sitk
    for p in paths:
        print(p, flush=True)
        img = sitk.ReadImage(p)
        _ = sitk.GetArrayViewFromImage(img).shape  # force the pixel buffer
    print("__ALL_OK__", flush=True)
    return 0


def run_chunk(paths: list[str]) -> tuple[list[str], list[tuple[str, str]]]:
    """Returns (files_verified, failures[(path, reason)])."""
    proc = subprocess.run(
        [sys.executable, str(Path(__file__).resolve()), CHILD_FLAG, *paths],
        capture_output=True, text=True,
    )
    lines = [ln.strip() for ln in proc.stdout.splitlines() if ln.strip()]
    if lines and lines[-1] == "__ALL_OK__":
        return paths, []

    # The chunk died. The last path printed is where it died, but re-check each
    # file on its own to be certain rather than inferring from ordering.
    if len(paths) == 1:
        reason = f"exit {proc.returncode}"
        if proc.returncode < 0 or proc.returncode > 0x80000000 - 1:
            reason += " (native crash)"
        err = (proc.stderr or "").strip().splitlines()
        if err:
            reason += f": {err[-1][:200]}"
        return [], [(paths[0], reason)]

    verified: list[str] = []
    failures: list[tuple[str, str]] = []
    for p in paths:
        ok, bad = run_chunk([p])
        verified += ok
        failures += bad
    return verified, failures


def main() -> None:
    if CHILD_FLAG in sys.argv:
        idx = sys.argv.index(CHILD_FLAG)
        sys.exit(child_main(sys.argv[idx + 1:]))

    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--jobs", type=int, default=6, help="parallel subprocesses")
    p.add_argument("--chunk", type=int, default=40, help="files checked per subprocess")
    p.add_argument("--pattern", default="*.nii.gz")
    args = p.parse_args()

    files = sorted(str(f) for f in args.root.rglob(args.pattern))
    if not files:
        raise SystemExit(f"No files matching {args.pattern} under {args.root}")
    print(f"Checking {len(files)} files with SimpleITK "
          f"({args.jobs} processes x {args.chunk} files per process)\n")

    chunks = [files[i:i + args.chunk] for i in range(0, len(files), args.chunk)]
    verified: list[str] = []
    failures: list[tuple[str, str]] = []
    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        for i, (ok, bad) in enumerate(pool.map(run_chunk, chunks), 1):
            verified += ok
            failures += bad
            print(f"  [{i}/{len(chunks)}] ok={len(verified)} failed={len(failures)}")

    print(f"\n=== Summary ===")
    print(f"readable : {len(verified)}/{len(files)}")
    print(f"FAILED   : {len(failures)}")
    for path, reason in failures:
        print(f"   {path}  --  {reason}")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
