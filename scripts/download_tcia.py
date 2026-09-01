#!/usr/bin/env python3
"""Download the FDG-PET-CT-Lesions (AutoPET) collection from TCIA.

Fetches CT + PET + segmentation for every patient as raw DICOM zip archives.
The full collection is roughly 419 GB across ~900 patients and takes many
hours; it is designed to be interrupted and resumed.

Standard library only -- no pip installs, and no shell-script execution
policy to fight on Windows. Runs identically on Windows, macOS, Linux and an
HPC login node.

Resumable: a patient is marked .done only when all three series are present
and verified, so rerunning the same command picks up where it stopped.
Each archive downloads to a .part file and is checked for corruption before
being kept, so an interrupted transfer cannot masquerade as a finished one.

Examples:
    python scripts/download_tcia.py /data/AutoPET_raw
    python scripts/download_tcia.py /data/AutoPET_raw --limit 20      # trial run
    python scripts/download_tcia.py /data/AutoPET_raw --retries 5

Behind a proxy (e.g. a campus network), either export HTTP_PROXY / HTTPS_PROXY,
which this script picks up automatically, or pass it explicitly:
    python scripts/download_tcia.py /data/AutoPET_raw --proxy proxy.example.ac.in:3128

To test connectivity before starting a multi-hour download:
    python scripts/download_tcia.py --check

Afterwards, convert to the layout the training pipeline expects:
    python scripts/dicom_to_nifti.py --source /data/AutoPET_raw --target /data/autopet_nifti
"""
from __future__ import annotations

import argparse
import json
import socket
import sys
import time
import urllib.error
import urllib.request
import zipfile
from collections import defaultdict
from pathlib import Path

socket.setdefaulttimeout(300)

API = "https://services.cancerimagingarchive.net/nbia-api/services/v1"
COLLECTION = "FDG-PET-CT-Lesions"
MODALITIES = ("CT", "PT", "SEG")

_EXPLICIT_PROXY: str | None = None


def install_proxy(proxy: str | None) -> None:
    """Route every request through `proxy`.

    urlretrieve() shares urlopen()'s global opener, so installing here covers
    the bulk archive downloads as well as the small JSON calls. With no --proxy
    the default opener is left alone, and it reads HTTP_PROXY / HTTPS_PROXY
    from the environment by itself.
    """
    global _EXPLICIT_PROXY
    if not proxy:
        return
    if "://" not in proxy:
        proxy = "http://" + proxy  # a bare host:port is the common way to write it
    handler = urllib.request.ProxyHandler({"http": proxy, "https": proxy})
    urllib.request.install_opener(urllib.request.build_opener(handler))
    _EXPLICIT_PROXY = proxy


def network_hint(e: BaseException) -> str:
    """Turn a connection failure into the specific thing to go and fix."""
    if _EXPLICIT_PROXY:
        active = f"{_EXPLICIT_PROXY} (from --proxy)"
    else:
        env = urllib.request.getproxies()
        active = (", ".join(f"{k}={v}" for k, v in env.items()) + " (from environment)"
                  if env else "none")
    text = str(e)

    if isinstance(e, urllib.error.HTTPError) and e.code == 407:
        why = ("The proxy requires authentication (HTTP 407). Include your "
               "credentials: --proxy http://USER:PASS@host:3128 "
               "(percent-encode any @ : / in the password).")
    elif isinstance(e, (TimeoutError, socket.timeout)) or "timed out" in text:
        why = ("The connection timed out -- typically no route to the internet "
               "without a proxy. Pass --proxy host:port, or export HTTPS_PROXY.")
    elif "Name or service not known" in text or "nodename nor servname" in text \
            or "getaddrinfo" in text:
        why = ("DNS could not resolve the host. If you set a proxy, check the "
               "proxy's own hostname is spelled correctly and is reachable.")
    elif "refused" in text:
        why = ("The connection was refused. The proxy address or port is "
               "probably wrong, or the proxy is not listening.")
    elif "CERTIFICATE_VERIFY_FAILED" in text:
        why = ("TLS verification failed. A proxy that intercepts HTTPS needs "
               "its CA certificate installed; ask your network admin for it. "
               "Do not disable certificate checking to work around this.")
    else:
        why = "Check that this machine can reach the public internet."

    return f"{why}\n  Proxy currently in effect: {active}"


def api_get_json(path: str, **params):
    qs = "&".join(f"{k}={v}" for k, v in params.items())
    url = f"{API}/{path}?{qs}" if qs else f"{API}/{path}"
    with urllib.request.urlopen(url, timeout=60) as r:
        return json.loads(r.read())


def download_series(uid: str, dest: Path, retries: int, log) -> bool:
    """Fetch one series to `dest`, verifying the archive before keeping it."""
    url = f"{API}/getImage?SeriesInstanceUID={uid}"
    tmp = dest.with_suffix(".part")

    for attempt in range(1, retries + 1):
        try:
            urllib.request.urlretrieve(url, tmp)
            # a truncated transfer produces a readable file but a broken zip;
            # check before promoting it, or resume would treat it as complete
            with zipfile.ZipFile(tmp) as zf:
                bad = zf.testzip()
            if bad is not None:
                raise RuntimeError(f"corrupt entry in archive: {bad}")
            tmp.replace(dest)
            return True
        except Exception as e:
            tmp.unlink(missing_ok=True)
            if attempt == retries:
                log(f"    failed after {retries} attempts: {e}")
                return False
            wait = 2 ** attempt
            log(f"    attempt {attempt}/{retries} failed ({e}); retrying in {wait}s")
            time.sleep(wait)
    return False


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("out_dir", nargs="?", default="./AutoPET_raw", type=Path,
                   help="where to write the download (default: ./AutoPET_raw)")
    p.add_argument("--collection", default=COLLECTION, help="TCIA collection name")
    p.add_argument("--limit", type=int, default=None,
                   help="download at most N patients (useful for a trial run)")
    p.add_argument("--retries", type=int, default=3, help="retries per series")
    p.add_argument("--proxy", default=None, metavar="HOST:PORT",
                   help="proxy to route all traffic through, e.g. proxy61.iitd.ac.in:3128 "
                        "or http://USER:PASS@host:3128. Overrides HTTP_PROXY/HTTPS_PROXY.")
    p.add_argument("--check", action="store_true",
                   help="test the connection to TCIA and exit, without downloading")
    p.add_argument("--sleep", type=float, default=0.0,
                   help="seconds to pause between patients, to be polite to the API")
    args = p.parse_args()
    install_proxy(args.proxy)

    if args.check:
        print(f"Contacting {API} …")
        try:
            n = len(api_get_json("getPatient", Collection=args.collection))
        except Exception as e:
            raise SystemExit(f"FAILED: {e}\n\n{network_hint(e)}")
        print(f"OK — reached TCIA and listed {n} patients in {args.collection}.")
        print("The connection works; rerun without --check to start downloading.")
        return

    out_dir: Path = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    log_file = out_dir / "download_log.txt"

    def log(msg: str) -> None:
        line = f"[{time.strftime('%H:%M:%S')}] {msg}"
        print(line, flush=True)
        with open(log_file, "a") as f:
            f.write(line + "\n")

    log(f"Fetching patient list for {args.collection}…")
    try:
        patients = api_get_json("getPatient", Collection=args.collection)
    except Exception as e:
        raise SystemExit(f"Could not reach TCIA: {e}\n\n{network_hint(e)}")

    ids = [p["PatientId"] for p in patients]
    if args.limit:
        ids = ids[:args.limit]
        log(f"--limit {args.limit}: downloading the first {len(ids)} patients only")
    total = len(ids)
    log(f"{total} patients to process. Output: {out_dir.resolve()}")

    counts = defaultdict(int)
    for i, patient_id in enumerate(ids, 1):
        patient_dir = out_dir / patient_id
        done_marker = patient_dir / ".done"

        if done_marker.exists():
            counts["already"] += 1
            continue

        log(f"[{i}/{total}] {patient_id}")
        try:
            series = api_get_json("getSeries", Collection=args.collection, PatientID=patient_id)
        except Exception as e:
            log(f"  error fetching series list: {e}")
            counts["error"] += 1
            continue

        patient_dir.mkdir(parents=True, exist_ok=True)
        by_mod = {m: [s for s in series if s["Modality"] == m] for m in MODALITIES}
        all_ok = True

        for mod in MODALITIES:
            candidates = by_mod[mod]
            if not candidates:
                log(f"  skip: no {mod} series")
                all_ok = False
                continue

            # multiple CT reconstructions can exist; take the most slices
            chosen = max(candidates, key=lambda s: int(s.get("ImageCount", 0)))
            dest = patient_dir / f"{mod}.zip"
            if dest.exists():
                continue

            size_mb = chosen.get("FileSize", 0) / 1e6
            log(f"  {mod}: {size_mb:.0f} MB")
            if not download_series(chosen["SeriesInstanceUID"], dest, args.retries, log):
                all_ok = False

        if all_ok:
            done_marker.touch()
            counts["ok"] += 1
        else:
            counts["partial"] += 1

        if args.sleep:
            time.sleep(args.sleep)

    log(f"DONE. complete={counts['ok']} already={counts['already']} "
        f"partial={counts['partial']} errors={counts['error']}")
    if counts["partial"] or counts["error"]:
        log("Rerun the same command to retry the incomplete patients.")
    log(f"\nNext: python scripts/dicom_to_nifti.py --source {out_dir} --target <nifti_dir>")


if __name__ == "__main__":
    main()
