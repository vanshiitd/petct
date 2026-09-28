#!/usr/bin/env python3
"""Inventory the collection's studies from TCIA metadata -- no image data.

Answers the questions that decide what has to be re-downloaded:

  * which patients have more than one study
  * which studies are *complete* (CT + PT + SEG all present)
  * how big a re-download of a given patient set would be
  * which patients the old "largest series per modality, across all studies"
    rule mixes, and which of those decisions hinged on an ImageCount tie

One `getSeries` call fetches the whole collection's series metadata, so this is
a single small request, not 900 of them. Nothing is downloaded.

    python scripts/tcia_study_inventory.py --out inventory.json
    python scripts/tcia_study_inventory.py --audit-csv pairing_audit.csv
"""
from __future__ import annotations

import argparse
import json
import socket
import urllib.request
from collections import defaultdict
from pathlib import Path

socket.setdefaulttimeout(300)

API = "https://services.cancerimagingarchive.net/nbia-api/services/v1"
COLLECTION = "FDG-PET-CT-Lesions"
MODALITIES = ("CT", "PT", "SEG")


def install_proxy(proxy: str | None) -> None:
    if not proxy:
        return
    if "://" not in proxy:
        proxy = "http://" + proxy
    handler = urllib.request.ProxyHandler({"http": proxy, "https": proxy})
    urllib.request.install_opener(urllib.request.build_opener(handler))


def api_get_json(path: str, **params):
    qs = "&".join(f"{k}={v}" for k, v in params.items())
    url = f"{API}/{path}?{qs}" if qs else f"{API}/{path}"
    with urllib.request.urlopen(url, timeout=120) as r:
        return json.loads(r.read())


def old_selection(series: list[dict]) -> dict[str, dict]:
    """Reproduce download_tcia.py's pre-fix choice: per modality, the series
    with the most images, chosen across *all* of the patient's studies."""
    chosen = {}
    for mod in MODALITIES:
        candidates = [s for s in series if s.get("Modality") == mod]
        if candidates:
            # max() returns the FIRST maximal element -- so with equal
            # ImageCounts the winner depends on API response order
            chosen[mod] = max(candidates, key=lambda s: int(s.get("ImageCount", 0)))
    return chosen


def had_tie(series: list[dict], mod: str) -> list[dict]:
    """Series of this modality tied for the maximum ImageCount (len > 1 = tie)."""
    candidates = [s for s in series if s.get("Modality") == mod]
    if not candidates:
        return []
    top = max(int(s.get("ImageCount", 0)) for s in candidates)
    return [s for s in candidates if int(s.get("ImageCount", 0)) == top]


def gb(n_bytes: float) -> float:
    return n_bytes / 1e9


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--collection", default=COLLECTION)
    p.add_argument("--proxy", default=None, metavar="HOST:PORT")
    p.add_argument("--out", type=Path, default=None, help="write the inventory as JSON")
    p.add_argument("--audit-csv", type=Path, default=None,
                   help="cross-check this check_study_pairing.py CSV against the metadata")
    args = p.parse_args()
    install_proxy(args.proxy)

    print(f"Fetching series metadata for {args.collection} …")
    series = api_get_json("getSeries", Collection=args.collection)
    print(f"  {len(series)} series returned\n")

    by_patient: dict[str, list[dict]] = defaultdict(list)
    for s in series:
        by_patient[s["PatientID"]].append(s)

    studies: dict[str, dict[str, list[dict]]] = {}
    for pid, rows in by_patient.items():
        per_study: dict[str, list[dict]] = defaultdict(list)
        for s in rows:
            per_study[s["StudyInstanceUID"]].append(s)
        studies[pid] = dict(per_study)

    def complete_studies(pid: str) -> list[str]:
        return [uid for uid, rows in studies[pid].items()
                if {r.get("Modality") for r in rows} >= set(MODALITIES)]

    multi = sorted(pid for pid in studies if len(studies[pid]) > 1)
    print(f"patients            : {len(studies)}")
    print(f"studies             : {sum(len(v) for v in studies.values())}")
    print(f"multi-study patients: {len(multi)}")

    # which patients does the OLD rule mix?
    mixed, tie_evidence = [], {}
    for pid in studies:
        chosen = old_selection(by_patient[pid])
        uids = {s["StudyInstanceUID"] for s in chosen.values()}
        if len(chosen) == len(MODALITIES) and len(uids) > 1:
            mixed.append(pid)
            ties = {mod: had_tie(by_patient[pid], mod) for mod in MODALITIES}
            ties = {m: v for m, v in ties.items() if len(v) > 1}
            if ties:
                tie_evidence[pid] = ties
    mixed.sort()
    print(f"mixed by the OLD rule (metadata simulation): {len(mixed)}")
    print(f"  of which the choice hinged on an ImageCount tie: {len(tie_evidence)}")

    # sizes
    def study_bytes(pid: str, uid: str) -> float:
        return sum(float(r.get("FileSize", 0)) for r in studies[pid][uid])

    multi_complete_bytes = sum(study_bytes(pid, uid) for pid in multi
                               for uid in complete_studies(pid))
    multi_complete_count = sum(len(complete_studies(pid)) for pid in multi)
    print(f"\ncomplete studies belonging to multi-study patients: {multi_complete_count}"
          f"  ({gb(multi_complete_bytes):.1f} GB)")

    if args.audit_csv:
        import csv
        rows = list(csv.DictReader(open(args.audit_csv, encoding="utf-8")))
        audited_mismatch = sorted(r["patient"] for r in rows if r["mismatch"] == "True")
        multiset = set(multi)
        not_multi = [p for p in audited_mismatch if p not in multiset]
        print(f"\n=== cross-check against {args.audit_csv.name} ===")
        print(f"audited mismatches            : {len(audited_mismatch)}")
        print(f"  that ARE multi-study in TCIA: {len(audited_mismatch) - len(not_multi)}")
        print(f"  that are SINGLE-study       : {len(not_multi)} {not_multi if not_multi else '(none)'}")
        extra = sorted(set(audited_mismatch) - set(mixed))
        missed = sorted(set(mixed) - set(audited_mismatch))
        print(f"on disk but not predicted by the simulation: {len(extra)} {extra}")
        print(f"predicted but not seen on disk            : {len(missed)} {missed}")
        print("\ntie evidence for the on-disk-only patients:")
        for pid in extra[:5]:
            ev = tie_evidence.get(pid)
            print(f"  {pid}: {'TIE -> ' + ', '.join(f'{m} x{len(v)}' for m, v in ev.items()) if ev else 'no tie'}")

    if args.out:
        payload = {
            "collection": args.collection,
            "n_patients": len(studies),
            "n_studies": sum(len(v) for v in studies.values()),
            "multi_study_patients": multi,
            "mixed_by_old_rule": mixed,
            "tie_patients": sorted(tie_evidence),
            "patients": {
                pid: {
                    "studies": {
                        uid: {
                            "modalities": sorted({r.get("Modality") for r in rows}),
                            "complete": {r.get("Modality") for r in rows} >= set(MODALITIES),
                            "bytes": sum(float(r.get("FileSize", 0)) for r in rows),
                            "study_date": next((r.get("StudyDate") for r in rows if r.get("StudyDate")), ""),
                            "series": [
                                {"uid": r["SeriesInstanceUID"], "modality": r.get("Modality"),
                                 "images": int(r.get("ImageCount", 0)),
                                 "bytes": float(r.get("FileSize", 0))}
                                for r in rows
                            ],
                        }
                        for uid, rows in studies[pid].items()
                    }
                }
                for pid in sorted(studies)
            },
        }
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(payload, indent=1), encoding="utf-8")
        print(f"\nInventory written to {args.out}")


if __name__ == "__main__":
    main()
