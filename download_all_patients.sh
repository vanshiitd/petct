#!/usr/bin/env bash
# Downloads CT + PET + SEG for all patients in TCIA's FDG-PET-CT-Lesions
# collection (~419GB total, ~900 patients). Safe to stop (Ctrl+C) and rerun --
# it skips any patient already fully downloaded.
#
# Requires: curl, jq
#
# Usage:
#   ./download_all_patients.sh /path/to/output/dir

set -uo pipefail

API="https://services.cancerimagingarchive.net/nbia-api/services/v1"
COLLECTION="FDG-PET-CT-Lesions"
OUT_DIR="${1:-./AutoPET_Full}"
LOG_FILE="$OUT_DIR/download_log.txt"

mkdir -p "$OUT_DIR"

log() {
    printf '[%s] %s\n' "$(date +%H:%M:%S)" "$1" | tee -a "$LOG_FILE"
}

log "Fetching patient list for $COLLECTION..."
patients_json=$(curl -sS -m 60 "$API/getPatient?Collection=$COLLECTION")
patient_ids=()
while IFS= read -r line; do
    patient_ids+=("$line")
done < <(echo "$patients_json" | jq -r '.[].PatientId')
total=${#patient_ids[@]}
log "Found $total patients."

i=0
for patient_id in "${patient_ids[@]}"; do
    i=$((i + 1))
    patient_dir="$OUT_DIR/$patient_id"
    done_marker="$patient_dir/.done"

    if [ -f "$done_marker" ]; then
        log "[$i/$total] $patient_id: already done, skipping"
        continue
    fi

    log "[$i/$total] $patient_id: fetching series list..."
    series_json=$(curl -sS -m 60 "$API/getSeries?Collection=$COLLECTION&PatientID=$patient_id")
    if [ -z "$series_json" ] || [ "$series_json" = "null" ]; then
        log "  ERROR fetching series list for $patient_id"
        continue
    fi

    mkdir -p "$patient_dir"
    all_ok=true

    for tag in CT PT SEG; do
        dest_zip="$patient_dir/$tag.zip"
        if [ -f "$dest_zip" ]; then
            continue  # already downloaded this one
        fi

        if [ "$tag" = "CT" ]; then
            # multiple CT series sometimes exist -- take the one with the most images
            uid=$(echo "$series_json" | jq -r '[.[] | select(.Modality=="CT")] | sort_by(.ImageCount) | reverse | .[0].SeriesInstanceUID // empty')
            size_mb=$(echo "$series_json" | jq -r '[.[] | select(.Modality=="CT")] | sort_by(.ImageCount) | reverse | .[0].FileSize // 0' )
        else
            uid=$(echo "$series_json" | jq -r --arg m "$tag" '[.[] | select(.Modality==$m)][0].SeriesInstanceUID // empty')
            size_mb=$(echo "$series_json" | jq -r --arg m "$tag" '[.[] | select(.Modality==$m)][0].FileSize // 0')
        fi

        if [ -z "$uid" ]; then
            log "  SKIP $patient_id: missing $tag series"
            all_ok=false
            continue
        fi

        size_mb=$(( size_mb / 1000000 ))
        log "  downloading $tag (~${size_mb}MB)..."
        if ! curl -sS -m 900 -o "$dest_zip" "$API/getImage?SeriesInstanceUID=$uid"; then
            log "  ERROR downloading $tag for $patient_id"
            rm -f "$dest_zip"
            all_ok=false
        fi
    done

    if [ "$all_ok" = true ]; then
        touch "$done_marker"
        log "  OK $patient_id"
    fi
done

log "DONE. Check $LOG_FILE for any SKIP/ERROR lines to retry individually."
