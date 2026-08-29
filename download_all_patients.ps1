# Downloads CT + PET + SEG for all patients in TCIA's FDG-PET-CT-Lesions
# collection (~419GB total, ~900 patients). Safe to stop (Ctrl+C) and rerun —
# it skips any patient already fully downloaded.
#
# Usage:
#   .\download_all_patients.ps1 -OutDir "D:\AutoPET_Full"

param(
    [string]$OutDir = "D:\AutoPET_Full"
)

$ErrorActionPreference = "Stop"
$ApiBase = "https://services.cancerimagingarchive.net/nbia-api/services/v1"
$Collection = "FDG-PET-CT-Lesions"
$LogFile = Join-Path $OutDir "download_log.txt"

New-Item -ItemType Directory -Force -Path $OutDir | Out-Null

function Log($msg) {
    $line = "[{0}] {1}" -f (Get-Date -Format "HH:mm:ss"), $msg
    Write-Host $line
    Add-Content -Path $LogFile -Value $line
}

Log "Fetching patient list for $Collection..."
$patients = Invoke-RestMethod -Uri "$ApiBase/getPatient?Collection=$Collection"
Log "Found $($patients.Count) patients."

$i = 0
foreach ($p in $patients) {
    $i++
    $patientId = $p.PatientId
    $patientDir = Join-Path $OutDir $patientId
    $doneMarker = Join-Path $patientDir ".done"

    if (Test-Path $doneMarker) {
        Log "[$i/$($patients.Count)] $patientId : already done, skipping"
        continue
    }

    Log "[$i/$($patients.Count)] $patientId : fetching series list..."
    try {
        $series = Invoke-RestMethod -Uri "$ApiBase/getSeries?Collection=$Collection&PatientID=$patientId"
    } catch {
        Log "  ERROR fetching series list for $patientId : $_"
        continue
    }

    New-Item -ItemType Directory -Force -Path $patientDir | Out-Null

    $ct  = $series | Where-Object { $_.Modality -eq "CT" }  | Sort-Object -Property ImageCount -Descending | Select-Object -First 1
    $pt  = $series | Where-Object { $_.Modality -eq "PT" }  | Select-Object -First 1
    $seg = $series | Where-Object { $_.Modality -eq "SEG" } | Select-Object -First 1

    $allOk = $true
    $targets = @(
        @{ Tag = "CT";  Series = $ct },
        @{ Tag = "PT";  Series = $pt },
        @{ Tag = "SEG"; Series = $seg }
    )

    foreach ($item in $targets) {
        if ($null -eq $item.Series) {
            Log "  SKIP $patientId : missing $($item.Tag) series"
            $allOk = $false
            continue
        }

        $destZip = Join-Path $patientDir "$($item.Tag).zip"
        if (Test-Path $destZip) { continue }  # already downloaded this one

        $uid = $item.Series.SeriesInstanceUID
        $url = "$ApiBase/getImage?SeriesInstanceUID=$uid"
        $sizeMb = [math]::Round($item.Series.FileSize / 1MB, 1)

        try {
            Log "  downloading $($item.Tag) ($sizeMb MB)..."
            Invoke-WebRequest -Uri $url -OutFile $destZip -TimeoutSec 900
        } catch {
            Log "  ERROR downloading $($item.Tag) for $patientId : $_"
            Remove-Item -Force $destZip -ErrorAction SilentlyContinue
            $allOk = $false
        }
    }

    if ($allOk) {
        New-Item -ItemType File -Force -Path $doneMarker | Out-Null
        Log "  OK $patientId"
    }
}

Log "DONE. Check $LogFile for any SKIP/ERROR lines to retry individually."
