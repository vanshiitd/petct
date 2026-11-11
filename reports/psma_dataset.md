# PSMA-PET-CT-Lesions: download and conversion record

**Source.** TCIA, PSMA-PET-CT-Lesions, version 2. This is the autoPET III
PSMA cohort, licensed CC BY 4.0. Cite it under the TCIA Data Usage Policy.

**Script.** `scripts/download_psma.py` (commits 2e3ebd0, de5901d).

**Output.** `D:\data\psma_nifti\`, 9.4 GB:
- one folder per study, `<PatientID>_<YYYYMMDD>_<uid8>/`, holding
  `PET.nii.gz` (SUV), `CT_resample.nii.gz` (int16, on the PET grid) and
  `tumorSeg.nii.gz`;
- `ledger.jsonl`, with per-study status and checks;
- `PSMA-PET-CT-Lesions-CLINICAL_v01_20251118.tsv`, holding age, scanner,
  radionuclide and CT contrast.

## Counts

| | |
|---|---|
| studies in TCIA (with CT, PT and SEG) | 597 |
| converted | **595** (377 patients) |
| tumour-positive / tumour-free | 537 / 58 |
| excluded | 2 (below) |

The class balance is very different from FDG, which has 385 tumour-free
scans out of 738 out-of-fold. PSMA has only 58 tumour-free studies, so a
PSMA evaluation has little power for the "tumour-free cleared" endpoint.
Any PSMA plan must say this.

## Exclusions

Both failed identically on retry, inside the organisers' converter, where
dicom2nifti produced no output.

| study | failing series |
|---|---|
| PSMA_738bc5d9946240f3_19970926_94254011 | CT (`dcm2nii_CT`) |
| PSMA_3959d1c381a5bcd6_19981005_26528057 | PET (`dcm2nii_PET`) |

dicom2nifti rejects series with inconsistent slice geometry by design.
Forcing a conversion could misalign images, so these two are excluded
rather than repaired.

## Mask fix (important)

The organisers' `dcm2nii_mask` assumes one SEG frame per PET slice, which
holds for FDG. PSMA SEGs store only the frames that contain lesion: one
195-slice PET had an 8-frame SEG. Used unchanged, the converter
**truncated and shifted every tumour-positive PSMA mask**.

`seg_on_pet_grid` places each frame at the PET slice given by its own
ImagePositionPatient, keeping the organisers' in-plane handling.

**Checks:**
- FDG regression: identical to the official masks on 12 of 12 random cases,
  both positive and negative;
- every PSMA study: slice offset <= 0.01, and no lesion voxel lost
  (enforced; 0 studies failed);
- trial PSMA studies: lesion SUVmean 10-13, against a body mean of about
  1.5.
