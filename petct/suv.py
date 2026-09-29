"""SUV (standardised uptake value) conversion for FDG PET.

A PET image in Bq/mL is not comparable between patients: the same tissue reads
differently depending on how much tracer was injected, how much the patient
weighs and how long uptake was allowed before scanning. SUV divides that out:

    SUVbw = activity_concentration [Bq/mL] * patient_weight [g] / decayed_dose [Bq]

so a value near 1 means "as much tracer per mL as the average over the whole
body", and thresholds like "SUV > 2.5 is suspicious" carry across patients.

The decay term matters. `DecayCorrection = START` means the pixel values were
already corrected back to the start of the series, so the *dose* has to be
decayed forward from injection to that same reference point:

    decayed_dose = total_dose * 2 ** (-(scan_time - injection_time) / half_life)

Everything needed lives in the PET DICOM header, most of it inside the
RadiopharmaceuticalInformationSequence, which is why this reads with pydicom
rather than SimpleITK (SimpleITK does not expose nested sequences).

Nothing is guessed: a missing or implausible tag returns a problem description
instead of a number, so those cases can be listed and looked at.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta

F18_HALF_LIFE_S = 6586.2  # seconds; only used to sanity-check the header value


@dataclass
class SUVFactor:
    """Multiply a Bq/mL volume by `factor` to get SUVbw."""
    factor: float | None
    problems: list[str] = field(default_factory=list)
    detail: dict = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return self.factor is not None and not self.problems


def _parse_dicom_datetime(date_str: str, time_str: str) -> datetime | None:
    if not date_str or not time_str:
        return None
    time_str = time_str.strip()
    frac = 0.0
    if "." in time_str:
        time_str, _, frac_str = time_str.partition(".")
        frac = float(f"0.{frac_str}") if frac_str else 0.0
    time_str = time_str.ljust(6, "0")[:6]
    try:
        dt = datetime.strptime(f"{date_str.strip()}{time_str}", "%Y%m%d%H%M%S")
    except ValueError:
        return None
    return dt + timedelta(seconds=frac)


def suv_factor_from_dataset(ds) -> SUVFactor:
    """Compute the Bq/mL -> SUVbw scale factor from a PET DICOM header.

    `ds` is a pydicom Dataset read from any slice of the series (these tags are
    constant across the series).
    """
    problems: list[str] = []
    detail: dict = {}

    units = str(ds.get("Units", "") or "").strip().upper()
    detail["units"] = units
    if units != "BQML":
        problems.append(f"Units is {units or 'missing'}, expected BQML")

    decay = str(ds.get("DecayCorrection", "") or "").strip().upper()
    detail["decay_correction"] = decay
    if decay not in ("START", "ADMIN"):
        problems.append(f"DecayCorrection is {decay or 'missing'}, expected START or ADMIN")

    weight_kg = ds.get("PatientWeight", None)
    detail["weight_kg"] = float(weight_kg) if weight_kg not in (None, "") else None
    if not weight_kg:
        problems.append("PatientWeight missing")
    elif not (20.0 <= float(weight_kg) <= 300.0):
        problems.append(f"PatientWeight {float(weight_kg)} kg is implausible")

    seq = ds.get("RadiopharmaceuticalInformationSequence", None)
    if not seq:
        problems.append("RadiopharmaceuticalInformationSequence missing")
        return SUVFactor(None, problems, detail)
    item = seq[0]

    dose = item.get("RadionuclideTotalDose", None)
    detail["dose_bq"] = float(dose) if dose not in (None, "") else None
    if not dose:
        problems.append("RadionuclideTotalDose missing")
    elif not (1e6 <= float(dose) <= 2e9):
        problems.append(f"RadionuclideTotalDose {float(dose):.3g} Bq is implausible")

    half_life = item.get("RadionuclideHalfLife", None)
    detail["half_life_s"] = float(half_life) if half_life not in (None, "") else None
    if not half_life:
        problems.append("RadionuclideHalfLife missing")
    elif abs(float(half_life) - F18_HALF_LIFE_S) > 0.1 * F18_HALF_LIFE_S:
        problems.append(f"RadionuclideHalfLife {float(half_life)} s is not F-18 "
                        f"(expected ~{F18_HALF_LIFE_S})")

    # Injection time: the explicit datetime if present, else the time-of-day
    # combined with the series date.
    series_date = str(ds.get("SeriesDate", "") or "")
    start_dt = item.get("RadiopharmaceuticalStartDateTime", None)
    injected = None
    if start_dt:
        injected = _parse_dicom_datetime(str(start_dt)[:8], str(start_dt)[8:])
    if injected is None:
        injected = _parse_dicom_datetime(series_date,
                                         str(item.get("RadiopharmaceuticalStartTime", "") or ""))
    if injected is None:
        problems.append("radiopharmaceutical start time unreadable")

    # Scan reference: with DecayCorrection START the pixels are corrected to the
    # series start, so that is the point to decay the dose to.
    scanned = _parse_dicom_datetime(series_date, str(ds.get("SeriesTime", "") or ""))
    if scanned is None:
        scanned = _parse_dicom_datetime(str(ds.get("AcquisitionDate", "") or series_date),
                                        str(ds.get("AcquisitionTime", "") or ""))
    if scanned is None:
        problems.append("series/acquisition time unreadable")

    if injected is not None and scanned is not None:
        delay_s = (scanned - injected).total_seconds()
        detail["uptake_delay_s"] = delay_s
        if delay_s < 0:
            problems.append(f"scan time precedes injection by {-delay_s:.0f} s")
        elif delay_s > 4 * 3600:
            problems.append(f"uptake delay {delay_s/60:.0f} min is implausible")

    if problems:
        return SUVFactor(None, problems, detail)

    decayed_dose = float(dose) * 2 ** (-detail["uptake_delay_s"] / float(half_life))
    detail["decayed_dose_bq"] = decayed_dose
    factor = (float(weight_kg) * 1000.0) / decayed_dose      # kg -> g
    detail["factor"] = factor
    return SUVFactor(factor, [], detail)


def suv_factor_from_file(path) -> SUVFactor:
    """Read one PET DICOM slice and compute the SUV factor from it."""
    import pydicom
    ds = pydicom.dcmread(str(path), stop_before_pixels=True)
    return suv_factor_from_dataset(ds)
