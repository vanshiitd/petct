#!/usr/bin/env python3
"""Phase IV step 2: the strategy library (plan section 10).

Every strategy has the same shape -- (scan, M_t, U_t, target) -> M_t+1 -- and
every one of them touches only the component or region it was pointed at. That
restriction is what makes the per-strategy table in step 2 interpretable and
what lets the agent in step 3 attribute a change in Q to a single action.

    1  topology      remove a spurious component, fill its holes, or merge the
                     fragments of one lesion
    2  recovery      add a sub-threshold candidate
    3  boundary      re-threshold P-bar inside the high-uncertainty shell of a
                     component, with a local threshold tuned on out-of-fold
    4  suv           re-segment inside a dilated neighbourhood with a standard
                     PET rule: 41% of the component SUVmax, or the PERCIST
                     liver threshold (1.5 x liver SUVmean + 2 x liver SD),
                     never crossing into a physiological-uptake organ
    5  volume        grow or shrink by one morphological step, or by moving the
                     probability threshold, when the volume is implausible

Strategies 6 (local re-prediction) and 7 (the refinement network) need the GPU
and live in their own scripts; they plug into the same interface.

`CaseContext` holds one case's arrays and the current mask, and can recompute
component features for an arbitrary voxel set, which is what makes Delta-Q
measurable: Q is a function of the mask's component features, so changing the
voxels of a component changes Q only through those features.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import SimpleITK as sitk
from scipy import ndimage as ndi

RAW = Path(r"C:\nnunet_raw\Dataset505_AutoPET_MAEprep_v2")
INV = RAW / "inverse"
TOTALSEG = Path(r"D:\data\totalseg_ml_v2")
PHYSIO = [2, 3, 21, 51, 90]
LIVER, SPLEEN = 5, 1
MIN_VOX = 5
T_LOW = 0.1

STRATEGIES = ["topology_remove", "topology_fill", "topology_merge", "recover",
              "boundary", "suv41", "suv_percist", "volume_grow", "volume_shrink"]


def label_path(case: str) -> Path:
    for sub in ("labelsTs_prep", "labelsTr"):
        f = RAW / sub / f"{case}.nii.gz"
        if f.exists():
            return f
    raise FileNotFoundError(case)


def crop_like(path: Path, bbox, shape) -> np.ndarray:
    a = sitk.GetArrayFromImage(sitk.ReadImage(str(path)))
    (z0, z1), (y0, y1), (x0, x1) = bbox
    a = a[z0:z1, y0:y1, x0:x1]
    if a.shape != shape:
        raise SystemExit(f"cropped {a.shape} != prepared {shape}")
    return a


def pad_slice(sl, shape, margin) -> tuple:
    out = []
    for s, n, m in zip(sl, shape, margin):
        out.append(slice(max(0, s.start - m), min(n, s.stop + m)))
    return tuple(out)


class CaseContext:
    """One case: images, uncertainty, anatomy, and the mask being edited."""

    def __init__(self, case: str, maps: Path, with_gt: bool = True):
        self.case = case
        z = np.load(maps / f"{case}.npz")
        self.P = z["p_bar"].astype(np.float32)
        self.ent = z["u_ent"].astype(np.float32)
        self.stab = z["u_stab"].astype(np.float32)
        self.spacing = tuple(float(v) for v in z["spacing"])
        self.vml = float(np.prod(self.spacing)) / 1000.0
        self.shape = self.P.shape

        meta = json.loads((INV / f"{case}.json").read_text())
        bbox = meta["crop_bbox_zyx"]
        self.suv = crop_like(Path(meta["source_pet"]), bbox, self.shape).astype(np.float32)
        self.ct = crop_like(Path(meta["source_pet"]).with_name("CT_resample.nii.gz"),
                            bbox, self.shape).astype(np.float32)
        ts = TOTALSEG / f"{case}.nii.gz"
        self.organ = crop_like(ts, bbox, self.shape) if ts.exists() else None

        self.liver_suv = self.liver_sd = float("nan")
        if self.organ is not None and (self.organ == LIVER).any():
            lv = self.suv[self.organ == LIVER]
            self.liver_suv, self.liver_sd = float(lv.mean()), float(lv.std())
        self.physio = (np.isin(self.organ, PHYSIO) if self.organ is not None
                       else np.zeros(self.shape, bool))
        self.dist_physio = (ndi.distance_transform_edt(~self.physio,
                                                       sampling=self.spacing)
                            if self.physio.any() else None)

        body = self.ct > -500
        zs = np.flatnonzero(body.any(axis=(1, 2)))
        self.z_lo, self.z_hi = ((int(zs[0]), int(zs[-1])) if zs.size
                                else (0, self.shape[0] - 1))

        self.gt = (sitk.GetArrayFromImage(sitk.ReadImage(str(label_path(case)))) > 0
                   if with_gt else None)
        if self.gt is not None:
            self.g_lab, self.g_n = ndi.label(self.gt)
            self.g_sizes = np.bincount(self.g_lab.ravel(), minlength=self.g_n + 1)
            self.gt_vox = int(self.gt.sum())

        self.reset()

    # ---------------------------------------------------------------- state
    def reset(self) -> None:
        """Back to M0: components of P > 0.5 with at least MIN_VOX voxels."""
        m0 = self.P > 0.5
        lab, n = ndi.label(m0)
        sizes = np.bincount(lab.ravel(), minlength=n + 1)
        self.labels = np.where(np.isin(lab, [c for c in range(1, n + 1)
                                             if sizes[c] >= MIN_VOX]), lab, 0)
        self.small = m0 & (self.labels == 0)
        self.comp_ids = sorted(int(c) for c in np.unique(self.labels) if c)
        self._slices = {c: s for c, s in
                        zip(range(1, n + 1), ndi.find_objects(lab)) if s is not None}
        self.next_id = (max(self.comp_ids) + 1) if self.comp_ids else 1
        self._rec_cache = None

    def mask(self) -> np.ndarray:
        return (self.labels > 0) | self.small

    def comp_mask(self, cid: int) -> np.ndarray:
        return self.labels == cid

    def comp_slice(self, cid: int) -> tuple:
        sl = self._slices.get(cid)
        if sl is None:
            sl = ndi.find_objects(ndi.label(self.labels == cid)[0])[0]
        return sl

    def set_component(self, cid: int, sl: tuple, new_local: np.ndarray) -> None:
        """Replace component `cid` inside `sl` with `new_local`."""
        block = self.labels[sl]
        block[block == cid] = 0
        block[new_local] = cid
        self.labels[sl] = block
        self._slices[cid] = sl
        if not (self.labels == cid).any():
            self.comp_ids = [c for c in self.comp_ids if c != cid]

    def add_component(self, sl: tuple, new_local: np.ndarray) -> int:
        cid = self.next_id
        self.next_id += 1
        block = self.labels[sl]
        block[new_local & (block == 0)] = cid
        self.labels[sl] = block
        self._slices[cid] = sl
        self.comp_ids.append(cid)
        return cid

    # ---------------------------------------------- recovery candidate pool
    def recovery_candidates(self) -> dict:
        """Components of P > T_LOW not touching the dilated current mask."""
        if self._rec_cache is not None:
            return self._rec_cache
        m = self.mask()
        low = (self.P > T_LOW) & ~m
        out = {}
        if low.any():
            lab, n = ndi.label(low)
            sizes = np.bincount(lab.ravel(), minlength=n + 1)
            touch = (np.bincount(lab[ndi.binary_dilation(m)].ravel(),
                                 minlength=n + 1) if m.any()
                     else np.zeros(n + 1, dtype=np.int64))
            slices = ndi.find_objects(lab)
            for c in range(1, n + 1):
                if sizes[c] >= MIN_VOX and touch[c] == 0:
                    out[c] = (slices[c - 1], lab[slices[c - 1]] == c)
        self._rec_cache = out
        return out

    # ------------------------------------------------------------- features
    def features(self, sl: tuple, local: np.ndarray) -> dict:
        """The component features Q consumes, for an arbitrary voxel set."""
        n = int(local.sum())
        if n == 0:
            return None
        suv = self.suv[sl][local]
        ct = self.ct[sl][local]
        vol = n * self.vml
        pos = np.argwhere(local)
        ext = (pos.max(axis=0) - pos.min(axis=0) + 1) * np.array(self.spacing)
        r_eq = (3.0 * vol * 1000.0 / (4.0 * np.pi)) ** (1.0 / 3.0)
        r_bb = float(max(ext)) / 2.0
        org = self.organ[sl][local] if self.organ is not None else None
        cz = float(np.argwhere(local)[:, 0].mean() + sl[0].start)
        f = {
            "n_vox": n, "vol_ml": vol,
            "p_mean": float(self.P[sl][local].mean()),
            "uent_mean": float(self.ent[sl][local].mean()),
            "ustab_mean": float(self.stab[sl][local].mean()),
            "suv_mean": float(suv.mean()), "suv_max": float(suv.max()),
            "suv_max_over_liver": float(suv.max() / self.liver_suv)
            if self.liver_suv == self.liver_suv and self.liver_suv > 0 else np.nan,
            "hu_mean": float(ct.mean()),
            "sphericity": float(r_eq / r_bb) if r_bb > 0 else np.nan,
            "elongation": float(max(ext) / max(min(ext), 1e-6)),
            "frac_physio": float(np.isin(org, PHYSIO).mean()) if org is not None else np.nan,
            "frac_liver": float((org == LIVER).mean()) if org is not None else np.nan,
            # build_candidates.py takes the MINIMUM over the component, i.e.
            # how close it comes to a physiological-uptake organ, not its
            # average distance. Q was trained on that definition, so Delta-Q is
            # only meaningful if this matches it exactly.
            "dist_physio_mm": float(self.dist_physio[sl][local].min())
            if self.dist_physio is not None else np.nan,
            "edge_body_mm": float(min(cz - self.z_lo, self.z_hi - cz)
                                  * self.spacing[0]),
        }
        return f

    # --------------------------------------------------------------- scoring
    def score(self) -> dict:
        """Exact Dice, FP/FN volume and lesion hits for the current mask."""
        m = self.mask()
        pv = int(m.sum())
        inter = int((m & self.gt).sum())
        hit = set(np.unique(self.g_lab[m]).tolist()) - {0}
        fn = sum(int(self.g_sizes[i]) for i in range(1, self.g_n + 1)
                 if i not in hit) * self.vml
        lab = self.labels
        fp_vox = 0
        for c in self.comp_ids:
            cm = lab == c
            if not (cm & self.gt).any():
                fp_vox += int(cm.sum())
        return {"dice": (2 * inter / (pv + self.gt_vox)) if (pv + self.gt_vox)
                else float("nan"),
                "pred_ml": pv * self.vml, "fp_ml": fp_vox * self.vml,
                "fn_ml": fn, "n_hit": len(hit), "n_gt": self.g_n,
                "n_fp_comp": sum(1 for c in self.comp_ids
                                 if not ((lab == c) & self.gt).any())}


# ====================================================================== 1-5
def topology_remove(ctx: CaseContext, cid: int, **kw):
    sl = ctx.comp_slice(cid)
    return sl, np.zeros_like(ctx.labels[sl], bool)


def topology_fill(ctx: CaseContext, cid: int, **kw):
    sl = ctx.comp_slice(cid)
    local = ctx.labels[sl] == cid
    return sl, ndi.binary_fill_holes(local)


def topology_merge(ctx: CaseContext, cid: int, radius: int = 2, **kw):
    """Close the gap to a neighbouring component, joining fragments."""
    sl = pad_slice(ctx.comp_slice(cid), ctx.shape, (radius + 1,) * 3)
    block = ctx.labels[sl]
    local = block == cid
    near = (block > 0) & ~local
    if not near.any():
        return sl, local
    bridge = ndi.binary_closing(local | near, iterations=radius)
    keep = ndi.label(bridge)[0]
    want = keep[local][0] if local.any() else 0
    return sl, (keep == want) & ~near if want else local


def recover(ctx: CaseContext, cid: int, rid: int = None, **kw):
    cands = ctx.recovery_candidates()
    if rid is None or rid not in cands:
        return None, None
    return cands[rid]


def boundary(ctx: CaseContext, cid: int, tau: float = 0.5, band: int = 2, **kw):
    """Re-threshold inside the uncertainty shell, leaving the interior alone."""
    sl = pad_slice(ctx.comp_slice(cid), ctx.shape, (band + 1,) * 3)
    local = ctx.labels[sl] == cid
    if not local.any():
        return sl, local
    outer = ndi.binary_dilation(local, iterations=band)
    inner = ndi.binary_erosion(local, iterations=band)
    shell = outer & ~inner
    new = local.copy()
    new[shell] = ctx.P[sl][shell] > tau
    new |= inner
    lab, n = ndi.label(new)
    if n > 1 and inner.any():                 # keep the piece holding the core
        want = np.bincount(lab[inner].ravel()).argmax()
        new = lab == want
    return sl, new


def _suv_region(ctx, cid, margin_mm=10.0):
    margin = tuple(int(np.ceil(margin_mm / s)) for s in ctx.spacing)
    sl = pad_slice(ctx.comp_slice(cid), ctx.shape, margin)
    local = ctx.labels[sl] == cid
    region = ndi.binary_dilation(local, iterations=max(margin))
    region &= ~ctx.physio[sl]
    other = (ctx.labels[sl] > 0) & ~local
    region &= ~other
    return sl, local, region


def suv41(ctx: CaseContext, cid: int, **kw):
    sl, local, region = _suv_region(ctx, cid)
    if not local.any():
        return sl, local
    thr = 0.41 * float(ctx.suv[sl][local].max())
    new = region & (ctx.suv[sl] > thr)
    lab, n = ndi.label(new)
    if n and local.any():
        ids = np.unique(lab[local])
        ids = ids[ids > 0]
        new = np.isin(lab, ids) if ids.size else local
    return sl, new if new.any() else local


def suv_percist(ctx: CaseContext, cid: int, **kw):
    if not (ctx.liver_suv == ctx.liver_suv):
        return None, None
    sl, local, region = _suv_region(ctx, cid)
    if not local.any():
        return sl, local
    thr = 1.5 * ctx.liver_suv + 2.0 * ctx.liver_sd
    new = region & (ctx.suv[sl] > thr)
    lab, n = ndi.label(new)
    if n and local.any():
        ids = np.unique(lab[local])
        ids = ids[ids > 0]
        new = np.isin(lab, ids) if ids.size else local
    return sl, new if new.any() else local


def volume_grow(ctx: CaseContext, cid: int, tau: float = 0.3, **kw):
    sl = pad_slice(ctx.comp_slice(cid), ctx.shape, (3, 3, 3))
    local = ctx.labels[sl] == cid
    other = (ctx.labels[sl] > 0) & ~local
    new = ndi.binary_dilation(local, iterations=1) & (ctx.P[sl] > tau) & ~other
    return sl, new | local


def volume_shrink(ctx: CaseContext, cid: int, tau: float = 0.7, **kw):
    sl = ctx.comp_slice(cid)
    local = ctx.labels[sl] == cid
    new = local & (ctx.P[sl] > tau)
    if not new.any():
        new = ndi.binary_erosion(local, iterations=1)
    return sl, new if new.any() else local


REGISTRY = {
    "topology_remove": topology_remove, "topology_fill": topology_fill,
    "topology_merge": topology_merge, "recover": recover, "boundary": boundary,
    "suv41": suv41, "suv_percist": suv_percist,
    "volume_grow": volume_grow, "volume_shrink": volume_shrink,
}
