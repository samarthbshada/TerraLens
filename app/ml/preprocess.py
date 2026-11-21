"""Preprocessing for a temporal image pair.

Order (matches the TerraLens workflow):
  1. quality screening (cloud-like pixels, blank images)
  2. size harmonisation to a common grid
  3. radiometric normalisation of T2 to T1 using *stable* pixels only
  4. geometric registration (ECC) of T2 onto T1
"""
from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from .imaging import Scene


@dataclass
class PreparedPair:
    t1: np.ndarray                    # HxWxC float32 in [0, 1]
    t2: np.ndarray                    # aligned + radiometrically matched
    names: tuple                      # channel names
    pixel_size_m: float               # metres per pixel of the arrays above
    valid: np.ndarray                 # HxW bool (False where warp left the image)
    info: dict = field(default_factory=dict)
    warnings: list = field(default_factory=list)

    @property
    def shape(self):
        return self.t1.shape[:2]

    def band(self, which: str, name: str):
        arr = self.t1 if which == "t1" else self.t2
        return arr[:, :, self.names.index(name)]

    def has(self, name: str) -> bool:
        return name in self.names

    def gray(self, which: str) -> np.ndarray:
        arr = self.t1 if which == "t1" else self.t2
        vis = [arr[:, :, self.names.index(n)] for n in ("red", "green", "blue") if n in self.names]
        if not vis:
            return arr.mean(axis=2)
        return np.mean(vis, axis=0).astype(np.float32)


# ----------------------------------------------------------------------------
def cloud_fraction(scene: Scene) -> float:
    """Rough cloud/haze indicator: bright and unsaturated pixels."""
    if not scene.has_visible:
        return 0.0
    r, g, b = (scene.bands[k] for k in ("red", "green", "blue"))
    bright = (r + g + b) / 3.0
    sat = (np.maximum(np.maximum(r, g), b) - np.minimum(np.minimum(r, g), b)) / (np.maximum(np.maximum(r, g), b) + 1e-6)
    return float(((bright > 0.82) & (sat < 0.12)).mean())


def _match_radiometry(t1: np.ndarray, t2: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Linear gain/offset per band fitted on the least-changed half of the image."""
    out = t2.copy()
    for _ in range(2):
        diff = np.abs(out - t1).mean(axis=2)
        diff = cv2.GaussianBlur(diff, (0, 0), 2.0)
        stable = (diff <= np.percentile(diff[valid], 60)) & valid
        if stable.sum() < 500:
            break
        for c in range(t1.shape[2]):
            a, b = t1[:, :, c][stable], out[:, :, c][stable]
            sb = b.std()
            gain = (a.std() / sb) if sb > 1e-6 else 1.0
            gain = float(np.clip(gain, 0.6, 1.6))
            offset = float(a.mean() - gain * b.mean())
            out[:, :, c] = np.clip(out[:, :, c] * gain + offset, 0.0, 1.0)
    return out


def _register(t1: np.ndarray, t2: np.ndarray):
    """Translation (ECC) alignment of T2 onto T1. Returns (aligned t2, valid mask, info dict).

    Translation-only is deliberate: it is stable even when a lot of the scene has genuinely
    changed, whereas affine/homography fits drift towards the changed areas."""
    h, w = t1.shape[:2]
    valid = np.ones((h, w), bool)
    info = {"registration": "skipped", "shift_px": (0.0, 0.0), "cc": None}
    if min(h, w) < 64:
        return t2, valid, info
    f = 256.0 / max(h, w) if max(h, w) > 256 else 1.0

    def small(a):
        g = a.mean(axis=2)
        g = cv2.resize(g, None, fx=f, fy=f, interpolation=cv2.INTER_AREA) if f != 1.0 else g
        return cv2.GaussianBlur(g, (0, 0), 1.5).astype(np.float32)

    s1, s2 = small(t1), small(t2)
    warp = np.eye(2, 3, dtype=np.float32)
    try:
        cc, warp = cv2.findTransformECC(s1, s2, warp, cv2.MOTION_TRANSLATION,
                                        (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 200, 1e-6), None, 5)
    except cv2.error:
        info["registration"] = "failed (kept original alignment)"
        return t2, valid, info
    tx, ty = float(warp[0, 2]) / f, float(warp[1, 2]) / f
    info["cc"] = round(float(cc), 3)
    if cc < 0.3 or abs(tx) > 0.10 * w or abs(ty) > 0.10 * h:
        info["registration"] = "kept original alignment (shift estimate unreliable)"
        return t2, valid, info
    if abs(tx) < 0.6 and abs(ty) < 0.6:
        info["registration"] = "already aligned"
        return t2, valid, info
    # the shift must actually reduce the difference on the least-changed pixels
    wsm = warp.copy()
    s2w = cv2.warpAffine(s2, wsm, (s1.shape[1], s1.shape[0]), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
                         borderMode=cv2.BORDER_REFLECT)
    def corr(x, y):
        x, y = x.ravel() - x.mean(), y.ravel() - y.mean()
        return float((x * y).sum() / (np.sqrt((x * x).sum() * (y * y).sum()) + 1e-9))
    if not corr(s1, s2w) > corr(s1, s2) + 0.003:
        info["registration"] = "already aligned"
        return t2, valid, info
    full = np.array([[1, 0, tx], [0, 1, ty]], np.float32)
    flags = cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP
    aligned = np.dstack([cv2.warpAffine(t2[:, :, c], full, (w, h), flags=flags, borderMode=cv2.BORDER_REFLECT)
                         for c in range(t2.shape[2])])
    ones = cv2.warpAffine(np.ones((h, w), np.float32), full, (w, h), flags=flags,
                          borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    # ignore a thin border where the shifted image has no data (and its blur halo)
    valid = cv2.erode((ones > 0.99).astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
    info.update(registration="aligned", shift_px=(round(tx, 2), round(ty, 2)))
    return aligned.astype(np.float32), valid, info


def prepare_pair(s1: Scene, s2: Scene, fallback_res_m: float = 10.0) -> PreparedPair:
    warnings = list(s1.warnings) + [f"T2: {w}" for w in s2.warnings]

    names = tuple(n for n in s1.bands if n in s2.bands)
    if len(names) < 3 and not (set(names) >= {"red", "green"}):
        raise ValueError("The two images share too few spectral bands to compare.")
    dropped = (set(s1.bands) | set(s2.bands)) - set(names)
    if dropped:
        warnings.append("Bands not present in both images were ignored: " + ", ".join(sorted(dropped)) + ".")

    cl1, cl2 = cloud_fraction(s1), cloud_fraction(s2)
    for label, frac in (("T1", cl1), ("T2", cl2)):
        if frac > 0.10:
            warnings.append(f"{label} looks cloudy/hazy (~{frac * 100:.0f}% bright, unsaturated pixels); "
                            "results may include false changes.")

    a = np.dstack([s1.bands[n] for n in names]).astype(np.float32)
    b = np.dstack([s2.bands[n] for n in names]).astype(np.float32)
    h, w = a.shape[:2]
    if b.shape[:2] != (h, w):
        warnings.append(f"Image sizes differ ({w}x{h} vs {b.shape[1]}x{b.shape[0]}); T2 was resampled to T1's grid.")
        b = np.dstack([cv2.resize(b[:, :, c], (w, h), interpolation=cv2.INTER_AREA if b.shape[0] > h else cv2.INTER_LINEAR)
                       for c in range(b.shape[2])])

    # metres per pixel of the analysis grid
    base_m = s1.pixel_size_m or fallback_res_m
    pixel_m = base_m / s1.scale if s1.scale else base_m

    b, valid, reg = _register(a, b)
    b = _match_radiometry(a, b, valid)
    if reg["registration"].startswith("failed") or (reg["cc"] is not None and reg["cc"] < 0.3):
        warnings.append("The two images correlate weakly; make sure both cover the same area. "
                        "Results may contain false changes.")
    info = {"cloud_t1": round(cl1, 4), "cloud_t2": round(cl2, 4), **reg,
            "analysis_size": (w, h), "pixel_size_m": round(pixel_m, 3)}
    return PreparedPair(t1=a, t2=b, names=names, pixel_size_m=pixel_m, valid=valid, info=info, warnings=warnings)
