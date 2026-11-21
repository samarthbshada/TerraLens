"""Human-activity attribution of detected changes.

IMPORTANT: this is a transparent, rule-based spectral/structural interpretation
- not a trained classifier.  Each changed pixel is scored for three kinds of
evidence and assigned to the best-supported class:

  1  Human-made    : change in 'built-up-ness' (low vegetation x high local
                     texture / structure) - construction, expansion, demolition
  2  Natural       : vegetation gain / loss without built-up structure
  3  Environmental : water-extent change (flooding, water level) or a broad
                     seasonal / illumination style shift

Regions are then smoothed by majority vote and given a confidence value.  Low
confidence regions are reported, never silently called human-made.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from .preprocess import PreparedPair

HUMAN, NATURAL, ENV = 1, 2, 3
CLASS_NAMES = {HUMAN: "Human-made", NATURAL: "Natural", ENV: "Environmental"}
LOW_CONFIDENCE = 0.45
EPS = 1e-6


@dataclass
class Attribution:
    labels: np.ndarray                      # HxW uint8: 0 none, 1 human, 2 natural, 3 environmental
    regions: list = field(default_factory=list)
    low_confidence_regions: int = 0
    notes: list = field(default_factory=list)


def _blur(x, s=2.0):
    return cv2.GaussianBlur(x.astype(np.float32), (0, 0), s)


def _vegetation(pair: PreparedPair, which: str) -> np.ndarray:
    arr = pair.t1 if which == "t1" else pair.t2
    ix = {n: arr[:, :, i] for i, n in enumerate(pair.names)}
    if "nir" in ix and "red" in ix:
        ndvi = (ix["nir"] - ix["red"]) / (ix["nir"] + ix["red"] + EPS)
        return np.clip((ndvi - 0.15) / 0.35, 0, 1)
    r, g = ix["red"], ix["green"]
    b = ix.get("blue", (r + g) / 2)
    vari = (g - r) / (g + r - b + EPS)                  # visible-band vegetation index
    return np.clip((vari - 0.02) / 0.20, 0, 1)


def _water(pair: PreparedPair, which: str) -> np.ndarray:
    arr = pair.t1 if which == "t1" else pair.t2
    ix = {n: arr[:, :, i] for i, n in enumerate(pair.names)}
    if "nir" in ix and "green" in ix:
        ndwi = (ix["green"] - ix["nir"]) / (ix["green"] + ix["nir"] + EPS)
        return np.clip(ndwi / 0.25, 0, 1)
    if all(k in ix for k in ("red", "green", "blue")):
        r, g, b = ix["red"], ix["green"], ix["blue"]
        blueish = (b + g - 2 * r) / (b + g + 2 * r + EPS)
        dark = 1.0 - np.clip(((r + g + b) / 3.0) / 0.35, 0, 1)
        return np.clip((blueish - 0.05) / 0.2, 0, 1) * dark
    return np.zeros(arr.shape[:2], np.float32)


def _impervious(pair: PreparedPair, which: str, veg: np.ndarray, wat: np.ndarray) -> np.ndarray:
    """Grey, non-vegetated, non-water surface (asphalt, concrete, roofs)."""
    arr = pair.t1 if which == "t1" else pair.t2
    ix = {n: arr[:, :, i] for i, n in enumerate(pair.names)}
    if not all(k in ix for k in ("red", "green", "blue")):
        return np.zeros(arr.shape[:2], np.float32)
    rgb = np.dstack([ix["red"], ix["green"], ix["blue"]])
    sat = (rgb.max(axis=2) - rgb.min(axis=2)) / (rgb.max(axis=2) + EPS)
    grey = np.clip((0.22 - sat) / 0.16, 0, 1)
    return (grey * (1 - veg) * (1 - wat)).astype(np.float32)


def _texture(gray: np.ndarray) -> np.ndarray:
    m = cv2.blur(gray, (7, 7))
    m2 = cv2.blur(gray * gray, (7, 7))
    return np.sqrt(np.maximum(m2 - m * m, 0)).astype(np.float32)


def attribute_changes(pair: PreparedPair, mask: np.ndarray) -> Attribution:
    h, w = mask.shape
    labels = np.zeros((h, w), np.uint8)
    if mask.sum() == 0:
        return Attribution(labels=labels, notes=["No changed pixels to attribute."])

    veg1, veg2 = _blur(_vegetation(pair, "t1")), _blur(_vegetation(pair, "t2"))
    wat1, wat2 = _blur(_water(pair, "t1")), _blur(_water(pair, "t2"))
    e1, e2 = _texture(pair.gray("t1")), _texture(pair.gray("t2"))
    scale = np.percentile(np.maximum(e1, e2), 95) + EPS
    tex1, tex2 = np.clip(e1 / scale, 0, 1), np.clip(e2 / scale, 0, 1)
    bu1 = (1 - veg1) * tex1 * (1 - wat1)
    bu2 = (1 - veg2) * tex2 * (1 - wat2)
    bu1, bu2 = _blur(bu1, 2.5), _blur(bu2, 2.5)

    imp1 = _blur(_impervious(pair, "t1", veg1, wat1), 1.5)
    imp2 = _blur(_impervious(pair, "t2", veg2, wat2), 1.5)
    d_bu = np.maximum(np.abs(bu2 - bu1), np.abs(imp2 - imp1) * 0.6)
    d_veg = np.abs(veg2 - veg1)
    d_wat = np.abs(wat2 - wat1)
    water_present = np.maximum(wat1, wat2)

    s_h = d_bu / 0.15
    s_n = d_veg / 0.40
    s_w = np.where(water_present > 0.25, d_wat / 0.20, 0.0)
    s_e = np.full((h, w), 0.50, np.float32)              # residual: seasonal / illumination

    stack = np.stack([s_h, s_n, s_w, s_e], axis=0).astype(np.float32)
    best = stack.argmax(axis=0)
    # map index -> class (water and residual are both 'environmental')
    to_class = np.array([HUMAN, NATURAL, ENV, ENV], np.uint8)
    pix = to_class[best]
    srt = np.sort(stack, axis=0)
    margin = (srt[-1] - srt[-2]) / (srt[-1] + EPS)

    pix = np.where(mask > 0, pix, 0).astype(np.uint8)
    pix = cv2.medianBlur(pix, 5)
    pix[mask == 0] = 0

    # ---- region-level voting ----------------------------------------------
    n, lab, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    counts = np.bincount((lab.astype(np.int64) * 4 + pix).ravel(), minlength=n * 4).reshape(n, 4)
    regions = []
    final = pix.copy()
    sub_water = (best == 2)
    for rid in range(1, n):
        area = int(stats[rid, cv2.CC_STAT_AREA])
        c = counts[rid, 1:4]
        tot = c.sum()
        if tot == 0:
            continue
        dom = int(c.argmax()) + 1
        share = float(c[dom - 1] / tot)
        region_px = lab == rid
        if share >= 0.55:
            final[region_px] = dom
        else:
            dom = int(np.bincount(final[region_px], minlength=4)[1:].argmax()) + 1
        m_reg = float(np.mean(margin[region_px & (pix == dom)])) if (region_px & (pix == dom)).any() else 0.0
        conf = float(np.clip(share * (0.35 + 0.65 * min(1.0, m_reg * 1.4)), 0, 1))
        x, y, bw, bh = (int(stats[rid, k]) for k in (cv2.CC_STAT_LEFT, cv2.CC_STAT_TOP, cv2.CC_STAT_WIDTH, cv2.CC_STAT_HEIGHT))
        rect = area / float(max(1, bw * bh))
        linear = False
        if dom == ENV and area >= 20 and sub_water[region_px].mean() <= 0.4:
            pts = np.column_stack(np.where(region_px))[:, ::-1].astype(np.float32)
            (_, _), (rw, rh), _ = cv2.minAreaRect(pts)
            long_side, short_side = max(rw, rh), max(1.0, min(rw, rh))
            if long_side / short_side >= 5.0 and short_side <= 14 and long_side >= 30:
                linear, dom = True, HUMAN
                final[region_px] = HUMAN
                conf = max(conf, 0.5)
        if dom == HUMAN and rect > 0.7:
            conf = min(1.0, conf + 0.08)
        if linear:
            reason = "Linear feature (road-like) with non-vegetated surface change"
        elif dom == HUMAN:
            reason = "Built-up structure changed (texture/brightness with little vegetation)"
        elif dom == NATURAL:
            reason = "Vegetation gained or lost without built-up structure"
        elif sub_water[region_px].mean() > 0.4:
            reason = "Water extent changed (flooding / water level)"
        else:
            reason = "Broad seasonal or illumination-type change"
        regions.append({"id": rid, "label": int(dom), "class": CLASS_NAMES[dom], "area_px": area,
                        "bbox": (x, y, bw, bh), "confidence": round(conf, 2), "reason": reason})

    low = sum(1 for r in regions if r["confidence"] < LOW_CONFIDENCE)
    notes = []
    if not pair.has("nir"):
        notes.append("No NIR band: vegetation/water use visible-band indices (less reliable).")
    return Attribution(labels=final, regions=regions, low_confidence_regions=low, notes=notes)
