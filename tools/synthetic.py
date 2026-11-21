"""Synthetic bitemporal 'satellite-like' scenes with known ground truth.

These are NOT real imagery.  They exist so the pipeline and the web app can be
tested and demonstrated without downloading data.  Each pair contains:
  * human-made change  (new buildings, a new road, a demolished block)
  * natural change     (vegetation growth, vegetation loss to bare soil)
  * environmental      (a lake that floods onto surrounding fields)
  * a mild global seasonal/illumination shift and sensor noise
"""
from __future__ import annotations

import cv2
import numpy as np

# band reflectances: R, G, B, NIR
SPECTRA = {
    "crop":    (0.10, 0.22, 0.07, 0.60),
    "forest":  (0.05, 0.11, 0.04, 0.50),
    "bare":    (0.30, 0.26, 0.20, 0.33),
    "water":   (0.04, 0.08, 0.11, 0.03),
    "asphalt": (0.17, 0.17, 0.18, 0.20),
    "roof_a":  (0.62, 0.60, 0.58, 0.42),
    "roof_b":  (0.45, 0.30, 0.25, 0.35),
    "roof_c":  (0.70, 0.70, 0.72, 0.45),
}
CLASS_HUMAN, CLASS_NATURAL, CLASS_ENV = 1, 2, 3


def _paint(img, mask, spectrum, rng, noise=0.012, texture=0.0):
    for i, v in enumerate(spectrum):
        layer = np.full(mask.shape, v, np.float32)
        if texture:
            layer += texture * (cv2.GaussianBlur(rng.standard_normal(mask.shape).astype(np.float32), (0, 0), 1.2))
        layer += noise * rng.standard_normal(mask.shape).astype(np.float32)
        img[:, :, i] = np.where(mask, layer, img[:, :, i])


def _blob(shape, cx, cy, rx, ry, rng, rough=0.25):
    h, w = shape
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    ang = np.arctan2(yy - cy, xx - cx)
    wob = 1 + rough * (np.sin(3 * ang + rng.uniform(0, 6)) * 0.5 + np.sin(5 * ang + rng.uniform(0, 6)) * 0.3)
    return (((xx - cx) / (rx * wob)) ** 2 + ((yy - cy) / (ry * wob)) ** 2) < 1


def _rect(shape, x, y, w, h):
    m = np.zeros(shape, bool)
    m[y:y + h, x:x + w] = True
    return m


def make_pair(size=512, seed=7, with_nir=True):
    """Return (t1, t2, truth) where t1/t2 are HxWx4 (R,G,B,NIR) float32 in [0,1] and
    truth is an HxW uint8 map (0 none, 1 human, 2 natural, 3 environmental)."""
    rng = np.random.default_rng(seed)
    S = (size, size)
    t1 = np.zeros((size, size, 4), np.float32)

    # --- base landscape: patchwork of crops / bare fields + forest ----------
    pts = rng.uniform(0, size, (24, 2))
    yy, xx = np.mgrid[0:size, 0:size]
    d = ((xx[..., None] - pts[:, 0]) ** 2 + (yy[..., None] - pts[:, 1]) ** 2)
    cell = d.argmin(-1)
    kinds = rng.choice(["crop", "bare", "forest"], size=24, p=[0.55, 0.2, 0.25])
    for k in range(24):
        _paint(t1, cell == k, SPECTRA[kinds[k]], rng, texture=0.02 if kinds[k] == "forest" else 0.006)

    # --- lake ----------------------------------------------------------------
    lake1 = _blob(S, size * 0.28, size * 0.55, size * 0.13, size * 0.17, rng, rough=0.2)
    _paint(t1, lake1, SPECTRA["water"], rng, noise=0.006)

    # --- existing roads & a small settlement --------------------------------
    for y in (int(size * 0.18), int(size * 0.82)):
        _paint(t1, _rect(S, 0, y, size, 4), SPECTRA["asphalt"], rng)
    _paint(t1, _rect(S, int(size * 0.62), 0, 4, size), SPECTRA["asphalt"], rng)
    for (bx, by) in [(330, 120), (350, 120), (370, 120), (330, 140), (350, 140), (370, 145)]:
        _paint(t1, _rect(S, bx, by, 13, 13), SPECTRA[rng.choice(["roof_a", "roof_b", "roof_c"])], rng, texture=0.03)

    # block that will be demolished (built-up in T1)
    demol = _rect(S, 400, 320, 56, 44)
    _paint(t1, demol, SPECTRA["bare"], rng)
    demol_blds = []
    for gx in range(3):
        for gy in range(2):
            m = _rect(S, 404 + gx * 18, 324 + gy * 20, 14, 15)
            demol_blds.append(m)
            _paint(t1, m, SPECTRA[rng.choice(["roof_a", "roof_c"])], rng, texture=0.03)

    t2 = t1.copy()
    truth = np.zeros(S, np.uint8)

    # === HUMAN: new buildings on bare/crop land ==============================
    new_blocks = [(110, 60, 60, 46), (230, 440, 52, 40), (440, 60, 50, 44)]
    for (bx, by, bw, bh) in new_blocks:
        area = _rect(S, bx, by, bw, bh)
        _paint(t2, area, SPECTRA["bare"], rng)
        for gx in range(0, bw - 12, 18):
            for gy in range(0, bh - 12, 20):
                m = _rect(S, bx + 3 + gx, by + 3 + gy, 14, 15)
                _paint(t2, m, SPECTRA[rng.choice(["roof_a", "roof_b", "roof_c"])], rng, texture=0.03)
        truth[area] = CLASS_HUMAN
    # new road
    road = _rect(S, 300, 0, 5, 150)
    _paint(t2, road, SPECTRA["asphalt"], rng)
    truth[road] = CLASS_HUMAN
    # demolition
    _paint(t2, demol, SPECTRA["bare"], rng)
    truth[demol] = CLASS_HUMAN

    # === NATURAL: vegetation growth on bare soil / loss to bare soil =========
    grow = _blob(S, 300, 250, 30, 24, rng)
    _paint(t1, grow, SPECTRA["bare"], rng)
    _paint(t2, grow, SPECTRA["crop"], rng, texture=0.006)
    truth[grow] = CLASS_NATURAL
    loss = _blob(S, 330, 400, 28, 24, rng)
    _paint(t1, loss, SPECTRA["crop"], rng, texture=0.006)
    _paint(t2, loss, SPECTRA["bare"], rng)
    truth[loss] = CLASS_NATURAL

    # === ENVIRONMENTAL: lake floods onto fields (water extent grows) =========
    lake2 = _blob(S, size * 0.28, size * 0.55, size * 0.19, size * 0.23, rng, rough=0.2)
    flood = lake2 & ~lake1
    _paint(t2, lake2, SPECTRA["water"], rng, noise=0.006)
    truth[flood] = CLASS_ENV

    # --- global seasonal / illumination shift + sensor noise ------------------
    t2 = t2 * np.array([1.03, 1.02, 0.99, 1.0], np.float32) + 0.012
    t1 = (t1 + 0.004 * rng.standard_normal(t1.shape)).astype(np.float32)
    t2 = (t2 + 0.004 * rng.standard_normal(t2.shape)).astype(np.float32)
    t1 = np.clip(t1, 0, 1)
    t2 = np.clip(t2, 0, 1)
    if not with_nir:
        t1, t2 = t1[:, :, :3], t2[:, :, :3]
    return t1, t2, truth


def make_no_change_pair(size=512, seed=3, with_nir=True):
    t1, _, _ = make_pair(size, seed, with_nir)
    rng = np.random.default_rng(seed + 100)
    t2 = t1 * 1.02 + 0.008 + 0.004 * rng.standard_normal(t1.shape).astype(np.float32)
    return t1, np.clip(t2, 0, 1).astype(np.float32), np.zeros(t1.shape[:2], np.uint8)


def to_uint8_rgb(arr):
    return (arr[:, :, :3].clip(0, 1) * 255).astype(np.uint8)


def write_png(path, arr):
    from PIL import Image
    Image.fromarray(to_uint8_rgb(arr)).save(path)


def write_rgbn_tiff(path, arr, pixel_size_m=10.0):
    """4-band uint16 GeoTIFF (R,G,B,NIR) with a ModelPixelScale tag."""
    import tifffile
    data = (arr.clip(0, 1) * 10000).astype(np.uint16)
    tifffile.imwrite(path, data, photometric="minisblack", planarconfig="contig",
                     extratags=[(33550, "d", 3, (pixel_size_m, pixel_size_m, 0.0), False)])
