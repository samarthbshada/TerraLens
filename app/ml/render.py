"""Turn arrays into the PNG images the web interface displays."""
from __future__ import annotations

import cv2
import numpy as np
from PIL import Image

from .preprocess import PreparedPair

CLASS_COLORS = {1: (239, 68, 68), 2: (34, 197, 94), 3: (234, 179, 8)}   # red, green, yellow
CLASS_HEX = {1: "#ef4444", 2: "#22c55e", 3: "#eab308"}


def display_rgb(pair: PreparedPair, which: str) -> np.ndarray:
    """uint8 HxWx3 display image of T1 or T2 (true colour, or NIR-R-G false colour)."""
    arr = pair.t1 if which == "t1" else pair.t2
    idx = {n: i for i, n in enumerate(pair.names)}
    order = ("red", "green", "blue") if all(k in idx for k in ("red", "green", "blue")) else ("nir", "red", "green")
    order = [k for k in order if k in idx]
    if len(order) < 3:
        g = arr.mean(axis=2)
        img = np.dstack([g, g, g])
    else:
        img = np.dstack([arr[:, :, idx[k]] for k in order])
    lo, hi = np.percentile(img, (1, 99.5))
    img = np.clip((img - lo) / max(hi - lo, 1e-6), 0, 1) ** 0.85
    return (img * 255).astype(np.uint8)


def save_png(path: str, arr: np.ndarray) -> None:
    Image.fromarray(arr).save(path, optimize=True)


def _outline(mask: np.ndarray) -> np.ndarray:
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    return (mask.astype(np.uint8) - cv2.erode(mask.astype(np.uint8), k)) > 0


def binary_map(mask: np.ndarray) -> np.ndarray:
    return (mask > 0).astype(np.uint8) * 255


def overlay_changes(base: np.ndarray, mask: np.ndarray) -> np.ndarray:
    out = base.astype(np.float32).copy()
    m = mask > 0
    red = np.array(CLASS_COLORS[1], np.float32)
    out[m] = out[m] * 0.7 + red * 0.3
    out[_outline(m)] = red
    return out.astype(np.uint8)


def class_layer(labels: np.ndarray, cls: int) -> np.ndarray:
    """RGBA layer for one class (translucent fill + solid outline)."""
    h, w = labels.shape
    m = labels == cls
    rgba = np.zeros((h, w, 4), np.uint8)
    color = CLASS_COLORS[cls]
    rgba[m] = (*color, 105)
    rgba[_outline(m)] = (*color, 255)
    return rgba


def composite(base: np.ndarray, labels: np.ndarray) -> np.ndarray:
    out = base.astype(np.float32).copy()
    for cls in (1, 2, 3):
        layer = class_layer(labels, cls).astype(np.float32)
        a = layer[:, :, 3:4] / 255.0
        out = out * (1 - a) + layer[:, :, :3] * a
    return out.astype(np.uint8)
