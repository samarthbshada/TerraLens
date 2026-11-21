"""Change detectors (Strategy pattern).

Every detector takes a PreparedPair and returns a DetectionOutput holding a
binary change mask and a continuous change-magnitude map.

* ClassicalDetector - always available, no weights needed (change-vector
  analysis + structural difference + adaptive threshold).  This is the
  baseline stage of the TerraLens experimental plan.
* OnnxDetector - OPTIONAL adapter for a trained deep model (BIT, ChangeFormer,
  ...) exported to ONNX.  Set CD_ONNX_MODEL=/path/model.onnx to enable it.
  If the model cannot be loaded the application falls back to the classical
  detector and records a warning.
"""
from __future__ import annotations

import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import cv2
import numpy as np

from .preprocess import PreparedPair


@dataclass
class DetectionOutput:
    mask: np.ndarray                  # HxW uint8 (0/1)
    magnitude: np.ndarray             # HxW float32, roughly 0..1
    detector: str
    notes: list = field(default_factory=list)


def _clean_mask(mask: np.ndarray, min_area: int) -> np.ndarray:
    m = mask.astype(np.uint8)
    k3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    k5 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, k3)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k5)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    keep = np.zeros(n, bool)
    keep[1:] = stats[1:, cv2.CC_STAT_AREA] >= min_area
    m = keep[lab].astype(np.uint8)
    # fill small holes
    inv = (1 - m).astype(np.uint8)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(inv, connectivity=4)
    holes = np.zeros(n, bool)
    holes[1:] = stats[1:, cv2.CC_STAT_AREA] < max(20, min_area // 2)
    m[holes[lab]] = 1
    return m


class ChangeDetector(ABC):
    name = "base"

    @abstractmethod
    def detect(self, pair: PreparedPair) -> DetectionOutput:
        ...


SENSITIVITY_K = {"low": 6.0, "medium": 4.5, "high": 3.0}


def _tolerant_difference(t1: np.ndarray, t2: np.ndarray, radius: int = 2) -> np.ndarray:
    """Spectral difference that forgives small misregistration / sharpness differences:
    for each T2 pixel take the smallest difference to any T1 pixel within +-radius pixels."""
    h, w = t1.shape[:2]
    pad = np.pad(t1, ((radius, radius), (radius, radius), (0, 0)), mode="edge")
    best = None
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            shifted = pad[radius + dy:radius + dy + h, radius + dx:radius + dx + w]
            d = np.sqrt(np.mean((t2 - shifted) ** 2, axis=2))
            best = d if best is None else np.minimum(best, d)
    return best.astype(np.float32)


class ClassicalDetector(ChangeDetector):
    name = "Classical baseline (tolerant CVA)"

    def __init__(self, abs_floor: float = 0.06, sensitivity: str = "medium", max_fraction: float = 0.35):
        self.abs_floor = abs_floor
        self.sensitivity = sensitivity if sensitivity in SENSITIVITY_K else "medium"
        self.k = SENSITIVITY_K[self.sensitivity]
        self.max_fraction = max_fraction

    def detect(self, pair: PreparedPair) -> DetectionOutput:
        h, w = pair.shape
        # blur first so differences in sharpness / resolution between the two images matter less
        b1 = np.dstack([cv2.GaussianBlur(pair.t1[:, :, c], (0, 0), 1.2) for c in range(pair.t1.shape[2])])
        b2 = np.dstack([cv2.GaussianBlur(pair.t2[:, :, c], (0, 0), 1.2) for c in range(pair.t2.shape[2])])
        mag = _tolerant_difference(b1, b2, radius=2)
        mag = cv2.GaussianBlur(mag, (0, 0), 1.5)
        mag[~pair.valid] = 0.0

        # noise level estimated from the LOWER half of the distribution only, so that a long tail of
        # genuine change (or edge differences) does not inflate the threshold.
        vals = mag[pair.valid]
        p10, p50 = np.percentile(vals, (10, 50))
        sigma_low = max(float(p50 - p10) / 1.28, 1e-4)
        thr = max(self.abs_floor, float(p50) + self.k * sigma_low)
        notes = [f"sensitivity={self.sensitivity}", f"threshold={thr:.3f}"]

        if (vals > thr).mean() > self.max_fraction:           # e.g. whole scene differs (season / sensor)
            thr = float(np.percentile(vals, 100 * (1 - self.max_fraction)))
            notes.append("very large difference between the images - threshold raised; "
                         "results may reflect seasonal or sensor differences")
        strong = (mag > thr * 1.2).astype(np.uint8)
        weak = (mag > thr).astype(np.uint8)
        n, lab = cv2.connectedComponents(weak, connectivity=8)
        has_strong = np.zeros(n, bool)
        has_strong[np.unique(lab[strong > 0])] = True
        has_strong[0] = False
        mask = has_strong[lab].astype(np.uint8)
        mask = _clean_mask(mask, max(24, int(h * w * 0.0002)))
        return DetectionOutput(mask=mask, magnitude=mag, detector=self.name, notes=notes)


class OnnxDetector(ChangeDetector):
    """Adapter for an ONNX-exported siamese change-detection network.

    Expected model contract (matches exports of BIT / ChangeFormer style nets):
      inputs : two float32 tensors (1,3,H,W) in [-1, 1]  (mean 0.5 / std 0.5)
      output : logits (1,2,H,W)  or  (1,1,H,W)  -> change probability
    The scene is processed in tiles of CD_ONNX_TILE pixels (default 256).
    """
    def __init__(self, model_path: str, tile: int = 256, threshold: float = 0.5):
        import onnxruntime as ort           # optional dependency
        self.session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
        self.inputs = [i.name for i in self.session.get_inputs()]
        if len(self.inputs) != 2:
            raise ValueError("ONNX model must have exactly two image inputs (T1, T2).")
        self.tile, self.threshold = tile, threshold
        self.name = f"ONNX deep model ({os.path.basename(model_path)})"

    @staticmethod
    def _rgb(arr, names):
        idx = [names.index(n) for n in ("red", "green", "blue")]
        return arr[:, :, idx]

    def detect(self, pair: PreparedPair) -> DetectionOutput:
        if not all(pair.has(n) for n in ("red", "green", "blue")):
            raise ValueError("The ONNX model needs RGB bands; this pair has no blue band.")
        a, b = self._rgb(pair.t1, pair.names), self._rgb(pair.t2, pair.names)
        h, w = a.shape[:2]
        T = self.tile
        prob = np.zeros((h, w), np.float32)
        for y in range(0, h, T):
            for x in range(0, w, T):
                ya, xa = min(y, max(0, h - T)), min(x, max(0, w - T))
                pa, pb = a[ya:ya + T, xa:xa + T], b[ya:ya + T, xa:xa + T]
                ph, pw = pa.shape[:2]
                if (ph, pw) != (T, T):
                    pa = cv2.resize(pa, (T, T)); pb = cv2.resize(pb, (T, T))
                ta = ((pa.transpose(2, 0, 1)[None] - 0.5) / 0.5).astype(np.float32)
                tb = ((pb.transpose(2, 0, 1)[None] - 0.5) / 0.5).astype(np.float32)
                out = self.session.run(None, {self.inputs[0]: ta, self.inputs[1]: tb})[0]
                if out.shape[1] >= 2:
                    e = np.exp(out - out.max(axis=1, keepdims=True))
                    p = (e[:, 1] / e.sum(axis=1))[0]
                else:
                    p = 1.0 / (1.0 + np.exp(-out[0, 0]))
                if (ph, pw) != (T, T):
                    p = cv2.resize(p, (pw, ph))
                prob[ya:ya + ph, xa:xa + pw] = p
        mask = _clean_mask(prob > self.threshold, max(24, int(h * w * 0.0002)))
        return DetectionOutput(mask=mask, magnitude=prob, detector=self.name)


def build_detector(kind: str = "classical", onnx_path: str | None = None, sensitivity: str = "medium"):
    """Return (detector, warnings). Falls back to classical on any problem."""
    warnings = []
    if kind == "onnx":
        try:
            if not onnx_path or not os.path.exists(onnx_path):
                raise FileNotFoundError("CD_ONNX_MODEL path not found")
            return OnnxDetector(onnx_path), warnings
        except Exception as exc:   # optional feature: never take the app down
            warnings.append(f"Deep model unavailable ({exc}); used the classical baseline instead.")
    return ClassicalDetector(sensitivity=sensitivity), warnings
