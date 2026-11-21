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


class ClassicalDetector(ChangeDetector):
    name = "Classical baseline (CVA + structure)"

    def __init__(self, abs_floor: float = 0.06, noise_k: float = 6.0):
        self.abs_floor = abs_floor
        self.noise_k = noise_k

    def detect(self, pair: PreparedPair) -> DetectionOutput:
        t1, t2 = pair.t1, pair.t2
        h, w = pair.shape
        # 1) spectral change-vector magnitude (reflectance units)
        spec = np.sqrt(np.mean((t2 - t1) ** 2, axis=2)).astype(np.float32)
        spec = cv2.GaussianBlur(spec, (0, 0), 1.2)

        # 2) structural change: difference of local edge energy
        def edge_energy(g):
            gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
            gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
            return cv2.GaussianBlur(np.sqrt(gx * gx + gy * gy), (0, 0), 2.0)
        struct = np.abs(edge_energy(pair.gray("t2")) - edge_energy(pair.gray("t1")))
        struct = struct / (np.percentile(struct, 99) + 1e-6) * 0.5      # comparable scale

        mag = (spec + 0.35 * struct).astype(np.float32)
        mag[~pair.valid] = 0.0

        # 3) adaptive threshold: Otsu on the magnitude, but never below a noise floor
        vals = mag[pair.valid]
        med = float(np.median(vals))
        mad = float(np.median(np.abs(vals - med)) * 1.4826)
        noise_thr = med + self.noise_k * mad
        u8 = np.clip(mag / (np.percentile(vals, 99.5) + 1e-6) * 255, 0, 255).astype(np.uint8)
        otsu_val, _ = cv2.threshold(u8[pair.valid].reshape(-1, 1), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        otsu_thr = float(otsu_val) / 255.0 * (np.percentile(vals, 99.5) + 1e-6)
        thr = max(noise_thr, self.abs_floor, 0.35 * otsu_thr)

        # hysteresis: keep weak pixels only when connected to strong ones
        strong = (mag > thr * 1.25).astype(np.uint8)
        weak = (mag > thr).astype(np.uint8)
        n, lab = cv2.connectedComponents(weak, connectivity=8)
        has_strong = np.zeros(n, bool)
        has_strong[np.unique(lab[strong > 0])] = True
        has_strong[0] = False
        mask = has_strong[lab].astype(np.uint8)

        min_area = max(24, int(h * w * 0.0002))
        mask = _clean_mask(mask, min_area)
        notes = [f"threshold={thr:.3f}"]
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


def build_detector(kind: str = "classical", onnx_path: str | None = None):
    """Return (detector, warnings). Falls back to classical on any problem."""
    warnings = []
    if kind == "onnx":
        try:
            if not onnx_path or not os.path.exists(onnx_path):
                raise FileNotFoundError("CD_ONNX_MODEL path not found")
            return OnnxDetector(onnx_path), warnings
        except Exception as exc:   # optional feature: never take the app down
            warnings.append(f"Deep model unavailable ({exc}); used the classical baseline instead.")
    return ClassicalDetector(), warnings
