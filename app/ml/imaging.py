"""Reading satellite scenes (GeoTIFF / TIFF / PNG / JPG) into normalised arrays.

Only light dependencies are used (tifffile, Pillow, OpenCV, NumPy) so the
application installs quickly on hosting platforms such as Railway.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field

import cv2
import numpy as np
from PIL import Image

Image.MAX_IMAGE_PIXELS = None  # we enforce our own pixel limit below

# band order of the uploaded file -> names of the channels
BAND_LAYOUTS = {
    "rgb": ("red", "green", "blue"),
    "rgbn": ("red", "green", "blue", "nir"),
    "gr_nir": ("green", "red", "nir"),   # LISS-IV B2, B3, B4
}
LAYOUT_LABELS = {
    "rgb": "RGB (3 bands: R, G, B)",
    "rgbn": "RGB + NIR (4 bands: R, G, B, NIR)",
    "gr_nir": "LISS-IV (3 bands: Green, Red, NIR)",
}
DEFAULT_RES_M = {"Sentinel-2": 10.0, "LISS-IV": 5.8}


class SceneError(ValueError):
    """Raised with a user-readable message when a file cannot be used."""


@dataclass
class Scene:
    bands: dict                      # name -> float32 HxW array in [0, 1]
    layout: str
    width: int                       # size of the stored arrays
    height: int
    orig_width: int
    orig_height: int
    pixel_size_m: float | None       # metres per ORIGINAL pixel if the file says so
    scale: float = 1.0               # stored size / original size
    warnings: list = field(default_factory=list)

    @property
    def has_nir(self) -> bool:
        return "nir" in self.bands

    @property
    def n_bands(self) -> int:
        return len(self.bands)

    @property
    def has_visible(self) -> bool:
        return all(k in self.bands for k in ("red", "green", "blue"))

    def stack(self, names=None) -> np.ndarray:
        names = names or list(self.bands)
        return np.dstack([self.bands[n] for n in names]).astype(np.float32)

    def gray(self) -> np.ndarray:
        vis = [self.bands[n] for n in ("red", "green", "blue") if n in self.bands]
        if not vis:
            vis = list(self.bands.values())
        return np.mean(vis, axis=0).astype(np.float32)

    def rgb_for_display(self) -> np.ndarray:
        """uint8 HxWx3 image that browsers can show (true colour when possible)."""
        if self.has_visible:
            order = ("red", "green", "blue")
        elif self.layout == "gr_nir":
            order = ("nir", "red", "green")       # standard false-colour composite
        else:
            g = (self.gray() * 255.0).clip(0, 255).astype(np.uint8)
            return np.dstack([g, g, g])
        img = np.dstack([self.bands[k] for k in order])
        return (img * 255.0).clip(0, 255).astype(np.uint8)


# ----------------------------------------------------------------------------
def _is_tiff(path: str) -> bool:
    return os.path.splitext(path)[1].lower() in (".tif", ".tiff")


def _tiff_pixel_size(tf) -> float | None:
    """Metres per pixel from the GeoTIFF ModelPixelScale tag (projected CRS only)."""
    try:
        tag = tf.pages[0].tags.get(33550)
        if tag is None:
            return None
        sx = float(tag.value[0])
        # values < 0.5 are almost certainly degrees (geographic CRS) -> unknown
        return sx if 0.5 <= sx <= 5000 else None
    except Exception:
        return None


def probe(path: str) -> dict:
    """Cheap metadata read (no full decode)."""
    try:
        if _is_tiff(path):
            import tifffile
            with tifffile.TiffFile(path) as tf:
                page = tf.pages[0]
                shape = page.shape
                if len(shape) == 2:
                    h, w, c = shape[0], shape[1], 1
                    if len(tf.pages) > 1:
                        c = len(tf.pages)
                elif shape[0] <= 16 and shape[0] < shape[-1]:
                    c, h, w = shape[0], shape[1], shape[2]
                else:
                    h, w, c = shape[0], shape[1], shape[2]
                return {"width": int(w), "height": int(h), "bands": int(c),
                        "pixel_size_m": _tiff_pixel_size(tf)}
        with Image.open(path) as im:
            w, h = im.size
            return {"width": w, "height": h, "bands": len(im.getbands()), "pixel_size_m": None}
    except Exception as exc:
        raise SceneError("Could not read this image file. Please upload a valid GeoTIFF, "
                         "TIFF, PNG or JPG.") from exc


def _read_array(path: str) -> np.ndarray:
    """Return HxWxC array in its native dtype."""
    if _is_tiff(path):
        import tifffile
        arr = tifffile.imread(path)
        if arr.ndim == 2:
            arr = arr[:, :, None]
        elif arr.ndim == 3 and arr.shape[0] <= 16 and arr.shape[0] < arr.shape[-1]:
            arr = np.transpose(arr, (1, 2, 0))
        elif arr.ndim > 3:
            arr = arr.reshape((-1,) + arr.shape[-2:])
            arr = np.transpose(arr, (1, 2, 0))
        return arr
    with Image.open(path) as im:
        if im.mode not in ("RGB", "L", "I;16", "I", "F"):
            im = im.convert("RGB")           # drops alpha / palette / CMYK so alpha is never read as a band
        arr = np.asarray(im)
    if arr.ndim == 2:
        arr = arr[:, :, None]
    return arr


def load_scene(path: str, layout: str = "rgb", max_side: int = 1024,
               max_pixels: int = 40_000_000) -> Scene:
    if layout not in BAND_LAYOUTS:
        raise SceneError("Unknown band order.")
    meta = probe(path)
    if meta["width"] * meta["height"] > max_pixels:
        raise SceneError(
            f"Image is {meta['width']}x{meta['height']} px, larger than the "
            f"{max_pixels / 1e6:.0f} megapixel limit. Please crop or resample it first.")
    arr = _read_array(path)
    h, w, c = arr.shape
    names = BAND_LAYOUTS[layout]
    warnings: list = []

    if c == 1:
        arr = np.repeat(arr, 3, axis=2)
        layout, names, c = "rgb", BAND_LAYOUTS["rgb"], 3
        warnings.append("Single-band image: vegetation and water indices are unavailable, "
                        "so human/natural attribution relies on structure only.")
    if c < len(names):
        if c == 3 and layout == "rgbn":
            layout, names = "rgb", BAND_LAYOUTS["rgb"]
            warnings.append("Only 3 bands found; treated as RGB (no NIR).")
        else:
            raise SceneError(f"The selected band order needs {len(names)} bands "
                             f"but the file has {c}.")
    arr = arr[:, :, :len(names)]

    valid = None
    if arr.dtype != np.uint8:
        valid = np.any(arr != 0, axis=2)           # 0 is the usual nodata value
    scale = max_side / float(max(h, w)) if max(h, w) > max_side else 1.0
    nh, nw = max(8, int(round(h * scale))), max(8, int(round(w * scale)))

    # One common gain for all bands keeps band ratios (NDVI / NDWI) meaningful.
    if arr.dtype == np.uint8:
        gain_hi = 255.0
    else:
        sample = arr[valid] if valid is not None and valid.any() else arr.reshape(-1, arr.shape[2])
        if len(sample) > 2_000_000:
            sample = sample[:: max(1, len(sample) // 2_000_000)]
        gain_hi = float(np.percentile(sample, 99.5)) or float(sample.max()) or 1.0
    bands = {}
    for i, name in enumerate(names):
        unit = np.clip(arr[:, :, i].astype(np.float32) / gain_hi, 0.0, 1.0)
        if (nh, nw) != (h, w):
            unit = cv2.resize(unit, (nw, nh), interpolation=cv2.INTER_AREA)
        bands[name] = unit.astype(np.float32)
    return Scene(bands=bands, layout=layout, width=nw, height=nh, orig_width=w, orig_height=h,
                 pixel_size_m=meta.get("pixel_size_m"), scale=nw / float(w), warnings=warnings)


def save_preview(scene: Scene, out_path: str) -> None:
    Image.fromarray(scene.rgb_for_display()).save(out_path, optimize=True)
