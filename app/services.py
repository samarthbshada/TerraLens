"""Application services: uploads, previews and running an analysis."""
from __future__ import annotations

import io
import os
import re
import uuid
import zipfile

from flask import current_app
from werkzeug.utils import secure_filename

from .ml.imaging import SceneError, load_scene, save_preview
from .ml.pipeline import run_pipeline

OUTPUT_FILES = ("t1.png", "t2.png", "overlay.png", "changemap.png", "classified.png",
                "layer_human.png", "layer_natural.png", "layer_env.png")
SAFE_NAME = re.compile(r"^[a-z0-9_]+\.png$")


def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in current_app.config["ALLOWED_EXTENSIONS"]


def output_dir(pair_id: int) -> str:
    return os.path.join(current_app.config["OUTPUT_FOLDER"], f"pair_{pair_id}")


def safe_remove(*paths):
    for p in paths:
        try:
            if p and os.path.isfile(p):
                os.remove(p)
        except OSError:
            pass


def remove_pair_files(pair_id: int, image_files, preview_files=()):
    safe_remove(*image_files, *preview_files)
    d = output_dir(pair_id)
    if os.path.isdir(d):
        for name in os.listdir(d):
            safe_remove(os.path.join(d, name))
        try:
            os.rmdir(d)
        except OSError:
            pass


def store_upload(file_storage, user_id: int, tag: str, layout: str):
    """Save an uploaded file, validate that it can be decoded and create its preview.

    Returns a dict with everything needed for the satellite_images row.
    Raises SceneError with a user-readable message on failure (and cleans up)."""
    if not file_storage or not file_storage.filename:
        raise SceneError(f"Please choose the {tag.upper()} image.")
    original = secure_filename(file_storage.filename) or f"{tag}.tif"
    if not allowed_file(original):
        raise SceneError(f"{tag.upper()}: unsupported file type. Use GeoTIFF, TIFF, PNG or JPG.")
    folder = os.path.join(current_app.config["UPLOAD_FOLDER"], str(user_id))
    os.makedirs(folder, exist_ok=True)
    stem = uuid.uuid4().hex[:12]
    path = os.path.join(folder, f"{stem}_{original}")
    file_storage.save(path)
    preview_name = f"{stem}_preview.png"
    preview_path = os.path.join(folder, preview_name)
    try:
        scene = load_scene(path, layout, max_side=1024, max_pixels=current_app.config["MAX_IMAGE_PIXELS"])
        save_preview(scene, preview_path)
    except SceneError:
        safe_remove(path, preview_path)
        raise
    except Exception as exc:                                   # corrupt file, unexpected format ...
        current_app.logger.warning("upload decode failed: %s", exc)
        safe_remove(path, preview_path)
        raise SceneError(f"{tag.upper()}: this file could not be read as an image.") from exc
    return {
        "original_filename": original, "file_path": path, "preview_name": preview_name,
        "file_size": os.path.getsize(path), "width": scene.orig_width, "height": scene.orig_height,
        "band_count": scene.n_bands, "band_layout": scene.layout, "pixel_size_m": scene.pixel_size_m,
        "warnings": scene.warnings,
    }


def preview_path(user_id: int, preview_name: str) -> str:
    return os.path.join(current_app.config["UPLOAD_FOLDER"], str(user_id), os.path.basename(preview_name))


def run_analysis(pair_row: dict):
    cfg = current_app.config
    return run_pipeline(
        pair_row["t1_path"], pair_row["t2_path"], output_dir(pair_row["id"]),
        layout_t1=pair_row["t1_layout"] or "rgb", layout_t2=pair_row["t2_layout"] or "rgb",
        fallback_res_m=float(pair_row.get("ground_res_m") or 10.0),
        detector_kind=cfg["CD_DETECTOR"], onnx_path=cfg.get("CD_ONNX_MODEL"),
        max_side=cfg["ANALYSIS_MAX_SIDE"], max_pixels=cfg["MAX_IMAGE_PIXELS"],
        sensitivity=pair_row.get("sensitivity") or "medium")


def results_zip(pair_row: dict, result: dict) -> bytes:
    """ZIP with a CSV summary + regions table and the map images."""
    import csv
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        out = io.StringIO()
        w = csv.writer(out)
        w.writerow(["TerraLens analysis report"])
        w.writerow(["Analysis", pair_row["name"]])
        w.writerow(["Source", pair_row.get("source") or ""])
        w.writerow(["T1", pair_row["t1_name"], pair_row.get("t1_date") or ""])
        w.writerow(["T2", pair_row["t2_name"], pair_row.get("t2_date") or ""])
        w.writerow(["Detector", result["detector"]])
        w.writerow(["Ground resolution (m/pixel, analysis grid)", result["pixel_size_m"]])
        w.writerow([])
        w.writerow(["Metric", "Value"])
        for label, key in [("Scene area (km2)", "scene_area_km2"), ("Total change area (km2)", "total_change_area_km2"),
                           ("Changed pixels", "changed_pixels"), ("Area changed (%)", "area_changed_pct"),
                           ("Change regions", "n_regions"), ("Low-confidence regions", "low_conf_regions"),
                           ("Human-made (km2)", "human_km2"), ("Natural (km2)", "natural_km2"),
                           ("Environmental (km2)", "env_km2"), ("Human-made (% of change)", "human_pct"),
                           ("Natural (% of change)", "natural_pct"), ("Environmental (% of change)", "env_pct")]:
            w.writerow([label, result[key]])
        w.writerow([])
        w.writerow(["Region", "Class", "Area (km2)", "Confidence", "Reason"])
        for r in result["regions"]:
            w.writerow([r["id"], r["class"], r.get("area_km2", ""), r["confidence"], r["reason"]])
        w.writerow([])
        w.writerow(["Note", "Attribution is a rule-based spectral/structural interpretation, not proof of cause."])
        z.writestr("terralens_report.csv", out.getvalue())
        d = output_dir(pair_row["id"])
        for name in ("overlay.png", "changemap.png", "classified.png"):
            p = os.path.join(d, name)
            if os.path.isfile(p):
                z.write(p, name)
    return buf.getvalue()
