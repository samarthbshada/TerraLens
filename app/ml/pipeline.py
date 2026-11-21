"""End-to-end TerraLens pipeline:

  load -> quality screening -> size harmonisation -> radiometric match ->
  registration -> change detection -> human-activity attribution -> maps
"""
from __future__ import annotations

import os
import time
from dataclasses import dataclass, field

import numpy as np

from . import render
from .attribute import HUMAN, NATURAL, ENV, attribute_changes
from .detectors import build_detector
from .imaging import load_scene
from .preprocess import prepare_pair


@dataclass
class PipelineResult:
    metrics: dict
    files: dict
    warnings: list = field(default_factory=list)
    regions: list = field(default_factory=list)
    insights: list = field(default_factory=list)


def _insights(m: dict) -> list:
    if m["changed_pixels"] == 0:
        return ["No significant change was detected between the two images.",
                "Check that both images cover the same area and were captured on different dates."]
    pct = {HUMAN: m["human_pct"], NATURAL: m["natural_pct"], ENV: m["env_pct"]}
    names = {HUMAN: "Human-made", NATURAL: "Natural", ENV: "Environmental"}
    major = max(pct, key=pct.get)
    out = [f"Major change type: {names[major]} ({pct[major]:.1f}%)"]
    if pct[HUMAN] >= 40:
        out.append("Indicative of construction, expansion or infrastructure development.")
    elif pct[HUMAN] > 0:
        out.append("Some built-up structure change was found; review the highlighted regions.")
    if pct[NATURAL] > 0:
        out.append("Natural changes are mainly vegetation growth or loss without built-up structure.")
    if pct[ENV] > 0:
        out.append("Environmental changes are likely water-extent shifts or seasonal / land-cover variation.")
    if m["low_conf_regions"]:
        out.append(f"{m['low_conf_regions']} region(s) have low confidence and need manual review.")
    return out


def run_pipeline(t1_path: str, t2_path: str, out_dir: str, *, layout_t1="rgb", layout_t2="rgb",
                 fallback_res_m=10.0, detector_kind="classical", onnx_path=None, max_side=1024,
                 max_pixels=40_000_000) -> PipelineResult:
    t0 = time.time()
    os.makedirs(out_dir, exist_ok=True)
    s1 = load_scene(t1_path, layout_t1, max_side=max_side, max_pixels=max_pixels)
    s2 = load_scene(t2_path, layout_t2, max_side=max_side, max_pixels=max_pixels)
    pair = prepare_pair(s1, s2, fallback_res_m=fallback_res_m)
    detector, det_warnings = build_detector(detector_kind, onnx_path)
    try:
        det = detector.detect(pair)
    except Exception as exc:        # a broken optional model must not break the analysis
        from .detectors import ClassicalDetector
        det_warnings.append(f"Detector '{detector.name}' failed ({exc}); used the classical baseline.")
        det = ClassicalDetector().detect(pair)
    att = attribute_changes(pair, det.mask)

    h, w = pair.shape
    valid_px = int(pair.valid.sum())
    changed_px = int((det.mask > 0).sum())
    px_km2 = (pair.pixel_size_m ** 2) / 1e6
    cls_px = {c: int((att.labels == c).sum()) for c in (HUMAN, NATURAL, ENV)}
    cls_tot = max(1, sum(cls_px.values()))

    base1, base2 = render.display_rgb(pair, "t1"), render.display_rgb(pair, "t2")
    files = {
        "t1.png": base1, "t2.png": base2,
        "overlay.png": render.overlay_changes(base2, det.mask),
        "changemap.png": render.binary_map(det.mask),
        "classified.png": render.composite(base2, att.labels),
        "layer_human.png": render.class_layer(att.labels, HUMAN),
        "layer_natural.png": render.class_layer(att.labels, NATURAL),
        "layer_env.png": render.class_layer(att.labels, ENV),
    }
    paths = {}
    for name, arr in files.items():
        p = os.path.join(out_dir, name)
        render.save_png(p, arr)
        paths[name] = p

    metrics = {
        "detector": det.detector,
        "width": w, "height": h,
        "pixel_size_m": round(pair.pixel_size_m, 3),
        "scene_area_km2": round(valid_px * px_km2, 3),
        "changed_pixels": changed_px,
        "area_changed_pct": round(100.0 * changed_px / max(1, valid_px), 2),
        "total_change_area_km2": round(changed_px * px_km2, 4),
        "n_regions": len(att.regions),
        "low_conf_regions": att.low_confidence_regions,
        "human_regions": sum(1 for r in att.regions if r["label"] == HUMAN),
        "natural_regions": sum(1 for r in att.regions if r["label"] == NATURAL),
        "env_regions": sum(1 for r in att.regions if r["label"] == ENV),
        "human_km2": round(cls_px[HUMAN] * px_km2, 4),
        "natural_km2": round(cls_px[NATURAL] * px_km2, 4),
        "env_km2": round(cls_px[ENV] * px_km2, 4),
        "human_pct": round(100.0 * cls_px[HUMAN] / cls_tot, 1) if changed_px else 0.0,
        "natural_pct": round(100.0 * cls_px[NATURAL] / cls_tot, 1) if changed_px else 0.0,
        "env_pct": round(100.0 * cls_px[ENV] / cls_tot, 1) if changed_px else 0.0,
        "registration": pair.info.get("registration"),
        "runtime_s": round(time.time() - t0, 2),
    }
    warnings = list(pair.warnings) + det_warnings + list(att.notes)
    if not (s1.pixel_size_m or s2.pixel_size_m):
        warnings.append(f"Ground resolution taken from your setting ({fallback_res_m:g} m/pixel); "
                        "areas are only as accurate as that value.")
    regions = sorted(att.regions, key=lambda r: -r["area_px"])[:60]
    for r in regions:
        r["area_km2"] = round(r["area_px"] * px_km2, 5)
    return PipelineResult(metrics=metrics, files=paths, warnings=warnings,
                          regions=regions, insights=_insights(metrics))
