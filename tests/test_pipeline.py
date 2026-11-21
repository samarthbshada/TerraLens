"""Accuracy sanity tests of the analysis engine on SYNTHETIC scenes with known ground truth.

These prove the plumbing and the rules behave as designed; they say nothing about
accuracy on real satellite imagery (see README, section 'What the model is').
"""
import os, sys, tempfile, shutil, unittest
import numpy as np
import cv2

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "tools"))

from synthetic import make_pair, make_no_change_pair, write_png            # noqa: E402
from app.ml.preprocess import PreparedPair, _register, _match_radiometry   # noqa: E402
from app.ml.detectors import ClassicalDetector, build_detector             # noqa: E402
from app.ml.attribute import attribute_changes                             # noqa: E402
from app.ml.pipeline import run_pipeline                                   # noqa: E402


def analyse(t1, t2, nir=True):
    names = ("red", "green", "blue", "nir") if nir else ("red", "green", "blue")
    t2r, valid, _ = _register(t1, t2)
    pair = PreparedPair(t1, _match_radiometry(t1, t2r, valid), names, 10.0, valid, {})
    det = ClassicalDetector().detect(pair)
    return det, attribute_changes(pair, det.mask)


class PipelineTest(unittest.TestCase):
    def test_detection_and_classes_with_nir(self):
        t1, t2, truth = make_pair(512, 7, True)
        det, att = analyse(t1, t2, True)
        gt = truth > 0
        inter = (det.mask > 0) & gt
        self.assertGreater(inter.sum() / ((det.mask > 0) | gt).sum(), 0.75, "detection IoU")
        for cls, minimum in ((1, 0.80), (2, 0.85), (3, 0.85)):
            g = (truth == cls) & (att.labels > 0)
            self.assertGreater(((att.labels == cls) & g).sum() / g.sum(), minimum, f"class {cls} correct share")

    def test_rgb_only_still_separates_classes(self):
        t1, t2, truth = make_pair(512, 7, False)
        det, att = analyse(t1, t2, False)
        for cls in (1, 2, 3):
            g = (truth == cls) & (att.labels > 0)
            self.assertGreater(g.sum(), 0)
            self.assertGreater(((att.labels == cls) & g).sum() / g.sum(), 0.70, f"class {cls}")

    def test_no_change_pair_is_quiet(self):
        for nir in (True, False):
            a, b, _ = make_no_change_pair(512, 3, nir)
            det, att = analyse(a, b, nir)
            self.assertLess(det.mask.mean(), 0.005)
            self.assertEqual(int((att.labels > 0).sum()), 0)

    def test_registration_corrects_translation(self):
        t1, t2, _ = make_pair(512, 7, True)
        shifted = cv2.warpAffine(t2, np.float32([[1, 0, 9], [0, 1, -6]]), (512, 512), borderMode=cv2.BORDER_REFLECT)
        det_ok, _ = analyse(t1, t2, True)
        det_sh, _ = analyse(t1, shifted, True)
        self.assertLess(abs(det_sh.mask.mean() - det_ok.mask.mean()), 0.02)

    def test_blurred_misaligned_recoloured_pair_still_detects(self):
        """Regression: real pairs differ in sharpness, colour and tiny misalignment everywhere.
        The old detector's noise estimate exploded and it reported zero change."""
        t1, t2, truth = make_pair(512, 7, True)
        t2 = cv2.GaussianBlur(t2, (0, 0), 1.3)
        t2 = cv2.warpAffine(t2, np.float32([[1, 0, 1.4], [0, 1, -1.1]]), (512, 512), borderMode=cv2.BORDER_REFLECT)
        t2 = np.clip(t2 * np.array([1.12, 1.08, 1.15, 1.0], np.float32) + 0.03, 0, 1)
        det, _ = analyse(t1, t2, True)
        gt = truth > 0
        inter = (det.mask > 0) & gt
        self.assertGreater(det.mask.sum(), 0, "must not report 'no change'")
        self.assertGreater(inter.sum() / gt.sum(), 0.6, "recall")
        self.assertLess(det.mask.mean(), 0.30, "must not flag most of the scene")

    def test_sensitivity_orders_results(self):
        t1, t2, _ = make_pair(512, 7, True)
        names = ("red", "green", "blue", "nir")
        pair = PreparedPair(t1, t2, names, 10.0, np.ones((512, 512), bool), {})
        frac = {k: ClassicalDetector(sensitivity=k).detect(pair).mask.mean() for k in ("low", "medium", "high")}
        self.assertLessEqual(frac["low"], frac["medium"]); self.assertLessEqual(frac["medium"], frac["high"])

    def test_png_alpha_is_never_read_as_a_band(self):
        from PIL import Image
        from app.ml.imaging import load_scene
        d = tempfile.mkdtemp()
        try:
            rgba = np.dstack([(np.random.rand(64, 64, 3) * 255).astype(np.uint8), np.full((64, 64), 255, np.uint8)])
            Image.fromarray(rgba, "RGBA").save(d + "/a.png")
            scene = load_scene(d + "/a.png", "rgbn")          # asks for NIR; alpha must not be used as NIR
            self.assertFalse(scene.has_nir)
            self.assertTrue(any("3 bands" in w for w in scene.warnings))
        finally:
            shutil.rmtree(d, ignore_errors=True)

    def test_files_pipeline_outputs_and_metrics(self):
        d = tempfile.mkdtemp()
        try:
            t1, t2, _ = make_pair(384, 7, True)
            write_png(d + "/a.png", t1); write_png(d + "/b.png", t2)
            r = run_pipeline(d + "/a.png", d + "/b.png", d + "/out", fallback_res_m=10.0)
            for f in ("overlay.png", "changemap.png", "classified.png", "layer_human.png", "t1.png", "t2.png"):
                self.assertTrue(os.path.getsize(os.path.join(d, "out", f)) > 0, f)
            m = r.metrics
            self.assertAlmostEqual(m["human_pct"] + m["natural_pct"] + m["env_pct"], 100.0, delta=0.5)
            self.assertAlmostEqual(m["scene_area_km2"], 384 * 384 * 100 / 1e6, places=2)
        finally:
            shutil.rmtree(d, ignore_errors=True)

    def test_onnx_request_without_model_falls_back(self):
        det, warnings = build_detector("onnx", "/does/not/exist.onnx")
        self.assertIsInstance(det, ClassicalDetector)
        self.assertTrue(warnings)


if __name__ == "__main__":
    unittest.main()
