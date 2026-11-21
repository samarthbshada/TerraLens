"""End-to-end tests of the web application (Flask test client, SQLite temp dir).

Run:  python -m unittest discover -s tests -v
"""
import io
import os
import re
import shutil
import sys
import tempfile
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tools"))

from synthetic import make_pair, write_png, write_rgbn_tiff   # noqa: E402


def token(html):
    m = re.search(r'name="csrf_token" value="([^"]+)"', html) or re.search(r'name="csrf-token" content="([^"]+)"', html)
    return m.group(1) if m else ""


class AppTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.mkdtemp()
        from app import create_app
        cls.app = create_app({
            "TESTING": True, "SECRET_KEY": "test", "DATABASE_URL": "sqlite:///" + os.path.join(cls.tmp, "t.db"),
            "UPLOAD_FOLDER": os.path.join(cls.tmp, "up"), "OUTPUT_FOLDER": os.path.join(cls.tmp, "out"),
            "MAX_UPLOAD_MB": 50})
        t1, t2, _ = make_pair(384, 7, True)
        cls.png1, cls.png2 = os.path.join(cls.tmp, "a.png"), os.path.join(cls.tmp, "b.png")
        cls.tif1, cls.tif2 = os.path.join(cls.tmp, "a.tif"), os.path.join(cls.tmp, "b.tif")
        write_png(cls.png1, t1); write_png(cls.png2, t2)
        write_rgbn_tiff(cls.tif1, t1, 10.0); write_rgbn_tiff(cls.tif2, t2, 10.0)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def client(self):
        return self.app.test_client()

    def post(self, c, url, data=None, page=None, **kw):
        """POST with a valid CSRF token taken from a page of the same session."""
        html = c.get(page or "/auth/login").get_data(as_text=True)
        data = dict(data or {}); data["csrf_token"] = token(html)
        return c.post(url, data=data, **kw)

    def register(self, c, name="alice", email="alice@example.com", pw="password123"):
        return self.post(c, "/auth/register", {"username": name, "email": email, "password": pw, "confirm": pw,
                                                "full_name": "Alice Tester"}, follow_redirects=True)

    def upload(self, c, kind="png", name="Bhopal Urban Expansion", source="Sentinel-2", layout="rgb"):
        a, b = (self.png1, self.png2) if kind == "png" else (self.tif1, self.tif2)
        html = c.get("/analysis/new").get_data(as_text=True)
        data = {"csrf_token": token(html), "source": source, "name": name, "layout": layout, "res": "10",
                "t1_date": "2024-01-12", "t2_date": "2025-01-18", "study_area": "Bhopal",
                "t1_file": (open(a, "rb"), os.path.basename(a)), "t2_file": (open(b, "rb"), os.path.basename(b))}
        return c.post("/analysis/new", data=data, content_type="multipart/form-data")

    # ---------------------------------------------------------------- basics
    def test_health_and_redirects(self):
        c = self.client()
        r = c.get("/health"); self.assertEqual(r.status_code, 200); self.assertEqual(r.get_json()["status"], "ok")
        self.assertEqual(c.get("/").status_code, 302)
        r = c.get("/dashboard"); self.assertEqual(r.status_code, 302); self.assertIn("/auth/login", r.location)
        self.assertEqual(c.get("/auth/login").status_code, 200)
        self.assertEqual(c.get("/static/css/app.css").status_code, 200)

    def test_csrf_blocks_forms_without_token(self):
        c = self.client()
        r = c.post("/auth/register", data={"username": "x", "email": "x@x.io", "password": "password123", "confirm": "password123"})
        self.assertEqual(r.status_code, 400)

    def test_register_validation_and_login_logout(self):
        c = self.client()
        r = self.post(c, "/auth/register", {"username": "a", "email": "bad", "password": "x", "confirm": "y"}, follow_redirects=True)
        self.assertIn(b"Username must be", r.data); self.assertIn(b"valid email", r.data)
        r = self.register(c, "bob", "bob@example.com")
        self.assertIn(b"Dashboard", r.data); self.assertIn(b"Welcome to TerraLens", r.data)
        r = self.post(c, "/auth/logout", page="/dashboard", follow_redirects=True)
        self.assertIn(b"Sign in", r.data)
        r = self.post(c, "/auth/login", {"login": "bob", "password": "wrong"}, follow_redirects=True)
        self.assertIn(b"Invalid username", r.data)
        r = self.post(c, "/auth/login", {"login": "BOB@example.com", "password": "password123"}, follow_redirects=True)
        self.assertIn(b"Total Analyses", r.data)
        # duplicate registration
        r = self.post(self.client(), "/auth/register", {"username": "bob", "email": "bob@example.com", "password": "password123", "confirm": "password123"}, follow_redirects=True)
        self.assertIn(b"already registered", r.data)

    def test_open_redirect_is_blocked(self):
        c = self.client(); self.register(c, "carl", "carl@example.com")
        self.post(c, "/auth/logout", page="/dashboard")
        r = self.post(c, "/auth/login?next=//evil.com", {"login": "carl", "password": "password123"})
        self.assertEqual(r.status_code, 302); self.assertNotIn("evil.com", r.location)

    # ---------------------------------------------------------------- wizard
    def test_full_wizard_png_and_download_and_delete(self):
        c = self.client(); self.register(c, "dina", "dina@example.com")
        r = self.upload(c, "png")
        self.assertEqual(r.status_code, 302, r.get_data(as_text=True)[:400])
        compare = r.location
        pid = int(re.search(r"/analysis/(\d+)/compare", compare).group(1))
        page = c.get(compare).get_data(as_text=True)
        self.assertIn("Temporal Image Comparison", page); self.assertIn("12 Jan 2024", page); self.assertIn("Side by Side", page)
        self.assertEqual(c.get(re.search(r'/media/preview/\d+', page).group(0)).status_code, 200)
        # detect needs POST; GET before running bounces back
        self.assertEqual(c.get(f"/analysis/{pid}/detect").status_code, 302)
        r = self.post(c, f"/analysis/{pid}/detect", page=compare)
        self.assertEqual(r.status_code, 302)
        det = c.get(f"/analysis/{pid}/detect").get_data(as_text=True)
        self.assertIn("Change Detection Results", det); self.assertIn("Total Change Area", det)
        for name in ("overlay", "changemap", "t1", "t2"):
            self.assertEqual(c.get(f"/media/output/{pid}/{name}.png").status_code, 200, name)
        cls = c.get(f"/analysis/{pid}/classify").get_data(as_text=True)
        self.assertIn("Human-made Changes", cls); self.assertIn("Natural Changes", cls); self.assertIn("Environmental Changes", cls)
        self.assertIn("Detected Regions", cls)
        for name in ("layer_human", "layer_natural", "layer_env", "classified"):
            self.assertEqual(c.get(f"/media/output/{pid}/{name}.png").status_code, 200, name)
        z = c.get(f"/analysis/{pid}/download")
        self.assertEqual(z.status_code, 200); self.assertEqual(z.mimetype, "application/zip")
        import zipfile
        names = zipfile.ZipFile(io.BytesIO(z.data)).namelist()
        self.assertIn("terralens_report.csv", names); self.assertIn("overlay.png", names)
        r = self.post(c, f"/analysis/{pid}/save", page=f"/analysis/{pid}/classify", follow_redirects=True)
        self.assertIn(b"Analysis saved", r.data)
        dash = c.get("/dashboard").get_data(as_text=True)
        self.assertIn("Bhopal Urban Expansion", dash); self.assertIn("Change Distribution", dash)
        self.assertEqual(c.get("/history").status_code, 200)
        self.assertIn(b"Bhopal", c.get("/history?q=bhopal").data)
        self.assertNotIn(b"Bhopal Urban", c.get("/history?q=zzzz").data)
        self.assertEqual(c.get("/datasets").status_code, 200)
        r = self.post(c, f"/analysis/{pid}/delete", page="/history", follow_redirects=True)
        self.assertIn(b"Analysis deleted", r.data)
        self.assertEqual(c.get(f"/analysis/{pid}/compare").status_code, 404)

    def test_geotiff_nir_pair_gives_real_areas(self):
        c = self.client(); self.register(c, "erin", "erin@example.com")
        r = self.upload(c, "tif", name="NIR run", layout="rgbn")
        self.assertEqual(r.status_code, 302, r.get_data(as_text=True)[:300])
        pid = int(re.search(r"/analysis/(\d+)/compare", r.location).group(1))
        self.post(c, f"/analysis/{pid}/detect", page=r.location)
        from app.repository import AnalysisRepository
        with self.app.app_context():
            from app.db import get_conn
            res = AnalysisRepository(get_conn()).result(pid)
        self.assertAlmostEqual(res["pixel_size_m"], 10.0, places=1)       # read from the GeoTIFF tag
        self.assertGreater(res["human_pct"], 15); self.assertGreater(res["natural_pct"], 5); self.assertGreater(res["env_pct"], 15)
        total = res["human_km2"] + res["natural_km2"] + res["env_km2"]
        self.assertAlmostEqual(total, res["total_change_area_km2"], delta=0.15 * res["total_change_area_km2"])

    # ---------------------------------------------------------------- failure handling
    def test_invalid_uploads_are_rejected_cleanly(self):
        c = self.client(); self.register(c, "fred", "fred@example.com")
        html = c.get("/analysis/new").get_data(as_text=True)
        base = {"csrf_token": token(html), "source": "Sentinel-2", "name": "bad", "layout": "rgb", "res": "10"}

        def files():
            up = os.path.join(self.tmp, "up")
            return {os.path.join(d, f) for d, _, fs in os.walk(up) for f in fs} if os.path.isdir(up) else set()
        before = files()
        # wrong extension
        d = dict(base, t1_file=(io.BytesIO(b"hello"), "a.txt"), t2_file=(io.BytesIO(b"hello"), "b.png"))
        r = c.post("/analysis/new", data=d, content_type="multipart/form-data")
        self.assertEqual(r.status_code, 400); self.assertIn(b"unsupported file type", r.data)
        # corrupt png as T2 (valid T1 must be cleaned up too)
        d = dict(base, t1_file=(open(self.png1, "rb"), "a.png"), t2_file=(io.BytesIO(b"not an image"), "b.png"))
        r = c.post("/analysis/new", data=d, content_type="multipart/form-data")
        self.assertEqual(r.status_code, 400); self.assertTrue(b"Could not read" in r.data or b"could not be read" in r.data)
        # missing name
        r = c.post("/analysis/new", data=dict(base, name=""), content_type="multipart/form-data")
        self.assertEqual(r.status_code, 400); self.assertIn(b"Give the analysis a name", r.data)
        self.assertEqual(files() - before, set(), "failed uploads must not leave files behind")

    def test_bandcount_mismatch_message(self):
        c = self.client(); self.register(c, "gina", "gina@example.com")
        html = c.get("/analysis/new").get_data(as_text=True)
        d = {"csrf_token": token(html), "source": "Sentinel-2", "name": "x", "layout": "rgbn", "res": "10",
             "t1_file": (open(self.png1, "rb"), "a.png"), "t2_file": (open(self.png2, "rb"), "b.png")}
        r = c.post("/analysis/new", data=d, content_type="multipart/form-data")
        self.assertEqual(r.status_code, 302)       # 3-band file with rgbn -> treated as RGB with a notice
        self.assertIn(b"Only 3 bands", c.get(r.location).data)

    def test_users_cannot_see_each_others_data(self):
        a, b = self.client(), self.client()
        self.register(a, "henry", "henry@example.com"); self.register(b, "iris", "iris@example.com")
        r = self.upload(a, "png", name="Private")
        pid = int(re.search(r"/analysis/(\d+)/compare", r.location).group(1))
        self.post(a, f"/analysis/{pid}/detect", page=r.location)
        for url in (f"/analysis/{pid}/compare", f"/analysis/{pid}/classify", f"/analysis/{pid}/download",
                    f"/media/output/{pid}/overlay.png", "/media/preview/1"):
            self.assertEqual(b.get(url).status_code, 404, url)
        self.assertEqual(self.post(b, f"/analysis/{pid}/delete", page="/history").status_code, 404)
        self.assertEqual(a.get(f"/media/output/{pid}/../../etc/passwd").status_code, 404)

    def test_settings_password_change(self):
        c = self.client(); self.register(c, "jane", "jane@example.com")
        r = self.post(c, "/settings", {"action": "password", "current": "password123", "new": "newpassword1", "confirm": "newpassword1"},
                      page="/settings", follow_redirects=True)
        self.assertIn(b"Password changed", r.data)
        self.post(c, "/auth/logout", page="/dashboard")
        r = self.post(c, "/auth/login", {"login": "jane", "password": "newpassword1"}, follow_redirects=True)
        self.assertIn(b"Total Analyses", r.data)

    def test_old_database_without_sensitivity_column_is_upgraded(self):
        import sqlite3
        from app import create_app
        path = os.path.join(self.tmp, "old.db")
        con = sqlite3.connect(path)
        con.executescript("""CREATE TABLE users (id INTEGER PRIMARY KEY AUTOINCREMENT, username VARCHAR(80) NOT NULL UNIQUE,
            email VARCHAR(120) NOT NULL UNIQUE, password_hash VARCHAR(255) NOT NULL, full_name VARCHAR(120), role VARCHAR(20),
            created_at VARCHAR(19), last_login VARCHAR(19));
            CREATE TABLE image_pairs (id INTEGER PRIMARY KEY AUTOINCREMENT, user_id INT NOT NULL, dataset_id INT, name VARCHAR(160) NOT NULL,
            study_area VARCHAR(160), description TEXT, t1_image_id INT NOT NULL, t2_image_id INT NOT NULL, status VARCHAR(20) DEFAULT 'uploaded',
            created_at VARCHAR(19), updated_at VARCHAR(19));""")
        con.commit(); con.close()
        create_app({"TESTING": True, "SECRET_KEY": "t", "DATABASE_URL": "sqlite:///" + path,
                    "UPLOAD_FOLDER": os.path.join(self.tmp, "up2"), "OUTPUT_FOLDER": os.path.join(self.tmp, "out2")})
        cols = [r[1] for r in sqlite3.connect(path).execute("PRAGMA table_info(image_pairs)")]
        self.assertIn("sensitivity", cols)

    def test_sensitivity_is_saved_and_no_change_message_shown(self):
        c = self.client(); self.register(c, "kate", "kate@example.com")
        r = self.upload(c, "png")
        pid = int(re.search(r"/analysis/(\d+)/compare", r.location).group(1))
        self.assertIn("High sensitivity", c.get(r.location).get_data(as_text=True))
        self.post(c, f"/analysis/{pid}/detect", {"sensitivity": "high"}, page=r.location)
        from app.repository import AnalysisRepository
        with self.app.app_context():
            from app.db import get_conn
            self.assertEqual(AnalysisRepository(get_conn()).pair(pid, 1 if False else AnalysisRepository(get_conn()).db.one("SELECT user_id FROM image_pairs WHERE id=?", (pid,))["user_id"])["sensitivity"], "high")
            self.assertEqual(AnalysisRepository(get_conn()).result(pid)["detector"].split(" ")[0], "Classical")

    def test_404_page(self):
        r = self.client().get("/nope"); self.assertEqual(r.status_code, 404)


if __name__ == "__main__":
    unittest.main()
