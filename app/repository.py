"""Data-access classes (one per entity) - keeps SQL out of the route handlers."""
from __future__ import annotations

import json

from .db import now_str

RESULT_COLS = ["detector", "width", "height", "pixel_size_m", "scene_area_km2", "total_change_area_km2",
               "changed_pixels", "area_changed_pct", "n_regions", "low_conf_regions",
               "human_km2", "natural_km2", "env_km2", "human_regions", "natural_regions", "env_regions",
               "human_pct", "natural_pct", "env_pct", "runtime_s"]


class UserRepository:
    def __init__(self, db):
        self.db = db

    def create(self, username, email, password_hash, full_name=""):
        uid = self.db.insert("users", username=username, email=email, password_hash=password_hash,
                             full_name=full_name or username, role="user", created_at=now_str())
        self.db.commit()
        return uid

    def by_id(self, uid):
        return self.db.one("SELECT * FROM users WHERE id = ?", (uid,))

    def by_login(self, login):
        login = (login or "").strip()
        return self.db.one("SELECT * FROM users WHERE LOWER(username) = LOWER(?) OR LOWER(email) = LOWER(?)",
                           (login, login))

    def exists(self, username, email):
        return self.db.one("SELECT id, username, email FROM users WHERE LOWER(username)=LOWER(?) OR LOWER(email)=LOWER(?)",
                           (username, email))

    def touch_login(self, uid):
        self.db.update("users", uid, last_login=now_str())
        self.db.commit()

    def set_password(self, uid, password_hash):
        self.db.update("users", uid, password_hash=password_hash)
        self.db.commit()

    def update_profile(self, uid, full_name):
        self.db.update("users", uid, full_name=full_name)
        self.db.commit()


class ActivityRepository:
    def __init__(self, db):
        self.db = db

    def log(self, user_id, kind, message):
        self.db.insert("activities", user_id=user_id, kind=kind, message=message[:300], created_at=now_str())
        self.db.commit()

    def recent(self, user_id, limit=6):
        return self.db.query("SELECT * FROM activities WHERE user_id = ? ORDER BY id DESC LIMIT ?", (user_id, limit))


class AnalysisRepository:
    """Datasets, satellite images, image pairs and detection results."""

    def __init__(self, db):
        self.db = db

    # ----------------------------------------------------------- datasets
    def dataset_for(self, user_id, source):
        row = self.db.one("SELECT * FROM datasets WHERE user_id = ? AND source = ?", (user_id, source))
        if row:
            return row["id"]
        did = self.db.insert("datasets", user_id=user_id, name=f"{source} imagery", source=source,
                             description=f"Images uploaded from {source}", created_at=now_str())
        return did

    def datasets_with_images(self, user_id):
        return self.db.query(
            """SELECT d.id, d.name, d.source, d.created_at, COUNT(i.id) AS n_images,
                      COALESCE(SUM(i.file_size), 0) AS total_size
               FROM datasets d LEFT JOIN satellite_images i ON i.dataset_id = d.id
               WHERE d.user_id = ? GROUP BY d.id, d.name, d.source, d.created_at ORDER BY d.id""", (user_id,))

    def images_for_user(self, user_id, limit=100):
        return self.db.query(
            """SELECT i.*, p.id AS pair_id, p.name AS pair_name FROM satellite_images i
               LEFT JOIN image_pairs p ON (p.t1_image_id = i.id OR p.t2_image_id = i.id)
               WHERE i.user_id = ? ORDER BY i.id DESC LIMIT ?""", (user_id, limit))

    # ------------------------------------------------------------- images
    def add_image(self, **values):
        values.setdefault("created_at", now_str())
        return self.db.insert("satellite_images", **values)

    # -------------------------------------------------------------- pairs
    _PAIR_SQL = """
        SELECT p.*, d.source AS source,
               a.original_filename AS t1_name, a.file_path AS t1_path, a.preview_name AS t1_preview,
               a.acquisition_date AS t1_date, a.band_layout AS t1_layout, a.file_size AS t1_size,
               a.ground_res_m AS ground_res_m,
               b.original_filename AS t2_name, b.file_path AS t2_path, b.preview_name AS t2_preview,
               b.acquisition_date AS t2_date, b.band_layout AS t2_layout, b.file_size AS t2_size
        FROM image_pairs p
        JOIN satellite_images a ON a.id = p.t1_image_id
        JOIN satellite_images b ON b.id = p.t2_image_id
        LEFT JOIN datasets d ON d.id = p.dataset_id"""

    def create_pair(self, **values):
        values.setdefault("status", "uploaded")
        values.setdefault("created_at", now_str())
        values.setdefault("updated_at", now_str())
        pid = self.db.insert("image_pairs", **values)
        self.db.commit()
        return pid

    def pair(self, pair_id, user_id):
        return self.db.one(self._PAIR_SQL + " WHERE p.id = ? AND p.user_id = ?", (pair_id, user_id))

    def pairs(self, user_id, search="", limit=100):
        sql = self._PAIR_SQL + " WHERE p.user_id = ?"
        params = [user_id]
        if search:
            sql += " AND (p.name LIKE ? OR p.study_area LIKE ?)"
            params += [f"%{search}%", f"%{search}%"]
        sql += " ORDER BY p.id DESC LIMIT ?"
        params.append(limit)
        return self.db.query(sql, params)

    def set_status(self, pair_id, status):
        self.db.update("image_pairs", pair_id, status=status, updated_at=now_str())
        self.db.commit()

    def delete_pair(self, pair_id, user_id):
        """Delete a pair and its images; returns the file paths that should be removed from disk."""
        row = self.pair(pair_id, user_id)
        if not row:
            return None
        files = [row["t1_path"], row["t2_path"]]
        self.db.execute("DELETE FROM change_detection_results WHERE pair_id = ?", (pair_id,)).close()
        self.db.execute("DELETE FROM reports WHERE pair_id = ?", (pair_id,)).close()
        self.db.execute("DELETE FROM image_pairs WHERE id = ?", (pair_id,)).close()
        self.db.execute("DELETE FROM satellite_images WHERE id IN (?, ?)", (row["t1_image_id"], row["t2_image_id"])).close()
        self.db.commit()
        return files

    # ------------------------------------------------------------ results
    def save_result(self, pair_id, metrics, warnings, insights, regions):
        self.db.execute("DELETE FROM change_detection_results WHERE pair_id = ?", (pair_id,)).close()
        vals = {k: metrics.get(k) for k in RESULT_COLS}
        self.db.insert("change_detection_results", pair_id=pair_id, warnings=json.dumps(warnings),
                       insights=json.dumps(insights), regions=json.dumps(regions), created_at=now_str(), **vals)
        self.db.commit()

    def result(self, pair_id):
        row = self.db.one("SELECT * FROM change_detection_results WHERE pair_id = ?", (pair_id,))
        if row:
            for k in ("warnings", "insights", "regions"):
                row[k] = json.loads(row[k] or "[]")
        return row

    def add_report(self, pair_id, title, report_type="csv"):
        self.db.insert("reports", pair_id=pair_id, title=title, report_type=report_type, generated_at=now_str())
        self.db.commit()

    # ---------------------------------------------------------- dashboard
    def dashboard_stats(self, user_id):
        d = self.db
        stats = {
            "total_analyses": d.scalar("SELECT COUNT(*) FROM image_pairs WHERE user_id = ?", (user_id,)),
            "images": d.scalar("SELECT COUNT(*) FROM satellite_images WHERE user_id = ?", (user_id,)),
        }
        agg = d.one(
            """SELECT COALESCE(SUM(r.n_regions),0) AS regions, COALESCE(SUM(r.human_regions),0) AS human_regions,
                      COALESCE(SUM(r.human_km2),0) AS human_km2,
                      COALESCE(SUM(r.natural_km2),0) + COALESCE(SUM(r.env_km2),0) AS other_km2
               FROM change_detection_results r JOIN image_pairs p ON p.id = r.pair_id WHERE p.user_id = ?""",
            (user_id,)) or {}
        stats.update(regions=int(agg.get("regions") or 0), human_regions=int(agg.get("human_regions") or 0),
                     human_km2=float(agg.get("human_km2") or 0), other_km2=float(agg.get("other_km2") or 0))
        tot = stats["human_km2"] + stats["other_km2"]
        stats["human_pct"] = round(100 * stats["human_km2"] / tot) if tot > 0 else 0
        stats["other_pct"] = 100 - stats["human_pct"] if tot > 0 else 0
        return stats

    def latest_completed(self, user_id):
        return self.db.one(self._PAIR_SQL + " WHERE p.user_id = ? AND p.status = 'completed' ORDER BY p.id DESC LIMIT 1",
                           (user_id,)) or self.db.one(self._PAIR_SQL + " WHERE p.user_id = ? ORDER BY p.id DESC LIMIT 1",
                                                      (user_id,))
