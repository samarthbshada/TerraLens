"""Tiny database layer.

SQLite is the default (zero setup, works on Railway with a Volume).  If
DATABASE_URL starts with mysql:// the same SQL runs on MySQL through PyMySQL.
SQL is written with ? placeholders and translated for MySQL.
"""
from __future__ import annotations

import os
import sqlite3
from datetime import datetime, timezone
from urllib.parse import unquote, urlparse

from flask import g


def now_str() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")


class Conn:
    def __init__(self, raw, dialect: str):
        self.raw, self.dialect = raw, dialect

    def _sql(self, sql: str) -> str:
        return sql.replace("?", "%s") if self.dialect == "mysql" else sql

    def execute(self, sql, params=()):
        cur = self.raw.cursor()
        cur.execute(self._sql(sql), tuple(params))
        return cur

    def query(self, sql, params=()) -> list:
        cur = self.execute(sql, params)
        rows = cur.fetchall()
        cur.close()
        return [dict(r) for r in rows]

    def one(self, sql, params=()):
        rows = self.query(sql, params)
        return rows[0] if rows else None

    def scalar(self, sql, params=(), default=0):
        cur = self.execute(sql, params)
        row = cur.fetchone()
        cur.close()
        if row is None:
            return default
        val = list(dict(row).values())[0] if not isinstance(row, (tuple, list)) else row[0]
        return default if val is None else val

    def insert(self, table: str, **values) -> int:
        cols = ", ".join(values)
        marks = ", ".join("?" for _ in values)
        cur = self.execute(f"INSERT INTO {table} ({cols}) VALUES ({marks})", list(values.values()))
        new_id = cur.lastrowid
        cur.close()
        return new_id

    def update(self, table: str, row_id: int, **values) -> None:
        sets = ", ".join(f"{k} = ?" for k in values)
        self.execute(f"UPDATE {table} SET {sets} WHERE id = ?", [*values.values(), row_id]).close()

    def commit(self):
        self.raw.commit()

    def close(self):
        try:
            self.raw.close()
        except Exception:
            pass


class Database:
    def __init__(self, url: str):
        self.url = url
        self.dialect = "mysql" if url.startswith(("mysql://", "mysql+pymysql://")) else "sqlite"
        if self.dialect == "sqlite":
            path = url.replace("sqlite:///", "", 1)
            os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
            self.path = path

    def connect(self) -> Conn:
        if self.dialect == "sqlite":
            raw = sqlite3.connect(self.path, timeout=30)
            raw.row_factory = sqlite3.Row
            raw.execute("PRAGMA foreign_keys = ON")
            raw.execute("PRAGMA journal_mode = WAL")
            return Conn(raw, "sqlite")
        import pymysql
        u = urlparse(self.url.replace("mysql+pymysql://", "mysql://"))
        raw = pymysql.connect(host=u.hostname, port=u.port or 3306, user=unquote(u.username or ""),
                              password=unquote(u.password or ""), database=(u.path or "/").lstrip("/"),
                              charset="utf8mb4", cursorclass=pymysql.cursors.DictCursor, autocommit=False)
        return Conn(raw, "mysql")

    # --------------------------------------------------------------- schema
    def init_schema(self):
        pk = "INTEGER PRIMARY KEY AUTOINCREMENT" if self.dialect == "sqlite" else "INT AUTO_INCREMENT PRIMARY KEY"
        tail = "" if self.dialect == "sqlite" else " ENGINE=InnoDB DEFAULT CHARSET=utf8mb4"
        ddl = [
            f"""CREATE TABLE IF NOT EXISTS users (
                id {pk}, username VARCHAR(80) NOT NULL UNIQUE, email VARCHAR(120) NOT NULL UNIQUE,
                password_hash VARCHAR(255) NOT NULL, full_name VARCHAR(120), role VARCHAR(20) DEFAULT 'user',
                created_at VARCHAR(19), last_login VARCHAR(19)){tail}""",
            f"""CREATE TABLE IF NOT EXISTS datasets (
                id {pk}, user_id INT NOT NULL, name VARCHAR(160) NOT NULL, source VARCHAR(50) NOT NULL,
                description TEXT, created_at VARCHAR(19),
                FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE){tail}""",
            f"""CREATE TABLE IF NOT EXISTS satellite_images (
                id {pk}, dataset_id INT, user_id INT NOT NULL, temporal_type VARCHAR(4) NOT NULL,
                original_filename VARCHAR(255) NOT NULL, file_path VARCHAR(500) NOT NULL,
                preview_name VARCHAR(255), file_size BIGINT, width INT, height INT, band_count INT,
                band_layout VARCHAR(12), acquisition_date VARCHAR(10), ground_res_m DOUBLE,
                satellite_source VARCHAR(50), created_at VARCHAR(19),
                FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE,
                FOREIGN KEY (dataset_id) REFERENCES datasets(id) ON DELETE CASCADE){tail}""",
            f"""CREATE TABLE IF NOT EXISTS image_pairs (
                id {pk}, user_id INT NOT NULL, dataset_id INT, name VARCHAR(160) NOT NULL,
                study_area VARCHAR(160), description TEXT, t1_image_id INT NOT NULL, t2_image_id INT NOT NULL,
                status VARCHAR(20) DEFAULT 'uploaded', sensitivity VARCHAR(10) DEFAULT 'medium',
                created_at VARCHAR(19), updated_at VARCHAR(19),
                FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE,
                FOREIGN KEY (t1_image_id) REFERENCES satellite_images(id),
                FOREIGN KEY (t2_image_id) REFERENCES satellite_images(id)){tail}""",
            f"""CREATE TABLE IF NOT EXISTS change_detection_results (
                id {pk}, pair_id INT NOT NULL UNIQUE, detector VARCHAR(160), width INT, height INT,
                pixel_size_m DOUBLE, scene_area_km2 DOUBLE, total_change_area_km2 DOUBLE, changed_pixels BIGINT,
                area_changed_pct DOUBLE, n_regions INT, low_conf_regions INT,
                human_km2 DOUBLE, natural_km2 DOUBLE, env_km2 DOUBLE,
                human_regions INT, natural_regions INT, env_regions INT,
                human_pct DOUBLE, natural_pct DOUBLE, env_pct DOUBLE,
                warnings TEXT, insights TEXT, regions TEXT, runtime_s DOUBLE, created_at VARCHAR(19),
                FOREIGN KEY (pair_id) REFERENCES image_pairs(id) ON DELETE CASCADE){tail}""",
            f"""CREATE TABLE IF NOT EXISTS reports (
                id {pk}, pair_id INT NOT NULL, title VARCHAR(200), report_type VARCHAR(20),
                generated_at VARCHAR(19),
                FOREIGN KEY (pair_id) REFERENCES image_pairs(id) ON DELETE CASCADE){tail}""",
            f"""CREATE TABLE IF NOT EXISTS activities (
                id {pk}, user_id INT NOT NULL, kind VARCHAR(20), message VARCHAR(300), created_at VARCHAR(19),
                FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE){tail}""",
        ]
        conn = self.connect()
        try:
            for stmt in ddl:
                conn.execute(stmt).close()
            self._migrate(conn)
            conn.commit()
        finally:
            conn.close()

    def _migrate(self, conn):
        """Add columns introduced after the first release (keeps existing databases working)."""
        if self.dialect == "sqlite":
            cols = [r["name"] for r in conn.query("PRAGMA table_info(image_pairs)")]
            missing = "sensitivity" not in cols
        else:
            missing = not conn.query("SHOW COLUMNS FROM image_pairs LIKE 'sensitivity'")
        if missing:
            conn.execute("ALTER TABLE image_pairs ADD COLUMN sensitivity VARCHAR(10) DEFAULT 'medium'").close()


def get_conn() -> Conn:
    if "db_conn" not in g:
        from flask import current_app
        g.db_conn = current_app.extensions["terralens_db"].connect()
    return g.db_conn


def close_conn(_exc=None):
    conn = g.pop("db_conn", None)
    if conn is not None:
        conn.close()
