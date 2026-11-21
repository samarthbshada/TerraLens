"""TerraLens - change detection due to human activities (Flask application factory)."""
from __future__ import annotations

import logging
import os
from datetime import datetime, timezone

from flask import Flask, g, render_template
from werkzeug.middleware.proxy_fix import ProxyFix

from config import Config
from . import security
from .db import Database, close_conn
from .ml.imaging import DEFAULT_RES_M, LAYOUT_LABELS


def _parse(ts):
    try:
        return datetime.strptime(ts[:19], "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc)
    except (TypeError, ValueError):
        return None


def create_app(test_config: dict | None = None) -> Flask:
    app = Flask(__name__, static_folder="static", template_folder="templates")
    app.config.from_object(Config)
    if test_config:
        app.config.update(test_config)
        if "MAX_UPLOAD_MB" in test_config:
            app.config["MAX_CONTENT_LENGTH"] = test_config["MAX_UPLOAD_MB"] * 1024 * 1024

    # Railway and most PaaS terminate HTTPS in a proxy: trust its forwarded headers.
    app.wsgi_app = ProxyFix(app.wsgi_app, x_for=1, x_proto=1, x_host=1)
    logging.basicConfig(level=logging.INFO)

    for key in ("UPLOAD_FOLDER", "OUTPUT_FOLDER"):
        os.makedirs(app.config[key], exist_ok=True)

    database = Database(app.config["DATABASE_URL"])
    database.init_schema()
    app.extensions["terralens_db"] = database
    app.teardown_appcontext(close_conn)

    app.before_request(security.load_user)
    app.before_request(security.check_csrf)

    @app.context_processor
    def inject():
        return {"csrf_token": security.csrf_token, "current_user": g.get("user"),
                "layout_labels": LAYOUT_LABELS, "default_res": DEFAULT_RES_M,
                "sources": app.config["SATELLITE_SOURCES"]}

    @app.template_filter("filesize")
    def filesize(n):
        n = float(n or 0)
        for unit in ("B", "KB", "MB", "GB"):
            if n < 1024 or unit == "GB":
                return f"{int(n)} B" if unit == "B" else f"{n:.1f} {unit}"
            n /= 1024

    @app.template_filter("date")
    def fmt_date(value):
        value = value or ""
        for fmt, ln in (("%Y-%m-%d", 10), ("%Y-%m-%d %H:%M:%S", 19)):
            try:
                return datetime.strptime(value[:ln], fmt).strftime("%d %b %Y")
            except ValueError:
                continue
        return value or "-"

    @app.template_filter("ago")
    def ago(value):
        dt = _parse(value)
        if not dt:
            return ""
        secs = int((datetime.now(timezone.utc) - dt).total_seconds())
        if secs < 60:
            return "just now"
        for size, label in ((86400, "day"), (3600, "hour"), (60, "minute")):
            if secs >= size:
                n = secs // size
                return f"{n} {label}{'s' if n != 1 else ''} ago"
        return ""

    from .routes.analysis import bp as analysis_bp
    from .routes.auth import bp as auth_bp
    from .routes.main import bp as main_bp
    app.register_blueprint(auth_bp)
    app.register_blueprint(main_bp)
    app.register_blueprint(analysis_bp)

    def _error(code, title, message):
        return render_template("error.html", code=code, title=title, message=message), code

    @app.errorhandler(400)
    def e400(e):
        return _error(400, "Request problem", getattr(e, "description", "Bad request."))

    @app.errorhandler(403)
    def e403(e):
        return _error(403, "Not allowed", "You do not have access to this page.")

    @app.errorhandler(404)
    def e404(e):
        return _error(404, "Page not found", "The page you are looking for does not exist.")

    @app.errorhandler(413)
    def e413(e):
        return _error(413, "Upload too large",
                      f"The files together exceed the {app.config['MAX_UPLOAD_MB']} MB limit. "
                      "Crop or resample the scenes and try again.")

    @app.errorhandler(500)
    def e500(e):
        return _error(500, "Something went wrong", "An unexpected error occurred. Please try again.")

    return app
