"""Dashboard, datasets, history, settings, media files and health check."""
import os

from flask import (Blueprint, abort, flash, g, redirect, render_template, request, send_file, url_for, jsonify)
from werkzeug.security import check_password_hash, generate_password_hash

from .. import services
from ..db import get_conn
from ..repository import ActivityRepository, AnalysisRepository, UserRepository
from ..security import login_required

bp = Blueprint("main", __name__)


@bp.route("/")
def index():
    return redirect(url_for("main.dashboard" if g.user else "auth.login"))


@bp.route("/health")
def health():
    """Used by Railway's health check; also verifies the database answers."""
    try:
        get_conn().scalar("SELECT 1")
        return jsonify(status="ok"), 200
    except Exception as exc:                                    # pragma: no cover
        return jsonify(status="error", detail=str(exc)), 503


@bp.route("/dashboard")
@login_required
def dashboard():
    repo = AnalysisRepository(get_conn())
    latest = repo.latest_completed(g.user["id"])
    result = repo.result(latest["id"]) if latest else None
    return render_template("dashboard.html", stats=repo.dashboard_stats(g.user["id"]), latest=latest, result=result,
                           activities=ActivityRepository(get_conn()).recent(g.user["id"], 6))


@bp.route("/datasets")
@login_required
def datasets():
    repo = AnalysisRepository(get_conn())
    return render_template("datasets.html", datasets=repo.datasets_with_images(g.user["id"]),
                           images=repo.images_for_user(g.user["id"]))


@bp.route("/history")
@login_required
def history():
    q = (request.args.get("q") or "").strip()
    repo = AnalysisRepository(get_conn())
    pairs = repo.pairs(g.user["id"], q)
    for p in pairs:
        p["result"] = repo.result(p["id"]) if p["status"] in ("completed", "detected") else None
    return render_template("history.html", pairs=pairs, q=q)


@bp.route("/settings", methods=["GET", "POST"])
@login_required
def settings():
    users = UserRepository(get_conn())
    if request.method == "POST":
        action = request.form.get("action")
        if action == "profile":
            name = (request.form.get("full_name") or "").strip()[:120]
            if name:
                users.update_profile(g.user["id"], name)
                flash("Profile updated.", "success")
        elif action == "password":
            cur, new, conf = (request.form.get(k, "") for k in ("current", "new", "confirm"))
            if not check_password_hash(g.user["password_hash"], cur):
                flash("Current password is incorrect.", "error")
            elif len(new) < 8:
                flash("New password must be at least 8 characters.", "error")
            elif new != conf:
                flash("New passwords do not match.", "error")
            else:
                users.set_password(g.user["id"], generate_password_hash(new))
                flash("Password changed.", "success")
        return redirect(url_for("main.settings"))
    from flask import current_app
    cfg = current_app.config
    detector = {"classical": "Classical baseline (always available)",
                "onnx": "ONNX deep model" + ("" if cfg.get("CD_ONNX_MODEL") else " (no model configured - using classical)")
                }.get(cfg["CD_DETECTOR"], "Classical baseline")
    return render_template("settings.html", detector=detector, db_kind=get_conn().dialect,
                           limits={"mb": cfg["MAX_UPLOAD_MB"], "mp": cfg["MAX_IMAGE_PIXELS"] // 1_000_000})


# ------------------------------------------------------------------ media
@bp.route("/media/preview/<int:image_id>")
@login_required
def preview(image_id):
    row = get_conn().one("SELECT user_id, preview_name FROM satellite_images WHERE id = ? AND user_id = ?",
                         (image_id, g.user["id"]))
    if not row or not row["preview_name"]:
        abort(404)
    path = services.preview_path(row["user_id"], row["preview_name"])
    if not os.path.isfile(path):
        abort(404)
    return send_file(path, mimetype="image/png", max_age=3600)


@bp.route("/media/output/<int:pair_id>/<name>")
@login_required
def output(pair_id, name):
    if not services.SAFE_NAME.match(name) or name not in services.OUTPUT_FILES:
        abort(404)
    if not AnalysisRepository(get_conn()).pair(pair_id, g.user["id"]):
        abort(404)
    path = os.path.join(services.output_dir(pair_id), name)
    if not os.path.isfile(path):
        abort(404)
    return send_file(path, mimetype="image/png", max_age=300)
