"""New-analysis wizard: Upload -> Compare -> Detect changes -> Classify & analyse."""
import re
from datetime import datetime

from flask import (Blueprint, Response, abort, current_app, flash, g, redirect, render_template, request,
                   url_for)

from .. import services
from ..db import get_conn
from ..ml.imaging import BAND_LAYOUTS, DEFAULT_RES_M, SceneError
from ..repository import ActivityRepository, AnalysisRepository
from ..security import login_required

bp = Blueprint("analysis", __name__, url_prefix="/analysis")


def _get_pair_or_404(pair_id):
    pair = AnalysisRepository(get_conn()).pair(pair_id, g.user["id"])
    if not pair:
        abort(404)
    return pair


def _valid_date(text):
    if not text:
        return ""
    try:
        datetime.strptime(text, "%Y-%m-%d")
        return text
    except ValueError:
        return None


@bp.route("/new", methods=["GET", "POST"])
@login_required
def new():
    if request.method == "GET":
        return render_template("analysis/new.html", form={"source": "Sentinel-2", "layout": "rgb", "res": "10"})

    form = {k: (request.form.get(k) or "").strip() for k in
            ("name", "source", "study_area", "t1_date", "t2_date", "layout", "res")}
    errors = []
    if form["source"] not in current_app.config["SATELLITE_SOURCES"]:
        errors.append("Choose a dataset (satellite source).")
    if not form["name"]:
        errors.append("Give the analysis a name.")
    elif len(form["name"]) > 160:
        errors.append("The analysis name is too long.")
    if form["layout"] not in BAND_LAYOUTS:
        errors.append("Choose a band order.")
    d1, d2 = _valid_date(form["t1_date"]), _valid_date(form["t2_date"])
    if d1 is None or d2 is None:
        errors.append("Dates must look like 2024-01-12.")
    try:
        res = float(form["res"] or DEFAULT_RES_M.get(form["source"], 10.0))
        if not 0.05 <= res <= 1000:
            raise ValueError
    except ValueError:
        res = None
        errors.append("Ground resolution must be a number between 0.05 and 1000 metres per pixel.")
    f1, f2 = request.files.get("t1_file"), request.files.get("t2_file")
    if not f1 or not f1.filename:
        errors.append("Choose the T1 (earlier) image.")
    if not f2 or not f2.filename:
        errors.append("Choose the T2 (later) image.")
    if errors:
        for e in errors:
            flash(e, "error")
        return render_template("analysis/new.html", form=form), 400

    stored = []
    try:
        for tag, fs in (("t1", f1), ("t2", f2)):
            stored.append(services.store_upload(fs, g.user["id"], tag, form["layout"]))
    except SceneError as exc:
        for s in stored:
            services.safe_remove(s["file_path"], services.preview_path(g.user["id"], s["preview_name"]))
        flash(str(exc), "error")
        return render_template("analysis/new.html", form=form), 400

    conn = get_conn()
    repo = AnalysisRepository(conn)
    dataset_id = repo.dataset_for(g.user["id"], form["source"])
    ids = []
    for tag, s, date in (("T1", stored[0], d1), ("T2", stored[1], d2)):
        ids.append(repo.add_image(
            dataset_id=dataset_id, user_id=g.user["id"], temporal_type=tag,
            original_filename=s["original_filename"], file_path=s["file_path"], preview_name=s["preview_name"],
            file_size=s["file_size"], width=s["width"], height=s["height"], band_count=s["band_count"],
            band_layout=s["band_layout"], acquisition_date=date or "", ground_res_m=res,
            satellite_source=form["source"]))
    pair_id = repo.create_pair(user_id=g.user["id"], dataset_id=dataset_id, name=form["name"],
                               study_area=form["study_area"], t1_image_id=ids[0], t2_image_id=ids[1])
    ActivityRepository(conn).log(g.user["id"], "upload", f"New dataset uploaded: {form['source']} ({form['name']})")
    for s in stored:
        for w in s["warnings"]:
            flash(w, "info")
    return redirect(url_for("analysis.compare", pair_id=pair_id))


@bp.route("/<int:pair_id>/compare")
@login_required
def compare(pair_id):
    return render_template("analysis/compare.html", pair=_get_pair_or_404(pair_id), step=2)


@bp.route("/<int:pair_id>/detect", methods=["GET", "POST"])
@login_required
def detect(pair_id):
    pair = _get_pair_or_404(pair_id)
    repo = AnalysisRepository(get_conn())
    if request.method == "POST":
        try:
            res = services.run_analysis(pair)
        except (SceneError, ValueError) as exc:
            flash(f"Analysis could not run: {exc}", "error")
            return redirect(url_for("analysis.compare", pair_id=pair_id))
        except Exception:
            current_app.logger.exception("analysis failed for pair %s", pair_id)
            flash("The analysis failed unexpectedly. Check that both images are valid and cover the same area.", "error")
            return redirect(url_for("analysis.compare", pair_id=pair_id))
        repo.save_result(pair_id, res.metrics, res.warnings, res.insights, res.regions)
        repo.set_status(pair_id, "detected")
        ActivityRepository(get_conn()).log(g.user["id"], "analysis", f"Change detection finished: {pair['name']}")
        return redirect(url_for("analysis.detect", pair_id=pair_id))
    result = repo.result(pair_id)
    if not result:
        flash("Run the change detection first.", "info")
        return redirect(url_for("analysis.compare", pair_id=pair_id))
    return render_template("analysis/detect.html", pair=pair, result=result, step=3)


@bp.route("/<int:pair_id>/classify")
@login_required
def classify(pair_id):
    pair = _get_pair_or_404(pair_id)
    result = AnalysisRepository(get_conn()).result(pair_id)
    if not result:
        flash("Run the change detection first.", "info")
        return redirect(url_for("analysis.compare", pair_id=pair_id))
    segs = [("human", result["human_pct"] or 0, "#ef4444"), ("natural", result["natural_pct"] or 0, "#22c55e"),
            ("env", result["env_pct"] or 0, "#eab308")]
    acc, stops = 0.0, []
    for _name, pct, color in segs:
        if pct > 0:
            stops.append(f"{color} {acc:.2f}% {acc + pct:.2f}%")
            acc += pct
    donut = "conic-gradient(" + ", ".join(stops) + ")" if stops else "conic-gradient(#24344d 0 100%)"
    return render_template("analysis/classify.html", pair=pair, result=result, donut=donut, step=4)


@bp.route("/<int:pair_id>/save", methods=["POST"])
@login_required
def save(pair_id):
    pair = _get_pair_or_404(pair_id)
    repo = AnalysisRepository(get_conn())
    if not repo.result(pair_id):
        abort(400, description="Nothing to save yet.")
    repo.set_status(pair_id, "completed")
    ActivityRepository(get_conn()).log(g.user["id"], "complete", f"Analysis completed: {pair['name']}")
    flash("Analysis saved to your history.", "success")
    return redirect(url_for("main.history"))


@bp.route("/<int:pair_id>/download")
@login_required
def download(pair_id):
    pair = _get_pair_or_404(pair_id)
    repo = AnalysisRepository(get_conn())
    result = repo.result(pair_id)
    if not result:
        abort(404)
    data = services.results_zip(pair, result)
    repo.add_report(pair_id, f"Report - {pair['name']}", "zip")
    safe = re.sub(r"[^A-Za-z0-9_-]+", "_", pair["name"]).strip("_") or "analysis"
    return Response(data, mimetype="application/zip",
                    headers={"Content-Disposition": f'attachment; filename="terralens_{safe}.zip"'})


@bp.route("/<int:pair_id>/delete", methods=["POST"])
@login_required
def delete(pair_id):
    pair = _get_pair_or_404(pair_id)
    previews = [services.preview_path(g.user["id"], p) for p in
                (pair["t1_preview"], pair["t2_preview"]) if p]
    files = AnalysisRepository(get_conn()).delete_pair(pair_id, g.user["id"])
    if files is not None:
        services.remove_pair_files(pair_id, files, previews)
        ActivityRepository(get_conn()).log(g.user["id"], "delete", f"Analysis deleted: {pair['name']}")
        flash("Analysis deleted.", "success")
    return redirect(url_for("main.history"))
