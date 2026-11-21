"""Sign in / sign up / sign out."""
import re

from flask import Blueprint, current_app, flash, g, redirect, render_template, request, session, url_for
from werkzeug.security import check_password_hash, generate_password_hash

from ..db import get_conn
from ..repository import ActivityRepository, UserRepository
from ..security import safe_next

bp = Blueprint("auth", __name__, url_prefix="/auth")
USERNAME_RE = re.compile(r"^[A-Za-z0-9_.-]{3,30}$")
EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")


@bp.route("/login", methods=["GET", "POST"])
def login():
    if g.user:
        return redirect(url_for("main.dashboard"))
    if request.method == "POST":
        users = UserRepository(get_conn())
        user = users.by_login(request.form.get("login"))
        if user and check_password_hash(user["password_hash"], request.form.get("password", "")):
            session.clear()
            session["uid"] = user["id"]
            session.permanent = bool(request.form.get("remember"))
            users.touch_login(user["id"])
            return redirect(safe_next(request.args.get("next"), url_for("main.dashboard")))
        flash("Invalid username/email or password.", "error")
    return render_template("auth/login.html")


@bp.route("/register", methods=["GET", "POST"])
def register():
    if g.user:
        return redirect(url_for("main.dashboard"))
    if not current_app.config["ALLOW_REGISTRATION"]:
        flash("Registration is disabled.", "error")
        return redirect(url_for("auth.login"))
    form = {}
    if request.method == "POST":
        form = {k: (request.form.get(k) or "").strip() for k in ("username", "email", "full_name")}
        pw, pw2 = request.form.get("password", ""), request.form.get("confirm", "")
        errors = []
        if not USERNAME_RE.match(form["username"]):
            errors.append("Username must be 3-30 characters (letters, numbers, . _ -).")
        if not EMAIL_RE.match(form["email"]):
            errors.append("Enter a valid email address.")
        if len(pw) < 8:
            errors.append("Password must be at least 8 characters.")
        if pw != pw2:
            errors.append("Passwords do not match.")
        users = UserRepository(get_conn())
        if not errors and users.exists(form["username"], form["email"]):
            errors.append("That username or email is already registered.")
        if errors:
            for e in errors:
                flash(e, "error")
        else:
            uid = users.create(form["username"], form["email"].lower(), generate_password_hash(pw), form["full_name"])
            ActivityRepository(get_conn()).log(uid, "account", "Account created")
            session.clear()
            session["uid"] = uid
            flash("Welcome to TerraLens!", "success")
            return redirect(url_for("main.dashboard"))
    return render_template("auth/register.html", form=form)


@bp.route("/logout", methods=["POST"])
def logout():
    session.clear()
    flash("You have been signed out.", "info")
    return redirect(url_for("auth.login"))
