"""Authentication helpers and CSRF protection (no third-party extensions needed)."""
from __future__ import annotations

import hmac
import secrets
from functools import wraps
from urllib.parse import urlparse

from flask import abort, current_app, flash, g, redirect, request, session, url_for

from .db import get_conn
from .repository import UserRepository


def load_user():
    g.user = None
    uid = session.get("uid")
    if uid:
        g.user = UserRepository(get_conn()).by_id(uid)
        if g.user is None:
            session.clear()


def csrf_token() -> str:
    if "_csrf" not in session:
        session["_csrf"] = secrets.token_urlsafe(32)
    return session["_csrf"]


def check_csrf():
    if request.method in ("GET", "HEAD", "OPTIONS"):
        return
    sent = request.form.get("csrf_token") or request.headers.get("X-CSRFToken") or ""
    expected = session.get("_csrf", "")
    if not expected or not hmac.compare_digest(sent, expected):
        if request.max_content_length and request.content_length and request.content_length > request.max_content_length:
            return  # let the 413 handler answer
        abort(400, description="Your session expired or the form was invalid. Please go back and try again.")


def login_required(view):
    @wraps(view)
    def wrapped(*args, **kwargs):
        if g.get("user") is None:
            flash("Please sign in to continue.", "info")
            return redirect(url_for("auth.login", next=request.path))
        return view(*args, **kwargs)
    return wrapped


def safe_next(target: str | None, default: str) -> str:
    """Only allow redirects to local paths."""
    if not target:
        return default
    p = urlparse(target)
    if p.scheme or p.netloc or not target.startswith("/") or target.startswith("//"):
        return default
    return target
