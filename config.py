"""TerraLens configuration (everything can be overridden with environment variables)."""
import os
import secrets

from dotenv import load_dotenv

BASE_DIR = os.path.abspath(os.path.dirname(__file__))
load_dotenv(os.path.join(BASE_DIR, ".env"))


def _bool(name, default=False):
    v = os.environ.get(name)
    return default if v is None else v.strip().lower() in ("1", "true", "yes", "on")


def _data_dir():
    # On Railway, attach a Volume and its mount path is exposed as RAILWAY_VOLUME_MOUNT_PATH.
    # Without a volume the container disk is wiped on every deploy.
    d = os.environ.get("DATA_DIR") or os.environ.get("RAILWAY_VOLUME_MOUNT_PATH") \
        or os.path.join(BASE_DIR, "instance")
    os.makedirs(d, exist_ok=True)
    return d


def _secret_key(data_dir):
    key = os.environ.get("SECRET_KEY")
    if key:
        return key
    # No key configured: create one once and keep it, so sessions survive restarts.
    path = os.path.join(data_dir, ".secret_key")
    try:
        if os.path.exists(path):
            return open(path).read().strip()
        key = secrets.token_hex(32)
        with open(path, "w") as fh:
            fh.write(key)
        return key
    except OSError:
        return secrets.token_hex(32)


DATA_DIR = _data_dir()


class Config:
    DATA_DIR = DATA_DIR
    SECRET_KEY = _secret_key(DATA_DIR)

    # sqlite by default; for MySQL use  mysql://user:pass@host:3306/dbname
    DATABASE_URL = os.environ.get("DATABASE_URL") or "sqlite:///" + os.path.join(DATA_DIR, "terralens.db")

    UPLOAD_FOLDER = os.environ.get("UPLOAD_FOLDER") or os.path.join(DATA_DIR, "uploads")
    OUTPUT_FOLDER = os.environ.get("OUTPUT_FOLDER") or os.path.join(DATA_DIR, "outputs")

    MAX_UPLOAD_MB = int(os.environ.get("MAX_UPLOAD_MB", "200"))      # both files together
    MAX_CONTENT_LENGTH = MAX_UPLOAD_MB * 1024 * 1024
    MAX_IMAGE_PIXELS = int(os.environ.get("MAX_IMAGE_PIXELS", str(40_000_000)))
    ANALYSIS_MAX_SIDE = int(os.environ.get("ANALYSIS_MAX_SIDE", "1024"))
    ALLOWED_EXTENSIONS = {"tif", "tiff", "png", "jpg", "jpeg"}

    # Change-detector: "classical" (default, always works) or "onnx" (BIT / ChangeFormer export)
    CD_DETECTOR = os.environ.get("CD_DETECTOR", "classical").lower()
    CD_ONNX_MODEL = os.environ.get("CD_ONNX_MODEL")

    SESSION_COOKIE_HTTPONLY = True
    SESSION_COOKIE_SAMESITE = "Lax"
    SESSION_COOKIE_SECURE = _bool("SESSION_COOKIE_SECURE", False)    # set to 1 behind HTTPS
    PERMANENT_SESSION_LIFETIME = 60 * 60 * 24 * 14
    SATELLITE_SOURCES = ["Sentinel-2", "LISS-IV"]
    ALLOW_REGISTRATION = _bool("ALLOW_REGISTRATION", True)
