# TerraLens - Change Detection Due to Human Activities

Upload two satellite images (T1 earlier, T2 later). TerraLens aligns them, finds what changed, and
labels every changed region as **Human-made**, **Natural** or **Environmental**, with maps,
area statistics (km²), a per-region confidence and a downloadable report.

Flask + SQLite/MySQL + OpenCV/NumPy. No external services, no CDN, no GPU needed.

---------------------------------------------------------------------------------------------------

## 1. Run it locally (2 minutes)

```bash
python -m venv .venv
.venv\Scripts\activate            # Windows      (Linux/Mac: source .venv/bin/activate)
pip install -r requirements.txt
python run.py                     # open http://127.0.0.1:5000  -> Create an account
```

Try it immediately with the synthetic demo pair in `samples/` (`demo_T1_rgb.png` + `demo_T2_rgb.png`,
or the 4-band GeoTIFFs `demo_T1_rgbn.tif` + `demo_T2_rgbn.tif` with band order **RGB + NIR**).
These are *generated* test scenes, not real satellite data (`python tools/make_samples.py` recreates them).

Run the tests: `python -m unittest discover -s tests -v`

## 2. Using it with real imagery

| Your files | Choose "Band order" | Notes |
|---|---|---|
| Screenshot / JPG / PNG / 3-band RGB GeoTIFF | RGB | works, but vegetation/water indices are less reliable without NIR |
| Sentinel-2 GeoTIFF stack of B4, B3, B2, B8 | RGB + NIR (R,G,B,NIR) | best results |
| LISS-IV 3-band GeoTIFF (B2, B3, B4) | LISS-IV (G,R,NIR) | |

* Both images should cover the **same area**. Different dates, ideally the **same season**.
* Area (km²) uses the GeoTIFF pixel size when it is in metres; otherwise the "ground resolution" field.
* Scenes are analysed at up to 1024 px on the long side (set `ANALYSIS_MAX_SIDE` to change). Images above
  `MAX_IMAGE_PIXELS` (default 40 megapixels) are rejected with a message - crop/resample first.
* Upload limit: `MAX_UPLOAD_MB` (default 200 MB for both files together).

## 3. What the "model" is - please read

* **Change detection**: change-vector analysis + structural (edge) difference, after radiometric
  matching and translation registration, with an adaptive noise-based threshold. Works on any image pair,
  needs no training data.
* **Human vs natural vs environmental**: a transparent, rule-based interpretation. Each changed pixel is scored for
  built-up structure change (construction / demolition / roads), vegetation gain-loss (natural) and
  water-extent / seasonal change (environmental); regions are smoothed by majority vote and given a confidence.
  Low-confidence regions are flagged, never silently called human-made.
* It is **not** a trained BIT / ChangeFormer network. No pretrained weights or training data ship with this
  project, so no deep model could honestly be claimed. Results support, but do not prove, the cause of a change.
  On the included synthetic scenes the tests show ~89-97 % correct class on detected human-made pixels - synthetic
  data is far easier than the real world, so validate on your own labelled study areas before reporting accuracy.

### Plugging in BIT / ChangeFormer (optional)
Train or download a model, export it to **ONNX** with:
* two inputs (T1, T2) of shape `(1,3,H,W)`, float32, normalised to `[-1, 1]` (mean 0.5, std 0.5), RGB order
* one output: logits `(1,2,H,W)` or `(1,1,H,W)`

then `pip install -r requirements-optional.txt` and set `CD_DETECTOR=onnx`, `CD_ONNX_MODEL=/path/model.onnx`
(Railway: put the file on the Volume). The model finds *where* things changed; the human/natural attribution
step still runs on top. If the model cannot be loaded the app falls back to the classical detector and says so.
This adapter (`OnnxDetector` in `app/ml/detectors.py`) was written to that contract but could not be exercised
here without a trained model - test it with yours.

## 4. Deploying on Railway (this fixes "no .railway.app domain")

What was wrong in the old project: the `Procfile` started `run:app`, but there was no `run.py`, and gunicorn was not
bound to Railway's `$PORT`. The app never started, so Railway had nothing to put a domain on. Now:

1. Push this folder to GitHub (the `.gitignore` already excludes secrets, data and caches).
2. Railway → **New Project → Deploy from GitHub repo** → pick it. Build uses `requirements.txt`, `.python-version`,
   and `railway.json` / `Procfile` (start command `gunicorn wsgi:app --bind 0.0.0.0:$PORT ...`; health check `/health`).
3. **Variables** tab: add `SECRET_KEY` (any long random string) and `SESSION_COOKIE_SECURE=1`.
4. **Settings → Networking → Generate Domain.** (A service only gets a `*.up.railway.app` address after you press
   this and the app is listening on `$PORT`.) If it asks for a port, leave the default / enter the value of `PORT`.
5. **Keep your data:** Service → **Volumes → New Volume**, mount path `/data`, and add variable `DATA_DIR=/data`.
   Without a volume Railway wipes the SQLite database and uploaded images on every deploy.
6. Open `https://<your-domain>/health` - it should answer `{"status":"ok"}`. Then open `/` and register.

MySQL instead of SQLite: add Railway's MySQL service and set `DATABASE_URL=mysql://user:pass@host:3306/db`
(uses PyMySQL; tables are created automatically; this path is untested here).

If a deploy fails, open **Deployments → View logs**: the first red line says why (missing variable, port, memory).
Large scenes need RAM - if the service is OOM-killed, lower `MAX_IMAGE_PIXELS` / `ANALYSIS_MAX_SIDE`.

## 5. Project layout

```
wsgi.py, run.py            entry points (gunicorn / local)
config.py                  all settings via environment variables
app/__init__.py           application factory
app/db.py, repository.py   database layer (SQLite default, MySQL optional), one repository class per entity
app/security.py           sessions, CSRF, login_required
app/services.py           uploads, previews, running an analysis, report ZIP
app/routes/               auth, main (dashboard/datasets/history/settings/media/health), analysis (wizard)
app/ml/                   imaging, preprocess, detectors (Strategy), attribute, render, pipeline
app/templates, static/    UI (no external assets)
tests/                    unit + end-to-end tests      tools/  synthetic data generator
```
Design notes: Strategy pattern for interchangeable change detectors (`ChangeDetector` → `ClassicalDetector`,
`OnnxDetector`), repository classes for persistence, services separate from routes.

## 6. Security notes
Passwords hashed (Werkzeug), CSRF token on every POST, per-user data isolation (404 for other users' files),
safe redirects, upload type/size/pixel limits, files stored under random names. Set `ALLOW_REGISTRATION=0`
to close sign-ups after you create your accounts.
