from __future__ import annotations

import csv
import hmac
import os
import shutil
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Dict, Optional
from urllib.parse import quote

from flask import (
    Flask,
    abort,
    flash,
    redirect,
    render_template,
    request,
    send_file,
    send_from_directory,
    session,
    url_for,
)
from werkzeug.utils import secure_filename

# ----------------------------
# Paths (Render Disk ready)
# ----------------------------
BASE_DIR = Path(__file__).resolve().parent

DATA_DIR = Path(os.environ.get("DATA_DIR", str(BASE_DIR / "data"))).resolve()
UPLOADS_DIR = Path(os.environ.get("UPLOADS_DIR", str(BASE_DIR / "uploads"))).resolve()

DATA_DIR.mkdir(parents=True, exist_ok=True)
UPLOADS_DIR.mkdir(parents=True, exist_ok=True)

SUBMISSIONS_CSV = DATA_DIR / "submissions.csv"

# ----------------------------
# Upload policy
# ----------------------------
ALLOWED_EXT = {"jpg", "jpeg", "png", "webp"}
MAX_FILES = int(os.environ.get("MAX_FILES", "5"))
MAX_TOTAL_MB = int(os.environ.get("MAX_TOTAL_MB", "25"))  # whole request cap
MAX_FILE_MB = int(os.environ.get("MAX_FILE_MB", "10"))    # per photo cap

app = Flask(__name__, static_folder="static", static_url_path="/static")

@app.template_filter('is_numeric')
def is_numeric_filter(value) -> bool:
    """Return True if the string looks like a plain number (after removing separators)."""
    if value is None:
        return False
    s = str(value).strip()
    if not s:
        return False
    # remove common thousands separators and spaces
    for ch in (' ', '\u00a0', ',', '.', '_'):
        s = s.replace(ch, '')
    return s.isdigit()

app.secret_key = os.environ.get("SECRET_KEY", "dev-secret-change-me")
app.config["MAX_CONTENT_LENGTH"] = MAX_TOTAL_MB * 1024 * 1024


# ----------------------------
# Helpers
# ----------------------------

def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _new_id() -> str:
    return uuid.uuid4().hex[:10].upper()


def _allowed_file(filename: str) -> bool:
    ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    return ext in ALLOWED_EXT


def _csv_columns() -> list[str]:
    # единый CSV для ПОКУПАЮ / ПРОДАЮ
    return ["id", "created_utc", "kind", "title", "price_tenge", "phone", "description", "photos"]


def _ensure_csv_header():
    if SUBMISSIONS_CSV.exists():
        # если файл уже есть, не трогаем (чтобы не потерять данные)
        return
    with SUBMISSIONS_CSV.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(_csv_columns())


def _save_submission_row(
    sid: str,
    created_utc: str,
    kind: str,
    title: str,
    price_tenge: str,
    phone: str,
    description: str,
    photos: List[str],
):
    _ensure_csv_header()
    with SUBMISSIONS_CSV.open("a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            sid,
            created_utc,
            kind,
            title,
            price_tenge,
            phone,
            description,
            ";".join(photos),
        ])


def _read_all_rows() -> list[dict]:
    if not SUBMISSIONS_CSV.exists():
        return []
    with SUBMISSIONS_CSV.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        rows: list[dict] = []
        for r in reader:
            if not (r.get("id") or "").strip():
                continue
            rows.append(r)
        return rows


def _write_all_rows(rows: list[dict]) -> None:
    # атомарная запись
    tmp = SUBMISSIONS_CSV.with_suffix(".tmp")
    cols = _csv_columns()
    with tmp.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            out = {c: (r.get(c, "") or "") for c in cols}
            w.writerow(out)
    tmp.replace(SUBMISSIONS_CSV)


def _find_row(rows: list[dict], sid: str) -> Optional[dict]:
    for r in rows:
        if (r.get("id") or "").strip() == sid:
            return r
    return None


def _list_photos(sid: str) -> list[str]:
    d = UPLOADS_DIR / sid
    if not d.exists() or not d.is_dir():
        return []
    return sorted([p.name for p in d.iterdir() if p.is_file()])


def _thumb_url(sid: str, kind: str, photos: list[str]) -> str:
    # покупатели: показываем лого
    if kind == "buy":
        return "/static/logo.jpeg"

    # продавцы: первое фото, иначе лого
    if photos:
        return f"/uploads/{sid}/{quote(photos[0])}"
    return "/static/logo.jpeg"


def _load_submissions(limit: int = 200) -> list[dict]:
    """Newer-first list for public page."""
    if not SUBMISSIONS_CSV.exists():
        return []

    items: list[dict] = []
    with SUBMISSIONS_CSV.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            sid = (row.get("id") or "").strip()
            if not sid:
                continue

            kind = (row.get("kind") or "sell").strip().lower()
            if kind not in {"buy", "sell"}:
                kind = "sell"

            photos_raw = (row.get("photos") or "").strip()
            photos = [p for p in photos_raw.split(";") if p] if photos_raw else []

            items.append({
                "id": sid,
                "created_utc": (row.get("created_utc") or "").strip(),
                "kind": kind,
                "title": (row.get("title") or "").strip(),
                "price_tenge": (row.get("price_tenge") or "").strip(),
                "phone": (row.get("phone") or "").strip(),
                "description": (row.get("description") or "").strip(),
                "photos": photos,
                "thumb_url": _thumb_url(sid, kind, photos),
            })

    # newest first: created_utc is ISO, so lexicographic sort works
    items.sort(key=lambda x: x.get("created_utc", ""), reverse=True)
    return items[:limit]


# ----------------------------
# Public routes
# ----------------------------

@app.get("/")
def index():
    submissions = _load_submissions(limit=int(os.environ.get("MAX_LISTINGS", "200")))
    return render_template(
        "index.html",
        max_files=MAX_FILES,
        max_file_mb=MAX_FILE_MB,
        data_dir=str(DATA_DIR),
        uploads_dir=str(UPLOADS_DIR),
        submissions=submissions,
    )


@app.post("/submit")
def submit():
    kind = (request.form.get("kind") or "sell").strip().lower()
    if kind not in {"buy", "sell"}:
        kind = "sell"

    title = (request.form.get("title") or "").strip()
    price_tenge = (request.form.get("price") or "").strip()
    phone = (request.form.get("phone") or "").strip()
    description = (request.form.get("description") or "").strip()

    # Файлы только для ПРОДАЮ
    files = request.files.getlist("photos") if kind == "sell" else []
    files = [f for f in files if f and f.filename]

    if kind == "sell":
        if len(files) > MAX_FILES:
            flash(f"Можно загрузить максимум {MAX_FILES} фото.")
            return redirect(url_for("index"))

        for f in files:
            if not _allowed_file(f.filename):
                flash("Разрешены только изображения: jpg, jpeg, png, webp.")
                return redirect(url_for("index"))

            pos = f.stream.tell()
            f.stream.seek(0, os.SEEK_END)
            size = f.stream.tell()
            f.stream.seek(pos)
            if size > MAX_FILE_MB * 1024 * 1024:
                flash(f"Файл {f.filename} слишком большой. Лимит {MAX_FILE_MB}MB на фото.")
                return redirect(url_for("index"))

    sid = _new_id()
    created_utc = _now_iso()

    saved_names: List[str] = []
    if kind == "sell" and files:
        sub_dir = UPLOADS_DIR / sid
        sub_dir.mkdir(parents=True, exist_ok=True)

        for f in files:
            safe = secure_filename(f.filename) or "photo.jpg"
            target = sub_dir / safe
            if target.exists():
                target = sub_dir / f"{target.stem}_{uuid.uuid4().hex[:6]}{target.suffix}"
            f.save(target)
            saved_names.append(target.name)

    _save_submission_row(
        sid=sid,
        created_utc=created_utc,
        kind=kind,
        title=title,
        price_tenge=price_tenge,
        phone=phone,
        description=description,
        photos=saved_names,
    )

    return redirect(url_for("thanks", sid=sid))


@app.get("/thanks/<sid>")
def thanks(sid: str):
    rows = _read_all_rows()
    r = _find_row(rows, sid)
    kind = ((r.get("kind") or "sell") if r else "sell").strip().lower()
    if kind not in {"buy", "sell"}:
        kind = "sell"

    photos: list[str] = []
    sub_dir = UPLOADS_DIR / sid
    if sub_dir.exists() and sub_dir.is_dir():
        photos = sorted([p.name for p in sub_dir.iterdir() if p.is_file()])

    return render_template("thanks.html", sid=sid, photos=photos, kind=kind)


# Если не хочешь публичные ссылки на фото — удали этот роут
@app.get("/uploads/<sid>/<path:filename>")
def uploads(sid: str, filename: str):
    return send_from_directory(UPLOADS_DIR / sid, filename)


@app.get("/health")
def health():
    return {"status": "ok"}


# ----------------------------
# Admin
# ----------------------------

ADMIN_KEY = os.environ.get("ADMIN_KEY", "").strip()


def _is_admin() -> bool:
    if not ADMIN_KEY:
        return False
    k = session.get("admin_key", "")
    return bool(k) and hmac.compare_digest(k, ADMIN_KEY)


def admin_required(fn):
    from functools import wraps

    @wraps(fn)
    def wrapper(*args, **kwargs):
        if not _is_admin():
            return redirect(url_for("admin_login"))
        return fn(*args, **kwargs)

    return wrapper


def _admin_submissions(limit: int = 500) -> list[dict]:
    rows = _read_all_rows()
    items: list[dict] = []
    for r in rows:
        sid = (r.get("id") or "").strip()
        kind = (r.get("kind") or "sell").strip().lower()
        if kind not in {"buy", "sell"}:
            kind = "sell"

        photos_raw = (r.get("photos") or "").strip()
        photos = [p for p in photos_raw.split(";") if p] if photos_raw else _list_photos(sid)

        items.append({
            "id": sid,
            "created_utc": (r.get("created_utc") or "").strip(),
            "kind": kind,
            "title": (r.get("title") or "").strip(),
            "price_tenge": (r.get("price_tenge") or "").strip(),
            "phone": (r.get("phone") or "").strip(),
            "description": (r.get("description") or "").strip(),
            "photos": photos,
            "thumb_url": _thumb_url(sid, kind, photos),
        })

    items.sort(key=lambda x: x.get("created_utc", ""), reverse=True)
    return items[:limit]


@app.get("/admin/login")
def admin_login():
    return render_template("admin/login.html")


@app.post("/admin/login")
def admin_login_post():
    key = (request.form.get("key") or "").strip()
    if not ADMIN_KEY:
        flash("ADMIN_KEY не задан в Render Environment.")
        return redirect(url_for("admin_login"))
    if hmac.compare_digest(key, ADMIN_KEY):
        session["admin_key"] = key
        return redirect("/admin")
    flash("Неверный ключ.")
    return redirect(url_for("admin_login"))


@app.get("/admin/logout")
def admin_logout():
    session.pop("admin_key", None)
    return redirect(url_for("index"))


@app.get("/admin")
@admin_required
def admin_index():
    subs = _admin_submissions()
    return render_template(
        "admin/index.html",
        submissions=subs,
        csv_path=str(SUBMISSIONS_CSV),
        uploads_path=str(UPLOADS_DIR),
    )


@app.get("/admin/edit/<sid>")
@admin_required
def admin_edit(sid: str):
    rows = _read_all_rows()
    r = _find_row(rows, sid)
    if not r:
        abort(404)

    photos = _list_photos(sid)
    first_photo = photos[0] if photos else ""
    row = {c: (r.get(c, "") or "") for c in _csv_columns()}
    return render_template("admin/edit.html", sid=sid, row=row, photos=photos, first_photo=first_photo)


@app.post("/admin/save/<sid>")
@admin_required
def admin_save(sid: str):
    rows = _read_all_rows()
    r = _find_row(rows, sid)
    if not r:
        abort(404)

    kind = (request.form.get("kind") or "sell").strip().lower()
    if kind not in {"buy", "sell"}:
        kind = "sell"

    r["kind"] = kind
    r["title"] = (request.form.get("title") or "").strip()
    r["price_tenge"] = (request.form.get("price_tenge") or "").strip()
    r["phone"] = (request.form.get("phone") or "").strip()
    r["description"] = (request.form.get("description") or "").strip()

    # синхронизируем список фото с папкой
    photos = _list_photos(sid)
    r["photos"] = ";".join(photos)

    _write_all_rows(rows)
    flash("Сохранено.")
    return redirect(f"/admin/edit/{sid}")


@app.post("/admin/delete/<sid>")
@admin_required
def admin_delete(sid: str):
    rows = _read_all_rows()
    rows2 = [r for r in rows if (r.get("id") or "").strip() != sid]
    _write_all_rows(rows2)

    d = UPLOADS_DIR / sid
    if d.exists() and d.is_dir():
        shutil.rmtree(d)

    flash(f"Удалено: {sid}")
    return redirect("/admin")


@app.post("/admin/photo_delete/<sid>/<path:filename>")
@admin_required
def admin_photo_delete(sid: str, filename: str):
    p = (UPLOADS_DIR / sid / filename).resolve()
    base = (UPLOADS_DIR / sid).resolve()
    if not str(p).startswith(str(base)):
        abort(400)

    if p.exists() and p.is_file():
        p.unlink()

    rows = _read_all_rows()
    r = _find_row(rows, sid)
    if r:
        r["photos"] = ";".join(_list_photos(sid))
        _write_all_rows(rows)

    return redirect(f"/admin/edit/{sid}")


@app.post("/admin/upload/<sid>")
@admin_required
def admin_upload(sid: str):
    files = request.files.getlist("photos")
    files = [f for f in files if f and f.filename]

    if not files:
        flash("Не выбраны файлы.")
        return redirect(f"/admin/edit/{sid}")

    sub_dir = UPLOADS_DIR / sid
    sub_dir.mkdir(parents=True, exist_ok=True)

    saved = 0
    for f in files:
        name = secure_filename(f.filename) or "photo.jpg"
        target = sub_dir / name
        if target.exists():
            target = sub_dir / f"{target.stem}_{uuid.uuid4().hex[:6]}{target.suffix}"
        f.save(target)
        saved += 1

    rows = _read_all_rows()
    r = _find_row(rows, sid)
    if r:
        r["photos"] = ";".join(_list_photos(sid))
        _write_all_rows(rows)

    flash(f"Загружено фото: {saved}")
    return redirect(f"/admin/edit/{sid}")


@app.get("/admin/csv")
@admin_required
def admin_csv_download():
    if not SUBMISSIONS_CSV.exists():
        abort(404)
    return send_file(SUBMISSIONS_CSV, as_attachment=True, download_name="submissions.csv")


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "10000"))
    app.run(host="0.0.0.0", port=port)