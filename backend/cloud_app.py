"""
cloud_app.py — the cloud-only service: templatization + matching, nothing else.

Per the on-device/cloud split (see the architecture plan), on-device now owns
capture, quality control, liveness, and preprocessing (segmentation + enhancement)
for both single-finger and slap. This service receives the already-preprocessed
image(s) an on-device client produced, and does exactly two things to them:

    1. Templatization — detect_minutiae() (minutiae.py), the same MinutiaeNet
       both flows have always used.
    2. Matching — match_templates() (matching.py) for single-finger, or
       score_against_user() (slap_matching.py) for slap's per-finger aggregate.

Also exposes a second, independent path using a CNN/Siamese embedding matcher
(embedding_matcher.py, Phase A: pretrained ResNet18 + cosine similarity, 92.5%
accuracy at EER=7.5% on the HKPU single-finger validation set, see
siamese_embedding/) for both single-finger and slap:
    - Single-finger: /v2/enroll_embedding, /v2/authenticate_embedding,
      /v2/verify_embedding — own table (embedding_templates).
    - Slap: /v2/enroll_slap_embedding, /v2/authenticate_slap_embedding,
      /v2/verify_slap_embedding — own tables (slap_embedding_templates,
      slap_embedding_auth_history), same position-constrained top-2-average
      aggregation as the minutiae slap path (slap_embedding_matching.py mirrors
      slap_matching.py). Slap-specific accuracy has NOT been measured -- only
      the single-finger protocol has a validated number.
Both run alongside the existing minutiae flow against their own tables, not
instead of it — the original endpoints and their tables are unchanged.

Nothing here loads YOLO, liveness, U2Net, or Zero-DCE — those are on-device-tier
models this process never needs, which is the entire point of the split
(see requirements-cloud.txt). app.py and slap_app.py keep the full on-device-tier
reference pipeline for local development; this file is what actually deploys.

Routes are the SAME /v2/* endpoints already built and tested in app.py/slap_app.py
— moved here, not duplicated, so there is exactly one implementation of each.
"""

import json
import os
import sqlite3
import traceback
from datetime import datetime

import cv2
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS

from minutiae import detect_minutiae, get_minutiae_model
from matching import match_templates
from slap_matching import score_against_user, label_by_order, MATCH_THRESH
from embedding_matcher import embed_image, cosine_similarity, EMBEDDING_THRESHOLD, get_embedding_model
from slap_embedding_matching import score_against_user_embedding, EMBEDDING_SLAP_MIN_FINGERS, EMBEDDING_SLAP_FINGER_THRESHOLD

app = Flask(__name__)
CORS(app)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

DATABASE = os.environ.get("DATABASE", os.path.join(BASE_DIR, "uidai.db"))
SLAP_DB  = os.environ.get("SLAP_DB", os.path.join(BASE_DIR, "slap.db"))

SINGLE_THRESHOLD = 0.20
MATCHING_ALGORITHM = "legacy"


# ══════════════════════════════════════════════════════════════════════════════
# DATABASE — same schema as app.py's init_db() / slap_app.py's init_slap_db()
# ══════════════════════════════════════════════════════════════════════════════

def get_connection():
    conn = sqlite3.connect(DATABASE)
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def init_db():
    conn = get_connection()
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            name        TEXT NOT NULL,
            uid         TEXT NOT NULL,
            batch       TEXT NOT NULL,
            template_json TEXT NOT NULL,
            created_at  TEXT NOT NULL,
            finger_position INTEGER NOT NULL DEFAULT 0
        )
    """)
    c.execute("""
        CREATE UNIQUE INDEX IF NOT EXISTS idx_uid_batch_finger
            ON users(uid, batch, finger_position)
    """)
    c.execute("""
        CREATE TABLE IF NOT EXISTS enrollments (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id     INTEGER,
            timestamp   TEXT,
            minutiae_count INTEGER,
            FOREIGN KEY(user_id) REFERENCES users(id)
        )
    """)
    c.execute("""
        CREATE TABLE IF NOT EXISTS authentications (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id     INTEGER,
            timestamp   TEXT,
            confidence  REAL,
            matched     INTEGER,
            FOREIGN KEY(user_id) REFERENCES users(id)
        )
    """)
    conn.commit()
    conn.close()


def init_embedding_db():
    """Separate table from `users` — the CNN/Siamese embedding path (Phase A,
    embedding_matcher.py) is additive, not a replacement for the minutiae flow.
    Keeping it in its own table means /v2/enroll_embedding etc. can't ever
    corrupt or collide with the existing minutiae-based enroll/authenticate/verify."""
    conn = get_connection()
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS embedding_templates (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            name            TEXT NOT NULL,
            uid             TEXT NOT NULL,
            batch           TEXT NOT NULL,
            finger_position INTEGER NOT NULL DEFAULT 0,
            embedding_json  TEXT NOT NULL,
            created_at      TEXT NOT NULL
        )
    """)
    c.execute("""
        CREATE UNIQUE INDEX IF NOT EXISTS idx_embedding_uid_batch_finger
            ON embedding_templates(uid, batch, finger_position)
    """)
    conn.commit()
    conn.close()


def init_slap_db():
    con = sqlite3.connect(SLAP_DB)
    con.execute("""
        CREATE TABLE IF NOT EXISTS slap_templates (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            uid             TEXT NOT NULL,
            name            TEXT,
            batch           TEXT NOT NULL,
            finger_position TEXT NOT NULL,
            iso_code        INTEGER,
            minutiae_count  INTEGER,
            template_b64    TEXT NOT NULL,
            minutiae_json   TEXT,
            created_at      TEXT NOT NULL,
            UNIQUE(uid, batch, finger_position)
        )
    """)
    con.execute("""
        CREATE TABLE IF NOT EXISTS slap_auth_history (
            id                   INTEGER PRIMARY KEY AUTOINCREMENT,
            uid                  TEXT,
            name                 TEXT,
            batch                TEXT NOT NULL,
            hand_side            TEXT,
            success              INTEGER NOT NULL,
            avg_confidence       REAL,
            best_confidence      REAL,
            matched_fingers_json TEXT,
            created_at           TEXT NOT NULL
        )
    """)
    # CNN/Siamese embedding slap path -- separate table from slap_templates,
    # additive alongside it. Same DB file (slap.db), same key shape.
    con.execute("""
        CREATE TABLE IF NOT EXISTS slap_embedding_templates (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            uid             TEXT NOT NULL,
            name            TEXT,
            batch           TEXT NOT NULL,
            finger_position TEXT NOT NULL,
            iso_code        INTEGER,
            embedding_json  TEXT NOT NULL,
            created_at      TEXT NOT NULL,
            UNIQUE(uid, batch, finger_position)
        )
    """)
    con.execute("""
        CREATE TABLE IF NOT EXISTS slap_embedding_auth_history (
            id                   INTEGER PRIMARY KEY AUTOINCREMENT,
            uid                  TEXT,
            name                 TEXT,
            batch                TEXT NOT NULL,
            hand_side            TEXT,
            success              INTEGER NOT NULL,
            avg_confidence       REAL,
            best_confidence      REAL,
            matched_fingers_json TEXT,
            created_at           TEXT NOT NULL
        )
    """)
    con.commit()
    con.close()


# ══════════════════════════════════════════════════════════════════════════════
# UPLOAD HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def _save_upload(field_name="image"):
    f  = request.files[field_name]
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(UPLOAD_FOLDER, f"{ts}_{f.filename}")
    f.save(path)
    return path


def _read_images_list(field_name="image"):
    imgs = []
    for f in request.files.getlist(field_name):
        data = np.frombuffer(f.read(), np.uint8)
        img = cv2.imdecode(data, cv2.IMREAD_COLOR)
        if img is not None:
            imgs.append(img)
    return imgs


def _parse_finger_order():
    raw = request.form.get("finger_order", "").strip()
    return [p.strip() for p in raw.split(",") if p.strip()] or None


# ══════════════════════════════════════════════════════════════════════════════
# HEALTH
# ══════════════════════════════════════════════════════════════════════════════

@app.route("/health", methods=["GET"])
def health():
    try:
        get_minutiae_model()
        minutiae_ok = True
    except Exception as e:
        minutiae_ok = str(e)
    try:
        get_embedding_model()
        embedding_ok = True
    except Exception as e:
        embedding_ok = str(e)
    return jsonify({
        "status": "ok",
        "service": "cloud-templatization-matching",
        "minutiae_model": minutiae_ok,
        "embedding_model": embedding_ok,
        "embedding_threshold": EMBEDDING_THRESHOLD,
        "embedding_slap_finger_threshold": EMBEDDING_SLAP_FINGER_THRESHOLD,
        "embedding_slap_min_fingers": EMBEDDING_SLAP_MIN_FINGERS,
    })


# ══════════════════════════════════════════════════════════════════════════════
# SINGLE-FINGER — /v2/enroll, /v2/authenticate, /v2/verify
# Moved from app.py, unchanged.
# ══════════════════════════════════════════════════════════════════════════════

@app.route("/v2/enroll", methods=["POST"])
def enroll_v2():
    try:
        name  = request.form.get("name", "").strip()
        uid   = request.form.get("uid", "").strip()
        batch = request.form.get("batch", "").strip()
        if not all([name, uid, batch]):
            return jsonify({"success": False, "error": "name, uid, batch required"}), 400
        if "image" not in request.files:
            return jsonify({"success": False, "error": "image required"}), 400

        try:
            finger_position = int(request.form.get("finger_position", 0))
        except ValueError:
            return jsonify({"success": False,
                            "error": "finger_position must be an integer 0-10"}), 400
        if not 0 <= finger_position <= 10:
            return jsonify({"success": False,
                            "error": "finger_position must be 0-10"}), 400

        path = _save_upload()
        preprocessed = cv2.imread(path)
        if preprocessed is None:
            return jsonify({"success": False, "error": "image could not be decoded"}), 400

        minutiae = detect_minutiae(preprocessed)
        if len(minutiae) < 5:
            return jsonify({"success": False, "error": "Too few minutiae — retake"}), 422

        conn = get_connection()
        try:
            c = conn.cursor()
            c.execute("""
                INSERT INTO users (name, uid, batch, template_json, created_at, finger_position)
                VALUES (?,?,?,?,?,?)
                ON CONFLICT(uid,batch,finger_position) DO UPDATE SET
                    name=excluded.name, template_json=excluded.template_json, created_at=excluded.created_at
            """, (name, uid, batch, json.dumps(minutiae), datetime.now().isoformat(),
                  finger_position))
            # lastrowid returns 0 when the upsert hits the UPDATE branch (no new
            # row) — always fetch the real id so the enrollments FK never fails.
            c.execute("SELECT id FROM users WHERE uid=? AND batch=? AND finger_position=?",
                      (uid, batch, finger_position))
            user_id = c.fetchone()[0]
            c.execute("INSERT INTO enrollments (user_id,timestamp,minutiae_count) VALUES (?,?,?)",
                      (user_id, datetime.now().isoformat(), len(minutiae)))
            conn.commit()
        finally:
            conn.close()

        return jsonify({"success": True, "minutiae_count": len(minutiae)})
    except Exception as e:
        traceback.print_exc()
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/v2/authenticate", methods=["POST"])
def authenticate_v2():
    try:
        batch = request.form.get("batch", "").strip()
        if not batch:
            return jsonify({"success": False, "error": "batch required"}), 400
        if "image" not in request.files:
            return jsonify({"success": False, "error": "image required"}), 400

        path = _save_upload()
        preprocessed = cv2.imread(path)
        if preprocessed is None:
            return jsonify({"success": False, "error": "image could not be decoded"}), 400

        input_minutiae = detect_minutiae(preprocessed)

        conn = get_connection()
        try:
            c = conn.cursor()
            c.execute("SELECT id,name,uid,template_json FROM users WHERE batch=?", (batch,))
            users = c.fetchall()
            if not users:
                return jsonify({"success": False, "message": "No users in this batch"})

            best_score = 0.0; best_user = None
            for row in users:
                uid_db, name, uid, tmpl_json = row
                score = match_templates(input_minutiae, json.loads(tmpl_json),
                                        algorithm=MATCHING_ALGORITHM)
                if score > best_score:
                    best_score = score; best_user = (uid_db, name, uid)

            if not best_user or best_score < SINGLE_THRESHOLD:
                return jsonify({"success": False, "message": "No match found",
                                "confidence": round(best_score, 4)})

            uid_db, name, uid = best_user
            c.execute("INSERT INTO authentications (user_id,timestamp,confidence,matched) VALUES (?,?,?,1)",
                      (uid_db, datetime.now().isoformat(), best_score))
            conn.commit()
        finally:
            conn.close()

        return jsonify({"success": True, "name": name, "uid": uid,
                        "confidence": round(best_score, 4)})
    except Exception as e:
        traceback.print_exc()
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/v2/verify", methods=["POST"])
def verify_v2():
    try:
        uid   = request.form.get("uid", "").strip()
        batch = request.form.get("batch", "").strip()
        if not uid or not batch:
            return jsonify({"success": False, "error": "uid and batch required"}), 400
        if "image" not in request.files:
            return jsonify({"success": False, "error": "image required"}), 400

        path = _save_upload()
        preprocessed = cv2.imread(path)
        if preprocessed is None:
            return jsonify({"success": False, "error": "image could not be decoded"}), 400

        input_minutiae = detect_minutiae(preprocessed)

        conn = get_connection()
        try:
            c = conn.cursor()
            c.execute("SELECT id,name,uid,template_json FROM users WHERE uid=? AND batch=?", (uid, batch))
            row = c.fetchone()
        finally:
            conn.close()
        if not row:
            return jsonify({"success": False, "error": f"No user found: {uid}"}), 404

        user_id, name, uid_db, tmpl_json = row
        score   = match_templates(input_minutiae, json.loads(tmpl_json),
                                  algorithm=MATCHING_ALGORITHM)
        matched = score >= SINGLE_THRESHOLD

        # Log every verify attempt (matched or not) so a history view can show them.
        log_conn = get_connection()
        try:
            log_conn.execute(
                "INSERT INTO authentications (user_id,timestamp,confidence,matched) VALUES (?,?,?,?)",
                (user_id, datetime.now().isoformat(), score, 1 if matched else 0)
            )
            log_conn.commit()
        finally:
            log_conn.close()

        return jsonify({
            "success":    True,
            "matched":    matched,
            "confidence": round(score, 4),
            "threshold":  SINGLE_THRESHOLD,
            "name":       name,
            "uid":        uid_db,
            "input_minutiae_count": len(input_minutiae),
        })
    except Exception as e:
        traceback.print_exc()
        return jsonify({"success": False, "error": str(e)}), 500


# ══════════════════════════════════════════════════════════════════════════════
# SINGLE-FINGER, CNN/SIAMESE EMBEDDING PATH — /v2/enroll_embedding,
# /v2/authenticate_embedding, /v2/verify_embedding
#
# Phase A (embedding_matcher.py): pretrained ResNet18, cosine similarity, no
# fingerprint-specific training. Runs alongside the minutiae flow above against
# a separate table (embedding_templates) — nothing here touches `users` or the
# existing /v2/enroll, /v2/authenticate, /v2/verify behavior.
#
# EMBEDDING_THRESHOLD (default 0.94) was calibrated on the HKPU lab dataset, not
# this app's real camera captures — see embedding_matcher.py's module docstring.
# ══════════════════════════════════════════════════════════════════════════════

@app.route("/v2/enroll_embedding", methods=["POST"])
def enroll_embedding():
    try:
        name  = request.form.get("name", "").strip()
        uid   = request.form.get("uid", "").strip()
        batch = request.form.get("batch", "").strip()
        if not all([name, uid, batch]):
            return jsonify({"success": False, "error": "name, uid, batch required"}), 400
        if "image" not in request.files:
            return jsonify({"success": False, "error": "image required"}), 400

        try:
            finger_position = int(request.form.get("finger_position", 0))
        except ValueError:
            return jsonify({"success": False,
                            "error": "finger_position must be an integer 0-10"}), 400
        if not 0 <= finger_position <= 10:
            return jsonify({"success": False,
                            "error": "finger_position must be 0-10"}), 400

        path = _save_upload()
        preprocessed = cv2.imread(path)
        if preprocessed is None:
            return jsonify({"success": False, "error": "image could not be decoded"}), 400

        embedding = embed_image(preprocessed)

        conn = get_connection()
        try:
            c = conn.cursor()
            c.execute("""
                INSERT INTO embedding_templates (name, uid, batch, finger_position, embedding_json, created_at)
                VALUES (?,?,?,?,?,?)
                ON CONFLICT(uid,batch,finger_position) DO UPDATE SET
                    name=excluded.name, embedding_json=excluded.embedding_json, created_at=excluded.created_at
            """, (name, uid, batch, finger_position, json.dumps(embedding.tolist()),
                  datetime.now().isoformat()))
            conn.commit()
        finally:
            conn.close()

        return jsonify({"success": True, "embedding_dim": len(embedding)})
    except Exception as e:
        traceback.print_exc()
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/v2/authenticate_embedding", methods=["POST"])
def authenticate_embedding():
    try:
        batch = request.form.get("batch", "").strip()
        if not batch:
            return jsonify({"success": False, "error": "batch required"}), 400
        if "image" not in request.files:
            return jsonify({"success": False, "error": "image required"}), 400

        path = _save_upload()
        preprocessed = cv2.imread(path)
        if preprocessed is None:
            return jsonify({"success": False, "error": "image could not be decoded"}), 400

        input_embedding = embed_image(preprocessed)

        conn = get_connection()
        try:
            c = conn.cursor()
            c.execute("SELECT id,name,uid,embedding_json FROM embedding_templates WHERE batch=?", (batch,))
            templates = c.fetchall()
        finally:
            conn.close()

        if not templates:
            return jsonify({"success": False, "message": "No users in this batch"})

        best_score = -1.0
        best_user = None
        for row in templates:
            _, name, uid, emb_json = row
            score = cosine_similarity(input_embedding, json.loads(emb_json))
            if score > best_score:
                best_score = score
                best_user = (name, uid)

        if not best_user or best_score < EMBEDDING_THRESHOLD:
            return jsonify({"success": False, "message": "No match found",
                            "confidence": round(best_score, 4),
                            "threshold": EMBEDDING_THRESHOLD})

        name, uid = best_user
        return jsonify({"success": True, "name": name, "uid": uid,
                        "confidence": round(best_score, 4),
                        "threshold": EMBEDDING_THRESHOLD})
    except Exception as e:
        traceback.print_exc()
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/v2/verify_embedding", methods=["POST"])
def verify_embedding():
    try:
        uid   = request.form.get("uid", "").strip()
        batch = request.form.get("batch", "").strip()
        if not uid or not batch:
            return jsonify({"success": False, "error": "uid and batch required"}), 400
        if "image" not in request.files:
            return jsonify({"success": False, "error": "image required"}), 400

        path = _save_upload()
        preprocessed = cv2.imread(path)
        if preprocessed is None:
            return jsonify({"success": False, "error": "image could not be decoded"}), 400

        input_embedding = embed_image(preprocessed)

        conn = get_connection()
        try:
            c = conn.cursor()
            c.execute("SELECT name,embedding_json FROM embedding_templates WHERE uid=? AND batch=?", (uid, batch))
            row = c.fetchone()
        finally:
            conn.close()
        if not row:
            return jsonify({"success": False, "error": f"No user found: {uid}"}), 404

        name, emb_json = row
        score = cosine_similarity(input_embedding, json.loads(emb_json))
        matched = score >= EMBEDDING_THRESHOLD

        return jsonify({
            "success":    True,
            "matched":    matched,
            "confidence": round(score, 4),
            "threshold":  EMBEDDING_THRESHOLD,
            "name":       name,
            "uid":        uid,
        })
    except Exception as e:
        traceback.print_exc()
        return jsonify({"success": False, "error": str(e)}), 500


# ══════════════════════════════════════════════════════════════════════════════
# SLAP — /v2/enroll_slap, /v2/authenticate_slap, /v2/verify_slap
# Moved from slap_app.py; aggregation now goes through slap_matching's
# score_against_user() instead of two separately-duplicated formulas
# (see slap_matching.py's docstring, bug.md #8).
# ══════════════════════════════════════════════════════════════════════════════

@app.route("/v2/enroll_slap", methods=["POST"])
def enroll_slap_v2():
    try:
        uid = request.form.get("uid", "").strip()
        if not uid:
            return jsonify({"success": False, "error": "uid required"}), 400
        name      = request.form.get("name", "")
        batch     = request.form.get("batch", "default")
        hand_side = request.form.get("hand_side", "right")

        imgs = _read_images_list()
        if not imgs:
            return jsonify({"success": False, "error": "at least one image required"}), 400
        labels = label_by_order(len(imgs), hand_side, _parse_finger_order())

        con = sqlite3.connect(SLAP_DB)
        try:
            now = datetime.utcnow().isoformat()
            saved = []
            for img, label in zip(imgs, labels):
                minutiae = detect_minutiae(img)
                con.execute(
                    """INSERT OR REPLACE INTO slap_templates
                       (uid,name,batch,finger_position,iso_code,minutiae_count,
                        template_b64,minutiae_json,created_at)
                       VALUES (?,?,?,?,?,?,?,?,?)""",
                    (uid, name, batch, label["finger_position"], label["iso_code"],
                     len(minutiae), "",  # template_b64 (ISO 19794-2 FMR) — not built
                     json.dumps(minutiae), now),                                   # cloud-side yet; minutiae_json is
                )                                                                  # what matching actually reads.
                saved.append({
                    "finger_position": label["finger_position"],
                    "iso_code": label["iso_code"],
                    "minutiae_count": len(minutiae),
                })
            con.commit()
        finally:
            con.close()

        return jsonify({
            "success": True,
            "uid": uid, "name": name, "batch": batch, "hand_side": hand_side,
            "enrolled_fingers": saved, "finger_count": len(saved),
        })
    except Exception as e:
        traceback.print_exc()
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/v2/authenticate_slap", methods=["POST"])
def authenticate_slap_v2():
    try:
        batch = request.form.get("batch", "").strip()
        hand_side = request.form.get("hand_side", "right")
        if not batch:
            return jsonify({"success": False, "error": "batch required"}), 400

        imgs = _read_images_list()
        if not imgs:
            return jsonify({"success": False, "error": "at least one image required"}), 400
        labels = label_by_order(len(imgs), hand_side, _parse_finger_order())
        fingers = [{"finger_position": lbl["finger_position"],
                    "minutiae": detect_minutiae(img)}
                   for img, lbl in zip(imgs, labels)]

        con = sqlite3.connect(SLAP_DB)
        con.row_factory = sqlite3.Row
        try:
            rows = con.execute(
                "SELECT uid, name, finger_position, minutiae_json "
                "FROM slap_templates WHERE batch = ? ORDER BY uid",
                (batch,),
            ).fetchall()
            if not rows:
                return jsonify({"success": False,
                                "message": "No enrolled slap users in batch"}), 404

            by_uid = {}
            for r in rows:
                by_uid.setdefault(r["uid"], []).append(
                    {"finger_position": r["finger_position"],
                     "name": r["name"],
                     "minutiae_json": r["minutiae_json"]})

            best_uid = None
            best_name = None
            best_agg = 0.0
            best_matches = []
            for uid, user_rows in by_uid.items():
                agg_score, finger_scores = score_against_user(fingers, user_rows)
                if agg_score > best_agg:
                    best_agg = agg_score
                    best_uid = uid
                    best_name = user_rows[0]["name"]
                    best_matches = finger_scores

            success = best_agg >= MATCH_THRESH and best_uid is not None
            now = datetime.utcnow().isoformat()
            if success:
                avg_conf = (sum(m["confidence"] for m in best_matches)
                            / max(1, len(best_matches)))
                con.execute(
                    """INSERT INTO slap_auth_history
                    (uid,name,batch,hand_side,success,avg_confidence,best_confidence,
                     matched_fingers_json,created_at) VALUES (?,?,?,?,?,?,?,?,?)""",
                    (best_uid, best_name, batch, hand_side, 1,
                     round(avg_conf, 5), round(best_agg, 5),
                     json.dumps(best_matches), now),
                )
                con.commit()
                return jsonify({
                    "success": True,
                    "uid": best_uid,
                    "name": best_name,
                    "hand_side": hand_side,
                    "avg_confidence": round(avg_conf, 5),
                    "confidence": round(best_agg, 5),
                    "threshold": MATCH_THRESH,
                    "matched_fingers": best_matches,
                })
            else:
                con.execute(
                    """INSERT INTO slap_auth_history
                    (uid,name,batch,hand_side,success,avg_confidence,best_confidence,
                     matched_fingers_json,created_at) VALUES (?,?,?,?,?,?,?,?,?)""",
                    (None, None, batch, hand_side, 0, 0.0, round(best_agg, 5),
                     "[]", now),
                )
                con.commit()
                return jsonify({
                    "success": False,
                    "message": "No match found",
                    "confidence": round(best_agg, 5),
                    "threshold": MATCH_THRESH,
                })
        finally:
            con.close()
    except Exception as e:
        traceback.print_exc()
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/v2/verify_slap", methods=["POST"])
def verify_slap_v2():
    try:
        uid = request.form.get("uid", "").strip()
        batch = request.form.get("batch", "").strip()
        hand_side = request.form.get("hand_side", "right")
        if not uid or not batch:
            return jsonify({"success": False,
                            "error": "uid and batch required"}), 400

        imgs = _read_images_list()
        if not imgs:
            return jsonify({"success": False, "error": "at least one image required"}), 400
        labels = label_by_order(len(imgs), hand_side, _parse_finger_order())
        fingers = [{"finger_position": lbl["finger_position"],
                    "minutiae": detect_minutiae(img)}
                   for img, lbl in zip(imgs, labels)]

        con = sqlite3.connect(SLAP_DB)
        con.row_factory = sqlite3.Row
        try:
            rows = con.execute(
                "SELECT uid, name, finger_position, minutiae_json "
                "FROM slap_templates WHERE uid=? AND batch=?",
                (uid, batch),
            ).fetchall()
            if not rows:
                return jsonify({"success": False,
                                "error": f"No slap enrollment for uid={uid}"}), 404
            name = rows[0]["name"]

            user_rows = [{"finger_position": r["finger_position"],
                         "minutiae_json": r["minutiae_json"]} for r in rows]
            score, matched_fingers = score_against_user(fingers, user_rows)
            matched = score >= MATCH_THRESH
            avg_conf = (sum(m["confidence"] for m in matched_fingers)
                        / max(1, len(matched_fingers)))

            return jsonify({
                "success": True,
                "matched": matched,
                "uid": uid,
                "name": name,
                "hand_side": hand_side,
                "confidence": round(score, 5),
                "avg_confidence": round(avg_conf, 5),
                "threshold": MATCH_THRESH,
                "matched_fingers": matched_fingers,
            })
        finally:
            con.close()
    except Exception as e:
        traceback.print_exc()
        return jsonify({"success": False, "error": str(e)}), 500


# ══════════════════════════════════════════════════════════════════════════════
# SLAP, CNN/SIAMESE EMBEDDING PATH — /v2/enroll_slap_embedding,
# /v2/authenticate_slap_embedding, /v2/verify_slap_embedding
#
# Same request contract as the minutiae slap endpoints above (multiple images,
# one per finger, labelled by upload order via label_by_order). Matching logic
# mirrors single-finger embedding matching's shape, run once per finger (see
# slap_embedding_matching.py's module docstring for the full history): each
# probe finger is compared only to its position-matched enrolled finger,
# judged independently against EMBEDDING_SLAP_FINGER_THRESHOLD (0.78 --
# deliberately its own, lower, real-photo-calibrated constant, NOT
# single-finger's lab-calibrated EMBEDDING_THRESHOLD of 0.94), and a candidate
# passes if at least EMBEDDING_SLAP_MIN_FINGERS (default 2) fingers
# individually clear it. Separate table (slap_embedding_templates, same
# slap.db file) — nothing about the existing minutiae slap flow, or the
# single-finger embedding flow, changes.
#
# Not validated for slap-specific accuracy — see slap_embedding_matching.py's
# module docstring.
# ══════════════════════════════════════════════════════════════════════════════

@app.route("/v2/enroll_slap_embedding", methods=["POST"])
def enroll_slap_embedding():
    try:
        uid = request.form.get("uid", "").strip()
        if not uid:
            return jsonify({"success": False, "error": "uid required"}), 400
        name      = request.form.get("name", "")
        batch     = request.form.get("batch", "default")
        hand_side = request.form.get("hand_side", "right")

        imgs = _read_images_list()
        if not imgs:
            return jsonify({"success": False, "error": "at least one image required"}), 400
        labels = label_by_order(len(imgs), hand_side, _parse_finger_order())

        con = sqlite3.connect(SLAP_DB)
        try:
            now = datetime.utcnow().isoformat()
            saved = []
            for img, label in zip(imgs, labels):
                embedding = embed_image(img)
                con.execute(
                    """INSERT OR REPLACE INTO slap_embedding_templates
                       (uid,name,batch,finger_position,iso_code,embedding_json,created_at)
                       VALUES (?,?,?,?,?,?,?)""",
                    (uid, name, batch, label["finger_position"], label["iso_code"],
                     json.dumps(embedding.tolist()), now),
                )
                saved.append({
                    "finger_position": label["finger_position"],
                    "iso_code": label["iso_code"],
                    "embedding_dim": len(embedding),
                })
            con.commit()
        finally:
            con.close()

        return jsonify({
            "success": True,
            "uid": uid, "name": name, "batch": batch, "hand_side": hand_side,
            "enrolled_fingers": saved, "finger_count": len(saved),
        })
    except Exception as e:
        traceback.print_exc()
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/v2/authenticate_slap_embedding", methods=["POST"])
def authenticate_slap_embedding():
    try:
        batch = request.form.get("batch", "").strip()
        hand_side = request.form.get("hand_side", "right")
        if not batch:
            return jsonify({"success": False, "error": "batch required"}), 400

        imgs = _read_images_list()
        if not imgs:
            return jsonify({"success": False, "error": "at least one image required"}), 400
        labels = label_by_order(len(imgs), hand_side, _parse_finger_order())
        fingers = [{"finger_position": lbl["finger_position"],
                    "embedding": embed_image(img)}
                   for img, lbl in zip(imgs, labels)]

        con = sqlite3.connect(SLAP_DB)
        con.row_factory = sqlite3.Row
        try:
            rows = con.execute(
                "SELECT uid, name, finger_position, embedding_json "
                "FROM slap_embedding_templates WHERE batch = ? ORDER BY uid",
                (batch,),
            ).fetchall()
            if not rows:
                return jsonify({"success": False,
                                "message": "No enrolled slap users in batch"}), 404

            by_uid = {}
            for r in rows:
                by_uid.setdefault(r["uid"], []).append(
                    {"finger_position": r["finger_position"],
                     "name": r["name"],
                     "embedding_json": r["embedding_json"]})

            best_uid = None
            best_name = None
            best_passed = -1
            best_avg = -1.0
            best_matches = []
            for uid, user_rows in by_uid.items():
                fingers_passed, finger_scores = score_against_user_embedding(fingers, user_rows)
                avg = sum(m["confidence"] for m in finger_scores) / max(1, len(finger_scores))
                # Rank candidates by how many fingers independently passed first,
                # average confidence only as a tiebreak -- fingers_passed is the
                # actual decision signal now, not a continuous aggregate score.
                if (fingers_passed, avg) > (best_passed, best_avg):
                    best_passed = fingers_passed
                    best_avg = avg
                    best_uid = uid
                    best_name = user_rows[0]["name"]
                    best_matches = finger_scores

            success = best_uid is not None and best_passed >= EMBEDDING_SLAP_MIN_FINGERS
            avg_conf = max(best_avg, 0.0)
            now = datetime.utcnow().isoformat()
            if success:
                con.execute(
                    """INSERT INTO slap_embedding_auth_history
                    (uid,name,batch,hand_side,success,avg_confidence,best_confidence,
                     matched_fingers_json,created_at) VALUES (?,?,?,?,?,?,?,?,?)""",
                    (best_uid, best_name, batch, hand_side, 1,
                     round(avg_conf, 5), round(avg_conf, 5),
                     json.dumps(best_matches), now),
                )
                con.commit()
                return jsonify({
                    "success": True,
                    "uid": best_uid,
                    "name": best_name,
                    "hand_side": hand_side,
                    "fingers_passed": best_passed,
                    "min_fingers_required": EMBEDDING_SLAP_MIN_FINGERS,
                    "avg_confidence": round(avg_conf, 5),
                    "threshold": EMBEDDING_SLAP_FINGER_THRESHOLD,
                    "matched_fingers": best_matches,
                })
            else:
                con.execute(
                    """INSERT INTO slap_embedding_auth_history
                    (uid,name,batch,hand_side,success,avg_confidence,best_confidence,
                     matched_fingers_json,created_at) VALUES (?,?,?,?,?,?,?,?,?)""",
                    (None, None, batch, hand_side, 0, 0.0, round(avg_conf, 5),
                     json.dumps(best_matches), now),
                )
                con.commit()
                return jsonify({
                    "success": False,
                    "message": "No match found",
                    "fingers_passed": max(best_passed, 0),
                    "min_fingers_required": EMBEDDING_SLAP_MIN_FINGERS,
                    "threshold": EMBEDDING_SLAP_FINGER_THRESHOLD,
                    "matched_fingers": best_matches,
                })
        finally:
            con.close()
    except Exception as e:
        traceback.print_exc()
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/v2/verify_slap_embedding", methods=["POST"])
def verify_slap_embedding():
    try:
        uid = request.form.get("uid", "").strip()
        batch = request.form.get("batch", "").strip()
        hand_side = request.form.get("hand_side", "right")
        if not uid or not batch:
            return jsonify({"success": False,
                            "error": "uid and batch required"}), 400

        imgs = _read_images_list()
        if not imgs:
            return jsonify({"success": False, "error": "at least one image required"}), 400
        labels = label_by_order(len(imgs), hand_side, _parse_finger_order())
        fingers = [{"finger_position": lbl["finger_position"],
                    "embedding": embed_image(img)}
                   for img, lbl in zip(imgs, labels)]

        con = sqlite3.connect(SLAP_DB)
        con.row_factory = sqlite3.Row
        try:
            rows = con.execute(
                "SELECT uid, name, finger_position, embedding_json "
                "FROM slap_embedding_templates WHERE uid=? AND batch=?",
                (uid, batch),
            ).fetchall()
            if not rows:
                return jsonify({"success": False,
                                "error": f"No slap embedding enrollment for uid={uid}"}), 404
            name = rows[0]["name"]

            user_rows = [{"finger_position": r["finger_position"],
                         "embedding_json": r["embedding_json"]} for r in rows]
            fingers_passed, matched_fingers = score_against_user_embedding(fingers, user_rows)
            matched = fingers_passed >= EMBEDDING_SLAP_MIN_FINGERS
            avg_conf = (sum(m["confidence"] for m in matched_fingers)
                        / max(1, len(matched_fingers))) if matched_fingers else 0.0

            return jsonify({
                "success": True,
                "matched": matched,
                "uid": uid,
                "name": name,
                "hand_side": hand_side,
                "fingers_passed": fingers_passed,
                "min_fingers_required": EMBEDDING_SLAP_MIN_FINGERS,
                "avg_confidence": round(avg_conf, 5),
                "threshold": EMBEDDING_SLAP_FINGER_THRESHOLD,
                "matched_fingers": matched_fingers,
            })
        finally:
            con.close()
    except Exception as e:
        traceback.print_exc()
        return jsonify({"success": False, "error": str(e)}), 500


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

# Module-level, not inside __main__: gunicorn imports this module as "cloud_app:app"
# rather than executing it directly, so anything gated behind __main__ (as app.py's
# init_db() is) silently never runs under a production WSGI server. Both are
# idempotent (CREATE TABLE IF NOT EXISTS), so this is safe to call on every import.
init_db()
init_embedding_db()
init_slap_db()

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("  SITAA cloud service — templatization + matching only")
    print("=" * 60)
    print("  POST /v2/enroll                  — single-finger enroll (minutiae)")
    print("  POST /v2/authenticate            — single-finger 1:N (minutiae)")
    print("  POST /v2/verify                  — single-finger 1:1 (minutiae)")
    print("  POST /v2/enroll_embedding        — single-finger enroll (CNN/Siamese, Phase A)")
    print("  POST /v2/authenticate_embedding  — single-finger 1:N (CNN/Siamese, Phase A)")
    print("  POST /v2/verify_embedding        — single-finger 1:1 (CNN/Siamese, Phase A)")
    print("  POST /v2/enroll_slap                  — slap enroll (minutiae)")
    print("  POST /v2/authenticate_slap            — slap 1:N (minutiae)")
    print("  POST /v2/verify_slap                  — slap 1:1 (minutiae)")
    print("  POST /v2/enroll_slap_embedding        — slap enroll (CNN/Siamese, Phase A)")
    print("  POST /v2/authenticate_slap_embedding  — slap 1:N (CNN/Siamese, Phase A)")
    print("  POST /v2/verify_slap_embedding        — slap 1:1 (CNN/Siamese, Phase A)")
    print("  GET  /health")
    print("=" * 60 + "\n")
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port, debug=False)
