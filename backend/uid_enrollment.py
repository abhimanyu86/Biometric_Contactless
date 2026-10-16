"""
uid_enrollment.py — UID-keyed multi-capture enrolment and 1:1 verification.

A person is enrolled under one unique 12-character alphanumeric UID with exactly
UID_CAPTURE_COUNT (5) captures taken at different angles. Each capture's processed image
and MinutiaeNet minutiae are stored under that UID. At verification the caller supplies
the UID plus one probe; the probe is matched only against that UID's stored captures
(1:1 verification, not a 1:N search).

This module holds the parts that need neither model weights nor Flask — UID validation,
the schema, storage, and the aggregate decision — so tests/test_uid_enrollment.py can run
without TensorFlow/torch. cloud_app.py wires it to the HTTP routes and to
minutiae.detect_minutiae().

Matcher: algorithm009 ("ransac_similarity"), the only registered algorithm that passes
every structural test in MATCHING_ALGORITHMS.md (rotation, outlier, uniform scale, partial
overlap) — the properties 5 angled captures vary. Its genuine-pair accuracy on real data
is not yet measured. UID_MATCH_THRESHOLD below is an uncalibrated starting point, same
caveat as MATCH_THRESH in slap_matching.py; calibrate from a measured FAR/FRR curve.
"""

import json
import os
import re
import sqlite3
from datetime import datetime

from matching import match_templates

UID_CAPTURE_COUNT = 5
MIN_MINUTIAE = 5

UID_MATCH_ALGORITHM = os.environ.get("UID_MATCH_ALGORITHM", "ransac_similarity")
UID_MATCH_THRESHOLD = float(os.environ.get("UID_MATCH_THRESHOLD", "0.20"))
# "max": best single capture. "top2_mean": mean of the two best — resists one lucky match.
UID_SCORE_RULE = os.environ.get("UID_SCORE_RULE", "top2_mean")

_UID_RE = re.compile(r"^[A-Za-z0-9]{12}$")


class UidError(Exception):
    """Client-correctable problem; `status` is the HTTP status to report."""

    def __init__(self, message, status=400):
        super().__init__(message)
        self.status = status


def validate_uid(uid):
    """Return the UID, or raise UidError. Exactly 12 ASCII letters/digits."""
    uid = (uid or "").strip()
    if not _UID_RE.match(uid):
        raise UidError("uid must be exactly 12 alphanumeric characters")
    return uid


def _connect(db_path):
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def init_uid_db(db_path):
    """Idempotent; safe to call at module import (gunicorn does not run __main__)."""
    conn = _connect(db_path)
    try:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS uid_users (
                uid        TEXT PRIMARY KEY
                           CHECK (length(uid) = 12 AND uid NOT GLOB '*[^A-Za-z0-9]*'),
                name       TEXT,
                created_at TEXT NOT NULL
            )
        """)
        conn.execute(f"""
            CREATE TABLE IF NOT EXISTS uid_captures (
                uid            TEXT NOT NULL,
                idx            INTEGER NOT NULL CHECK (idx BETWEEN 1 AND {UID_CAPTURE_COUNT}),
                processed_png  BLOB,
                minutiae_json  TEXT NOT NULL,
                minutiae_count INTEGER NOT NULL,
                created_at     TEXT NOT NULL,
                PRIMARY KEY (uid, idx),
                FOREIGN KEY (uid) REFERENCES uid_users(uid) ON DELETE CASCADE
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS uid_verifications (
                id         INTEGER PRIMARY KEY AUTOINCREMENT,
                uid        TEXT NOT NULL,
                score      REAL,
                matched    INTEGER NOT NULL,
                created_at TEXT NOT NULL
            )
        """)
        conn.commit()
    finally:
        conn.close()


def enroll_uid(db_path, uid, name, captures):
    """Store exactly UID_CAPTURE_COUNT captures under a new UID, atomically.

    captures: list of {"minutiae": [...], "png": bytes | None}.
    Raises UidError 409 if the UID already exists (no silent overwrite of a biometric
    identity), 400/422 for a wrong capture count or a capture with too few minutiae.
    """
    uid = validate_uid(uid)
    if len(captures) != UID_CAPTURE_COUNT:
        raise UidError(f"exactly {UID_CAPTURE_COUNT} images required, got {len(captures)}")
    for i, cap in enumerate(captures, 1):
        if len(cap["minutiae"]) < MIN_MINUTIAE:
            raise UidError(f"image {i}: too few minutiae ({len(cap['minutiae'])}) — retake",
                           status=422)

    now = datetime.now().isoformat()
    conn = _connect(db_path)
    try:
        try:
            conn.execute("INSERT INTO uid_users (uid, name, created_at) VALUES (?,?,?)",
                         (uid, (name or "").strip() or None, now))
        except sqlite3.IntegrityError:
            raise UidError("uid already enrolled", status=409)
        conn.executemany(
            "INSERT INTO uid_captures (uid, idx, processed_png, minutiae_json,"
            " minutiae_count, created_at) VALUES (?,?,?,?,?,?)",
            [(uid, i, cap.get("png"), json.dumps(cap["minutiae"]), len(cap["minutiae"]), now)
             for i, cap in enumerate(captures, 1)])
        conn.commit()  # one transaction: all 5 captures or none
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()
    return [len(c["minutiae"]) for c in captures]


def load_templates(db_path, uid):
    """Minutiae lists for a UID, ordered by capture index; [] if the UID is unknown."""
    conn = _connect(db_path)
    try:
        rows = conn.execute(
            "SELECT minutiae_json FROM uid_captures WHERE uid=? ORDER BY idx", (uid,)
        ).fetchall()
    finally:
        conn.close()
    return [json.loads(r[0]) for r in rows]


def aggregate(scores, rule=None):
    """Collapse per-capture scores to one number."""
    rule = rule or UID_SCORE_RULE
    if not scores:
        return 0.0
    ranked = sorted(scores, reverse=True)
    if rule == "max" or len(ranked) < 2:
        return ranked[0]
    if rule == "top2_mean":
        return (ranked[0] + ranked[1]) / 2.0
    raise ValueError(f"unknown score rule {rule!r}")


def verify_probe(db_path, uid, probe_minutiae, algorithm=None, threshold=None, rule=None):
    """Match one probe against the UID's stored captures. Returns a result dict.

    Raises UidError 404 for an unknown UID. An unknown UID and a failed match are kept
    distinct here; the HTTP layer may choose to present them identically.
    """
    uid = validate_uid(uid)
    algorithm = algorithm or UID_MATCH_ALGORITHM
    threshold = UID_MATCH_THRESHOLD if threshold is None else threshold

    templates = load_templates(db_path, uid)
    if not templates:
        raise UidError("uid not enrolled", status=404)

    per_capture = [match_templates(probe_minutiae, t, algorithm=algorithm) for t in templates]
    score = aggregate(per_capture, rule)
    matched = score >= threshold

    conn = _connect(db_path)
    try:
        conn.execute("INSERT INTO uid_verifications (uid, score, matched, created_at)"
                     " VALUES (?,?,?,?)", (uid, score, int(matched), datetime.now().isoformat()))
        conn.commit()
    finally:
        conn.close()

    return {"matched": matched, "score": round(score, 4), "threshold": threshold,
            "per_capture": [round(s, 4) for s in per_capture],
            "rule": rule or UID_SCORE_RULE, "algorithm": algorithm}
