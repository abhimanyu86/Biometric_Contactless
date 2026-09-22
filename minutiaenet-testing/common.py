"""
Shared setup for the without-preprocessing / with-preprocessing MinutiaeNet
matching test: points both phases at single_finger_backend_models/ (rather
than backend/, which doesn't hold the weight files) and gives both scripts
one place to load images and sweep thresholds.
"""
import csv
import os
import sys

import cv2

# Source photos are raw 8192x6144 (50MP) camera dumps. YOLO and MinutiaeNet
# resize internally before doing real work, so phase 1 is unaffected, but
# app.py's Zero-DCE step (with-preprocessing) runs on the YOLO-cropped image
# at ITS NATIVE resolution with no resize -- on a crop that's still several
# thousand pixels per side, that CNN call on CPU took over 3 hours for a
# single run and never finished. Downscaling to a realistic capture
# resolution here (not touching backend/app.py, which is production code)
# fixes this for both phases so the comparison stays apples-to-apples.
MAX_DIM = 1600

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(BASE_DIR)
BACKEND_DIR = os.path.join(REPO_ROOT, "backend")
MODELS_DIR = os.path.join(REPO_ROOT, "single_finger_backend_models")

ENROLL_IMAGE = os.path.join(REPO_ROOT, "image_db", "Rahul.jpg")
PROBE_DIR = os.path.join(REPO_ROOT, "test_db", "Rahul")

THRESHOLDS = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)

# app.py and minutiae.py read these env vars at import time to resolve model
# paths -- must be set before either module is imported.
os.environ.setdefault("SINGLE_MODELS_DIR", MODELS_DIR)
os.environ.setdefault("MINUTIAE_MODEL_PATH", os.path.join(MODELS_DIR, "best_f1.pth"))

if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)


def load_image(path):
    img = cv2.imread(path)
    if img is None:
        return None
    h, w = img.shape[:2]
    longest = max(h, w)
    if longest > MAX_DIM:
        scale = MAX_DIM / longest
        img = cv2.resize(img, (round(w * scale), round(h * scale)), interpolation=cv2.INTER_AREA)
    return img


GALLERY_DIR = os.path.join(REPO_ROOT, "image_db")


def list_gallery_images():
    """
    31-identity enrollment gallery: Person1..Person30 (image_db's other
    identities, mixed .jpeg/.png/.jpg extensions) plus Rahul -- used as
    impostor identities against Rahul's 106 test_db probes, since Rahul is
    the only identity with probe photos in test_db.
    Returns [(identity_label, path), ...] sorted Person1..Person30, Rahul last.
    """
    import glob as glob_mod

    entries = []
    for path in glob_mod.glob(os.path.join(GALLERY_DIR, "Person*.*")):
        name = os.path.splitext(os.path.basename(path))[0]
        entries.append((name, path))
    entries.sort(key=lambda e: int("".join(ch for ch in e[0] if ch.isdigit())))
    entries.append(("Rahul", ENROLL_IMAGE))
    return entries


def list_probe_images():
    names = sorted(
        os.listdir(PROBE_DIR),
        key=lambda n: int(os.path.splitext(n)[0]) if os.path.splitext(n)[0].isdigit() else n,
    )
    return [os.path.join(PROBE_DIR, n) for n in names]


def sweep_accuracy(scores):
    """scores: list of match scores for genuine (Rahul vs Rahul) pairs only."""
    total = len(scores)
    rows = []
    for threshold in THRESHOLDS:
        matched = sum(1 for s in scores if s >= threshold)
        rows.append(
            {
                "threshold": threshold,
                "matched": matched,
                "total": total,
                "accuracy": matched / total if total else float("nan"),
            }
        )
    return rows


def compute_far_frr(genuine_scores, impostor_scores):
    """Same FAR/FRR/EER definition as backend/benchmark.py's compute_metrics."""
    from bisect import bisect_left

    genuine_sorted = sorted(genuine_scores)
    impostor_sorted = sorted(impostor_scores)
    g_count = len(genuine_sorted)
    i_count = len(impostor_sorted)

    rows = []
    for threshold in THRESHOLDS:
        false_accepts = i_count - bisect_left(impostor_sorted, threshold) if i_count else 0
        false_rejects = bisect_left(genuine_sorted, threshold) if g_count else 0
        far = false_accepts / i_count if i_count else float("nan")
        frr = false_rejects / g_count if g_count else float("nan")
        rows.append(
            {
                "threshold": threshold,
                "far": far,
                "frr": frr,
                "false_accepts": false_accepts,
                "false_rejects": false_rejects,
                "total_genuine": g_count,
                "total_impostor": i_count,
            }
        )

    eer_row = None
    if rows:
        eer_row = min(rows, key=lambda r: abs(r["far"] - r["frr"]))
        eer_row = {**eer_row, "eer": (eer_row["far"] + eer_row["frr"]) / 2.0}
    return rows, eer_row


def write_per_image_csv(path, records):
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["image", "status", "score", "minutiae_count"])
        writer.writeheader()
        for row in records:
            writer.writerow(row)


def write_summary_csv(path, rows):
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["threshold", "matched", "total", "accuracy"])
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def print_summary(label, rows, failures, total_probes):
    print(f"\n=== {label} ===")
    print(f"Probes: {total_probes}  |  Extraction failures: {failures}")
    print(f"{'threshold':>9} | {'matched':>7} | {'total':>5} | {'accuracy':>8}")
    best = None
    for row in rows:
        print(f"{row['threshold']:>9.1f} | {row['matched']:>7} | {row['total']:>5} | {row['accuracy']:>8.2%}")
        if best is None or row["accuracy"] > best["accuracy"]:
            best = row
    if best:
        print(f"Best accuracy: {best['accuracy']:.2%} at threshold {best['threshold']:.1f}")
    return best
