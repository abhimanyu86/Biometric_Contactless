"""
Pilot comparison: this project's own MinutiaeNet (backend/minutiae.py) vs. a
pure-Python classical extractor (fingerprint_feature_extractor, PyPI --
Gabor-filter enhancement + skeletonization + crossing-number minutiae
detection) vs. NIST NBIS MINDTCT (the reference public-domain fingerprint
minutiae extractor, compiled from source in WSL since it has no native
Windows build path -- see minutiae-labelling-and-training/README notes),
run on images AFTER the same YOLO detect-and-crop step the on-device
pipeline actually uses (backend/app.py:detect_and_crop_image) -- not raw,
uncropped photos. The first pilot run used raw photos and the comparison
was dominated by background noise (points scattered on wallpaper behind an
out-of-focus finger), which was a capture-quality artifact, not a real
extractor-quality signal -- this version removes that confound.

Purpose unchanged: get a first, cheap read on whether independent
extractors agree or disagree with MinutiaeNet's own output on the same
real (now properly cropped) images, before committing to any ground-truth-
labeling approach.
"""
import os
import re
import subprocess
import sys

import cv2
import numpy as np
import fingerprint_feature_extractor as ffe

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
BACKEND_DIR = os.path.join(os.path.dirname(BASE_DIR), "backend")
sys.path.insert(0, BACKEND_DIR)

from minutiae import detect_minutiae  # noqa: E402
from app import detect_and_crop_image  # noqa: E402 -- same YOLO crop the on-device pipeline uses

OUT_DIR = os.path.join(BASE_DIR, "pilot_output")
os.makedirs(OUT_DIR, exist_ok=True)

# MINDTCT toolchain: compiled inside WSL Ubuntu (native Windows MinGW build
# hit unfixable path-translation bugs in NBIS's ancient Makefiles -- see
# session notes). cjpegl converts a raw pixmap to lossless JPEG first because
# this NBIS build used --STDLIBS (no JASPER/OpenJP2/PNG), so mindtct's own
# image-type sniffer only recognizes WSQ/JPEGL/JPEGB/IHead/ANSI_NIST -- a
# plain PGM is reported as "image type UNKNOWN".
WSL_DISTRO = "Ubuntu"
MINDTCT_TMP = os.path.join(OUT_DIR, "_mindtct_tmp")
os.makedirs(MINDTCT_TMP, exist_ok=True)
CJPEGL_BIN = "~/nbis-build/src/imgtools/bin/cjpegl"
MINDTCT_BIN = "~/nbis-build/src/mindtct/bin/mindtct"
MINDTCT_MIN_LINE_RE = re.compile(
    r"^\s*\d+\s*:\s*(\d+),\s*(\d+)\s*:\s*(\d+)\s*:\s*[\d.]+\s*:(RIG|BIF)\s*:"
)


def to_wsl_path(win_path):
    win_path = os.path.abspath(win_path)
    drive, rest = os.path.splitdrive(win_path)
    return f"/mnt/{drive.rstrip(':').lower()}{rest.replace(os.sep, '/')}"

# Placeholder samples -- swap for the real target-domain "contactless enhanced"
# images once available (drop them into this folder and re-point this list).
SAMPLE_IMAGES = [
    r"D:\Contactless\image_db\Person_01.jpeg",
    r"D:\Contactless\image_db\Person_05.jpeg",
    r"D:\Contactless\image_db\Person10.png",
    r"D:\Contactless\image_db\Person_03.jpeg",
    r"D:\Contactless\image_db\Person15.png",
]


def run_classical_extractor(gray):
    """Returns list of (x, y, orientation_deg, type) -- 'RIG' or 'BIF'."""
    terms, bifs = ffe.extract_minutiae_features(
        gray, spuriousMinutiaeThresh=10, invertImage=False, showResult=False, saveResult=False,
    )
    points = []
    for m in terms:
        points.append((m.locX, m.locY, m.Orientation[0] if m.Orientation else 0.0, "RIG"))
    for m in bifs:
        points.append((m.locX, m.locY, m.Orientation[0] if m.Orientation else 0.0, "BIF"))
    return points


def run_minutiaenet(img_bgr):
    """Returns list of (x, y, direction_deg, type)."""
    minutiae = detect_minutiae(img_bgr)
    return [(m["x"], m["y"], np.degrees(m["direction"]), m["type"]) for m in minutiae]


# MINDTCT hardcodes DEFAULT_PPI=500 (mindtct/include/lfs.h) whenever the input
# doesn't carry an explicit PPI -- and our first pass never told it one, so
# it silently treated every phone-camera crop as if it were a 500-PPI contact
# scan. These crops are at wildly different effective scales depending on
# how close the phone was held, so that assumption is wrong by a different
# amount every time -- a very plausible cause of the wild minutiae-count
# swings (2 vs 81 vs 0) seen in the first run. Fix: resize each crop so its
# ridge frequency actually corresponds to ~500 PPI before handing it to
# MINDTCT, using an assumed average adult fingertip pad width as the scale
# reference (flagged explicitly -- this is an approximation, not a
# calibrated measurement, since we have no physical reference object in
# frame). Coordinates are scaled back to the original crop's pixel space
# before returning so overlays still align with the unscaled image.
ASSUMED_FINGER_WIDTH_MM = 16.0
MINDTCT_TARGET_PPI = 500


def run_mindtct(gray, name):
    """Returns list of (x, y, direction_deg, type) via NIST NBIS MINDTCT (WSL)."""
    orig_h, orig_w = gray.shape
    target_w = round(MINDTCT_TARGET_PPI * ASSUMED_FINGER_WIDTH_MM / 25.4)
    scale = target_w / orig_w
    scaled = cv2.resize(gray, (target_w, round(orig_h * scale)), interpolation=cv2.INTER_CUBIC)
    h, w = scaled.shape

    raw_path = os.path.join(MINDTCT_TMP, f"{name}.raw")
    with open(raw_path, "wb") as f:
        f.write(scaled.tobytes())

    wsl_dir = to_wsl_path(MINDTCT_TMP)
    cmd = (
        f"cd {wsl_dir} && "
        f"{CJPEGL_BIN} jpl {name}.raw -raw_in {w},{h},8,{MINDTCT_TARGET_PPI} && "
        f"{MINDTCT_BIN} -m1 {name}.jpl {name}_out"
    )
    result = subprocess.run(
        ["wsl", "-d", WSL_DISTRO, "--", "bash", "-c", cmd],
        capture_output=True, text=True, timeout=60,
    )
    if result.returncode != 0:
        raise RuntimeError(f"mindtct failed: {result.stdout.strip()} {result.stderr.strip()}")

    min_path = os.path.join(MINDTCT_TMP, f"{name}_out.min")
    points = []
    with open(min_path) as f:
        for line in f:
            m = MINDTCT_MIN_LINE_RE.match(line)
            if m:
                x, y, angle, typ = m.groups()
                # NBIS quantizes direction into 32 compass units (360/32 = 11.25 deg/unit).
                # Scale x,y back from the 500-PPI-normalized image to the original crop.
                points.append((int(int(x) / scale), int(int(y) / scale), int(angle) * 11.25, typ))
    return points


def draw_overlay(img_bgr, points, color_rig, color_bif, radius=4):
    out = img_bgr.copy()
    if out.ndim == 2:
        out = cv2.cvtColor(out, cv2.COLOR_GRAY2BGR)
    for x, y, direction, typ in points:
        color = color_rig if typ == "RIG" else color_bif
        cv2.circle(out, (int(x), int(y)), radius, color, 1)
        dx = int(radius * 2 * np.cos(np.radians(direction)))
        dy = int(radius * 2 * np.sin(np.radians(direction)))
        cv2.line(out, (int(x), int(y)), (int(x) + dx, int(y) + dy), color, 1)
    return out


def main():
    for path in SAMPLE_IMAGES:
        if not os.path.isfile(path):
            print(f"[pilot] WARNING: missing {path}, skipping")
            continue

        # Disambiguate by parent folder too -- fixed the earlier filename
        # collision bug where two different people's "p1.bmp" overwrote
        # each other's output.
        parent = os.path.basename(os.path.dirname(path))
        stem = os.path.splitext(os.path.basename(path))[0]
        name = f"{parent}_{stem}" if parent else stem

        raw_img = cv2.imread(path)
        if raw_img is None:
            print(f"[pilot] WARNING: could not read {path}, skipping")
            continue

        print(f"\n=== {name} ===")

        try:
            img, det_conf = detect_and_crop_image(raw_img, conf=0.1)
            print(f"  YOLO crop: {raw_img.shape[:2]} -> {img.shape[:2]} (det_conf={det_conf:.3f})")
        except ValueError as e:
            print(f"  YOLO crop FAILED ({e}) -- skipping, no usable finger detected")
            continue

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        mn_points = run_minutiaenet(img)
        print(f"  MinutiaeNet (project's own):        {len(mn_points)} minutiae")

        try:
            cl_points = run_classical_extractor(gray)
            print(f"  fingerprint_feature_extractor:      {len(cl_points)} minutiae "
                  f"({sum(1 for p in cl_points if p[3]=='RIG')} ridge-endings, "
                  f"{sum(1 for p in cl_points if p[3]=='BIF')} bifurcations)")
        except Exception as e:
            print(f"  fingerprint_feature_extractor: FAILED ({e})")
            cl_points = []

        try:
            mt_points = run_mindtct(gray, name)
            print(f"  MINDTCT (NIST NBIS):                {len(mt_points)} minutiae "
                  f"({sum(1 for p in mt_points if p[3]=='RIG')} ridge-endings, "
                  f"{sum(1 for p in mt_points if p[3]=='BIF')} bifurcations)")
        except Exception as e:
            print(f"  MINDTCT: FAILED ({e})")
            mt_points = []

        mn_overlay = draw_overlay(img, mn_points, color_rig=(0, 200, 0), color_bif=(0, 255, 255))
        cl_overlay = draw_overlay(img, cl_points, color_rig=(0, 0, 255), color_bif=(255, 0, 255))
        mt_overlay = draw_overlay(img, mt_points, color_rig=(0, 140, 255), color_bif=(255, 0, 0))

        h, w = img.shape[:2]
        gap = 20
        combined = np.zeros((h, w * 3 + gap * 2, 3), dtype=np.uint8)
        combined[:, :w] = mn_overlay
        combined[:, w+gap:w*2+gap] = cl_overlay
        combined[:, w*2+gap*2:w*3+gap*2] = mt_overlay
        cv2.putText(combined, "MinutiaeNet (ours)", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 0), 1)
        cv2.putText(combined, "Classical extractor", (w+gap+10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        cv2.putText(combined, "MINDTCT (NIST)", (w*2+gap*2+10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 140, 255), 1)

        out_path = os.path.join(OUT_DIR, f"{name}_cropped_comparison.png")
        cv2.imwrite(out_path, combined)
        print(f"  wrote {out_path}")


if __name__ == "__main__":
    main()
