"""
Single-panel MinutiaeNet-only labeled overlays (YOLO-cropped first, same as
the extractor pilot) for the user-uploaded photos -- just MinutiaeNet's own
detections drawn on each image, not the 3-way comparison layout.
"""
import glob
import os
import sys

import cv2
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
BACKEND_DIR = os.path.join(os.path.dirname(BASE_DIR), "backend")
sys.path.insert(0, BACKEND_DIR)

from minutiae import detect_minutiae  # noqa: E402
from app import detect_and_crop_image  # noqa: E402

IN_DIR = os.path.join(BASE_DIR, "user_photos")
OUT_DIR = os.path.join(BASE_DIR, "pilot_output", "minutiaenet_labeled")
os.makedirs(OUT_DIR, exist_ok=True)


def run_minutiaenet(img_bgr):
    minutiae = detect_minutiae(img_bgr)
    return [(m["x"], m["y"], np.degrees(m["direction"]), m["type"]) for m in minutiae]


def draw_overlay(img_bgr, points, color_rig=(0, 200, 0), color_bif=(0, 255, 255), radius=4):
    out = img_bgr.copy()
    for x, y, direction, typ in points:
        color = color_rig if typ == "RIG" else color_bif
        cv2.circle(out, (int(x), int(y)), radius, color, 1)
        dx = int(radius * 2 * np.cos(np.radians(direction)))
        dy = int(radius * 2 * np.sin(np.radians(direction)))
        cv2.line(out, (int(x), int(y)), (int(x) + dx, int(y) + dy), color, 1)
    return out


def main():
    paths = sorted(glob.glob(os.path.join(IN_DIR, "*.jpg")))
    print(f"Found {len(paths)} photos in {IN_DIR}")

    for path in paths:
        name = os.path.splitext(os.path.basename(path))[0]
        raw_img = cv2.imread(path)
        if raw_img is None:
            print(f"  {name}: could not read, skipping")
            continue

        try:
            img, det_conf = detect_and_crop_image(raw_img, conf=0.1)
        except ValueError as e:
            print(f"  {name}: YOLO crop failed ({e}), skipping")
            continue

        points = run_minutiaenet(img)
        overlay = draw_overlay(img, points)

        rig = sum(1 for p in points if p[3] == "RIG")
        bif = sum(1 for p in points if p[3] == "BIF")
        cv2.putText(overlay, f"MinutiaeNet: {len(points)} ({rig} RIG, {bif} BIF)",
                    (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)

        # Upscale small crops so the overlay is actually viewable.
        h, w = overlay.shape[:2]
        if max(h, w) < 500:
            scale = 500 / max(h, w)
            overlay = cv2.resize(overlay, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_CUBIC)

        out_path = os.path.join(OUT_DIR, f"{name}_minutiaenet.png")
        cv2.imwrite(out_path, overlay)
        print(f"  {name}: YOLO crop {raw_img.shape[:2]} -> {img.shape[:2]} (det_conf={det_conf:.3f}), "
              f"{len(points)} minutiae ({rig} RIG, {bif} BIF) -> {out_path}")


if __name__ == "__main__":
    main()
