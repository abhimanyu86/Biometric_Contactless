"""
Runs the with-preprocessing pipeline (YOLO crop -> U2Net -> Zero-DCE/hist-eq
-> MinutiaeNet) on a handful of test_db/Rahul probes and saves labelled
minutiae overlays via app.py's own create_visualization (green=RIG ridge
ending, yellow=BIF bifurcation, arrow=ridge direction).
"""
import functools
import os
import sys

print = functools.partial(print, flush=True)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import common  # noqa: E402

import cv2  # noqa: E402
import app as app_module  # noqa: E402

N = int(sys.argv[1]) if len(sys.argv) > 1 else 10
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "minutiae_visualizations")
os.makedirs(OUT_DIR, exist_ok=True)


def main():
    print("Loading models...")
    app_module.load_models()

    probe_paths = common.list_probe_images()[:N]
    for path in probe_paths:
        name = os.path.basename(path)
        img = common.load_image(path)
        cropped, _conf = app_module.detect_and_crop_image(img)
        cropped = common.cap_crop(cropped)
        mask = app_module.get_segmentation_mask(cropped)
        preprocessed = app_module.preprocess_fingerprint(cropped, mask=mask)
        minutiae = app_module.detect_minutiae(preprocessed)
        vis = app_module.create_visualization(preprocessed, minutiae)
        out_path = os.path.join(OUT_DIR, name)
        cv2.imwrite(out_path, vis)
        print(f"{name}: minutiae={len(minutiae)} -> {out_path}")


if __name__ == "__main__":
    main()
