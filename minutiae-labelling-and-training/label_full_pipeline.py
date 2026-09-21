"""
Full production preprocessing chain (backend/app.py, same order compute_readiness_score
uses) applied to the user-uploaded photos, then MinutiaeNet, then the app's own
create_visualization() overlay -- not a re-implementation, the actual pipeline functions:

    detect_and_crop_image (YOLO)
      -> get_segmentation_mask (U2-Net)
      -> preprocess_fingerprint (Zero-DCE/CLAHE + adaptive threshold, mask-gated)
      -> detect_minutiae (MinutiaeNet)
      -> create_visualization
"""
import glob
import os
import sys

import cv2

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
BACKEND_DIR = os.path.join(os.path.dirname(BASE_DIR), "backend")
sys.path.insert(0, BACKEND_DIR)

from app import (  # noqa: E402
    detect_and_crop_image,
    get_segmentation_mask,
    preprocess_fingerprint,
    detect_minutiae,
    create_visualization,
    global_models,
    load_models,
)

load_models()  # app.py only calls this under __main__; importing app as a module skips it

IN_DIR = os.path.join(BASE_DIR, "user_photos")
OUT_DIR = os.path.join(BASE_DIR, "pilot_output", "full_pipeline_labeled")
os.makedirs(OUT_DIR, exist_ok=True)


def main():
    print("Loaded models:", sorted(global_models.keys()))
    missing = [m for m in ("u2net", "zero_dce", "minutiae") if m not in global_models]
    if missing:
        print(f"WARNING: missing model(s) {missing} -- their stage will degrade silently per CLAUDE.md")

    paths = sorted(glob.glob(os.path.join(IN_DIR, "*.jpg")))
    print(f"\nFound {len(paths)} photos in {IN_DIR}")

    for path in paths:
        name = os.path.splitext(os.path.basename(path))[0]
        raw_img = cv2.imread(path)
        if raw_img is None:
            print(f"  {name}: could not read, skipping")
            continue

        try:
            cropped, det_conf = detect_and_crop_image(raw_img, conf=0.1)
        except ValueError as e:
            print(f"  {name}: YOLO crop FAILED ({e}), skipping")
            continue

        mask = get_segmentation_mask(cropped)
        fg_px = int(mask.sum())
        preprocessed = preprocess_fingerprint(cropped, mask=mask)
        minutiae = detect_minutiae(preprocessed)
        vis = create_visualization(preprocessed, minutiae)

        rig = sum(1 for m in minutiae if m["type"] == "RIG")
        bif = sum(1 for m in minutiae if m["type"] == "BIF")

        h, w = vis.shape[:2]
        if max(h, w) < 500:
            scale = 500 / max(h, w)
            vis = cv2.resize(vis, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_NEAREST)

        out_path = os.path.join(OUT_DIR, f"{name}_full_pipeline.png")
        cv2.imwrite(out_path, vis)
        print(f"  {name}: YOLO {raw_img.shape[:2]}->{cropped.shape[:2]} (det_conf={det_conf:.3f}), "
              f"mask_fg_px={fg_px}, {len(minutiae)} minutiae ({rig} RIG, {bif} BIF) -> {out_path}")


if __name__ == "__main__":
    main()
