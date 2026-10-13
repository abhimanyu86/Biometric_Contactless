"""
Phase 2: YOLO crop -> U2Net segmentation -> Zero-DCE/hist-eq enhancement
(app.py's preprocess_fingerprint) -> MinutiaeNet. Same enrollment/probe set
and same accuracy-by-threshold report as run_without_preprocessing.py, so the
two results_summary.csv files are directly comparable.

Uses app.py's own load_models() + detect_and_crop_image/get_segmentation_mask/
preprocess_fingerprint/detect_minutiae as one consistent set, all backed by
the same global_models dict app.py populates -- avoids mixing app.py's model
loading with minutiae.py's standalone loader.
"""
import functools
import os
import sys
import time

print = functools.partial(print, flush=True)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import common  # noqa: E402

import cv2  # noqa: E402
import app as app_module  # noqa: E402
from matching import match_templates  # noqa: E402

OUT_DIR = os.path.dirname(os.path.abspath(__file__))
PROCESSED_DIR = os.path.join(OUT_DIR, "processed_images")
os.makedirs(PROCESSED_DIR, exist_ok=True)


def extract_template(image_path, verbose=False):
    img = common.load_image(image_path)
    if img is None:
        return None, "unreadable", None

    t0 = time.time()
    try:
        cropped, _conf = app_module.detect_and_crop_image(img)
        cropped = common.cap_crop(cropped)
    except Exception:
        return None, "no_finger_detected", None
    t1 = time.time()

    try:
        mask = app_module.get_segmentation_mask(cropped)
        if cv2.countNonZero(mask) == 0:
            return None, "no_segmentation", None
        t2 = time.time()
        preprocessed = app_module.preprocess_fingerprint(cropped, mask=mask)
        t3 = time.time()
    except Exception:
        return None, "preprocessing_failed", None

    minutiae = app_module.detect_minutiae(preprocessed)
    t4 = time.time()

    if verbose:
        print(
            f"    crop_shape={cropped.shape} "
            f"crop={t1-t0:.2f}s seg={t2-t1:.2f}s enhance={t3-t2:.2f}s minutiae={t4-t3:.2f}s"
        )

    if not minutiae:
        return None, "no_minutiae", preprocessed
    return minutiae, "ok", preprocessed


def main():
    print("Loading models (YOLO, U2Net, Zero-DCE, MinutiaeNet)...")
    t0 = time.time()
    app_module.load_models()
    print(f"Models loaded in {time.time() - t0:.1f}s")

    print(f"Enrolling: {common.ENROLL_IMAGE}")
    enroll_template, status, enroll_preprocessed = extract_template(common.ENROLL_IMAGE, verbose=True)
    if enroll_template is None:
        raise RuntimeError(f"Could not build enrollment template: {status}")
    print(f"Enrollment minutiae: {len(enroll_template)}")
    cv2.imwrite(os.path.join(PROCESSED_DIR, os.path.basename(common.ENROLL_IMAGE)), enroll_preprocessed)

    probe_paths = common.list_probe_images()
    print(f"Probes found: {len(probe_paths)}")

    records = []
    genuine_scores = []
    failures = 0

    for i, path in enumerate(probe_paths, 1):
        name = os.path.basename(path)
        t_start = time.time()
        template, status, preprocessed = extract_template(path, verbose=(i <= 3))
        if preprocessed is not None:
            cv2.imwrite(os.path.join(PROCESSED_DIR, name), preprocessed)
        if template is None:
            failures += 1
            records.append({"image": name, "status": status, "score": "", "minutiae_count": ""})
            print(f"[{i}/{len(probe_paths)}] {name}: FAILED ({status}) in {time.time()-t_start:.2f}s")
            continue
        score = match_templates(enroll_template, template)
        genuine_scores.append(score)
        records.append(
            {"image": name, "status": "ok", "score": round(score, 4), "minutiae_count": len(template)}
        )
        print(f"[{i}/{len(probe_paths)}] {name}: score={score:.4f} minutiae={len(template)} ({time.time()-t_start:.2f}s)")

    rows = common.sweep_accuracy(genuine_scores)
    common.write_per_image_csv(os.path.join(OUT_DIR, "results_per_image.csv"), records)
    common.write_summary_csv(os.path.join(OUT_DIR, "results_summary.csv"), rows)
    common.print_summary("With preprocessing", rows, failures, len(probe_paths))


if __name__ == "__main__":
    main()
