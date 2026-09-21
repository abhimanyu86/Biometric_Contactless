"""
Phase 1: YOLO crop -> MinutiaeNet, no U2Net segmentation and no Zero-DCE/
histogram-equalization enhancement in between. Enrolls image_db/Rahul.jpg
once, matches it against every photo in test_db/Rahul/, and reports
genuine-match accuracy at each threshold in common.THRESHOLDS.

Deliberately avoids app.py's load_models()/global_models path (which pulls in
U2Net, Zero-DCE, liveness and brightspot) -- this phase only needs the YOLO
detector (app.py.get_finger_detector, lazy-loaded) and MinutiaeNet
(minutiae.py's standalone loader), so it stays cheaper to run than phase 2.
"""
import functools
import os
import sys

print = functools.partial(print, flush=True)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import common  # noqa: E402

from app import detect_and_crop_image  # noqa: E402
from minutiae import detect_minutiae  # noqa: E402
from matching import match_templates  # noqa: E402

OUT_DIR = os.path.dirname(os.path.abspath(__file__))


def extract_template(image_path):
    img = common.load_image(image_path)
    if img is None:
        return None, "unreadable"
    try:
        cropped, _conf = detect_and_crop_image(img)
    except Exception:
        return None, "no_finger_detected"
    minutiae = detect_minutiae(cropped)
    if not minutiae:
        return None, "no_minutiae"
    return minutiae, "ok"


def main():
    print(f"Enrolling: {common.ENROLL_IMAGE}")
    enroll_template, status = extract_template(common.ENROLL_IMAGE)
    if enroll_template is None:
        raise RuntimeError(f"Could not build enrollment template: {status}")
    print(f"Enrollment minutiae: {len(enroll_template)}")

    probe_paths = common.list_probe_images()
    print(f"Probes found: {len(probe_paths)}")

    records = []
    genuine_scores = []
    failures = 0

    for path in probe_paths:
        name = os.path.basename(path)
        template, status = extract_template(path)
        if template is None:
            failures += 1
            records.append({"image": name, "status": status, "score": "", "minutiae_count": ""})
            continue
        score = match_templates(enroll_template, template)
        genuine_scores.append(score)
        records.append(
            {"image": name, "status": "ok", "score": round(score, 4), "minutiae_count": len(template)}
        )
        print(f"{name}: score={score:.4f} minutiae={len(template)}")

    rows = common.sweep_accuracy(genuine_scores)
    common.write_per_image_csv(os.path.join(OUT_DIR, "results_per_image.csv"), records)
    common.write_summary_csv(os.path.join(OUT_DIR, "results_summary.csv"), rows)
    common.print_summary("Without preprocessing", rows, failures, len(probe_paths))


if __name__ == "__main__":
    main()
