"""
31-identity gallery match, with-preprocessing pipeline (YOLO crop -> U2Net
segmentation -> Zero-DCE/hist-eq enhancement -> MinutiaeNet).

Enrollment gallery: image_db/Person1..Person30 + image_db/Rahul.jpg (31
identities, one enrollment photo each).
Probes: the same 106 photos in test_db/Rahul/ used by the single-enrollment
scripts -- Rahul is the only identity with a probe set in test_db.

Every one of the 31 enrolled templates is matched against every one of the
106 probes (31 x 106 = 3286 comparisons). Since only the "Rahul" row is the
same identity as the probes, this splits into:
  - genuine pairs (106): Rahul-enrollment vs each Rahul probe
  - impostor pairs (3180): Person1..30-enrollment vs each Rahul probe
which is what makes FAR/FRR/EER possible here, unlike the single-enrollment
run_with_preprocessing.py script (genuine-only, no impostor data).

Does not touch run_with_preprocessing.py or its results -- separate script,
separate output folder (gallery_results/), so the original single-enrollment
numbers stay intact for reference.
"""
import csv
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

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "gallery_results")
os.makedirs(OUT_DIR, exist_ok=True)


def extract_template(image_path):
    img = common.load_image(image_path)
    if img is None:
        return None, "unreadable"
    try:
        cropped, _conf = app_module.detect_and_crop_image(img)
        cropped = common.cap_crop(cropped)
    except Exception:
        return None, "no_finger_detected"
    try:
        mask = app_module.get_segmentation_mask(cropped)
        if cv2.countNonZero(mask) == 0:
            return None, "no_segmentation"
        preprocessed = app_module.preprocess_fingerprint(cropped, mask=mask)
    except Exception:
        return None, "preprocessing_failed"
    minutiae = app_module.detect_minutiae(preprocessed)
    if not minutiae:
        return None, "no_minutiae"
    return minutiae, "ok"


def build_templates(entries, label):
    templates = {}
    statuses = []
    for i, (identity, path) in enumerate(entries, 1):
        t0 = time.time()
        template, status = extract_template(path)
        templates[identity] = template
        statuses.append({"identity": identity, "image": os.path.basename(path), "status": status,
                          "minutiae_count": len(template) if template else ""})
        tag = f"minutiae={len(template)}" if template else f"FAILED ({status})"
        print(f"[{label} {i}/{len(entries)}] {identity}: {tag} ({time.time()-t0:.2f}s)")
    return templates, statuses


def main():
    print("Loading models (YOLO, U2Net, Zero-DCE, MinutiaeNet)...")
    app_module.load_models()

    gallery_entries = common.list_gallery_images()
    print(f"Gallery identities: {len(gallery_entries)}")
    gallery_templates, gallery_status = build_templates(gallery_entries, "gallery")

    probe_paths = common.list_probe_images()
    print(f"Probes: {len(probe_paths)}")
    probe_entries = [(os.path.basename(p), p) for p in probe_paths]
    probe_templates, probe_status = build_templates(probe_entries, "probe")

    common.write_per_image_csv(
        os.path.join(OUT_DIR, "gallery_enrollment_status.csv"),
        [{"image": r["image"], "status": r["status"], "score": "", "minutiae_count": r["minutiae_count"]} for r in gallery_status],
    )
    common.write_per_image_csv(
        os.path.join(OUT_DIR, "probe_extraction_status.csv"),
        [{"image": r["image"], "status": r["status"], "score": "", "minutiae_count": r["minutiae_count"]} for r in probe_status],
    )

    matrix_rows = []
    genuine_scores = []
    impostor_scores = []

    for identity, _path in gallery_entries:
        g_template = gallery_templates[identity]
        for probe_name, _probe_path in probe_entries:
            p_template = probe_templates[probe_name]
            if g_template is None or p_template is None:
                matrix_rows.append({"identity": identity, "probe": probe_name, "score": "", "pair_type": ""})
                continue
            score = match_templates(g_template, p_template)
            pair_type = "genuine" if identity == "Rahul" else "impostor"
            matrix_rows.append({"identity": identity, "probe": probe_name, "score": round(score, 4), "pair_type": pair_type})
            (genuine_scores if pair_type == "genuine" else impostor_scores).append(score)

    with open(os.path.join(OUT_DIR, "score_matrix.csv"), "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["identity", "probe", "score", "pair_type"])
        writer.writeheader()
        writer.writerows(matrix_rows)

    print(f"\nTotal comparisons: {len(matrix_rows)}")
    print(f"Genuine pairs: {len(genuine_scores)}  Impostor pairs: {len(impostor_scores)}")

    # -- accuracy-at-threshold (genuine-only, same definition as run_with_preprocessing.py) --
    accuracy_rows = common.sweep_accuracy(genuine_scores)
    common.write_summary_csv(os.path.join(OUT_DIR, "accuracy_by_threshold.csv"), accuracy_rows)

    print("\n=== Accuracy at threshold (genuine pairs only) ===")
    print(f"{'threshold':>9} | {'matched':>7} | {'total':>5} | {'accuracy':>8}")
    best = None
    for row in accuracy_rows:
        print(f"{row['threshold']:>9.1f} | {row['matched']:>7} | {row['total']:>5} | {row['accuracy']:>8.2%}")
        if best is None or row["accuracy"] > best["accuracy"]:
            best = row
    print(f"Best accuracy: {best['accuracy']:.2%} at threshold {best['threshold']:.1f}")

    # -- FAR/FRR/EER (genuine vs impostor) --
    far_frr_rows, eer_row = common.compute_far_frr(genuine_scores, impostor_scores)
    with open(os.path.join(OUT_DIR, "far_frr_eer.csv"), "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["threshold", "far", "frr", "false_accepts", "false_rejects",
                        "total_genuine", "total_impostor", "eer_threshold", "eer"],
        )
        writer.writeheader()
        for row in far_frr_rows:
            writer.writerow({**row, "eer_threshold": eer_row["threshold"] if eer_row else "",
                              "eer": eer_row["eer"] if eer_row else ""})

    print("\n=== FAR / FRR by threshold ===")
    print(f"{'threshold':>9} | {'FAR':>8} | {'FRR':>8}")
    for row in far_frr_rows:
        print(f"{row['threshold']:>9.1f} | {row['far']:>8.2%} | {row['frr']:>8.2%}")
    if eer_row:
        print(f"\nEER ~ {eer_row['eer']:.2%} at threshold {eer_row['threshold']:.1f} "
              f"(FAR={eer_row['far']:.2%}, FRR={eer_row['frr']:.2%})")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        plt.figure(figsize=(8, 5))
        thresholds = [r["threshold"] for r in far_frr_rows]
        plt.plot(thresholds, [r["far"] for r in far_frr_rows], marker="o", label="FAR")
        plt.plot(thresholds, [r["frr"] for r in far_frr_rows], marker="s", label="FRR")
        if eer_row:
            plt.scatter([eer_row["threshold"]], [eer_row["eer"]], color="red", zorder=3, label=f"EER ~ {eer_row['eer']:.2%}")
            plt.axvline(eer_row["threshold"], color="red", linestyle="--", alpha=0.4)
        plt.legend()
        plt.title("Gallery match FAR/FRR (with preprocessing)")
        plt.xlabel("Threshold")
        plt.ylabel("Rate")
        plt.ylim(0, 1)
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(OUT_DIR, "eer_plot.png"), dpi=160)
        plt.close()
        print(f"Wrote {os.path.join(OUT_DIR, 'eer_plot.png')}")
    except ImportError:
        print("matplotlib not available -- skipped eer_plot.png")


if __name__ == "__main__":
    main()
