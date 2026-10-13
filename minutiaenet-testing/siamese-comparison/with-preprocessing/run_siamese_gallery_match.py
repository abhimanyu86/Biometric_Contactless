"""
30-identity impostor-only gallery match: same YOLO crop -> U2Net segmentation
-> Zero-DCE/hist-eq preprocessing as run_gallery_match.py, but the enrollment
gallery here is image_db/Person1..Person30 ONLY (no Rahul) -- so every one of
the 30 x 106 = 3180 comparisons against Rahul's test_db probes is, by
definition, an impostor pair. There is no genuine pair possible in this
setup (Rahul isn't enrolled), so this reports a pure false-accept-rate table,
not accuracy/FAR/FRR/EER.

The final matching step swaps MinutiaeNet extraction + matching.py's graph
matcher for the trained Siamese ResNet18 (siamese_embedding/model_v2.py +
checkpoints/exp2_best_model.pth) scored by cosine similarity between its
512-D embeddings.

The preprocessed fingerprint image (a single-channel uint8 array, same
object that would otherwise go into MinutiaeNet) is fed straight into the
Siamese net's own eval transform (PIL grayscale -> resize 224x224 ->
Normalize(mean=0.5, std=0.5)) instead of a re-read from disk, since we
already have it in memory from preprocess_fingerprint.
"""
import csv
import functools
import os
import sys
import time

import numpy as np

print = functools.partial(print, flush=True)

TESTING_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, TESTING_ROOT)
import common  # noqa: E402

BACKEND_DIR = os.path.join(os.path.dirname(TESTING_ROOT), "backend")
SIAMESE_DIR = os.path.join(os.path.dirname(TESTING_ROOT), "siamese_embedding")
sys.path.insert(0, SIAMESE_DIR)

import cv2  # noqa: E402
from PIL import Image  # noqa: E402
import torch  # noqa: E402
import app as app_module  # noqa: E402
from embed_v2 import build_embedder, _eval_transform  # noqa: E402

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "gallery_results")
os.makedirs(OUT_DIR, exist_ok=True)

DEVICE = "cpu"
THRESHOLDS = common.THRESHOLDS


def preprocess_to_fingerprint_image(image_path):
    """YOLO crop -> U2Net mask -> Zero-DCE/hist-eq. Returns a single-channel
    uint8 array (same object app.py's own pipeline would hand to MinutiaeNet)
    or None with a status string on failure."""
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
    return preprocessed, "ok"


def embed_fingerprint_image(model, preprocessed):
    """preprocessed: single-channel uint8 array -> 512-D L2-normalized embedding."""
    pil_img = Image.fromarray(preprocessed).convert("L")
    tensor = _eval_transform(pil_img).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        vec = model.forward_once(tensor).cpu().numpy()[0]
    return vec


def build_embeddings(entries, model, label):
    embeddings = {}
    statuses = []
    for i, (identity, path) in enumerate(entries, 1):
        t0 = time.time()
        preprocessed, status = preprocess_to_fingerprint_image(path)
        if preprocessed is None:
            embeddings[identity] = None
            statuses.append({"identity": identity, "image": os.path.basename(path), "status": status, "minutiae_count": ""})
            print(f"[{label} {i}/{len(entries)}] {identity}: FAILED ({status}) ({time.time()-t0:.2f}s)")
            continue
        vec = embed_fingerprint_image(model, preprocessed)
        embeddings[identity] = vec
        statuses.append({"identity": identity, "image": os.path.basename(path), "status": "ok", "minutiae_count": ""})
        print(f"[{label} {i}/{len(entries)}] {identity}: embedded ({time.time()-t0:.2f}s)")
    return embeddings, statuses


def main():
    print("Loading YOLO/U2Net/Zero-DCE (app.py)...")
    app_module.load_models()

    print("Loading Siamese ResNet18 (exp2_best_model.pth)...")
    model = build_embedder(device=DEVICE)

    gallery_entries = common.list_gallery_images(include_rahul=False)
    print(f"Gallery identities (Rahul excluded): {len(gallery_entries)}")
    gallery_embeddings, gallery_status = build_embeddings(gallery_entries, model, "gallery")

    probe_paths = common.list_probe_images()
    probe_entries = [(os.path.basename(p), p) for p in probe_paths]
    print(f"Probes: {len(probe_entries)}")
    probe_embeddings, probe_status = build_embeddings(probe_entries, model, "probe")

    common.write_per_image_csv(
        os.path.join(OUT_DIR, "gallery_enrollment_status.csv"),
        [{"image": r["image"], "status": r["status"], "score": "", "minutiae_count": ""} for r in gallery_status],
    )
    common.write_per_image_csv(
        os.path.join(OUT_DIR, "probe_extraction_status.csv"),
        [{"image": r["image"], "status": r["status"], "score": "", "minutiae_count": ""} for r in probe_status],
    )

    matrix_rows = []
    impostor_scores = []

    for identity, _path in gallery_entries:
        g_vec = gallery_embeddings[identity]
        for probe_name, _probe_path in probe_entries:
            p_vec = probe_embeddings[probe_name]
            if g_vec is None or p_vec is None:
                matrix_rows.append({"identity": identity, "probe": probe_name, "score": "", "pair_type": ""})
                continue
            score = float(np.dot(g_vec, p_vec) / (np.linalg.norm(g_vec) * np.linalg.norm(p_vec)))
            matrix_rows.append({"identity": identity, "probe": probe_name, "score": round(score, 4), "pair_type": "impostor"})
            impostor_scores.append(score)

    with open(os.path.join(OUT_DIR, "score_matrix.csv"), "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["identity", "probe", "score", "pair_type"])
        writer.writeheader()
        writer.writerows(matrix_rows)

    print(f"\nTotal comparisons (30 identities x {len(probe_entries)} probes): {len(matrix_rows)}")
    print(f"All pairs are impostor pairs -- Rahul is not enrolled, so no genuine pair exists in this run.")
    if impostor_scores:
        print(f"Cosine similarity range: min={min(impostor_scores):.4f} max={max(impostor_scores):.4f} "
              f"mean={np.mean(impostor_scores):.4f} median={np.median(impostor_scores):.4f}")

    rows = []
    with open(os.path.join(OUT_DIR, "false_accept_rate_by_threshold.csv"), "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["threshold", "wrongly_accepted", "total", "false_accept_rate"])
        writer.writeheader()
        for t in THRESHOLDS:
            accepted = sum(1 for s in impostor_scores if s >= t)
            row = {"threshold": t, "wrongly_accepted": accepted, "total": len(impostor_scores),
                   "false_accept_rate": accepted / len(impostor_scores) if impostor_scores else float("nan")}
            rows.append(row)
            writer.writerow(row)

    print("\n=== False-accept rate by threshold (30-identity gallery, no Rahul enrolled) ===")
    print(f"{'threshold':>9} | {'wrongly accepted':>17} | {'total':>6} | {'false-accept rate':>18}")
    for row in rows:
        print(f"{row['threshold']:>9.1f} | {row['wrongly_accepted']:>17} | {row['total']:>6} | {row['false_accept_rate']:>17.2%}")


if __name__ == "__main__":
    main()
