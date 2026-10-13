"""
Focused diagnostic: does YOLO-cropping to just the finger (removing the
background/lighting bleed-through found by visually inspecting the 33
rejected images) fix Rahul's genuine-match rate against his own gallery
photo?

embed_resnet50.py's preprocess_image() currently does zero cropping -- raw
photo straight to 224x224 -- so backgrounds (patterned cloth, walls, warm
lamp lighting) get embedded as if they were fingerprint signal. This script
re-embeds Rahul's gallery + all 106 probes AFTER running them through the
same detect_and_crop_image() YOLO step used everywhere else in this
project, and compares the new genuine scores against the old (uncropped)
ones and the old EER threshold, pair by pair.

Scope: Rahul's own genuine scores only (this is a before/after check, not
a new full-dataset EER/FAR/FRR -- that would need all 31 people re-cropped
and the full pipeline rerun).
"""
import json
import os
import sys

import cv2
import numpy as np
import torch

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
BACKEND_DIR = os.path.join(os.path.dirname(BASE_DIR), "backend")
sys.path.insert(0, BACKEND_DIR)
sys.path.insert(0, BASE_DIR)

from app import detect_and_crop_image  # noqa: E402
from embed_resnet50 import build_embedder, INPUT_SIZE, _IMAGENET_MEAN, _IMAGENET_STD  # noqa: E402

GALLERY_PATH = r"D:\Contactless\image_db\Rahul.jpg"
PROBE_DIR = r"D:\Contactless\test_db\Rahul"
OLD_CACHE = os.path.join(BASE_DIR, "embeddings_cache_dynamic_resnet50.json")
OLD_EER_THRESHOLD = 0.5893110036849976
OUT_CSV = os.path.join(BASE_DIR, "rahul_cropped_vs_uncropped.csv")


def preprocess_cropped_bgr(cropped_bgr):
    """Same normalization as embed_resnet50.preprocess_image, but starting
    from an already-cropped in-memory BGR array instead of reading a path."""
    gray = cv2.cvtColor(cropped_bgr, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (INPUT_SIZE, INPUT_SIZE), interpolation=cv2.INTER_AREA)
    rgb = np.stack([resized] * 3, axis=-1).astype(np.float32) / 255.0
    normalized = (rgb - _IMAGENET_MEAN) / _IMAGENET_STD
    chw = np.transpose(normalized, (2, 0, 1))
    return torch.from_numpy(chw).float()


def embed_cropped(model, path, device):
    raw = cv2.imread(path)
    if raw is None:
        return None, "could not read image"
    try:
        cropped, det_conf = detect_and_crop_image(raw, conf=0.1)
    except ValueError as e:
        return None, f"YOLO crop failed: {e}"
    tensor = preprocess_cropped_bgr(cropped).unsqueeze(0).to(device)
    with torch.no_grad():
        vec = model(tensor).cpu().numpy()[0]
    return vec, f"det_conf={det_conf:.3f}"


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = build_embedder(device)

    with open(OLD_CACHE, encoding="utf-8") as f:
        old_raw = json.load(f)
    old_gallery_vec = np.array(old_raw[GALLERY_PATH], dtype=np.float32)
    old_gallery_vec = old_gallery_vec / np.linalg.norm(old_gallery_vec)

    print("Cropping + embedding gallery image (Rahul.jpg)...")
    new_gallery_vec, gallery_note = embed_cropped(model, GALLERY_PATH, device)
    if new_gallery_vec is None:
        print(f"FATAL: could not crop gallery image ({gallery_note}); aborting.")
        return
    print(f"  gallery: {gallery_note}")
    new_gallery_vec = new_gallery_vec / np.linalg.norm(new_gallery_vec)

    rows = []
    crop_failures = []
    for i in range(1, 107):
        path = os.path.join(PROBE_DIR, f"{i}.jpg")

        old_vec = np.array(old_raw[path], dtype=np.float32)
        old_vec = old_vec / np.linalg.norm(old_vec)
        old_score = float(np.dot(old_vec, old_gallery_vec))
        old_matched = old_score >= OLD_EER_THRESHOLD

        new_vec, note = embed_cropped(model, path, device)
        if new_vec is None:
            crop_failures.append((i, note))
            rows.append({
                "image": f"{i}.jpg", "old_score": round(old_score, 4), "old_matched": old_matched,
                "new_score": "", "new_matched_at_old_threshold": "", "note": note,
            })
            continue

        new_vec = new_vec / np.linalg.norm(new_vec)
        new_score = float(np.dot(new_vec, new_gallery_vec))
        new_matched = new_score >= OLD_EER_THRESHOLD

        rows.append({
            "image": f"{i}.jpg", "old_score": round(old_score, 4), "old_matched": old_matched,
            "new_score": round(new_score, 4), "new_matched_at_old_threshold": new_matched, "note": note,
        })

        if (i - 1) % 20 == 0:
            print(f"  processed {i}/106")

    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        import csv
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    scored_rows = [r for r in rows if r["new_score"] != ""]
    old_matched_count = sum(1 for r in rows if r["old_matched"])
    new_matched_count = sum(1 for r in scored_rows if r["new_matched_at_old_threshold"])

    flipped_to_match = [r for r in scored_rows if not r["old_matched"] and r["new_matched_at_old_threshold"]]
    flipped_to_reject = [r for r in scored_rows if r["old_matched"] and not r["new_matched_at_old_threshold"]]
    still_rejected = [r for r in scored_rows if not r["old_matched"] and not r["new_matched_at_old_threshold"]]

    print(f"\n=== RESULTS (compared at the OLD EER threshold={OLD_EER_THRESHOLD:.4f} for a fair before/after) ===")
    print(f"Old (uncropped): {old_matched_count}/106 matched")
    print(f"New (YOLO-cropped): {new_matched_count}/{len(scored_rows)} matched"
          f" ({len(crop_failures)} images failed YOLO detection, excluded)")
    print(f"\nFlipped REJECT -> MATCH after cropping: {len(flipped_to_match)}")
    for r in flipped_to_match:
        print(f"  {r['image']}: {r['old_score']} -> {r['new_score']}")
    print(f"\nFlipped MATCH -> REJECT after cropping: {len(flipped_to_reject)}")
    for r in flipped_to_reject:
        print(f"  {r['image']}: {r['old_score']} -> {r['new_score']}")
    print(f"\nStill REJECTED after cropping: {len(still_rejected)}")
    for r in still_rejected:
        print(f"  {r['image']}: {r['old_score']} -> {r['new_score']}")

    if crop_failures:
        print(f"\nYOLO crop FAILED on {len(crop_failures)} images (no finger detected, excluded from comparison):")
        for i, note in crop_failures:
            print(f"  {i}.jpg: {note}")

    print(f"\nFull results written to {OUT_CSV}")


if __name__ == "__main__":
    main()
