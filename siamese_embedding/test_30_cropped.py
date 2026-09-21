"""
Same treatment as test_rahul_cropped.py, applied to the original 30 people
(image_db/PersonN.* + test_db/PersonN/image1.*,image2.*) instead of Rahul.

For each person: YOLO-crop the gallery photo AND both probe photos, SAVE
every cropped image to disk (image_db_cropped/, test_db_cropped/PersonN/),
re-embed the cropped versions with the same ResNet50 pipeline, and compare
the new genuine score against the old (uncropped) score at the SAME old
EER threshold used for Rahul's comparison -- same MATCH/REJECT flip report,
same low-confidence-crop flagging.
"""
import csv
import glob
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

IMAGE_DB = r"D:\UIDAI\image_db"
TEST_DB = r"D:\UIDAI\test_db"
IMAGE_DB_CROPPED = r"D:\UIDAI\image_db_cropped"
TEST_DB_CROPPED = r"D:\UIDAI\test_db_cropped"
OLD_CACHE = os.path.join(BASE_DIR, "embeddings_cache_dynamic_resnet50.json")
OLD_EER_THRESHOLD = 0.5893110036849976
OUT_CSV = os.path.join(BASE_DIR, "30people_cropped_vs_uncropped.csv")
LOW_CONF_THRESHOLD = 0.5

os.makedirs(IMAGE_DB_CROPPED, exist_ok=True)


def find_gallery_path(person_num):
    for ext in (".jpeg", ".jpg", ".png"):
        p = os.path.join(IMAGE_DB, f"Person{person_num}{ext}")
        if os.path.isfile(p):
            return p
    return None


def find_probe_paths(person_num):
    d = os.path.join(TEST_DB, f"Person{person_num}")
    if not os.path.isdir(d):
        return []
    return sorted(glob.glob(os.path.join(d, "image*.*")))


def preprocess_cropped_bgr(cropped_bgr):
    gray = cv2.cvtColor(cropped_bgr, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (INPUT_SIZE, INPUT_SIZE), interpolation=cv2.INTER_AREA)
    rgb = np.stack([resized] * 3, axis=-1).astype(np.float32) / 255.0
    normalized = (rgb - _IMAGENET_MEAN) / _IMAGENET_STD
    chw = np.transpose(normalized, (2, 0, 1))
    return torch.from_numpy(chw).float()


def crop_embed_and_save(model, src_path, save_path, device):
    raw = cv2.imread(src_path)
    if raw is None:
        return None, None, "could not read image"
    try:
        cropped, det_conf = detect_and_crop_image(raw, conf=0.1)
    except ValueError as e:
        return None, None, f"YOLO crop failed: {e}"
    cv2.imwrite(save_path, cropped)
    tensor = preprocess_cropped_bgr(cropped).unsqueeze(0).to(device)
    with torch.no_grad():
        vec = model(tensor).cpu().numpy()[0]
    return vec, det_conf, "ok"


def find_old_score(old_raw, old_gallery_vec, path):
    vec = np.array(old_raw[path], dtype=np.float32)
    vec = vec / np.linalg.norm(vec)
    return float(np.dot(vec, old_gallery_vec))


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = build_embedder(device)

    with open(OLD_CACHE, encoding="utf-8") as f:
        old_raw = json.load(f)

    rows = []
    low_conf_crops = []
    crop_failures = []

    for person_num in range(1, 31):
        gallery_src = find_gallery_path(person_num)
        probe_srcs = find_probe_paths(person_num)
        if gallery_src is None or not probe_srcs:
            print(f"Person{person_num}: WARNING missing gallery or probes, skipping")
            continue

        old_gallery_vec_raw = np.array(old_raw[os.path.abspath(gallery_src)], dtype=np.float32)
        old_gallery_vec = old_gallery_vec_raw / np.linalg.norm(old_gallery_vec_raw)

        gallery_save = os.path.join(IMAGE_DB_CROPPED, f"Person{person_num}.jpg")
        new_gallery_vec, gallery_conf, gallery_note = crop_embed_and_save(model, gallery_src, gallery_save, device)
        if new_gallery_vec is None:
            print(f"Person{person_num}: FATAL gallery crop failed ({gallery_note}), skipping person")
            crop_failures.append((f"Person{person_num} (gallery)", gallery_note))
            continue
        if gallery_conf < LOW_CONF_THRESHOLD:
            low_conf_crops.append((f"Person{person_num} gallery", gallery_conf))
        new_gallery_vec = new_gallery_vec / np.linalg.norm(new_gallery_vec)

        person_probe_dir = os.path.join(TEST_DB_CROPPED, f"Person{person_num}")
        os.makedirs(person_probe_dir, exist_ok=True)

        for probe_src in probe_srcs:
            probe_name = os.path.splitext(os.path.basename(probe_src))[0]
            probe_save = os.path.join(person_probe_dir, f"{probe_name}.jpg")

            old_score = find_old_score(old_raw, old_gallery_vec, os.path.abspath(probe_src))
            old_matched = old_score >= OLD_EER_THRESHOLD

            new_vec, det_conf, note = crop_embed_and_save(model, probe_src, probe_save, device)
            if new_vec is None:
                crop_failures.append((f"Person{person_num}/{probe_name}", note))
                rows.append({
                    "person": f"Person{person_num}", "probe": probe_name,
                    "old_score": round(old_score, 4), "old_matched": old_matched,
                    "new_score": "", "new_matched_at_old_threshold": "", "det_conf": "", "note": note,
                })
                continue

            if det_conf < LOW_CONF_THRESHOLD:
                low_conf_crops.append((f"Person{person_num}/{probe_name}", det_conf))

            new_vec = new_vec / np.linalg.norm(new_vec)
            new_score = float(np.dot(new_vec, new_gallery_vec))
            new_matched = new_score >= OLD_EER_THRESHOLD

            rows.append({
                "person": f"Person{person_num}", "probe": probe_name,
                "old_score": round(old_score, 4), "old_matched": old_matched,
                "new_score": round(new_score, 4), "new_matched_at_old_threshold": new_matched,
                "det_conf": round(det_conf, 3), "note": "ok",
            })

        print(f"Person{person_num}: done")

    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    scored_rows = [r for r in rows if r["new_score"] != ""]
    old_matched_count = sum(1 for r in rows if r["old_matched"])
    new_matched_count = sum(1 for r in scored_rows if r["new_matched_at_old_threshold"])

    flipped_to_match = [r for r in scored_rows if not r["old_matched"] and r["new_matched_at_old_threshold"]]
    flipped_to_reject = [r for r in scored_rows if r["old_matched"] and not r["new_matched_at_old_threshold"]]
    still_rejected = [r for r in scored_rows if not r["old_matched"] and not r["new_matched_at_old_threshold"]]

    print(f"\n=== RESULTS (30 people, {len(rows)} probe pairs, at OLD EER threshold={OLD_EER_THRESHOLD:.4f}) ===")
    print(f"Old (uncropped): {old_matched_count}/{len(rows)} matched")
    print(f"New (YOLO-cropped): {new_matched_count}/{len(scored_rows)} matched"
          f" ({len(crop_failures)} crop failures excluded)")

    print(f"\nFlipped REJECT -> MATCH after cropping: {len(flipped_to_match)}")
    for r in flipped_to_match:
        print(f"  {r['person']}/{r['probe']}: {r['old_score']} -> {r['new_score']}")

    print(f"\nFlipped MATCH -> REJECT after cropping: {len(flipped_to_reject)}")
    for r in flipped_to_reject:
        print(f"  {r['person']}/{r['probe']}: {r['old_score']} -> {r['new_score']}")

    print(f"\nStill REJECTED after cropping: {len(still_rejected)}")
    for r in still_rejected:
        print(f"  {r['person']}/{r['probe']}: {r['old_score']} -> {r['new_score']}")

    if low_conf_crops:
        print(f"\nLOW-CONFIDENCE crops (det_conf < {LOW_CONF_THRESHOLD}), verify visually:")
        for name, conf in low_conf_crops:
            print(f"  {name}: det_conf={conf:.3f}")

    if crop_failures:
        print(f"\nYOLO crop FAILURES:")
        for name, note in crop_failures:
            print(f"  {name}: {note}")

    print(f"\nFull results written to {OUT_CSV}")
    print(f"Cropped galleries saved to {IMAGE_DB_CROPPED}")
    print(f"Cropped probes saved to {TEST_DB_CROPPED}")


if __name__ == "__main__":
    main()
