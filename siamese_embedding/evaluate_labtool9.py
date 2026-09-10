"""
Runs ResNet50 (embed_resnet50.py) against the 9-person dataset originally used
in the Streamlit lab bench (backend/labtool/dataset/, cross-referenced against
backend/lab_gallery.db's subjects/templates tables to confirm identity).

IMPORTANT LIMITATION: only one photo per person survives on disk (the enrolled
gallery image). The 57 probe attempts logged in lab_gallery.db's `probes` table
during the original testing session only stored a SHA-256 hash of each probe
photo, not the photo itself -- so no genuine (same-person, different-photo)
pair can be reconstructed. This script computes IMPOSTOR pairs only (every
person's photo vs. every other person's photo) and reports FAR across a
threshold sweep. It does NOT and CANNOT report FRR, EER, or an overall
"accuracy" number -- those require genuine pairs, which don't exist for this
dataset anymore. Report FAR honestly as a partial result, not the full picture.
"""
import os
import glob

import numpy as np

from embed_resnet50 import build_embedder, preprocess_image
import torch

DATASET_DIR = r"D:\UIDAI\backend\labtool\dataset"

# From lab_gallery.db's subjects table.
NAMES = {
    "UID001": "Aftab", "UID002": "Aslam", "UID003": "Ishak", "UID004": "Maa",
    "UID005": "Majju", "UID006": "Mujju", "UID007": "Salman", "UID008": "Sameer",
    "UID009": "Shoaib",
}

THRESHOLDS = np.round(np.arange(0.80, 1.0001, 0.01), 3)


def cosine_similarity(a, b):
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    return float(np.dot(a, b) / denom) if denom else 0.0


def main():
    paths = sorted(glob.glob(os.path.join(DATASET_DIR, "*.jpeg")))
    print(f"[evaluate_labtool9] found {len(paths)} images")
    for p in paths:
        uid = os.path.basename(p).split("_")[0]
        print(f"  {uid} ({NAMES.get(uid, '?')}): {os.path.basename(p)}")

    print("\n[evaluate_labtool9] loading ResNet50...")
    model = build_embedder()

    embeddings = {}
    for p in paths:
        tensor = preprocess_image(p).unsqueeze(0)
        with torch.no_grad():
            vec = model(tensor).cpu().numpy()[0]
        uid = os.path.basename(p).split("_")[0]
        embeddings[uid] = vec
    print(f"[evaluate_labtool9] computed {len(embeddings)} embeddings")

    uids = list(embeddings.keys())
    impostor_scores = []
    impostor_pairs = []
    for i in range(len(uids)):
        for j in range(i + 1, len(uids)):
            s = cosine_similarity(embeddings[uids[i]], embeddings[uids[j]])
            impostor_scores.append(s)
            impostor_pairs.append((uids[i], uids[j], s))

    impostor_scores = np.array(impostor_scores)
    print(f"\n[evaluate_labtool9] {len(impostor_scores)} impostor pairs "
          f"(no genuine pairs available -- see module docstring)")
    print(f"[evaluate_labtool9] impostor score mean={impostor_scores.mean():.4f} "
          f"median={np.median(impostor_scores):.4f} std={impostor_scores.std():.4f} "
          f"min={impostor_scores.min():.4f} max={impostor_scores.max():.4f}")

    print("\n[evaluate_labtool9] all impostor pairs, sorted by score (highest = most confusable):")
    for a, b, s in sorted(impostor_pairs, key=lambda x: -x[2]):
        print(f"  {a} ({NAMES[a]:8s}) vs {b} ({NAMES[b]:8s}): {s:.4f}")

    print("\n[evaluate_labtool9] FAR across threshold sweep (fraction of impostor "
          "pairs that would be WRONGLY accepted as a match at that threshold):")
    print(f"{'threshold':>10s}  {'FAR':>8s}  {'false_accepts':>14s}")
    for t in THRESHOLDS:
        false_accepts = int(np.sum(impostor_scores >= t))
        far = false_accepts / len(impostor_scores)
        print(f"{t:10.3f}  {far:8.4f}  {false_accepts:14d}/{len(impostor_scores)}")

    print("\n[evaluate_labtool9] NOTE: FRR/EER/accuracy are NOT computable -- no "
          "genuine pairs exist for this dataset (original probe photos were "
          "never saved, only their SHA-256 hash was logged).")


if __name__ == "__main__":
    main()
