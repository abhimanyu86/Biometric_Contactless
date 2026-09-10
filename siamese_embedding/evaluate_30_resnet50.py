"""
Full matching-accuracy/FAR/FRR/EER test using ResNet50 on the 30-person
image_db (gallery) + test_db (2 probes/person) dataset -- real, uncontrolled
phone-camera photos, unlike the HKPU lab dataset. Reuses evaluate.py's
compute_metrics/evaluate_embeddings so the numbers are produced by identical
scoring logic to the earlier 336-person and exp2 runs.
"""
import os
import json

import cv2
import numpy as np
import torch

from dataset_30 import load_people
from embed_resnet50 import build_embedder, preprocess_image
from evaluate import evaluate_embeddings

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CACHE_PATH = os.path.join(BASE_DIR, "embeddings_cache_30_resnet50.json")
CSV_PATH = os.path.join(BASE_DIR, "benchmark_results_30_resnet50.csv")
PLOT_PATH = os.path.join(BASE_DIR, "eer_plot_30_resnet50.png")


def compute_all_embeddings(people, device="cpu"):
    model = build_embedder(device=device)
    all_paths = []
    for info in people.values():
        all_paths.append(info["gallery"])
        all_paths.extend(info["probes"])
    all_paths = list(dict.fromkeys(all_paths))

    print(f"[evaluate_30_resnet50] computing embeddings for {len(all_paths)} images on {device}...")
    embeddings = {}
    for path in all_paths:
        try:
            tensor = preprocess_image(path).unsqueeze(0).to(device)
        except ValueError as exc:
            print(f"[evaluate_30_resnet50] WARNING: {exc}, skipping")
            continue
        with torch.no_grad():
            vec = model(tensor).cpu().numpy()[0]
        embeddings[path] = vec
    print(f"[evaluate_30_resnet50] done, {len(embeddings)} embeddings computed")
    return embeddings


def get_embeddings(force_recompute=False):
    if not force_recompute and os.path.isfile(CACHE_PATH):
        with open(CACHE_PATH, "r", encoding="utf-8") as handle:
            raw = json.load(handle)
        print(f"[evaluate_30_resnet50] loaded {len(raw)} embeddings from cache")
        return {path: np.array(vec, dtype=np.float32) for path, vec in raw.items()}

    people = load_people()
    embeddings = compute_all_embeddings(people)
    with open(CACHE_PATH, "w", encoding="utf-8") as handle:
        json.dump({path: vec.tolist() for path, vec in embeddings.items()}, handle)
    return embeddings


def main():
    people = load_people()
    embeddings = get_embeddings()
    result = evaluate_embeddings(
        people, embeddings,
        csv_path=CSV_PATH, plot_path=PLOT_PATH,
        title_prefix="ResNet50 on 30-person image_db/test_db",
    )
    eer_accuracy = result["eer_accuracy"]
    print(f"\n[evaluate_30_resnet50] Accuracy at EER: {eer_accuracy*100:.2f}%")


if __name__ == "__main__":
    main()
