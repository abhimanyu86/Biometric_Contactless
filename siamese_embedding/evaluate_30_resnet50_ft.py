"""
Full matching-accuracy/FAR/FRR/EER test using the fine-tuned ResNet50
(best_e01.pt) on the 30-person image_db (gallery) + test_db (2 probes/person)
dataset. Reuses evaluate.py's compute_metrics/evaluate_embeddings so the
numbers are produced by identical scoring logic to the earlier ResNet18 and
untrained-ResNet50 runs on this same dataset.
"""
import os
import json

import numpy as np
import torch

from dataset_30 import load_people
from embed_resnet50_ft import build_embedder, preprocess_image
from evaluate import evaluate_embeddings

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CACHE_PATH = os.path.join(BASE_DIR, "embeddings_cache_30_resnet50_ft.json")
CSV_PATH = os.path.join(BASE_DIR, "benchmark_results_30_resnet50_ft.csv")
PLOT_PATH = os.path.join(BASE_DIR, "eer_plot_30_resnet50_ft.png")


def compute_all_embeddings(people, device="cpu"):
    model = build_embedder(device=device)
    all_paths = []
    for info in people.values():
        all_paths.append(info["gallery"])
        all_paths.extend(info["probes"])
    all_paths = list(dict.fromkeys(all_paths))

    print(f"[evaluate_30_resnet50_ft] computing embeddings for {len(all_paths)} images on {device}...")
    embeddings = {}
    for path in all_paths:
        try:
            tensor = preprocess_image(path).unsqueeze(0).to(device)
        except (OSError, ValueError) as exc:
            print(f"[evaluate_30_resnet50_ft] WARNING: {exc}, skipping")
            continue
        with torch.no_grad():
            vec = model.forward_once(tensor).cpu().numpy()[0]
        embeddings[path] = vec
    print(f"[evaluate_30_resnet50_ft] done, {len(embeddings)} embeddings computed")
    return embeddings


def get_embeddings(force_recompute=False):
    if not force_recompute and os.path.isfile(CACHE_PATH):
        with open(CACHE_PATH, "r", encoding="utf-8") as handle:
            raw = json.load(handle)
        print(f"[evaluate_30_resnet50_ft] loaded {len(raw)} embeddings from cache")
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
        title_prefix="Fine-tuned ResNet50 (best_e01) on 30-person image_db/test_db",
    )
    eer_accuracy = result["eer_accuracy"]
    print(f"\n[evaluate_30_resnet50_ft] Accuracy at EER: {eer_accuracy*100:.2f}%")


if __name__ == "__main__":
    main()
