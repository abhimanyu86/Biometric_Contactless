"""
Full matching-accuracy/FAR/FRR/EER test using the deployed minutiae matcher
(backend/matching.py, algorithm="legacy" -- the exact algorithm cloud_app.py
actually runs in production) on the same 30-person image_db/test_db dataset
already used for the embedding runs. Lets us compare minutiae vs. embedding
on identical pairs (see compare_matchers_30.py).
"""
import os
import sys
import json

import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
BACKEND_DIR = os.path.join(os.path.dirname(BASE_DIR), "backend")
sys.path.insert(0, BACKEND_DIR)

from matching import match_templates  # noqa: E402

from dataset_30 import load_people  # noqa: E402
from extract_minutiae_30 import get_minutiae  # noqa: E402
from evaluate import compute_metrics, write_csv, write_plot, accuracy_at_threshold, ILLUSTRATIVE_THRESHOLD  # noqa: E402

CSV_PATH = os.path.join(BASE_DIR, "benchmark_results_30_minutiae.csv")
PLOT_PATH = os.path.join(BASE_DIR, "eer_plot_30_minutiae.png")
SCORES_CACHE = os.path.join(BASE_DIR, "minutiae_scores_30.json")
ALGORITHM = "legacy"  # matches cloud_app.py's MATCHING_ALGORITHM in production


def compute_scores(people, minutiae, force_recompute=False):
    if not force_recompute and os.path.isfile(SCORES_CACHE):
        with open(SCORES_CACHE, "r", encoding="utf-8") as handle:
            cached = json.load(handle)
        print(f"[evaluate_30_minutiae] loaded cached scores")
        return cached["genuine"], cached["impostor"], cached["genuine_pairs"], cached["impostor_pairs"]

    person_ids = sorted(people.keys())
    genuine_scores = []
    impostor_scores = []
    genuine_pairs = []   # (probe_path, gallery_path)
    impostor_pairs = []

    total = sum(len(people[pid]["probes"]) * len(person_ids) for pid in person_ids)
    done = 0
    for pid in person_ids:
        for probe_path in people[pid]["probes"]:
            probe_min = minutiae.get(probe_path)
            if probe_min is None:
                continue
            for other_pid in person_ids:
                gallery_path = people[other_pid]["gallery"]
                gallery_min = minutiae.get(gallery_path)
                if gallery_min is None:
                    continue
                score = match_templates(probe_min, gallery_min, algorithm=ALGORITHM)
                if other_pid == pid:
                    genuine_scores.append(score)
                    genuine_pairs.append((probe_path, gallery_path))
                else:
                    impostor_scores.append(score)
                    impostor_pairs.append((probe_path, gallery_path))
                done += 1
                if done % 200 == 0:
                    print(f"[evaluate_30_minutiae] {done}/{total} pairs scored...")

    with open(SCORES_CACHE, "w", encoding="utf-8") as handle:
        json.dump({
            "genuine": genuine_scores, "impostor": impostor_scores,
            "genuine_pairs": genuine_pairs, "impostor_pairs": impostor_pairs,
        }, handle)

    return genuine_scores, impostor_scores, genuine_pairs, impostor_pairs


def main():
    people = load_people()
    minutiae = get_minutiae()
    genuine_scores, impostor_scores, genuine_pairs, impostor_pairs = compute_scores(people, minutiae)

    genuine_scores = np.array(genuine_scores)
    impostor_scores = np.array(impostor_scores)
    print(f"\n[evaluate_30_minutiae] {len(genuine_scores)} genuine pairs, {len(impostor_scores)} impostor pairs")
    print(f"[evaluate_30_minutiae] genuine score  mean={genuine_scores.mean():.4f} median={np.median(genuine_scores):.4f}")
    print(f"[evaluate_30_minutiae] impostor score mean={impostor_scores.mean():.4f} median={np.median(impostor_scores):.4f}")

    rows, eer_row = compute_metrics(genuine_scores, impostor_scores)
    write_csv(rows, eer_row, csv_path=CSV_PATH)
    write_plot(rows, eer_row, genuine_scores, impostor_scores, plot_path=PLOT_PATH,
               title_prefix="Minutiae (legacy) on 30-person image_db/test_db")

    eer_accuracy = 1.0 - eer_row["eer"]
    print(f"\n[evaluate_30_minutiae] EER threshold = {eer_row['threshold']:.4f}")
    print(f"[evaluate_30_minutiae] At EER: FAR={eer_row['far']:.4f}, FRR={eer_row['frr']:.4f}, EER={eer_row['eer']:.4f}")
    print(f"[evaluate_30_minutiae] Accuracy at EER: {eer_accuracy*100:.2f}%")


if __name__ == "__main__":
    main()
