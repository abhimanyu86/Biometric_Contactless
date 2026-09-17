"""
Extracts minutiae for all 90 images (30 gallery + 60 probes) in the 30-person
real-photo dataset (dataset_30.py), using the exact same detect_minutiae()
the deployed backend uses. Caches results so the comparison script doesn't
re-run the model every time.
"""
import os
import sys
import json

import cv2

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
BACKEND_DIR = os.path.join(os.path.dirname(BASE_DIR), "backend")
sys.path.insert(0, BACKEND_DIR)

from minutiae import detect_minutiae  # noqa: E402

from dataset_30 import load_people  # noqa: E402

CACHE_PATH = os.path.join(BASE_DIR, "minutiae_cache_30.json")


def compute_all_minutiae(people):
    all_paths = []
    for info in people.values():
        all_paths.append(info["gallery"])
        all_paths.extend(info["probes"])
    all_paths = list(dict.fromkeys(all_paths))

    print(f"[extract_minutiae_30] extracting minutiae for {len(all_paths)} images...")
    results = {}
    for i, path in enumerate(all_paths, 1):
        img = cv2.imread(path)
        if img is None:
            print(f"[extract_minutiae_30] WARNING: could not read {path}, skipping")
            continue
        minutiae = detect_minutiae(img)
        results[path] = minutiae
        print(f"  [{i}/{len(all_paths)}] {os.path.basename(os.path.dirname(path))}/"
              f"{os.path.basename(path)}: {len(minutiae)} minutiae")
    return results


def get_minutiae(force_recompute=False):
    if not force_recompute and os.path.isfile(CACHE_PATH):
        with open(CACHE_PATH, "r", encoding="utf-8") as handle:
            cached = json.load(handle)
        print(f"[extract_minutiae_30] loaded cached minutiae for {len(cached)} images")
        return cached

    people = load_people()
    results = compute_all_minutiae(people)
    with open(CACHE_PATH, "w", encoding="utf-8") as handle:
        json.dump(results, handle)
    return results


if __name__ == "__main__":
    results = get_minutiae(force_recompute=True)
    counts = [len(v) for v in results.values()]
    print(f"\n[extract_minutiae_30] done. {len(results)} images, "
          f"minutiae count: min={min(counts)} max={max(counts)} "
          f"mean={sum(counts)/len(counts):.1f}")
