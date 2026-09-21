"""
Dynamic ResNet50 evaluation for a gallery/probe directory pair.

For each image in image_db, this script uses the filename stem as the person
identifier and loads every image from the matching folder in test_db. For
example, image_db/rahul.jpeg is evaluated against every image in
test_db/rahul/.

The aggregate metrics use the same genuine/impostor scoring pipeline as the
existing evaluation scripts. Genuine pairs compare a probe with its own
gallery image; impostor pairs compare it with every other enrolled gallery
image.
"""
import argparse
import csv
import json
import os

import numpy as np
import torch

from embed_resnet50 import build_embedder, embed_paths
from evaluate import evaluate_embeddings


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_GALLERY_DIR = r"D:\UIDAI\image_db"
DEFAULT_PROBE_DIR = r"D:\UIDAI\test_db"
DEFAULT_CACHE_PATH = os.path.join(BASE_DIR, "embeddings_cache_dynamic_resnet50.json")
DEFAULT_CSV_PATH = os.path.join(BASE_DIR, "benchmark_results_dynamic_resnet50.csv")
DEFAULT_PLOT_PATH = os.path.join(BASE_DIR, "eer_plot_dynamic_resnet50.png")
DEFAULT_PERSON_CSV_PATH = os.path.join(BASE_DIR, "per_person_results_dynamic_resnet50.csv")
IMAGE_EXTENSIONS = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}


def _image_files(directory):
    """Return image files below directory in deterministic order."""
    paths = []
    for root, _, filenames in os.walk(directory):
        for filename in filenames:
            if os.path.splitext(filename)[1].lower() in IMAGE_EXTENSIONS:
                paths.append(os.path.abspath(os.path.join(root, filename)))
    return sorted(paths, key=lambda path: os.path.normcase(path))


def load_people(gallery_dir, probe_dir):
    """Discover {person: {gallery, probes}} from matching file/folder stems."""
    gallery_dir = os.path.abspath(gallery_dir)
    probe_dir = os.path.abspath(probe_dir)
    if not os.path.isdir(gallery_dir):
        raise FileNotFoundError(f"gallery directory does not exist: {gallery_dir}")
    if not os.path.isdir(probe_dir):
        raise FileNotFoundError(f"probe directory does not exist: {probe_dir}")

    people = {}
    for gallery_path in _image_files(gallery_dir):
        if os.path.dirname(gallery_path) != gallery_dir:
            continue
        person = os.path.splitext(os.path.basename(gallery_path))[0]
        if person in people:
            raise ValueError(f"multiple gallery images found for person '{person}'")

        person_probe_dir = os.path.join(probe_dir, person)
        if not os.path.isdir(person_probe_dir):
            print(f"[dynamic_dataset] WARNING: missing probe folder for {person}, skipping")
            continue
        probes = _image_files(person_probe_dir)
        if not probes:
            print(f"[dynamic_dataset] WARNING: no probe images for {person}, skipping")
            continue
        people[person] = {"gallery": gallery_path, "probes": probes}

    if not people:
        raise ValueError("no gallery/probe pairs were discovered")
    if len(people) < 2:
        raise ValueError("at least two people are required to calculate impostor metrics")
    return people


def _load_cached_embeddings(cache_path):
    if not os.path.isfile(cache_path):
        return {}
    with open(cache_path, "r", encoding="utf-8") as handle:
        raw = json.load(handle)
    return {path: np.array(vector, dtype=np.float32) for path, vector in raw.items()}


def get_embeddings(people, cache_path, device, force_recompute=False):
    all_paths = []
    for info in people.values():
        all_paths.append(info["gallery"])
        all_paths.extend(info["probes"])
    all_paths = list(dict.fromkeys(all_paths))

    embeddings = {} if force_recompute else _load_cached_embeddings(cache_path)
    missing_paths = [path for path in all_paths if path not in embeddings]
    if missing_paths:
        print(f"[dynamic_resnet50] computing embeddings for {len(missing_paths)} images on {device}...")
        model = build_embedder(device=device)
        embeddings.update(embed_paths(model, missing_paths, device=device))
        print(f"[dynamic_resnet50] total embeddings available: {len(embeddings)}")
        with open(cache_path, "w", encoding="utf-8") as handle:
            json.dump({path: vector.tolist() for path, vector in embeddings.items()}, handle)
    else:
        print(f"[dynamic_resnet50] loaded {len(embeddings)} embeddings from cache")
    return {path: embeddings[path] for path in all_paths if path in embeddings}


def write_per_person_results(people, embeddings, threshold, csv_path):
    rows = []
    for person, info in sorted(people.items()):
        gallery = embeddings.get(info["gallery"])
        scores = []
        if gallery is not None:
            gallery_norm = np.linalg.norm(gallery)
            if gallery_norm:
                gallery = gallery / gallery_norm
                for probe_path in info["probes"]:
                    probe = embeddings.get(probe_path)
                    if probe is not None:
                        probe_norm = np.linalg.norm(probe)
                        if probe_norm:
                            scores.append(float(np.dot(probe / probe_norm, gallery)))
        accepted = sum(score >= threshold for score in scores)
        rows.append({
            "person": person,
            "probe_count": len(scores),
            "accepted_at_eer": accepted,
            "rejected_at_eer": len(scores) - accepted,
            "genuine_match_rate_at_eer": accepted / len(scores) if scores else "",
            "mean_genuine_score": float(np.mean(scores)) if scores else "",
            "median_genuine_score": float(np.median(scores)) if scores else "",
        })

    with open(csv_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gallery-dir", default=DEFAULT_GALLERY_DIR)
    parser.add_argument("--probe-dir", default=DEFAULT_PROBE_DIR)
    parser.add_argument("--cache", default=DEFAULT_CACHE_PATH)
    parser.add_argument("--csv", default=DEFAULT_CSV_PATH)
    parser.add_argument("--plot", default=DEFAULT_PLOT_PATH)
    parser.add_argument("--person-csv", default=DEFAULT_PERSON_CSV_PATH)
    parser.add_argument("--force-recompute", action="store_true")
    parser.add_argument("--device", default=None, choices=("cpu", "cuda"))
    return parser.parse_args()


def main():
    args = parse_args()
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    people = load_people(args.gallery_dir, args.probe_dir)
    total_probes = sum(len(info["probes"]) for info in people.values())
    print(f"[dynamic_dataset] loaded {len(people)} people and {total_probes} probe images")

    embeddings = get_embeddings(
        people, args.cache, device=device, force_recompute=args.force_recompute
    )
    result = evaluate_embeddings(
        people,
        embeddings,
        csv_path=args.csv,
        plot_path=args.plot,
        title_prefix="ResNet50 on dynamic image_db/test_db",
    )
    write_per_person_results(
        people, embeddings, result["eer_row"]["threshold"], args.person_csv
    )
    print(f"[dynamic_resnet50] per-person results written to {args.person_csv}")


if __name__ == "__main__":
    main()
