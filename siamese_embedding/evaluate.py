"""
Phase A evaluation: scores every genuine (probe vs. own gallery image) and impostor
(probe vs. every other person's gallery image) pair by cosine similarity of the
ResNet18 embeddings from embed.py, sweeps a threshold grid for FAR/FRR/EER, and
reports accuracy against the ~52% MinutiaNet baseline from
Contactless_Fingerprint_Matching_Next_Plan.pdf.

Mirrors the FAR/FRR/EER sweep already used in backend/benchmark.py
(compute_metrics/write_csv/write_plot) rather than reinventing it, adapted to take
score arrays directly instead of reading backend/uidai.db.
"""
import os
import csv
import tempfile

import numpy as np
from bisect import bisect_left

os.environ.setdefault("MPLCONFIGDIR", os.path.join(tempfile.gettempdir(), "matplotlib"))

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:
    plt = None

from dataset import load_people
from embed import get_embeddings

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(BASE_DIR, "benchmark_results.csv")
PLOT_PATH = os.path.join(BASE_DIR, "eer_plot.png")
NUM_THRESHOLDS = 181  # 0.005 resolution over [-0.10, 1.00]
MINUTIAE_BASELINE_ACCURACY = 0.52
ILLUSTRATIVE_THRESHOLD = 0.75


def build_score_matrices(people, embeddings):
    """Returns (genuine_scores, impostor_scores) as flat numpy arrays, computed via
    one matrix multiply of L2-normalized embeddings rather than a python double loop
    over ~562K pairs."""
    person_ids = sorted(people.keys())
    gallery_vecs = []
    valid_person_ids = []
    for pid in person_ids:
        path = people[pid]["gallery"]
        if path in embeddings:
            gallery_vecs.append(embeddings[path])
            valid_person_ids.append(pid)
    gallery_matrix = np.stack(gallery_vecs)
    gallery_matrix = gallery_matrix / np.linalg.norm(gallery_matrix, axis=1, keepdims=True)
    pid_to_gallery_row = {pid: i for i, pid in enumerate(valid_person_ids)}

    probe_vecs = []
    probe_owner_row = []  # index into valid_person_ids / gallery_matrix rows
    for pid in valid_person_ids:
        for probe_path in people[pid]["probes"]:
            if probe_path not in embeddings:
                continue
            probe_vecs.append(embeddings[probe_path])
            probe_owner_row.append(pid_to_gallery_row[pid])

    probe_matrix = np.stack(probe_vecs)
    probe_matrix = probe_matrix / np.linalg.norm(probe_matrix, axis=1, keepdims=True)

    similarity = probe_matrix @ gallery_matrix.T  # (num_probes, num_people)

    owner_rows = np.array(probe_owner_row)
    probe_indices = np.arange(len(owner_rows))

    genuine_scores = similarity[probe_indices, owner_rows]

    impostor_mask = np.ones_like(similarity, dtype=bool)
    impostor_mask[probe_indices, owner_rows] = False
    impostor_scores = similarity[impostor_mask]

    return genuine_scores, impostor_scores


def compute_metrics(genuine_scores, impostor_scores):
    """Same threshold-sweep + min(|FAR-FRR|) EER-point approach as
    backend/benchmark.py:compute_metrics, but with a threshold grid sized to the
    actual cosine-similarity score range instead of hardcoded 0.1 steps -- generic
    CNN embeddings of same-domain images (all fingerprints) can cluster tightly, so
    coarse steps would miss the real EER."""
    genuine_scores = sorted(float(s) for s in genuine_scores)
    impostor_scores = sorted(float(s) for s in impostor_scores)
    genuine_count = len(genuine_scores)
    impostor_count = len(impostor_scores)

    lo = min(genuine_scores[0], impostor_scores[0])
    hi = max(genuine_scores[-1], impostor_scores[-1])
    thresholds = np.linspace(lo, hi, NUM_THRESHOLDS)

    rows = []
    for threshold in thresholds:
        false_accepts = impostor_count - bisect_left(impostor_scores, threshold) if impostor_count else 0
        false_rejects = bisect_left(genuine_scores, threshold) if genuine_count else 0

        far = false_accepts / impostor_count if impostor_count else float("nan")
        frr = false_rejects / genuine_count if genuine_count else float("nan")
        rows.append({
            "threshold": float(threshold),
            "far": far,
            "frr": frr,
            "false_accepts": false_accepts,
            "false_rejects": false_rejects,
            "total_genuine_pairs": genuine_count,
            "total_impostor_pairs": impostor_count,
        })

    eer_row = min(rows, key=lambda row: abs(row["far"] - row["frr"]))
    eer_row = {**eer_row, "eer": (eer_row["far"] + eer_row["frr"]) / 2.0}

    return rows, eer_row


def accuracy_at_threshold(genuine_scores, impostor_scores, threshold):
    genuine_scores = np.asarray(genuine_scores)
    impostor_scores = np.asarray(impostor_scores)
    true_accepts = np.sum(genuine_scores >= threshold)
    true_rejects = np.sum(impostor_scores < threshold)
    total = len(genuine_scores) + len(impostor_scores)
    return (true_accepts + true_rejects) / total if total else float("nan")


def write_csv(rows, eer_row, csv_path=CSV_PATH):
    with open(csv_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "threshold", "far", "frr", "false_accepts", "false_rejects",
                "total_genuine_pairs", "total_impostor_pairs", "eer_threshold", "eer",
            ],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow({
                **row,
                "eer_threshold": eer_row["threshold"] if eer_row else "",
                "eer": eer_row["eer"] if eer_row else "",
            })


def write_plot(rows, eer_row, genuine_scores, impostor_scores, plot_path=PLOT_PATH, title_prefix="CNN/Siamese Embedding"):
    if plt is None:
        print("[evaluate] matplotlib not available, skipping plot")
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    thresholds = [row["threshold"] for row in rows]
    fars = [row["far"] for row in rows]
    frrs = [row["frr"] for row in rows]
    ax1.plot(thresholds, fars, label="FAR")
    ax1.plot(thresholds, frrs, label="FRR")
    if eer_row:
        ax1.scatter([eer_row["threshold"]], [eer_row["eer"]], color="red", zorder=3,
                    label=f"EER ~ {eer_row['eer']:.4f}")
        ax1.axvline(eer_row["threshold"], color="red", linestyle="--", alpha=0.4)
    ax1.set_title(f"{title_prefix}: FAR/FRR vs. Threshold")
    ax1.set_xlabel("Cosine similarity threshold")
    ax1.set_ylabel("Rate")
    ax1.set_ylim(0, 1)
    ax1.grid(alpha=0.3)
    ax1.legend()

    ax2.hist(impostor_scores, bins=50, alpha=0.6, label="Impostor (different finger)", density=True)
    ax2.hist(genuine_scores, bins=50, alpha=0.6, label="Genuine (same finger)", density=True)
    ax2.set_title("Genuine vs. Impostor Score Distributions")
    ax2.set_xlabel("Cosine similarity")
    ax2.set_ylabel("Density")
    ax2.legend()

    plt.tight_layout()
    plt.savefig(plot_path, dpi=160)
    plt.close()


def evaluate_embeddings(people, embeddings, csv_path=CSV_PATH, plot_path=PLOT_PATH, title_prefix="CNN/Siamese Embedding"):
    """Shared scoring pipeline: given a people dict and an {path: embedding} map,
    computes genuine/impostor scores, FAR/FRR/EER, writes CSV+plot, and returns a
    results dict. Used by both the Phase A (pretrained ResNet18) and exp2
    (trained Siamese) evaluations so the metric logic isn't duplicated."""
    genuine_scores, impostor_scores = build_score_matrices(people, embeddings)

    print(f"\n[evaluate] {len(genuine_scores)} genuine pairs, {len(impostor_scores)} impostor pairs")
    print(f"[evaluate] genuine score  mean={np.mean(genuine_scores):.4f} median={np.median(genuine_scores):.4f} "
          f"std={np.std(genuine_scores):.4f}")
    print(f"[evaluate] impostor score mean={np.mean(impostor_scores):.4f} median={np.median(impostor_scores):.4f} "
          f"std={np.std(impostor_scores):.4f}")

    rows, eer_row = compute_metrics(genuine_scores, impostor_scores)
    write_csv(rows, eer_row, csv_path=csv_path)
    write_plot(rows, eer_row, genuine_scores, impostor_scores, plot_path=plot_path, title_prefix=title_prefix)

    eer_accuracy = 1.0 - eer_row["eer"]
    fixed_accuracy = accuracy_at_threshold(genuine_scores, impostor_scores, ILLUSTRATIVE_THRESHOLD)

    print(f"\n[evaluate] EER threshold = {eer_row['threshold']:.4f}")
    print(f"[evaluate] At EER: FAR={eer_row['far']:.4f}, FRR={eer_row['frr']:.4f}, EER={eer_row['eer']:.4f}")
    print(f"[evaluate] Accuracy at EER threshold: {eer_accuracy:.4f} ({eer_accuracy*100:.1f}%)")
    print(f"[evaluate] Accuracy at illustrative threshold {ILLUSTRATIVE_THRESHOLD}: "
          f"{fixed_accuracy:.4f} ({fixed_accuracy*100:.1f}%)")
    print(f"[evaluate] Results written to {csv_path} and {plot_path}")

    return {
        "genuine_scores": genuine_scores,
        "impostor_scores": impostor_scores,
        "rows": rows,
        "eer_row": eer_row,
        "eer_accuracy": eer_accuracy,
        "fixed_accuracy": fixed_accuracy,
    }


def main():
    people = load_people()
    embeddings = get_embeddings()
    result = evaluate_embeddings(people, embeddings)
    eer_accuracy = result["eer_accuracy"]

    print(f"\n[evaluate] Baseline (MinutiaNet, PDF): {MINUTIAE_BASELINE_ACCURACY*100:.1f}% accuracy")
    print(f"[evaluate] Phase A (pretrained CNN/Siamese embedding): {eer_accuracy*100:.1f}% accuracy at EER "
          f"-> {'BETTER' if eer_accuracy > MINUTIAE_BASELINE_ACCURACY else 'NOT better'} than baseline")


if __name__ == "__main__":
    main()
