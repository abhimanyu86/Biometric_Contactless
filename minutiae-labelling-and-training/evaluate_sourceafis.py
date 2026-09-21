"""
FAR/FRR/EER for the SourceAFIS match-score CSV produced by
sourceafis-runner/EvalMain.java, reusing the same compute_metrics/write_csv/
write_plot pipeline already used for the MinutiaeNet baseline and the
CNN/Siamese embedding evaluation (siamese_embedding/evaluate.py) so results
are directly comparable.
"""
import csv
import os
import sys

import numpy as np

SIAMESE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "siamese_embedding")
sys.path.insert(0, os.path.abspath(SIAMESE_DIR))

from evaluate import compute_metrics, write_csv, write_plot, accuracy_at_threshold  # noqa: E402

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MINUTIAE_BASELINE_ACCURACY = 0.52
SOURCEAFIS_SUGGESTED_THRESHOLD = 40  # per SourceAFIS docs, "reasonable starting point"


def load_scores(csv_path):
    genuine, impostor = [], []
    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            score = float(row["score"])
            if row["pair_type"] == "genuine":
                genuine.append(score)
            else:
                impostor.append(score)
    return np.array(genuine), np.array(impostor)


def main():
    csv_in = sys.argv[1] if len(sys.argv) > 1 else os.path.join(BASE_DIR, "sourceafis_scores_336.csv")
    out_csv = os.path.join(BASE_DIR, "sourceafis_benchmark_results.csv")
    out_plot = os.path.join(BASE_DIR, "sourceafis_eer_plot.png")

    genuine_scores, impostor_scores = load_scores(csv_in)
    zero_genuine = int(np.sum(genuine_scores == 0))
    zero_impostor = int(np.sum(impostor_scores == 0))

    print(f"[evaluate] {len(genuine_scores)} genuine pairs, {len(impostor_scores)} impostor pairs")
    print(f"[evaluate] genuine  mean={np.mean(genuine_scores):.4f} median={np.median(genuine_scores):.4f} "
          f"std={np.std(genuine_scores):.4f} ({zero_genuine} scored exactly 0, "
          f"{100*zero_genuine/len(genuine_scores):.1f}% -- SourceAFIS found no usable alignment)")
    print(f"[evaluate] impostor mean={np.mean(impostor_scores):.4f} median={np.median(impostor_scores):.4f} "
          f"std={np.std(impostor_scores):.4f} ({zero_impostor} scored exactly 0, "
          f"{100*zero_impostor/len(impostor_scores):.1f}%)")

    rows, eer_row = compute_metrics(genuine_scores, impostor_scores)
    write_csv(rows, eer_row, csv_path=out_csv)
    write_plot(rows, eer_row, genuine_scores, impostor_scores, plot_path=out_plot,
               title_prefix="SourceAFIS")

    eer_accuracy = 1.0 - eer_row["eer"]
    fixed_accuracy = accuracy_at_threshold(genuine_scores, impostor_scores, SOURCEAFIS_SUGGESTED_THRESHOLD)

    print(f"\n[evaluate] EER threshold = {eer_row['threshold']:.4f}")
    print(f"[evaluate] At EER: FAR={eer_row['far']:.4f}, FRR={eer_row['frr']:.4f}, EER={eer_row['eer']:.4f}")
    print(f"[evaluate] Accuracy at EER threshold: {eer_accuracy:.4f} ({eer_accuracy*100:.1f}%)")
    print(f"[evaluate] Accuracy at SourceAFIS-suggested threshold "
          f"({SOURCEAFIS_SUGGESTED_THRESHOLD}): {fixed_accuracy:.4f} ({fixed_accuracy*100:.1f}%)")
    print(f"[evaluate] Results written to {out_csv} and {out_plot}")

    print(f"\n[evaluate] Baseline (MinutiaNet, PDF): {MINUTIAE_BASELINE_ACCURACY*100:.1f}% accuracy")
    print(f"[evaluate] SourceAFIS: {eer_accuracy*100:.1f}% accuracy at EER -> "
          f"{'BETTER' if eer_accuracy > MINUTIAE_BASELINE_ACCURACY else 'NOT better'} than baseline")


if __name__ == "__main__":
    main()
