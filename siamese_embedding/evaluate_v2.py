"""
Runs the exact same 336-person gallery/probe protocol as evaluate.py (Phase A),
but against the teammate's exp2 trained Siamese checkpoint instead of the stock
pretrained ResNet18. Reuses evaluate.py's compute_metrics/evaluate_embeddings so
the two results are produced by identical scoring logic -- a fair comparison, not
two different measurement methods.
"""
import os

from dataset import load_people
from embed_v2 import get_embeddings
from evaluate import evaluate_embeddings, MINUTIAE_BASELINE_ACCURACY

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(BASE_DIR, "benchmark_results_v2.csv")
PLOT_PATH = os.path.join(BASE_DIR, "eer_plot_v2.png")


def main():
    people = load_people()
    embeddings = get_embeddings()
    result = evaluate_embeddings(
        people, embeddings,
        csv_path=CSV_PATH, plot_path=PLOT_PATH,
        title_prefix="exp2 Trained Siamese (contrastive loss)",
    )
    eer_accuracy = result["eer_accuracy"]

    print(f"\n[evaluate_v2] Baseline (MinutiaNet, PDF): {MINUTIAE_BASELINE_ACCURACY*100:.1f}% accuracy")
    print(f"[evaluate_v2] exp2 trained Siamese: {eer_accuracy*100:.1f}% accuracy at EER "
          f"-> {'BETTER' if eer_accuracy > MINUTIAE_BASELINE_ACCURACY else 'NOT better'} than baseline")


if __name__ == "__main__":
    main()
