"""
Step 0 of the hybrid-matching investigation: on the exact same 60 genuine +
1,740 impostor pairs (30-person real-photo dataset), does minutiae and
ResNet18 embedding fail on the SAME fingerprints, or DIFFERENT ones?

If different: a hybrid has real headroom (one matcher can catch what the
other misses). If the same: hybrid can't help -- no combination of two
matchers both wrong about the same case fixes that case.

Each matcher is judged against its OWN measured EER threshold from its own
30-person run (not a shared threshold -- minutiae and embedding scores live
on completely different scales, ~0.05-0.15 vs ~0.7-0.9).
"""
import json
import os

import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MINUTIAE_SCORES_PATH = os.path.join(BASE_DIR, "minutiae_scores_30.json")
RESNET18_EMBEDDINGS_PATH = os.path.join(BASE_DIR, "embeddings_cache_30_resnet18.json")

MINUTIAE_EER_THRESHOLD = 0.0693   # from evaluate_30_minutiae.py's own run
RESNET18_EER_THRESHOLD = 0.7805   # from evaluate_30_resnet18.py's own run


def cosine_similarity(a, b):
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    return float(np.dot(a, b) / denom) if denom else 0.0


def load_data():
    with open(MINUTIAE_SCORES_PATH, "r", encoding="utf-8") as f:
        minutiae_data = json.load(f)
    with open(RESNET18_EMBEDDINGS_PATH, "r", encoding="utf-8") as f:
        raw_embeddings = json.load(f)
    embeddings = {p: np.array(v, dtype=np.float32) for p, v in raw_embeddings.items()}
    return minutiae_data, embeddings


def build_comparison(pair_type, pairs, minutiae_scores, embeddings, minutiae_pass_is_correct):
    """pair_type: 'genuine' or 'impostor'. For genuine pairs, a matcher is
    "correct" when it passes (score >= its threshold) -- same finger should
    be accepted. For impostor pairs, correct means it FAILS to pass --
    different finger should be rejected. minutiae_pass_is_correct captures
    which direction counts as correct for this pair_type."""
    rows = []
    for (probe_path, gallery_path), m_score in zip(pairs, minutiae_scores):
        e_score = cosine_similarity(embeddings[probe_path], embeddings[gallery_path])
        m_passed = m_score >= MINUTIAE_EER_THRESHOLD
        e_passed = e_score >= RESNET18_EER_THRESHOLD
        m_correct = m_passed if minutiae_pass_is_correct else not m_passed
        e_correct = e_passed if minutiae_pass_is_correct else not e_passed
        rows.append({
            "probe": probe_path, "gallery": gallery_path,
            "minutiae_score": round(m_score, 4), "embedding_score": round(e_score, 4),
            "minutiae_correct": m_correct, "embedding_correct": e_correct,
        })
    return rows


def summarize(rows, label):
    both_right = sum(1 for r in rows if r["minutiae_correct"] and r["embedding_correct"])
    both_wrong = sum(1 for r in rows if not r["minutiae_correct"] and not r["embedding_correct"])
    only_minutiae = sum(1 for r in rows if r["minutiae_correct"] and not r["embedding_correct"])
    only_embedding = sum(1 for r in rows if not r["minutiae_correct"] and r["embedding_correct"])
    total = len(rows)
    print(f"\n=== {label} ({total} pairs) ===")
    print(f"  Both correct:            {both_right:5d} ({100*both_right/total:.1f}%)")
    print(f"  Both wrong:              {both_wrong:5d} ({100*both_wrong/total:.1f}%)")
    print(f"  Only minutiae correct:   {only_minutiae:5d} ({100*only_minutiae/total:.1f}%)  <- hybrid upside")
    print(f"  Only embedding correct:  {only_embedding:5d} ({100*only_embedding/total:.1f}%)  <- hybrid downside risk")
    return {"both_right": both_right, "both_wrong": both_wrong,
            "only_minutiae": only_minutiae, "only_embedding": only_embedding, "total": total}


def main():
    minutiae_data, embeddings = load_data()

    genuine_rows = build_comparison(
        "genuine", minutiae_data["genuine_pairs"], minutiae_data["genuine"],
        embeddings, minutiae_pass_is_correct=True,
    )
    impostor_rows = build_comparison(
        "impostor", minutiae_data["impostor_pairs"], minutiae_data["impostor"],
        embeddings, minutiae_pass_is_correct=False,
    )

    genuine_summary = summarize(genuine_rows, "GENUINE PAIRS (should match)")
    impostor_summary = summarize(impostor_rows, "IMPOSTOR PAIRS (should NOT match)")

    print("\n\nWhere minutiae rescues a genuine pair embedding missed:")
    for r in genuine_rows:
        if r["minutiae_correct"] and not r["embedding_correct"]:
            print(f"  {os.path.basename(os.path.dirname(r['probe']))}/{os.path.basename(r['probe'])} "
                  f"vs {os.path.basename(os.path.dirname(r['gallery']))}/{os.path.basename(r['gallery'])}: "
                  f"minutiae={r['minutiae_score']} (pass) embedding={r['embedding_score']} (fail)")

    print("\nWhere minutiae wrongly accepts an impostor embedding correctly rejects:")
    count = 0
    for r in impostor_rows:
        if r["minutiae_correct"] and not r["embedding_correct"]:
            continue
        if (not r["minutiae_correct"]) and r["embedding_correct"]:
            count += 1
    print(f"  {count} impostor pairs where minutiae falsely accepted but embedding correctly rejected")

    total_only_minutiae = genuine_summary["only_minutiae"] + impostor_summary["only_minutiae"]
    total_only_embedding = genuine_summary["only_embedding"] + impostor_summary["only_embedding"]
    print(f"\n\n=== BOTTOM LINE ===")
    print(f"Pairs where ONLY minutiae was right (hybrid could rescue): {total_only_minutiae}")
    print(f"Pairs where ONLY embedding was right (hybrid could break): {total_only_embedding}")


if __name__ == "__main__":
    main()
