"""
hybrid_matcher.py — fuses the minutiae matcher (matching.py) and the CNN/
Siamese embedding matcher (embedding_matcher.py) into one score.

Motivated by a real feasibility check (siamese_embedding/compare_matchers_30.py):
on the 30-person real-photo dataset, minutiae and embedding disagree on a
meaningful number of pairs (200 where only minutiae was right, 404 where only
embedding was right) -- real complementary signal exists, but embedding's
unique wins outnumber minutiae's roughly 2-to-1, so a naive combination
(simple average, either-can-veto) would likely hurt more than help.

The two raw scores live on completely different scales -- minutiae's
"legacy" algorithm score sits around 0.05-0.15, embedding's cosine
similarity around 0.7-0.95. Combining them meaningfully requires putting
both on a comparable footing first: z-score normalize each against its OWN
matcher's measured impostor mean/std (from the same 30-person dataset), so
both read as "how many standard deviations above a typical impostor,"
before combining. The weights (0.20 minutiae / 0.80 embedding) and the
fused decision threshold were fit by grid search on that same dataset to
minimize EER -- not guessed. Result: 85.37% accuracy / EER 14.63%, beating
embedding alone (82.73% / 17.27%) and minutiae alone (71.24% / 28.76%) on
identical pairs.

Not validated on real slap captures specifically (same caveat as
embedding_matcher.py and slap_embedding_matching.py) -- calibrated on
single real-photo images, not actual multi-finger slap photos.
"""
import os

from matching import match_templates
from embedding_matcher import cosine_similarity

# Calibrated via siamese_embedding/compare_matchers_30.py's grid search on
# the 30-person real-photo dataset (60 genuine + 1740 impostor pairs).
WEIGHT_MINUTIAE = float(os.environ.get("HYBRID_WEIGHT_MINUTIAE", "0.20"))
WEIGHT_EMBEDDING = float(os.environ.get("HYBRID_WEIGHT_EMBEDDING", "0.80"))
FUSED_THRESHOLD = float(os.environ.get("HYBRID_FUSED_THRESHOLD", "1.1162"))

MINUTIAE_IMPOSTOR_MEAN = float(os.environ.get("HYBRID_MINUTIAE_IMPOSTOR_MEAN", "0.05504561884888432"))
MINUTIAE_IMPOSTOR_STD = float(os.environ.get("HYBRID_MINUTIAE_IMPOSTOR_STD", "0.0385722408583879"))
EMBEDDING_IMPOSTOR_MEAN = float(os.environ.get("HYBRID_EMBEDDING_IMPOSTOR_MEAN", "0.7137894791090625"))
EMBEDDING_IMPOSTOR_STD = float(os.environ.get("HYBRID_EMBEDDING_IMPOSTOR_STD", "0.06915622500308251"))

MATCHING_ALGORITHM = "legacy"  # matches cloud_app.py's minutiae algorithm in production


def normalize_minutiae(raw_score):
    """z-score against minutiae's own measured impostor distribution."""
    return (raw_score - MINUTIAE_IMPOSTOR_MEAN) / MINUTIAE_IMPOSTOR_STD


def normalize_embedding(raw_score):
    """z-score against embedding's own measured impostor distribution."""
    return (raw_score - EMBEDDING_IMPOSTOR_MEAN) / EMBEDDING_IMPOSTOR_STD


def fuse(minutiae_raw, embedding_raw):
    """Returns (fused_score, minutiae_z, embedding_z). fused_score compares
    against FUSED_THRESHOLD, not either raw score's own threshold."""
    m_z = normalize_minutiae(minutiae_raw)
    e_z = normalize_embedding(embedding_raw)
    fused = WEIGHT_MINUTIAE * m_z + WEIGHT_EMBEDDING * e_z
    return fused, m_z, e_z


def hybrid_score_for_pair(probe_minutiae, gallery_minutiae, probe_embedding, gallery_embedding):
    """One finger pair, both matchers, fused. Returns
    (fused_score, minutiae_raw, embedding_raw, passed)."""
    minutiae_raw = match_templates(probe_minutiae, gallery_minutiae, algorithm=MATCHING_ALGORITHM)
    embedding_raw = cosine_similarity(probe_embedding, gallery_embedding)
    fused, _, _ = fuse(minutiae_raw, embedding_raw)
    passed = fused >= FUSED_THRESHOLD
    return fused, minutiae_raw, embedding_raw, passed
