"""
slap_embedding_matching.py — slap-specific orchestration for the CNN/Siamese
embedding matcher (embedding_matcher.py), mirroring slap_matching.py's structure
exactly (position-constrained per-finger scoring, then top-2-average aggregation)
so the two matching families behave the same way at the slap level and are easy
to compare. embedding_matcher.py's cosine_similarity() replaces matching.py's
match_templates() as the per-finger score function; everything else -- the
position constraint, the "ignore non-matching fingers rather than scoring them
zero" rule (see slap_matching.py's docstring, bug.md #8) -- is unchanged.

Not validated for slap-specific accuracy/FAR/FRR -- only the single-finger
protocol has been measured (siamese_embedding/, 92.5% accuracy at EER=7.5%).
This is the same matcher run per-finger with the same aggregation shape as the
minutiae slap path; treat EMBEDDING_SLAP_THRESHOLD as a placeholder until slap
data is actually used to calibrate it.
"""
import json
import os

from embedding_matcher import cosine_similarity, EMBEDDING_THRESHOLD

# Defaults to the same value as single-finger's EMBEDDING_THRESHOLD, but is
# independently overridable -- mirrors how MATCH_THRESH (slap) and
# SINGLE_THRESHOLD (single-finger) are separate constants in the minutiae path
# even though they currently happen to share a value.
EMBEDDING_SLAP_THRESHOLD = float(os.environ.get("EMBEDDING_SLAP_THRESHOLD", str(EMBEDDING_THRESHOLD)))


def best_finger_embedding_match(probe_embedding, user_rows, probe_position=None):
    """Match one probe embedding against a user's enrolled finger(s); return
    (best_score, matched_position, all_scores_per_pos).

    When probe_position is given, only compares against the stored embedding at
    that SAME position -- same position-constraint rule as
    slap_matching.best_finger_match(), for the same reason (bug.md #2): a probe
    finger's declared identity shouldn't be free to search every stored position
    for a lucky match."""
    best_score = -1.0
    matched_pos = None
    per_pos = {}
    for user_row in user_rows:
        pos = user_row["finger_position"]
        if probe_position is not None and pos != probe_position:
            continue
        try:
            stored_embedding = json.loads(user_row["embedding_json"])
        except Exception:
            continue
        if not stored_embedding:
            continue
        s = cosine_similarity(probe_embedding, stored_embedding)
        per_pos[pos] = round(s, 5)
        if s > best_score:
            best_score = s
            matched_pos = pos
    if matched_pos is None:
        return 0.0, None, per_pos
    return best_score, matched_pos, per_pos


def score_against_user_embedding(probe_fingers, user_rows):
    """
    Aggregate score for one candidate user against a set of probe fingers.

    probe_fingers: list of {"finger_position": str, "embedding": list[float]}
    user_rows:     list of {"finger_position": str, "embedding_json": str}

    Same aggregation as slap_matching.score_against_user(): the average of the
    top 2 scores among fingers that actually matched something (nonzero score);
    0.0 if none did. A finger with no stored counterpart at its position doesn't
    contribute -- it isn't scored as a zero that dilutes the average.
    """
    matched_fingers = []
    scores = []
    for f in probe_fingers:
        s, mpos, _ = best_finger_embedding_match(f["embedding"], user_rows,
                                                  probe_position=f["finger_position"])
        # mpos is None means no stored finger existed at this probe's position --
        # that's "nothing to compare", not "compared and scored low". Cosine
        # similarity can be negative in principle (unlike matching.py's minutiae
        # scores, which are always >= 0), so checking mpos rather than s > 0.0
        # is what actually distinguishes the two cases here.
        if mpos is not None:
            matched_fingers.append({
                "probe_position": f["finger_position"],
                "matched_position": mpos,
                "confidence": round(s, 5),
            })
            scores.append(s)

    if not scores:
        return 0.0, matched_fingers

    top = sorted(scores, reverse=True)[:2]
    agg_score = sum(top) / len(top)
    return agg_score, matched_fingers
