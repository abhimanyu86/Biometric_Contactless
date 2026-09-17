"""
slap_hybrid_matching.py — slap-specific orchestration for the fused minutiae
+ embedding matcher (hybrid_matcher.py). Same shape as slap_embedding_matching.py:
each probe finger is compared only to its position-matched enrolled finger
(bug.md #2's constraint), judged independently via the fused score, and a
candidate passes if at least HYBRID_SLAP_MIN_FINGERS individually clear
hybrid_matcher.FUSED_THRESHOLD. No averaging across fingers.

Requires BOTH a minutiae template and an embedding for each enrolled and
probe finger -- enrollment and probing both run detect_minutiae() AND
embed_image() per image, roughly double the per-finger compute of either
matcher alone.
"""
import os

from hybrid_matcher import hybrid_score_for_pair

HYBRID_SLAP_MIN_FINGERS = int(os.environ.get("HYBRID_SLAP_MIN_FINGERS", "2"))


def best_finger_hybrid_match(probe_minutiae, probe_embedding, user_rows, probe_position=None):
    """Match one probe finger (both minutiae + embedding) against a user's
    enrolled finger(s); return (best_fused, minutiae_raw, embedding_raw,
    matched_position).

    Position-constrained: only compares against the stored finger at the
    SAME position, same rule and reasoning as slap_matching.best_finger_match()
    and slap_embedding_matching.best_finger_embedding_match() (bug.md #2)."""
    best_fused = None
    best_detail = (0.0, 0.0)
    matched_pos = None
    for user_row in user_rows:
        pos = user_row["finger_position"]
        if probe_position is not None and pos != probe_position:
            continue
        stored_minutiae = user_row.get("minutiae")
        stored_embedding = user_row.get("embedding")
        if not stored_minutiae or stored_embedding is None:
            continue
        fused, m_raw, e_raw, _ = hybrid_score_for_pair(
            probe_minutiae, stored_minutiae, probe_embedding, stored_embedding,
        )
        if best_fused is None or fused > best_fused:
            best_fused = fused
            best_detail = (m_raw, e_raw)
            matched_pos = pos
    if matched_pos is None:
        return None, 0.0, 0.0, None
    return best_fused, best_detail[0], best_detail[1], matched_pos


def score_against_user_hybrid(probe_fingers, user_rows, min_fingers_required=None):
    """
    Judge one candidate user against a set of probe fingers using the fused
    minutiae+embedding score.

    probe_fingers: list of {"finger_position": str, "minutiae": list, "embedding": array}
    user_rows:     list of {"finger_position": str, "minutiae": list, "embedding": array}

    Returns (fingers_passed, matched_fingers). matched_fingers is the
    per-finger detail (probe_position, matched_position, fused_score,
    minutiae_score, embedding_score, passed) for every finger that found a
    same-position stored counterpart.
    """
    from hybrid_matcher import FUSED_THRESHOLD

    if min_fingers_required is None:
        min_fingers_required = HYBRID_SLAP_MIN_FINGERS

    matched_fingers = []
    fingers_passed = 0
    for f in probe_fingers:
        fused, m_raw, e_raw, mpos = best_finger_hybrid_match(
            f["minutiae"], f["embedding"], user_rows, probe_position=f["finger_position"],
        )
        if mpos is None:
            continue  # nothing enrolled at this probe's position -- no data, not a failure
        passed = fused >= FUSED_THRESHOLD
        matched_fingers.append({
            "probe_position": f["finger_position"],
            "matched_position": mpos,
            "fused_score": round(fused, 5),
            "minutiae_score": round(m_raw, 5),
            "embedding_score": round(e_raw, 5),
            "passed": passed,
        })
        if passed:
            fingers_passed += 1

    return fingers_passed, matched_fingers
