"""
slap_embedding_matching.py — slap-specific orchestration for the CNN/Siamese
embedding matcher (embedding_matcher.py).

Per-finger matching mirrors the single-finger embedding matcher's own logic,
run once per finger: each probe finger is compared to its position-matched
enrolled finger via plain cosine similarity -- no averaging across fingers.
A candidate passes if at least EMBEDDING_SLAP_MIN_FINGERS individual fingers
independently clear the per-finger threshold (default 2 of the up to 4
captured).

EMBEDDING_SLAP_FINGER_THRESHOLD (default 0.78) is deliberately its OWN
constant, separate from single-finger's EMBEDDING_THRESHOLD (0.94) -- an
earlier version of this file reused 0.94 directly for slap too, on the
reasoning "same matcher, same threshold." That was wrong: 0.94 was measured
on the HKPU lab dataset (clean, consistent studio captures). Every real-photo
measurement we've since run (siamese_embedding/evaluate_30_resnet18.py, the
30-person image_db/test_db set of real uncontrolled phone photos) puts the
actual genuine/impostor separation point much lower, at EER threshold
~0.78 -- so requiring each finger to clear 0.94 was very likely rejecting
real, genuine slap captures far more often than the lab number would
suggest. 0.78 is that measured real-photo number, used here as the better-
grounded starting point.

The position constraint is unchanged (a probe finger only ever compares
against the SAME declared position's enrolled finger -- bug.md #2's fix for
the minutiae path, kept here for the same reason: without it, a probe
finger could get a lucky cross-position match).

Still not validated against real MULTI-FINGER slap captures specifically --
0.78 comes from single-finger real-photo data (the closest real-world
evidence we have), not a slap-specific measurement. Watch for the opposite
failure mode after this change: a lower per-finger bar means impostor
fingers are now more likely to individually pass too (we measured real
impostor scores routinely landing in the high 0.7s-0.8s on this same
dataset), so EMBEDDING_SLAP_MIN_FINGERS (requiring 2+, not just 1) is now
doing more of the real security work than the per-finger threshold is --
worth re-checking FAR specifically after this change, not just FRR.
"""
import json
import os

from embedding_matcher import cosine_similarity, EMBEDDING_THRESHOLD  # noqa: F401 -- EMBEDDING_THRESHOLD kept for reference/back-compat imports

EMBEDDING_SLAP_MIN_FINGERS = int(os.environ.get("EMBEDDING_SLAP_MIN_FINGERS", "2"))
EMBEDDING_SLAP_FINGER_THRESHOLD = float(os.environ.get("EMBEDDING_SLAP_FINGER_THRESHOLD", "0.78"))


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


def score_against_user_embedding(probe_fingers, user_rows, min_fingers_required=None):
    """
    Judge one candidate user against a set of probe fingers.

    probe_fingers: list of {"finger_position": str, "embedding": list[float]}
    user_rows:     list of {"finger_position": str, "embedding_json": str}

    Each probe finger is compared ONLY to its position-matched enrolled finger
    (bug.md #2's constraint) via plain cosine similarity, then judged
    independently against EMBEDDING_SLAP_FINGER_THRESHOLD -- a real-photo-
    calibrated threshold (0.78), deliberately lower than single-finger's
    lab-calibrated EMBEDDING_THRESHOLD (0.94) -- see this module's docstring
    for why. No averaging across fingers.

    Returns (fingers_passed, matched_fingers). matched_fingers is the
    per-finger detail list (probe_position, matched_position, confidence,
    passed) for every finger that found a same-position stored counterpart --
    a finger with nothing to compare against isn't included at all (it's
    "no data", not "failed"). fingers_passed is how many of those included
    fingers individually cleared EMBEDDING_SLAP_FINGER_THRESHOLD. The caller
    decides accept/reject by comparing fingers_passed to min_fingers_required
    (defaults to EMBEDDING_SLAP_MIN_FINGERS).
    """
    if min_fingers_required is None:
        min_fingers_required = EMBEDDING_SLAP_MIN_FINGERS

    matched_fingers = []
    fingers_passed = 0
    for f in probe_fingers:
        s, mpos, _ = best_finger_embedding_match(f["embedding"], user_rows,
                                                  probe_position=f["finger_position"])
        if mpos is None:
            continue  # nothing enrolled at this probe's position -- no data, not a failure
        passed = s >= EMBEDDING_SLAP_FINGER_THRESHOLD
        matched_fingers.append({
            "probe_position": f["finger_position"],
            "matched_position": mpos,
            "confidence": round(s, 5),
            "passed": passed,
        })
        if passed:
            fingers_passed += 1

    return fingers_passed, matched_fingers
