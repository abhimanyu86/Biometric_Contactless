"""
slap_embedding_matching.py — slap-specific orchestration for the CNN/Siamese
embedding matcher (embedding_matcher.py).

Redesigned (was: top-2-average aggregation into one continuous slap score,
mirroring slap_matching.py's minutiae aggregation). The averaged-score
approach wasn't working well in practice, so this now matches the
single-finger embedding matcher's own logic exactly, just run once per
finger: each probe finger is compared to its position-matched enrolled
finger via plain cosine similarity and judged against the SAME
EMBEDDING_THRESHOLD single-finger uses (0.94 by default) -- no separate
slap-specific threshold, no averaging across fingers. A candidate passes if
at least EMBEDDING_SLAP_MIN_FINGERS individual fingers independently clear
that threshold (default 2 of the up to 4 captured).

The position constraint is unchanged (a probe finger only ever compares
against the SAME declared position's enrolled finger -- bug.md #2's fix for
the minutiae path, kept here for the same reason: without it, a probe
finger could get a lucky cross-position match).

Still not validated for slap-specific accuracy/FAR/FRR -- only the
single-finger protocol has a measured number (siamese_embedding/, 92.5%
accuracy at EER=7.5%). This redesign makes slap decisions using that same
measured threshold per finger, which is a better-grounded starting point
than the old placeholder aggregate threshold, but "at least 2 fingers must
independently clear 0.94" as a *slap-level* accept/reject rule is still
unmeasured and should be revisited once real slap capture data exists.
"""
import json
import os

from embedding_matcher import cosine_similarity, EMBEDDING_THRESHOLD

EMBEDDING_SLAP_MIN_FINGERS = int(os.environ.get("EMBEDDING_SLAP_MIN_FINGERS", "2"))


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
    independently against EMBEDDING_THRESHOLD -- the exact same threshold and
    decision rule /v2/verify_embedding uses for a single finger. No averaging
    across fingers.

    Returns (fingers_passed, matched_fingers). matched_fingers is the
    per-finger detail list (probe_position, matched_position, confidence,
    passed) for every finger that found a same-position stored counterpart --
    a finger with nothing to compare against isn't included at all (it's
    "no data", not "failed"). fingers_passed is how many of those included
    fingers individually cleared EMBEDDING_THRESHOLD. The caller decides
    accept/reject by comparing fingers_passed to min_fingers_required
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
        passed = s >= EMBEDDING_THRESHOLD
        matched_fingers.append({
            "probe_position": f["finger_position"],
            "matched_position": mpos,
            "confidence": round(s, 5),
            "passed": passed,
        })
        if passed:
            fingers_passed += 1

    return fingers_passed, matched_fingers
