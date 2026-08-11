"""
Invariance tests for the minutiae matcher.

These run with no model weights, no database and no TensorFlow, so they can be used to
validate a matcher change before any fingerprint data has been collected.

The rotation test is the important one. The team's own analysis identified "no rotation
handling" as the reason genuine scores sat at 0.130-0.201 against a 0.25 threshold, i.e.
0% TAR. test_legacy_collapses_under_rotation pins that failure in place; the matching
test on the current path proves it is fixed.
"""

import math
import os
import random
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from matching import match_templates  # noqa: E402


# ── Synthetic data ───────────────────────────────────────────────────────────

def make_template(n=40, width=200, height=250, seed=0):
    """A plausible minutiae set. Spatially spread, mixed types, random directions."""
    rng = random.Random(seed)
    return [{
        "x": rng.uniform(0, width),
        "y": rng.uniform(0, height),
        "direction": rng.uniform(0, 2 * math.pi),
        "type": "BIF" if rng.random() > 0.5 else "RIG",
    } for _ in range(n)]


def rotate(tmpl, degrees):
    """
    Rotate a template as a rigid body.

    Both the coordinates and each minutia's ridge direction must rotate. Rotating only
    the coordinates would not simulate a turned finger, it would simulate a scrambled
    one, and the test would prove nothing.
    """
    th = math.radians(degrees)
    c, s = math.cos(th), math.sin(th)
    return [{**m,
             "x": m["x"] * c - m["y"] * s,
             "y": m["x"] * s + m["y"] * c,
             "direction": (m["direction"] + th) % (2 * math.pi)} for m in tmpl]


def translate(tmpl, dx, dy):
    return [{**m, "x": m["x"] + dx, "y": m["y"] + dy} for m in tmpl]


def rescale(tmpl, k):
    return [{**m, "x": m["x"] * k, "y": m["y"] * k} for m in tmpl]


def jitter(tmpl, px=2.0, rad=0.05, seed=99):
    """Small perturbation, standing in for capture-to-capture variation."""
    rng = random.Random(seed)
    return [{**m,
             "x": m["x"] + rng.uniform(-px, px),
             "y": m["y"] + rng.uniform(-px, px),
             "direction": (m["direction"] + rng.uniform(-rad, rad)) % (2 * math.pi)}
            for m in tmpl]


ROTATIONS = [0, 5, 10, 15, 20, 30, 45]


# ── Baseline ─────────────────────────────────────────────────────────────────

def test_identical_templates_score_high():
    t = make_template(seed=1)
    assert match_templates(t, t) > 0.5


def test_empty_input_is_zero():
    t = make_template(seed=1)
    assert match_templates([], t) == 0.0
    assert match_templates(t, []) == 0.0


def test_count_ratio_gate():
    """Wildly different minutiae counts are rejected before any geometry runs."""
    big = make_template(n=40, seed=1)
    small = big[:10]
    assert match_templates(big, small) == 0.0


# ── The fix ──────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("degrees", ROTATIONS)
def test_rotation_invariance(degrees):
    """A rotated capture of the same finger must still match."""
    t = make_template(seed=2)
    baseline = match_templates(t, t)
    rotated = match_templates(t, rotate(t, degrees))
    assert rotated >= 0.8 * baseline, (
        "rotation by %d deg dropped the score from %.3f to %.3f"
        % (degrees, baseline, rotated)
    )


def test_legacy_collapses_under_rotation():
    """
    Pins the original defect.

    Not a regression guard — it documents why the matcher scored 0% TAR, and it will
    start failing if someone reverts the current path to the old descriptor.
    """
    t = make_template(seed=2)
    baseline = match_templates(t, t, legacy=True)
    rotated = match_templates(t, rotate(t, 30), legacy=True)
    assert baseline > 0.3, "legacy should still match a template against itself"
    assert rotated < 0.5 * baseline, (
        "expected the legacy matcher to fail under rotation, got %.3f vs %.3f"
        % (rotated, baseline)
    )


def test_rotation_invariance_beats_legacy():
    """Direct before/after on identical data."""
    t = make_template(seed=3)
    rotated = rotate(t, 30)
    assert match_templates(t, rotated) > match_templates(t, rotated, legacy=True)


# ── Other invariances (these should already have held) ───────────────────────

@pytest.mark.parametrize("dx,dy", [(50, 30), (-120, 75), (500, -400)])
def test_translation_invariance(dx, dy):
    t = make_template(seed=4)
    assert match_templates(t, translate(t, dx, dy)) == pytest.approx(
        match_templates(t, t), rel=0.02)


@pytest.mark.parametrize("k", [0.85, 1.15, 1.5])
def test_scale_invariance(k):
    t = make_template(seed=5)
    assert match_templates(t, rescale(t, k)) >= 0.8 * match_templates(t, t)


def test_single_outlier_does_not_wreck_the_score():
    """
    The original scaled by bounding-box span, so one stray minutia at the edge of the
    frame rescaled the whole template and shifted every distance bin.
    """
    t = make_template(seed=6)
    with_outlier = t + [{"x": 5000.0, "y": 5000.0,
                         "direction": 0.0, "type": "RIG"}]
    assert match_templates(t, with_outlier) >= 0.7 * match_templates(t, t)


# ── Discrimination ───────────────────────────────────────────────────────────

def test_genuine_scores_above_impostor():
    """A matcher that accepts everyone would pass every test above but not this one."""
    genuine, impostor = [], []
    for s in range(12):
        base = make_template(seed=100 + s)
        genuine.append(match_templates(base, jitter(base, seed=200 + s)))
        impostor.append(match_templates(base, make_template(seed=900 + s)))

    mean_g = sum(genuine) / len(genuine)
    mean_i = sum(impostor) / len(impostor)
    assert mean_g > mean_i, "genuine %.3f vs impostor %.3f" % (mean_g, mean_i)
    assert mean_g - mean_i > 0.05, (
        "separation too small to threshold: genuine %.3f, impostor %.3f"
        % (mean_g, mean_i))


def recapture(t, drop=0.3, spurious=0.2, rotation=20, seed=0):
    """
    A second capture of the same finger, simulated honestly.

    Two captures never yield the same minutiae set: extraction misses some, invents
    others, and the finger sits at a different angle. Testing only against a jittered
    copy of the same set flatters the matcher and hides exactly the failure mode that
    matters in the field.
    """
    rng = random.Random(seed)
    kept = [m for m in t if rng.random() > drop]
    xs = [m["x"] for m in t]
    ys = [m["y"] for m in t]
    for _ in range(int(len(t) * spurious)):
        kept.append({
            "x": rng.uniform(min(xs), max(xs)),
            "y": rng.uniform(min(ys), max(ys)),
            "direction": rng.uniform(0, 2 * math.pi),
            "type": "BIF" if rng.random() > 0.5 else "RIG",
        })
    return rotate(jitter(kept, px=3.0, rad=0.10, seed=seed), rotation)


@pytest.mark.parametrize("drop", [0.2, 0.3, 0.4])
def test_separation_survives_partial_minutiae_overlap(drop):
    """Genuine must still outscore impostor when up to 40% of minutiae go missing."""
    genuine, impostor = [], []
    for s in range(10):
        base = make_template(seed=500 + s)
        genuine.append(match_templates(base, recapture(base, drop=drop, seed=600 + s)))
        impostor.append(match_templates(
            base, recapture(make_template(seed=800 + s), drop=drop, seed=610 + s)))

    mean_g = sum(genuine) / len(genuine)
    mean_i = sum(impostor) / len(impostor)
    assert mean_g - mean_i > 0.08, (
        "drop=%.0f%%: genuine %.3f vs impostor %.3f — not separable"
        % (drop * 100, mean_g, mean_i))


def test_rotated_genuine_still_beats_impostor():
    """The property that actually matters in the field: rotation must not cause a miss."""
    for s in range(6):
        base = make_template(seed=300 + s)
        rotated_genuine = match_templates(base, rotate(jitter(base, seed=400 + s), 25))
        impostor = match_templates(base, make_template(seed=700 + s))
        assert rotated_genuine > impostor, (
            "seed %d: rotated genuine %.3f did not beat impostor %.3f"
            % (s, rotated_genuine, impostor))
