"""
Tests for the frequency-domain screen/print recapture heuristic.

No weights, no database, no TensorFlow — synthetic ridge and synthetic
screen-grid images only, same approach as test_fir.py's ridge_image().
"""

import math
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from spoof_detection import CHROMA_DIFF_THRESH, detect_screen_replay  # noqa: E402


def ridge_image(period=10.0, size=256, angle=0.0, noise=0.0, seed=0):
    """A synthetic fingerprint-like pattern: parallel ridges at a known spacing."""
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float32)
    proj = xx * math.cos(angle) + yy * math.sin(angle)
    wave = np.sin(2 * math.pi * proj / period)
    img = (wave * 0.5 + 0.5) * 255
    if noise:
        rng = np.random.RandomState(seed)
        img = img + rng.normal(0, noise, size=img.shape)
    return np.clip(img, 0, 255).astype(np.uint8)


def screen_recapture_image(ridge_period=10.0, grid_period=8.0, size=256, angle=0.0):
    """
    Stand-in for "photo of a finger photo shown on a screen": the same ridge
    pattern, with a fine repeating grid superimposed. A display's pixel/subpixel
    raster is much closer to a repeating rectangle than a pure sinusoid, so it's
    modeled as a square wave here — a square wave's fundamental-plus-odd-harmonics
    structure is exactly what shows up as more than one sharp radial peak, unlike
    the single peak a real ridge pattern produces.
    """
    base = ridge_image(period=ridge_period, size=size, angle=angle).astype(np.float32)
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float32)
    grid = (np.sign(np.sin(2 * math.pi * xx / grid_period)) +
            np.sign(np.sin(2 * math.pi * yy / grid_period)))
    return np.clip(base + grid * 18, 0, 255).astype(np.uint8)


def to_bgr(gray):
    return np.dstack([gray] * 3)


# ── Real fingerprint texture should not be flagged ──────────────────────────

def test_clean_ridge_pattern_not_flagged():
    result = detect_screen_replay(to_bgr(ridge_image()))
    assert result["is_screen"] is False


@pytest.mark.parametrize("angle", [0, math.pi / 6, math.pi / 3, math.pi / 2])
def test_ridge_pattern_not_flagged_at_any_orientation(angle):
    result = detect_screen_replay(to_bgr(ridge_image(angle=angle)))
    assert result["is_screen"] is False


def test_noisy_ridge_pattern_not_flagged():
    result = detect_screen_replay(to_bgr(ridge_image(noise=15.0)))
    assert result["is_screen"] is False


# ── Screen recapture grid should be flagged ──────────────────────────────────

def test_screen_recapture_grid_is_flagged():
    result = detect_screen_replay(to_bgr(screen_recapture_image()))
    assert result["is_screen"] is True
    assert result["peak_count"] >= 2


def test_screen_recapture_flagged_even_when_ridge_pattern_present():
    """The spoof still carries the original finger's ridge frequency (it's a
    photo of a real ridge pattern) — the grid has to be caught on top of it,
    not instead of it."""
    result = detect_screen_replay(to_bgr(screen_recapture_image(ridge_period=10.0)))
    assert result["is_screen"] is True


@pytest.mark.parametrize("grid_period", [5.0, 8.0, 12.0])
def test_screen_recapture_flagged_across_grid_scales(grid_period):
    """Different phone-to-screen distances change the apparent grid period —
    this shouldn't be tuned to one specific spacing."""
    result = detect_screen_replay(to_bgr(screen_recapture_image(grid_period=grid_period)))
    assert result["is_screen"] is True


# ── Chromatic spectral divergence ────────────────────────────────────────────
# Real-data-informed second signal (see spoof_detection.py's module docstring):
# on an actual real-vs-spoof photo set, this caught spoofs the peak-count check
# missed. Synthetic coverage here can't reproduce THAT number, only prove the
# code path itself behaves as designed: identical channels -> no divergence,
# genuinely different per-channel content -> flagged, even with only one peak
# per channel (so it isn't just re-deriving the peak-count signal).

def channels_image(r_img, g_img, b_img):
    """Build a BGR image (cv2 channel order) from three independent 2D arrays."""
    return np.dstack([b_img, g_img, r_img])


def test_identical_channels_have_zero_chroma_diff():
    gray = ridge_image(period=10)
    result = detect_screen_replay(channels_image(gray, gray, gray))
    assert result["chroma_diff"] == pytest.approx(0.0, abs=1e-6)
    assert result["is_screen"] is False


def test_diverging_channel_spectra_flagged():
    """R and B carrying genuinely different frequency content (rather than the
    same pattern replicated across channels) should read as a large chroma_diff
    and flag the capture. It's expected that grayscale peak-count also fires
    here -- a synthetic difference this large shows up both ways -- the point
    is that chroma_diff itself is computed correctly from the split channels,
    not derived from (or accidentally always equal to) the peak-count path."""
    r_channel = ridge_image(period=8)
    b_channel = ridge_image(period=20)
    img = channels_image(r_channel, r_channel, b_channel)
    result = detect_screen_replay(img)
    assert result["chroma_diff"] > CHROMA_DIFF_THRESH
    assert result["is_screen"] is True


def test_grayscale_input_has_no_chroma_diff_but_still_works():
    gray = ridge_image(period=10)  # genuinely 2D, no channel dimension at all
    result = detect_screen_replay(gray)
    assert result["chroma_diff"] is None
    assert result["is_screen"] is False


# ── Degenerate input ──────────────────────────────────────────────────────────

def test_blank_image_not_flagged_but_marked_insufficient():
    blank = np.full((256, 256, 3), 128, dtype=np.uint8)
    result = detect_screen_replay(blank)
    assert result["is_screen"] is False
    assert result["note"] == "insufficient_signal"


def test_tiny_image_handled_gracefully():
    tiny = to_bgr(ridge_image(size=16))
    result = detect_screen_replay(tiny)
    assert result["is_screen"] is False
    assert result["note"] == "insufficient_signal"


def test_empty_image_handled_gracefully():
    result = detect_screen_replay(np.zeros((0, 0, 3), dtype=np.uint8))
    assert result["is_screen"] is False
    assert result["note"] == "insufficient_signal"
