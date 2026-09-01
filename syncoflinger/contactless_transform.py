"""
SynCoLFinGer's contactless transformation layer (Priesnitz, Rathgeb, Buchmann,
Busch -- Pattern Recognition Letters 157 (2022) 127-134), applied on top of a
clean SFinGe ridge pattern from Anguli.

f_i = r_cl + c_s + c_e_i, where:
  r_cl = d(r_cb)  -- contactless capturing simulation (Stage 1: deformation,
                     rotation, ridge thinning + blur)
  c_s             -- subject-related characteristics (Stage 2: low-contrast
                     regions, skin color, skin tone variation) -- persistent
                     across all mated samples of one finger
  c_e_i           -- environmental characteristics per sample i (Stage 3:
                     brightness/shadow, dirt, ridge inversion, camera noise,
                     slight rotation/scale) -- varies per mated sample

Anguli's raw output is black ridges on a white background (near-binary,
simulating a 500 dpi contact scan with no further processing) -- exactly the
r_cb this pipeline expects as input.

The paper's deformation formula didn't extract cleanly from the PDF (garbled
math), so Stage 1's warp is a principled reconstruction of the *described*
behavior (border-weighted inward pull, stronger at the top than the sides --
per Fig. 3/4 of the paper), not a byte-exact reproduction. Flagged inline
where a choice was made rather than transcribed.

Default skin tones are a small placeholder palette, not calibrated against
real data (the paper's own calibration used 25 real ISPFDv2 subjects, which
we don't have). Worth recalibrating once this project's own data collection
has real contactless captures -- see syncoflinger/README.md.
"""

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# Stage 1 -- contactless capturing simulation
# ---------------------------------------------------------------------------

def _border_weighted_warp(img, strength=14.0, top_bias=1.6):
    """Inward-pulling warp, stronger at the top than the sides -- simulates
    a contactless ridge pattern's natural finger contour, vs. SFinGe's
    flat/pressed simulated-contact look. See module docstring re: formula."""
    h, w = img.shape[:2]
    cx, cy = w / 2.0, h / 2.0

    ys, xs = np.mgrid[0:h, 0:w].astype(np.float32)
    dx = (xs - cx) / cx
    dy = (ys - cy) / cy

    weight_y = np.where(dy < 0, top_bias, 1.0)
    r2 = dx ** 2 + dy ** 2

    # Pull source sample points outward from center (== pushes image content
    # inward at the border) proportional to r^2, asymmetric top/side weight.
    map_x = xs + dx * r2 * strength * weight_y
    map_y = ys + dy * r2 * strength * weight_y

    return cv2.remap(img, map_x.astype(np.float32), map_y.astype(np.float32),
                      interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)


def _simulate_rolling(img, max_degrees=7.0, rng=None):
    """Slight rotation simulating finger presentation angle variance.

    The paper reuses its deformation algorithm with horizontally-facing
    vectors (a shear-like stretch/compress) to simulate rolling. A small
    real rotation achieves the same qualitative "slight tilt between
    captures" effect with far less code -- a deliberate simplification,
    not an oversight.

    Returns (image, M) -- M is needed to propagate minutiae through the
    same transform (see minutiae_ground_truth.warp_points_affine)."""
    rng = rng or np.random.default_rng()
    angle = rng.uniform(-max_degrees, max_degrees)
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), angle, 1.0)
    return cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REPLICATE), M


def _thin_and_blur(img, thin_kernel=3, blur_sigma=1.2, border_blur_sigma=6.0):
    """Ridge thinning + smoothing + out-of-focus border blur.

    Input convention: black ridges (low intensity) on white background.
    Thinning ridges == growing the white background == cv2.dilate on the
    grayscale image (dilate takes the local MAX, i.e. grows bright pixels),
    not erode -- erode would thicken the dark ridges instead."""
    k = np.ones((thin_kernel, thin_kernel), np.uint8)
    thinned = cv2.dilate(img, k, iterations=1)
    smoothed = cv2.GaussianBlur(thinned, (0, 0), sigmaX=blur_sigma)

    # Depth-of-field: sharp center, blurred border (radial blend).
    h, w = img.shape[:2]
    ys, xs = np.mgrid[0:h, 0:w].astype(np.float32)
    cx, cy = w / 2.0, h / 2.0
    r = np.sqrt(((xs - cx) / cx) ** 2 + ((ys - cy) / cy) ** 2)
    focus_mask = np.clip(1.0 - r, 0.0, 1.0) ** 0.5  # 1 at center, 0 at border

    heavily_blurred = cv2.GaussianBlur(smoothed, (0, 0), sigmaX=border_blur_sigma)
    out = smoothed.astype(np.float32) * focus_mask + heavily_blurred.astype(np.float32) * (1 - focus_mask)
    return np.clip(out, 0, 255).astype(np.uint8)


def contactless_capture_simulation(r_cb, rng=None, minutiae=None):
    """r_cl = d(r_cb) -- Stage 1 end to end.

    If `minutiae` (list of {x,y,direction,type} from r_cb) is given,
    propagates them through the same warp + rotation the image goes
    through and returns (image, transformed_minutiae) instead of just
    the image. Thinning/blur are intensity-only, so minutiae positions
    pass through those two steps unchanged."""
    warped = _border_weighted_warp(r_cb)
    rolled, M = _simulate_rolling(warped, rng=rng)
    out = _thin_and_blur(rolled)

    if minutiae is None:
        return out

    from minutiae_ground_truth import warp_points_border_weighted, warp_points_affine
    m = warp_points_border_weighted(minutiae, r_cb.shape)
    m = warp_points_affine(m, M)
    return out, m


# ---------------------------------------------------------------------------
# Stage 2 -- subject-related characteristics (persistent across mated samples)
# ---------------------------------------------------------------------------

# Placeholder skin-tone palette (RGB). NOT calibrated against real data --
# recalibrate from this project's own collected contactless captures once
# available, the way the paper calibrated from 25 real ISPFDv2 subjects.
DEFAULT_SKIN_TONES = [
    (255, 219, 172),
    (241, 194, 125),
    (224, 172, 105),
    (198, 134, 66),
    (141, 85, 36),
]


def _low_contrast_regions(img, rng, n_regions=(1, 3), max_radius_frac=0.18):
    """Simulates worn-out/dermatological low-contrast patches: local blur
    blended into a few random elliptical regions."""
    rng = rng or np.random.default_rng()
    h, w = img.shape[:2]
    out = img.copy().astype(np.float32)
    blurred = cv2.GaussianBlur(img, (0, 0), sigmaX=3.5).astype(np.float32)

    for _ in range(rng.integers(*n_regions, endpoint=True)):
        cx, cy = rng.uniform(0.2, 0.8) * w, rng.uniform(0.2, 0.8) * h
        rx = rng.uniform(0.06, max_radius_frac) * w
        ry = rng.uniform(0.06, max_radius_frac) * h
        mask = np.zeros((h, w), np.float32)
        cv2.ellipse(mask, (int(cx), int(cy)), (int(rx), int(ry)), 0, 0, 360, 1.0, -1)
        mask = cv2.GaussianBlur(mask, (0, 0), sigmaX=8.0)
        out = out * (1 - mask[..., None] if out.ndim == 3 else 1 - mask) + \
              blurred * (mask[..., None] if out.ndim == 3 else mask)

    return np.clip(out, 0, 255).astype(np.uint8)


def apply_subject_characteristics(r_cl, skin_rgb=None, rng=None):
    """c_s -- colorizes the (grayscale) ridge pattern with a skin tone and
    adds low-contrast regions + subtle local skin-tone variation. Returns a
    3-channel BGR image; persistent across all mated samples of one finger."""
    rng = rng or np.random.default_rng()
    skin_rgb = skin_rgb or DEFAULT_SKIN_TONES[rng.integers(0, len(DEFAULT_SKIN_TONES))]

    gray = _low_contrast_regions(r_cl, rng)
    intensity = gray.astype(np.float32) / 255.0  # 1.0 = valley (white), 0.0 = ridge (black)

    skin = np.array(skin_rgb[::-1], dtype=np.float32)  # RGB -> BGR
    # Ridges read darker than valleys against the base skin tone.
    colored = skin[None, None, :] * (0.55 + 0.45 * intensity[..., None])

    # Subtle local skin-tone/brightness variation (smoothed noise field).
    h, w = gray.shape
    noise = rng.normal(0, 1, (h // 8 + 1, w // 8 + 1)).astype(np.float32)
    noise = cv2.resize(noise, (w, h), interpolation=cv2.INTER_CUBIC)
    noise = cv2.GaussianBlur(noise, (0, 0), sigmaX=15.0)
    colored *= (1.0 + 0.06 * noise[..., None])

    return np.clip(colored, 0, 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# Stage 3 -- environmental characteristics (one call per mated sample)
# ---------------------------------------------------------------------------

def _shadow_mask(shape, rng):
    """Single-light-source shadow: linear gradient from a random angle."""
    h, w = shape[:2]
    angle = rng.uniform(0, 2 * np.pi)
    ys, xs = np.mgrid[0:h, 0:w].astype(np.float32)
    proj = (xs - w / 2.0) * np.cos(angle) + (ys - h / 2.0) * np.sin(angle)
    proj = proj / (max(h, w) / 2.0)
    mask = np.clip(0.5 + 0.5 * proj, 0.0, 1.0)  # 0 (dark side) .. 1 (lit side)
    return cv2.GaussianBlur(mask, (0, 0), sigmaX=25.0)


def _add_dirt(img, rng, n_specks=(3, 10)):
    out = img.copy()
    h, w = img.shape[:2]
    for _ in range(rng.integers(*n_specks, endpoint=True)):
        x, y = rng.integers(0, w), rng.integers(0, h)
        r = rng.integers(1, 4)
        val = int(rng.uniform(0, 80)) if rng.random() < 0.5 else int(rng.uniform(180, 255))
        color = (val, val, val) if out.ndim == 3 else val
        cv2.circle(out, (x, y), r, color, -1)
    return out


def apply_environmental(img, rng=None, noise_level=1.0, minutiae=None):
    """c_e_i -- one mated sample's worth of environmental variation:
    brightness/shadow, ridge-line inversion under shadow, dirt, camera
    noise, slight rotation/scale (segmentation-imprecision simulation).

    If `minutiae` is given, returns (image, transformed_minutiae) --
    only the final rotation/scale step is geometric; everything else
    here is intensity-only and leaves positions unchanged."""
    rng = rng or np.random.default_rng()
    h, w = img.shape[:2]
    shadow = _shadow_mask(img.shape, rng)

    base = img.astype(np.float32)
    is_color = base.ndim == 3

    # Ridge-line inversion must act on luminance only -- inverting raw BGR
    # channels flips hue too (skin's low-blue/high-red channels invert into
    # their complementary blue/orange), not just brightness. Derive a
    # luminance proxy, invert *that*, then reapply as a per-pixel brightness
    # ratio so skin color is preserved and only contrast polarity changes.
    luma = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32) if is_color else base
    inverted_luma = 255.0 - luma

    # Blending luma with (255-luma) at weight ~0.5 mathematically collapses
    # to flat mid-grey regardless of original contrast -- sharpen the
    # transition so most of the image commits to one polarity, with only a
    # narrow band actually blending.
    shadow_sharp = np.clip((shadow - 0.5) * 6.0 + 0.5, 0.0, 1.0)
    blended_luma = luma * shadow_sharp + inverted_luma * (1 - shadow_sharp)

    if is_color:
        ratio = ((blended_luma + 1.0) / (luma + 1.0))[..., None]
        blended = base * ratio
    else:
        blended = blended_luma

    # Brightness/shadow multiply (mild, using the *unsharpened* mask so
    # overall lighting still grades smoothly) + global brightness/color jitter.
    lit_mask = shadow[..., None] if is_color else shadow
    brightness_jitter = rng.uniform(0.85, 1.15)
    color_jitter = rng.uniform(0.95, 1.05, size=3) if is_color else 1.0
    lit = blended * (0.8 + 0.2 * lit_mask) * brightness_jitter * color_jitter

    # Camera noise, heavier in darker (shadowed) regions.
    noise_amount = (1.0 - lit_mask) * 18.0 * noise_level + 4.0 * noise_level
    noise = rng.normal(0, 1, base.shape).astype(np.float32) * noise_amount
    noisy = lit + noise

    out = np.clip(noisy, 0, 255).astype(np.uint8)
    out = _add_dirt(out, rng, n_specks=(int(3 * noise_level), int(10 * noise_level) + 1))

    # Slight rotation/scale, simulating segmentation imprecision.
    angle = rng.uniform(-2.0, 2.0)
    scale = rng.uniform(0.97, 1.03)
    M = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), angle, scale)
    out = cv2.warpAffine(out, M, (w, h), borderMode=cv2.BORDER_REPLICATE)

    if minutiae is None:
        return out

    from minutiae_ground_truth import warp_points_affine
    return out, warp_points_affine(minutiae, M)


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

QUALITY_PRESETS = {
    "high": dict(noise_level=0.5),
    "medium": dict(noise_level=1.0),
    "low": dict(noise_level=2.0),
}


def generate_contactless_samples(sfinge_image_path, n_samples=5, quality="high", seed=None,
                                  with_ground_truth=True):
    """f_i = r_cl + c_s + c_e_i for i in 1..n_samples, from one Anguli/SFinGe
    ridge pattern.

    If with_ground_truth (default): extracts minutiae from r_cb (see
    minutiae_ground_truth.extract_minutiae_skeleton) and propagates them
    through every geometric transform alongside the image. Returns a list
    of (image, minutiae) tuples -- each sample's minutiae correctly
    aligned to *that* sample's framing/rotation, not just r_cb's.
    If False, returns a plain list of images (old behavior)."""
    rng = np.random.default_rng(seed)
    r_cb = cv2.imread(sfinge_image_path, cv2.IMREAD_GRAYSCALE)
    if r_cb is None:
        raise FileNotFoundError(sfinge_image_path)

    preset = QUALITY_PRESETS[quality]

    if not with_ground_truth:
        r_cl = contactless_capture_simulation(r_cb, rng=rng)
        subject_base = apply_subject_characteristics(r_cl, rng=rng)
        return [apply_environmental(subject_base, rng=rng, **preset) for _ in range(n_samples)]

    from minutiae_ground_truth import extract_minutiae_skeleton
    minutiae_r_cb = extract_minutiae_skeleton(r_cb)

    r_cl, minutiae_r_cl = contactless_capture_simulation(r_cb, rng=rng, minutiae=minutiae_r_cb)
    subject_base = apply_subject_characteristics(r_cl, rng=rng)  # no geometric change

    results = []
    for _ in range(n_samples):
        img, m = apply_environmental(subject_base, rng=rng, minutiae=minutiae_r_cl, **preset)
        results.append((img, m))
    return results


if __name__ == "__main__":
    import sys
    import os

    src = sys.argv[1] if len(sys.argv) > 1 else \
        "syncoflinger/anguli/test_output/Fingerprints/fp_1/1.png"
    outdir = sys.argv[2] if len(sys.argv) > 2 else "syncoflinger/test_output"
    os.makedirs(outdir, exist_ok=True)

    from minutiae_ground_truth import draw_minutiae
    import json

    for quality in ("high", "medium", "low"):
        samples = generate_contactless_samples(src, n_samples=3, quality=quality, seed=42)
        for i, (img, minutiae) in enumerate(samples, 1):
            cv2.imwrite(os.path.join(outdir, f"{quality}_{i}.png"), img)
            with open(os.path.join(outdir, f"{quality}_{i}_minutiae.json"), "w") as f:
                json.dump(minutiae, f, indent=2)
            overlay = draw_minutiae(img, minutiae)
            cv2.imwrite(os.path.join(outdir, f"{quality}_{i}_overlay.png"), overlay)
    print(f"Wrote 9 samples + ground-truth minutiae + overlays (3 quality tiers x 3 mated samples) to {outdir}")
