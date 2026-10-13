"""
Robust ridge-period estimate for choosing a 500 DPI scale factor.

ground_truth_tool.measure_scale looks at one centre patch, which fails when the
finger is diagonal or small in its crop, or the photo is soft. This version
scans a grid of patches over a working copy, measures each one's radial-spectrum
peak and how prominent it is, and takes the median period of the most prominent
patches. Returns (scale, period_at_work_scale, prominence), or (None, ...) when
no patch shows a clear ridge frequency.
"""
import cv2
import numpy as np

from fir import TARGET_RIDGE_PERIOD_PX

WORK_MAX_DIM = 800
WHITEN = 0.0   # spectrum weight exponent; chosen by validation against the verified Rahul scales
CLAHE = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))


def patch_peak(p, lo_period=4.0, hi_period=30.0):
    p = p.astype(np.float32)
    p = p - cv2.GaussianBlur(p, (0, 0), 10)
    n = p.shape[0]
    spec = np.abs(np.fft.fftshift(np.fft.fft2(p * np.outer(np.hanning(n), np.hanning(n)))))
    yy, xx = np.mgrid[0:n, 0:n]
    rad = np.hypot(yy - n // 2, xx - n // 2).astype(int)
    radial = np.bincount(rad.ravel(), spec.ravel()) / np.maximum(np.bincount(rad.ravel()), 1)
    lo, hi = max(2, int(n / hi_period)), min(len(radial) - 1, int(n / lo_period))
    # Natural images fall off roughly as 1/f, so without whitening the lowest bin
    # (shading, the finger outline) always wins. Multiply by radius to flatten it.
    band = radial[lo:hi] * np.arange(lo, hi) ** WHITEN
    if band.size < 3:
        return None, 0.0
    k = int(np.argmax(band))
    if k == 0 or k == band.size - 1:          # peak on the band edge: no real ridge frequency
        return None, 0.0
    a, b, c = band[k - 1], band[k], band[k + 1]
    off = 0.5 * (a - c) / (a - 2 * b + c) if (a - 2 * b + c) != 0 else 0.0   # sub-bin peak
    return n / (k + lo + off), float(b / (np.median(band) + 1e-9))


def estimate(gray, min_prominence=2.0, top=3):
    s0 = min(1.0, WORK_MAX_DIM / max(gray.shape))
    work = CLAHE.apply(cv2.resize(gray, None, fx=s0, fy=s0, interpolation=cv2.INTER_AREA))
    h, w = work.shape
    # large patches (as in the centre-patch method) keep the frequency resolution fine;
    # scanning several of them handles fingers that are off-centre or diagonal in the crop
    patch = max(64, int(min(h, w) * 0.5))
    found = []
    for y in range(0, h - patch + 1, patch // 2):
        for x in range(0, w - patch + 1, patch // 2):
            per, prom = patch_peak(work[y:y + patch, x:x + patch])
            if per and prom >= min_prominence:
                found.append((prom, per))
    if not found:
        return None, None, 0.0
    found.sort(reverse=True)
    best = found[:top]
    period = float(np.median([p for _, p in best]))
    return s0 * TARGET_RIDGE_PERIOD_PX / period, period, float(np.median([q for q, _ in best]))
