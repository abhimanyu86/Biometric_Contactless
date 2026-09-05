"""
Ground-truth minutiae extraction for Anguli's raw synthetic ridge patterns,
plus point-transform companions for every geometric operation in
contactless_transform.py, so minutiae stay correctly aligned with the image
through the full contactless-simulation pipeline.

WHY THIS EXISTS: Anguli (unlike the original SFinGe) doesn't track or export
minutiae internally -- confirmed by grepping the whole vendored source for
"minutia"/"bifurcation"/etc: zero matches. Its "-meta" output is pattern
type + core/delta only. So minutiae have to be *extracted*, not read out.

Anguli's raw output is a clean, near-binary ridge pattern -- much cleaner
than any real or even enhanced-photo capture. That's exactly the condition
where classical skeleton + crossing-number extraction is reliable: no deep
model, no MINDTCT noise, a deterministic algorithm on an unambiguous input.
Output format matches this project's {x, y, direction, type: RIG|BIF}
contract (see backend/app.py's detect_minutiae).
"""

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# Skeleton + crossing-number minutiae extraction
# ---------------------------------------------------------------------------

# 8-neighborhood in a fixed cyclic (clockwise) order for crossing-number.
_CN_OFFSETS = [(-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1)]


def extract_minutiae_skeleton(img_gray, border_margin=28, spur_min_dist=8, direction_radius=9):
    """Ridge endings (RIG) and bifurcations (BIF) via skeleton + crossing
    number. Input: grayscale, black ridges on white background (Anguli's
    raw convention). Returns a list of {x, y, direction, type} dicts,
    direction in radians via atan2(sin, cos) as elsewhere in this project.
    """
    # Binarize: ridge (foreground) = 1, background = 0.
    _, binary = cv2.threshold(img_gray, 128, 1, cv2.THRESH_BINARY_INV)
    skeleton = cv2.ximgproc.thinning((binary * 255).astype(np.uint8),
                                      thinningType=cv2.ximgproc.THINNING_ZHANGSUEN)
    skel = (skeleton > 0).astype(np.uint8)

    h, w = skel.shape
    ys, xs = np.nonzero(skel)

    raw_points = []
    for y, x in zip(ys, xs):
        if x < border_margin or x >= w - border_margin or y < border_margin or y >= h - border_margin:
            continue
        neighbors = []
        for dy, dx in _CN_OFFSETS:
            ny, nx = y + dy, x + dx
            # int(), not the raw numpy uint8 -- uint8 subtraction wraps
            # around (0 - 1 == 255), silently corrupting the CN count.
            neighbors.append(int(skel[ny, nx]) if 0 <= ny < h and 0 <= nx < w else 0)
        cn = sum(abs(neighbors[i] - neighbors[(i + 1) % 8]) for i in range(8)) // 2
        if cn == 1:
            raw_points.append((x, y, "RIG"))
        elif cn == 3:
            raw_points.append((x, y, "BIF"))

    # Spur removal: skeleton thinning artifacts cluster spurious minutiae
    # close together -- keep only one per cluster.
    filtered = []
    for x, y, t in raw_points:
        if all((x - fx) ** 2 + (y - fy) ** 2 >= spur_min_dist ** 2 for fx, fy, _ in filtered):
            filtered.append((x, y, t))

    minutiae = []
    for x, y, t in filtered:
        direction = _local_direction(skel, x, y, direction_radius)
        if direction is None:
            continue
        minutiae.append({"x": int(x), "y": int(y), "direction": float(direction), "type": t})

    return minutiae


def _local_direction(skel, x, y, radius):
    """Average direction to nearby skeleton pixels -- for a ridge ending,
    this is the single branch's direction (pointing into the ridge, away
    from the termination); for a bifurcation, the mean of the outgoing
    branches. Roughly-right convention, consistent within this dataset --
    matches this project's existing +-20-30deg consensus tolerance."""
    h, w = skel.shape
    y0, y1 = max(0, y - radius), min(h, y + radius + 1)
    x0, x1 = max(0, x - radius), min(w, x + radius + 1)
    patch = skel[y0:y1, x0:x1]
    pys, pxs = np.nonzero(patch)
    if len(pys) <= 1:
        return None
    dys = (pys + y0 - y).astype(np.float64)
    dxs = (pxs + x0 - x).astype(np.float64)
    dist = np.sqrt(dys ** 2 + dxs ** 2)
    keep = dist > 0.5
    if not np.any(keep):
        return None
    angles = np.arctan2(dys[keep], dxs[keep])
    sin_mean = np.mean(np.sin(angles))
    cos_mean = np.mean(np.cos(angles))
    if abs(sin_mean) < 1e-9 and abs(cos_mean) < 1e-9:
        return None
    return float(np.arctan2(sin_mean, cos_mean))


def draw_minutiae(img, minutiae, radius=4, length=14):
    """Debug overlay -- green=RIG, yellow=BIF, matching this project's
    existing visualization convention (create_visualization in app.py)."""
    out = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR) if img.ndim == 2 else img.copy()
    for m in minutiae:
        color = (0, 255, 0) if m["type"] == "RIG" else (0, 255, 255)
        x, y = m["x"], m["y"]
        cv2.circle(out, (x, y), radius, color, -1)
        x2 = int(x + length * np.cos(m["direction"]))
        y2 = int(y + length * np.sin(m["direction"]))
        cv2.line(out, (x, y), (x2, y2), color, 2)
    return out


# ---------------------------------------------------------------------------
# Point-transform companions for contactless_transform.py's geometric stages
# ---------------------------------------------------------------------------

def warp_points_border_weighted(minutiae, shape, strength=14.0, top_bias=1.6, n_iter=6):
    """Forward-transforms points through _border_weighted_warp's backward
    (remap-style) displacement field via fixed-point iteration -- the field
    is smooth and displacements are modest, so this converges quickly.
    Direction is updated via the warp's local Jacobian (finite difference).
    """
    h, w = shape[:2]
    cx, cy = w / 2.0, h / 2.0

    def backward_map(x, y):
        dx = (x - cx) / cx
        dy = (y - cy) / cy
        weight_y = np.where(dy < 0, top_bias, 1.0)
        r2 = dx ** 2 + dy ** 2
        return x + dx * r2 * strength * weight_y, y + dy * r2 * strength * weight_y

    def invert_point(mx, my):
        ox, oy = mx, my
        for _ in range(n_iter):
            bx, by = backward_map(ox, oy)
            ox += mx - bx
            oy += my - by
        return ox, oy

    out = []
    eps = 1.5
    for m in minutiae:
        ox, oy = invert_point(m["x"], m["y"])
        if not (0 <= ox < w and 0 <= oy < h):
            continue
        # Local Jacobian via finite difference along current direction.
        dxu, dyu = np.cos(m["direction"]), np.sin(m["direction"])
        ox2, oy2 = invert_point(m["x"] + eps * dxu, m["y"] + eps * dyu)
        new_dir = np.arctan2(oy2 - oy, ox2 - ox)
        out.append({"x": int(round(ox)), "y": int(round(oy)), "direction": float(new_dir), "type": m["type"]})
    return out


def warp_points_affine(minutiae, M):
    """Forward-transforms points through a cv2.warpAffine matrix M --
    OpenCV's default convention (no WARP_INVERSE_MAP) treats M as the
    forward src->dst mapping, so direct application is correct here,
    unlike the remap-based border warp above."""
    angle_rad = np.arctan2(M[1, 0], M[0, 0])  # rotation component of M
    out = []
    for m in minutiae:
        vec = np.array([m["x"], m["y"], 1.0])
        nx, ny = M @ vec
        out.append({"x": int(round(nx)), "y": int(round(ny)),
                    "direction": float(m["direction"] + angle_rad), "type": m["type"]})
    return out
