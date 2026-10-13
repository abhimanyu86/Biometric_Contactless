"""
Align captures that SIFT couldn't register to any consensus reference, using the
minutiae themselves, the way a fingerprint matcher does.

For a capture (FingerNet minutiae) and a consensus template, every
(capture minutia, template minutia) pair proposes a rigid transform: rotation
= angle difference, translation from the positions. Scale is fixed at 1, since
both are at 500 DPI. The hypothesis matching the most template points within
STRICT_D px / STRICT_O degrees wins, and is refined by least squares on its
matched pairs.

Because the transform is fitted to FingerNet's own points, a high match count
could be chance. The acceptance threshold is therefore set from two null
distributions:
  * impostors: 30 other people's fingers (image_db_cropped), prepared the same way;
  * mirror: Rahul's own captures flipped left-right (same minutia density and
    ridge statistics, no true correspondence).
Accuracy is checked on captures that SIFT *did* register: how far the
minutiae-based transform moves points relative to the SIFT one.

    python align_by_minutiae.py      # writes ground_truth/consensus/minutiae_alignment.json

Transforms found this way are used only to propagate reviewed labels. They
never feed the consensus support counts, which would be circular.
"""
import csv
import glob
import json
import math
import os

import cv2
import numpy as np

import fingernet_consistency as fc

GT = fc.GT
CONS = os.path.join(GT, "consensus")
IMP = os.path.join(GT, "_impostors")
STRICT_D, STRICT_O = 8.0, math.radians(20)


def load_mnt(path):
    with open(path) as f:
        return np.array([[m["x"], m["y"], m["direction"]] for m in json.load(f)["minutiae"]]).reshape(-1, 3)


def score(a, t):
    """How many template points have a transformed capture point within tolerance (greedy)."""
    if len(a) == 0:
        return 0, []
    d = np.hypot(t[:, None, 0] - a[None, :, 0], t[:, None, 1] - a[None, :, 1])
    o = np.abs((t[:, None, 2] - a[None, :, 2] + np.pi) % (2 * np.pi) - np.pi)
    ok = (d <= STRICT_D) & (o <= STRICT_O)
    pairs, used = [], set()
    for j in np.argsort(d.min(1)):
        cand = [i for i in np.nonzero(ok[j])[0] if i not in used]
        if cand:
            i = min(cand, key=lambda i: d[j, i])
            used.add(i)
            pairs.append((i, j))
    return len(pairs), pairs


def apply(M, pts):
    return fc.transform(pts, M)


def align(cap, tmpl):
    if len(cap) == 0 or len(tmpl) == 0:
        return 0, None
    # every (capture i, template j) pair as a rigid hypothesis, scored all at once (loose count)
    I, J = np.meshgrid(np.arange(len(cap)), np.arange(len(tmpl)), indexing="ij")
    I, J = I.ravel(), J.ravel()
    r = tmpl[J, 2] - cap[I, 2]
    c, s = np.cos(r), np.sin(r)
    tx = tmpl[J, 0] - (c * cap[I, 0] - s * cap[I, 1])
    ty = tmpl[J, 1] - (s * cap[I, 0] + c * cap[I, 1])
    X = c[:, None] * cap[None, :, 0] - s[:, None] * cap[None, :, 1] + tx[:, None]      # H x N
    Y = s[:, None] * cap[None, :, 0] + c[:, None] * cap[None, :, 1] + ty[:, None]
    O = cap[None, :, 2] + r[:, None]
    loose = np.zeros(len(r), int)
    for j in range(len(tmpl)):                                                        # template point j covered?
        d = np.hypot(X - tmpl[j, 0], Y - tmpl[j, 1])
        o = np.abs((O - tmpl[j, 2] + np.pi) % (2 * np.pi) - np.pi)
        loose += ((d <= STRICT_D) & (o <= STRICT_O)).any(1)
    best = (0, None, [])
    for h in np.argsort(-loose)[:5]:                                                  # exact one-to-one on the top 5
        M = np.array([[c[h], -s[h], tx[h]], [s[h], c[h], ty[h]]])
        n, pairs = score(apply(M, cap), tmpl)
        if n > best[0]:
            best = (n, M, pairs)
    n, M, pairs = best
    if M is not None and len(pairs) >= 3:     # least-squares refinement on the matched pairs
        src = np.float32([cap[i, :2] for i, _ in pairs])
        dst = np.float32([tmpl[j, :2] for _, j in pairs])
        M2, _ = cv2.estimateAffinePartial2D(src, dst, method=cv2.LMEDS)
        if M2 is not None and 0.9 <= np.hypot(M2[0, 0], M2[1, 0]) <= 1.1:
            n2, pairs2 = score(apply(M2, cap), tmpl)
            if n2 >= n:
                n, M, pairs = n2, M2, pairs2
    return n, M


def main():
    refs = sorted(os.path.basename(p)[4:] for p in glob.glob(os.path.join(CONS, "ref_*")))
    tmpl = {r: load_mnt(os.path.join(CONS, "ref_" + r, "consensus_%s.json" % r)) for r in refs}
    sift_T = {}
    for r in refs:
        with open(os.path.join(CONS, "ref_" + r, "transforms.json")) as f:
            for n, M in json.load(f)["to_reference"].items():
                sift_T.setdefault(n, (r, np.array(M)))
    with open(os.path.join(GT, "manifest.csv"), newline="") as f:
        rows = [r for r in csv.DictReader(f) if r["status"] != "rejected"]
    names = [r["image"] for r in rows]
    width = {r["image"]: int(r["w"]) for r in rows}
    rest = [n for n in names if n not in sift_T]
    print("references:", refs, "| captures without a SIFT transform:", len(rest))

    # null 1: impostors
    imp = []
    for p in sorted(glob.glob(os.path.join(IMP, "prelabels", "*.json"))):
        cap = load_mnt(p)
        imp.append(max(align(cap, tmpl[r])[0] for r in refs))
    # null 2: Rahul mirrored
    mir = []
    for n in names[::3]:
        cap = load_mnt(os.path.join(GT, "prelabels", n + ".json"))
        m = cap.copy()
        m[:, 0] = width[n] - 1 - m[:, 0]
        m[:, 2] = (math.pi - m[:, 2]) % (2 * math.pi)
        mir.append(max(align(m, tmpl[r])[0] for r in refs))
    null_max = max(imp + mir)
    thresh = null_max + 3
    print("null: impostors max %d (p95 %.1f), mirrored max %d (p95 %.1f) -> accept >= %d matches" % (
        max(imp), np.percentile(imp, 95), max(mir), np.percentile(mir, 95), thresh))

    # accuracy vs SIFT, on captures SIFT registered (not the references themselves)
    errs, genuine = [], []
    for n, (r, Ms) in list(sift_T.items()):
        if n == r:
            continue
        cap = load_mnt(os.path.join(GT, "prelabels", n + ".json"))
        k, Mm = align(cap, tmpl[r])
        genuine.append(k)
        if Mm is not None:
            e = np.hypot(*(apply(Mm, cap)[:, :2] - apply(Ms, cap)[:, :2]).T)
            errs.append(float(np.median(e)))
    acc = [e for e, k in zip(errs, genuine) if k >= thresh]
    print("SIFT-registered captures: matches median %d (min %d); %d/%d pass threshold; "
          "median displacement vs SIFT among those %.1f px (p90 %.1f)" % (
              np.median(genuine), min(genuine), len(acc), len(genuine), np.median(acc), np.percentile(acc, 90)))

    out = {}
    for n in rest:
        cap = load_mnt(os.path.join(GT, "prelabels", n + ".json"))
        res = [(align(cap, tmpl[r]), r) for r in refs]
        (k, M), r = max(res, key=lambda t: t[0][0])
        out[n] = dict(reference=r, matches=int(k), accepted=bool(k >= thresh),
                      to_reference=M.tolist() if M is not None else None)
        print("%-4s best ref %-3s matches %2d %s" % (n, r, k, "ACCEPT" if k >= thresh else "reject"))
    summary = dict(threshold=thresh, null_impostor=imp, null_mirror=mir, genuine_sift_registered=genuine,
                   displacement_vs_sift_px=errs, accepted=sum(v["accepted"] for v in out.values()),
                   rest=len(rest), captures=out)
    with open(os.path.join(CONS, "minutiae_alignment.json"), "w") as f:
        json.dump(summary, f, indent=1)
    print("accepted %d of %d" % (summary["accepted"], len(rest)))


if __name__ == "__main__":
    main()
