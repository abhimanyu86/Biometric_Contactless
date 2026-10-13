"""
31-identity gallery test on 500 DPI images: FingerNet vs MinutiaeNet, two matchers.

Gallery (enrolment): Rahul (image_db/Rahul.jpg, cropped by the backend's YOLO) +
Person1..30 (image_db_cropped/). Probes: the 106 Rahul captures
(ground_truth/images_500dpi, test_db/Rahul_cropped). 31 x 106 = 3286 comparisons:
106 genuine (Rahul vs Rahul) + 3180 impostor.

Every image is prepared the same way: grayscale, scaled to 500 DPI (ridge period
10 px), CLAHE. Probes use the centre-patch scale already in ground_truth/manifest.csv.
The gallery uses the same centre-patch measurement where it works; otherwise the
multi-patch estimate (scale_estimate.py), divided by the 1.09 bias it shows against
the verified Rahul scales; otherwise the median gallery scale. The method is
recorded per image. Both extractors see exactly these images:
  * FingerNet (fingernet/fingernet_tf2.py); no minutia type, so all points are
    "RIG" and matching.py's type filter can't help it;
  * MinutiaeNet (backend detect_minutiae, the model the product pipeline uses).
Matchers:
  * backend/matching.py match_templates, algorithm "modern" (as in
    minutiaenet-testing/with-preprocessing/run_gallery_match.py);
  * NIST BOZORTH3 (NBIS, run in WSL), an independent reference matcher.
Metrics: verification ROC (EER, TAR at FAR 1%) and rank-1 identification
(is Rahul the top score among the 31 for each probe?). Sensitivity: FingerNet
gallery minutiae rescaled x0.8 and x1.25 about the image centre, i.e. what a
gallery scale error would do.

    python eval_gallery.py        # writes ground_truth/eval_gallery/
"""
import csv
import json
import math
import os
import subprocess
import sys
import time

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
GT = os.path.join(HERE, "ground_truth")
OUT = os.path.join(GT, "eval_gallery")
sys.path.insert(0, os.path.join(HERE, "fingernet"))
sys.path.insert(0, os.path.join(REPO, "minutiaenet-testing"))   # common.py: model paths/env for the backend
sys.path.insert(0, os.path.join(REPO, "backend"))

import ground_truth_tool as gtt  # noqa: E402
import scale_estimate as se  # noqa: E402

MULTI_PATCH_BIAS = 1.09
BOZORTH = "~/nbis-build/src/bozorth3/bin/bozorth3"


def to_wsl(p):
    p = os.path.abspath(p)
    return "/mnt/%s%s" % (p[0].lower(), p[2:].replace("\\", "/"))


def prepare_gallery():
    srcs = [("Rahul", os.path.join(GT, "_enroll_rahul", "Rahul.png"))]
    srcs += [("Person%d" % i, os.path.join(REPO, "image_db_cropped", "Person%d.jpg" % i)) for i in range(1, 31)]
    # squeeze: the backend's app module wraps cv2.imread and returns grayscale as (h, w, 1)
    grays = {n: np.squeeze(cv2.imread(p, cv2.IMREAD_GRAYSCALE)) for n, p in srcs}
    scales = {}
    for n, g in grays.items():
        s = gtt.measure_scale(g)
        if s:
            scales[n] = (s, "centre_patch")
        else:
            s2 = se.estimate(g)[0]
            scales[n] = (s2 / MULTI_PATCH_BIAS, "multi_patch") if s2 else (None, "median")
    med = float(np.median([s for s, _ in scales.values() if s]))
    # Plausibility check against finger size. At 500 DPI Rahul's fingertip crops have a
    # short side of ~260-400 px. A ridge-period estimate that would put a finger outside
    # 200-450 px locked onto something else (blurry or low-res photos where the ridges are
    # barely resolved), so fall back to matching the median probe crop area.
    with open(os.path.join(GT, "manifest.csv"), newline="") as f:
        probe_area = float(np.median([int(r["w"]) * int(r["h"]) for r in csv.DictReader(f)]))
    for n, g in grays.items():
        s, m = scales[n]
        short = min(g.shape) * (s if s else med)
        if s is None or not (200 <= short <= 450):
            scales[n] = (math.sqrt(probe_area / (g.shape[0] * g.shape[1])), "size_prior")
    imgs, meta = {}, {}
    od = os.path.join(OUT, "gallery_500dpi")
    os.makedirs(od, exist_ok=True)
    for n, g in grays.items():
        s, m = scales[n]
        if s is None:
            s = med
        img = gtt.CLAHE.apply(cv2.resize(g, None, fx=s, fy=s, interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_CUBIC))
        cv2.imwrite(os.path.join(od, n + ".png"), img)
        imgs[n] = img
        meta[n] = dict(scale=round(s, 4), method=m, w=img.shape[1], h=img.shape[0])
    return imgs, meta


def fingernet_minutiae(net, run_image, img):
    return [dict(x=float(x), y=float(y), direction=float(o), type="RIG", confidence=float(s))
            for x, y, o, s in run_image(net, img)["mnt"]]


def rescale(mnts, img, f):
    cy, cx = img.shape[0] / 2, img.shape[1] / 2
    return [dict(m, x=cx + (m["x"] - cx) * f, y=cy + (m["y"] - cy) * f) for m in mnts]


def write_xyt(path, mnts, h):
    # NBIS xyt: origin bottom-left, angle counter-clockwise in degrees
    with open(path, "w") as f:
        for m in mnts:
            f.write("%d %d %d %d\n" % (round(m["x"]), round(h - 1 - m["y"]),
                                       round(math.degrees(-m["direction"])) % 360,
                                       max(1, min(100, round(100 * m.get("confidence", 1.0))))))


def bozorth_scores(xyt_dir, probes, gallery):
    """score[g][p] via BOZORTH3 in WSL, one call per probe against the whole gallery."""
    out = {g: {} for g in gallery}
    gfiles = " ".join("%s/g_%s.xyt" % (to_wsl(xyt_dir), g) for g in gallery)
    for p in probes:
        cmd = "%s -A outfmt=s -p %s/p_%s.xyt %s" % (BOZORTH, to_wsl(xyt_dir), p, gfiles)
        res = subprocess.run(["wsl", "-d", "Ubuntu", "--", "bash", "-lc", cmd], capture_output=True, text=True)
        vals = [int(v) for v in res.stdout.split()]
        if len(vals) != len(gallery):
            raise RuntimeError("bozorth3 output mismatch for probe %s: %r %r" % (p, res.stdout[:200], res.stderr[:200]))
        for g, v in zip(gallery, vals):
            out[g][p] = float(v)
    return out


def metrics(S, probes, gallery, genuine="Rahul"):
    gen = np.array([S[genuine][p] for p in probes])
    imp = np.array([S[g][p] for g in gallery if g != genuine for p in probes])
    thr = np.unique(np.concatenate([gen, imp]))
    far = np.array([(imp >= t).mean() for t in thr])
    frr = np.array([(gen < t).mean() for t in thr])
    k = int(np.argmin(np.abs(far - frr)))
    eer = float((far[k] + frr[k]) / 2)
    ok = far <= 0.01
    tar1 = float(1 - frr[ok].min()) if ok.any() else 0.0
    thr1 = float(thr[ok][np.argmin(frr[ok])]) if ok.any() else None
    rank1 = 0
    for p in probes:
        s = S[genuine][p]
        if s > 0 and all(s > S[g][p] for g in gallery if g != genuine):   # strict: ties are not a win
            rank1 += 1
    return dict(eer=round(eer, 4), eer_threshold=float(thr[k]), tar_at_far_1pct=round(tar1, 4),
                threshold_at_far_1pct=thr1, rank1_identification=round(rank1 / len(probes), 4),
                rank1_correct=rank1, genuine_mean=round(float(gen.mean()), 4), impostor_mean=round(float(imp.mean()), 4),
                genuine=gen.tolist(), impostor_n=len(imp))


def main():
    os.makedirs(OUT, exist_ok=True)
    t0 = time.time()
    import common  # noqa: F401  (sets SINGLE_MODELS_DIR / MINUTIAE_MODEL_PATH)
    import app as app_module
    from matching import match_templates
    from fingernet_tf2 import FingerNet, run_image
    app_module.load_models()
    net = FingerNet()

    gal_imgs, gal_meta = prepare_gallery()
    gallery = list(gal_imgs)
    with open(os.path.join(GT, "manifest.csv"), newline="") as f:
        probes = [r["image"] for r in csv.DictReader(f)]        # all 106, incl. rejected 57 (a failed capture)
    # squeeze: the backend's app module wraps cv2.imread and returns grayscale as (h, w, 1)
    probe_imgs = {p: np.squeeze(cv2.imread(os.path.join(GT, "images_500dpi", p + ".png"), cv2.IMREAD_GRAYSCALE))
                  for p in probes}
    print("gallery", len(gallery), "probes", len(probes))
    print("gallery scale methods:", {m: sum(v["method"] == m for v in gal_meta.values())
                                     for m in ("centre_patch", "multi_patch", "size_prior")})

    T = {"fingernet": {}, "minutiaenet": {}}
    for kind, imgs in (("g", gal_imgs), ("p", probe_imgs)):
        for n, img in imgs.items():
            T["fingernet"][(kind, n)] = fingernet_minutiae(net, run_image, img)
            T["minutiaenet"][(kind, n)] = app_module.detect_minutiae(img)
    counts = {e: dict(gallery_median=float(np.median([len(T[e][("g", g)]) for g in gallery])),
                      probe_median=float(np.median([len(T[e][("p", p)]) for p in probes])),
                      empty=sum(1 for k, v in T[e].items() if not v)) for e in T}
    print("minutiae counts:", counts)

    # "modern" normalises template scale internally, so scale sensitivity is tested with BOZORTH3
    variants = [("fingernet", "modern", 1.0), ("minutiaenet", "modern", 1.0),
                ("fingernet", "bozorth3", 1.0), ("minutiaenet", "bozorth3", 1.0),
                ("fingernet", "bozorth3", 0.8), ("fingernet", "bozorth3", 1.25)]
    results, scores = {}, {}
    for ext, matcher, f in variants:
        gal_t = {g: T[ext][("g", g)] for g in gallery}
        if f != 1.0:   # simulate an impostor-gallery scale error (Rahul's enrolment untouched)
            gal_t = {g: (t if g == "Rahul" else rescale(t, gal_imgs[g], f)) for g, t in gal_t.items()}
        if matcher == "modern":
            S = {g: {p: float(match_templates(gal_t[g], T[ext][("p", p)])) for p in probes} for g in gallery}
        else:
            xd = os.path.join(OUT, "xyt_%s" % ext)
            os.makedirs(xd, exist_ok=True)
            for g in gallery:
                write_xyt(os.path.join(xd, "g_%s.xyt" % g), gal_t[g], gal_imgs[g].shape[0])
            for p in probes:
                write_xyt(os.path.join(xd, "p_%s.xyt" % p), T[ext][("p", p)], probe_imgs[p].shape[0])
            S = bozorth_scores(xd, probes, gallery)
        key = "%s+%s%s" % (ext, matcher, "" if f == 1.0 else " (impostor gallery x%.2f)" % f)
        enrolled = [g for g in gallery if gal_t[g]]              # failure-to-enrol: empty template
        results[key] = metrics(S, probes, gallery)
        results[key]["failed_to_enrol"] = [g for g in gallery if g not in enrolled]
        results[key]["enrolled_only"] = {k: v for k, v in metrics(S, probes, enrolled).items() if k != "genuine"}             if "Rahul" in enrolled else None
        scores[key] = S
        r = results[key]
        print("%-45s EER %5.1f%%  TAR@FAR1%% %5.1f%%  rank-1 %5.1f%% (%d/%d)  FTE %d" % (
            key, 100 * r["eer"], 100 * r["tar_at_far_1pct"], 100 * r["rank1_identification"], r["rank1_correct"],
            len(probes), len(r["failed_to_enrol"])))
        if r["failed_to_enrol"] and r["enrolled_only"]:
            e = r["enrolled_only"]
            print("%-45s EER %5.1f%%  TAR@FAR1%% %5.1f%%  rank-1 %5.1f%%  (only the %d enrolled identities)" % (
                "   enrolled-only", 100 * e["eer"], 100 * e["tar_at_far_1pct"], 100 * e["rank1_identification"],
                len(gallery) - len(r["failed_to_enrol"])))

    with open(os.path.join(OUT, "score_matrix.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["variant", "gallery", "probe", "pair_type", "score"])
        for key, S in scores.items():
            for g in gallery:
                for p in probes:
                    w.writerow([key, g, p, "genuine" if g == "Rahul" else "impostor", S[g][p]])
    with open(os.path.join(OUT, "results.json"), "w") as f:
        json.dump(dict(gallery_meta=gal_meta, minutiae_counts=counts,
                       results={k: {kk: vv for kk, vv in v.items() if kk != "genuine"} for k, v in results.items()},
                       seconds=round(time.time() - t0)), f, indent=1)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    main_keys = [k for k in results if "gallery x" not in k]
    fig, axes = plt.subplots(1, len(main_keys), figsize=(4.2 * len(main_keys), 3.4))
    for ax, k in zip(axes, main_keys):
        S = scores[k]
        gen = [S["Rahul"][p] for p in probes]
        imp = [S[g][p] for g in gallery if g != "Rahul" for p in probes]
        bins = np.linspace(min(gen + imp), max(gen + imp) + 1e-9, 40)
        ax.hist(imp, bins=bins, density=True, alpha=0.6, label="impostor (3180)", color="#c0504d")
        ax.hist(gen, bins=bins, density=True, alpha=0.6, label="genuine (106)", color="#4f81bd")
        r = results[k]
        ax.set_title("%s\nEER %.1f%%, rank-1 %.0f%%" % (k, 100 * r["eer"], 100 * r["rank1_identification"]), fontsize=9)
        ax.set_xlabel("match score")
        ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "score_distributions.png"), dpi=110)
    sys.stdout.flush()
    print("done in %.0fs -> %s" % (time.time() - t0, OUT))


if __name__ == "__main__":
    main()
