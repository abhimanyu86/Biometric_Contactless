"""Builds a small labeled synthetic contactless fingerprint dataset:
N distinct fingers (Anguli) x M mated samples each (contactless_transform),
each with propagated ground-truth minutiae. Organized for direct use as
Stage B training-target input.
"""
import os
import sys
import json
import glob

import cv2

sys.path.insert(0, os.path.dirname(__file__))
from contactless_transform import generate_contactless_samples  # noqa: E402

BASE_DIR = os.path.join(os.path.dirname(__file__), "dataset_base")
OUT_DIR = os.path.join(os.path.dirname(__file__), "dataset")

# (quality, count) per finger -- 5 samples/finger, skewed toward higher
# quality (realistic capture mix), matching the paper's tiered evaluation.
SAMPLE_PLAN = [("high", 2), ("medium", 2), ("low", 1)]


def main():
    os.makedirs(os.path.join(OUT_DIR, "images"), exist_ok=True)
    os.makedirs(os.path.join(OUT_DIR, "minutiae"), exist_ok=True)

    base_images = sorted(glob.glob(os.path.join(BASE_DIR, "*.png")),
                          key=lambda p: int(os.path.splitext(os.path.basename(p))[0]))

    manifest = []
    total_minutiae = 0

    for finger_idx, base_path in enumerate(base_images, 1):
        finger_id = f"finger{finger_idx:02d}"
        sample_num = 0
        for quality, count in SAMPLE_PLAN:
            samples = generate_contactless_samples(
                base_path, n_samples=count, quality=quality, seed=1000 * finger_idx + sample_num
            )
            for img, minutiae in samples:
                sample_num += 1
                name = f"{finger_id}_{quality}_{sample_num:02d}"
                img_path = os.path.join(OUT_DIR, "images", f"{name}.png")
                mnt_path = os.path.join(OUT_DIR, "minutiae", f"{name}.json")
                cv2.imwrite(img_path, img)
                with open(mnt_path, "w") as f:
                    json.dump(minutiae, f, indent=2)

                total_minutiae += len(minutiae)
                manifest.append({
                    "name": name,
                    "finger_id": finger_id,
                    "base_fingerprint": os.path.basename(base_path),
                    "quality": quality,
                    "image": f"images/{name}.png",
                    "minutiae": f"minutiae/{name}.json",
                    "minutiae_count": len(minutiae),
                })

    with open(os.path.join(OUT_DIR, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"Wrote {len(manifest)} samples across {len(base_images)} distinct fingers to {OUT_DIR}")
    print(f"Total minutiae: {total_minutiae} (avg {total_minutiae / len(manifest):.1f} per image)")


if __name__ == "__main__":
    main()
