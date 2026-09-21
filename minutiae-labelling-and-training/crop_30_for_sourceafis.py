"""
YOLO-crop the 30-person real-photo set (image_db gallery + test_db probes --
see siamese_embedding/dataset_30.py) into a flat folder SourceAFIS can read,
for the same reason the extractor pilot crops first: SourceAFIS's docs say
it "is not scale-invariant" and these are raw, uncontrolled phone photos
with lots of background -- feeding it uncropped would confound "can
SourceAFIS match" with "can SourceAFIS ignore a wall behind an out-of-focus
finger", which isn't the question being asked.

Output: sourceafis_input_30/Person{id}/gallery.png, probe1.png, probe2.png
"""
import os
import sys

import cv2

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
BACKEND_DIR = os.path.join(os.path.dirname(BASE_DIR), "backend")
sys.path.insert(0, BACKEND_DIR)
sys.path.insert(0, os.path.join(os.path.dirname(BASE_DIR), "siamese_embedding"))

from app import detect_and_crop_image  # noqa: E402
from dataset_30 import load_people  # noqa: E402

OUT_DIR = os.path.join(BASE_DIR, "sourceafis_input_30")
os.makedirs(OUT_DIR, exist_ok=True)


def crop_and_save(src_path, dst_path):
    img = cv2.imread(src_path)
    if img is None:
        print(f"  WARNING: could not read {src_path}")
        return False
    try:
        cropped, det_conf = detect_and_crop_image(img, conf=0.1)
    except ValueError as e:
        print(f"  WARNING: YOLO crop failed for {src_path} ({e})")
        return False
    gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
    cv2.imwrite(dst_path, gray)
    print(f"  {os.path.basename(src_path)} -> {cropped.shape[:2]} (det_conf={det_conf:.3f}) -> {dst_path}")
    return True


def main():
    people = load_people()
    print(f"Loaded {len(people)} people from image_db/test_db\n")

    ok_people = 0
    for person_id, info in sorted(people.items()):
        person_out = os.path.join(OUT_DIR, f"Person{person_id}")
        os.makedirs(person_out, exist_ok=True)
        print(f"=== Person{person_id} ===")

        gallery_ok = crop_and_save(info["gallery"], os.path.join(person_out, "gallery.png"))
        probe_ok_count = 0
        for i, probe_path in enumerate(info["probes"], 1):
            if crop_and_save(probe_path, os.path.join(person_out, f"probe{i}.png")):
                probe_ok_count += 1

        if gallery_ok and probe_ok_count > 0:
            ok_people += 1
        else:
            print(f"  WARNING: Person{person_id} unusable (gallery_ok={gallery_ok}, probes_ok={probe_ok_count})")

    print(f"\nDone: {ok_people}/{len(people)} people have a usable cropped gallery + at least one probe")


if __name__ == "__main__":
    main()
