"""
Full FAR/FRR/EER for the 30-person dataset using the YOLO-cropped images
already saved in image_db_cropped/ and test_db_cropped/ (from
test_30_cropped.py). Unlike that script (which only checked genuine scores
against the old uncropped EER threshold), this computes genuine AND
impostor pairs fresh on the cropped data and sweeps for a real EER --
a proper accuracy number for the cropped pipeline, not a before/after flip
check.

Person14 has only 1 probe (image1's crop failed -- no finger detected),
so genuine pairs = 59, not 60.
"""
import os

import cv2
import numpy as np
import torch

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
from embed_resnet50 import build_embedder, INPUT_SIZE, _IMAGENET_MEAN, _IMAGENET_STD  # noqa: E402
from evaluate import compute_metrics, write_csv, write_plot, accuracy_at_threshold  # noqa: E402

IMAGE_DB_CROPPED = r"D:\Contactless\image_db_cropped"
TEST_DB_CROPPED = r"D:\Contactless\test_db_cropped"
OUT_CSV = os.path.join(BASE_DIR, "benchmark_results_30_cropped.csv")
OUT_PLOT = os.path.join(BASE_DIR, "eer_plot_30_cropped.png")
ILLUSTRATIVE_THRESHOLD = 0.75


def preprocess(path):
    img = cv2.imread(path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (INPUT_SIZE, INPUT_SIZE), interpolation=cv2.INTER_AREA)
    rgb = np.stack([resized] * 3, axis=-1).astype(np.float32) / 255.0
    normalized = (rgb - _IMAGENET_MEAN) / _IMAGENET_STD
    chw = np.transpose(normalized, (2, 0, 1))
    return torch.from_numpy(chw).float()


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = build_embedder(device)

    gallery_vecs = {}
    for person_num in range(1, 31):
        path = os.path.join(IMAGE_DB_CROPPED, f"Person{person_num}.jpg")
        if not os.path.isfile(path):
            print(f"WARNING: missing cropped gallery for Person{person_num}")
            continue
        tensor = preprocess(path).unsqueeze(0).to(device)
        with torch.no_grad():
            vec = model(tensor).cpu().numpy()[0]
        gallery_vecs[person_num] = vec / np.linalg.norm(vec)

    probe_vecs = {}  # person_num -> list of vectors
    for person_num in range(1, 31):
        person_dir = os.path.join(TEST_DB_CROPPED, f"Person{person_num}")
        if not os.path.isdir(person_dir):
            continue
        vecs = []
        for fname in sorted(os.listdir(person_dir)):
            path = os.path.join(person_dir, fname)
            tensor = preprocess(path).unsqueeze(0).to(device)
            with torch.no_grad():
                vec = model(tensor).cpu().numpy()[0]
            vecs.append(vec / np.linalg.norm(vec))
        probe_vecs[person_num] = vecs

    genuine_scores = []
    impostor_scores = []
    for person_num, vecs in probe_vecs.items():
        if person_num not in gallery_vecs:
            continue
        own_gallery = gallery_vecs[person_num]
        for vec in vecs:
            genuine_scores.append(float(np.dot(vec, own_gallery)))
            for other_num, other_gallery in gallery_vecs.items():
                if other_num == person_num:
                    continue
                impostor_scores.append(float(np.dot(vec, other_gallery)))

    genuine_scores = np.array(genuine_scores)
    impostor_scores = np.array(impostor_scores)

    print(f"genuine pairs: {len(genuine_scores)}, impostor pairs: {len(impostor_scores)}")
    print(f"genuine  mean={genuine_scores.mean():.4f} median={np.median(genuine_scores):.4f}")
    print(f"impostor mean={impostor_scores.mean():.4f} median={np.median(impostor_scores):.4f}")

    rows, eer_row = compute_metrics(genuine_scores, impostor_scores)
    write_csv(rows, eer_row, csv_path=OUT_CSV)
    write_plot(rows, eer_row, genuine_scores, impostor_scores, plot_path=OUT_PLOT,
               title_prefix="ResNet50 on YOLO-cropped 30-person set")

    eer_accuracy = 1.0 - eer_row["eer"]
    fixed_accuracy = accuracy_at_threshold(genuine_scores, impostor_scores, ILLUSTRATIVE_THRESHOLD)

    print(f"\n=== FAR/FRR/EER (30 people, YOLO-cropped, fresh threshold sweep) ===")
    print(f"EER threshold = {eer_row['threshold']:.4f}")
    print(f"At EER: FAR = {eer_row['far']*100:.2f}%,  FRR = {eer_row['frr']*100:.2f}%,  EER = {eer_row['eer']*100:.2f}%")
    print(f"Matching accuracy at EER threshold = {eer_accuracy*100:.2f}%")
    print(f"Accuracy at illustrative threshold {ILLUSTRATIVE_THRESHOLD} = {fixed_accuracy*100:.2f}%")
    print(f"\nResults written to {OUT_CSV} and {OUT_PLOT}")


if __name__ == "__main__":
    main()
