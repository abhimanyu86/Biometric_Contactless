"""
Enumerates the first_session portion of the HKPU contactless fingerprint database
as a gallery/probe split for the Siamese/CNN embedding experiment.

Dataset layout (see D:\\Cross_Fingerprint_Images_Database\\CrossFingerprint_Database_Readme.txt):
first_session has 336 unique fingers, pX/pY.bmp, X=1..336, Y=1..6. second_session's 160
people are recaptures of a subset of these same 336 fingers months later, not new
identities, so it's excluded here per the current experiment's scope.
"""
import os

DATASET_ROOT = r"D:\Cross_Fingerprint_Images_Database\processed_contactless_2d_fingerprint_images\first_session"

NUM_PEOPLE = 336
IMAGES_PER_PERSON = 6
GALLERY_SAMPLE = 1
PROBE_SAMPLES = range(2, IMAGES_PER_PERSON + 1)


def load_people(root=DATASET_ROOT):
    """Returns {person_id: {"gallery": path, "probes": [paths]}}, skipping and
    warning on any person/sample that isn't actually on disk instead of assuming
    the numbering is gap-free."""
    people = {}
    for person_id in range(1, NUM_PEOPLE + 1):
        person_dir = os.path.join(root, f"p{person_id}")
        if not os.path.isdir(person_dir):
            print(f"[dataset] WARNING: missing person folder p{person_id}, skipping")
            continue

        gallery_path = os.path.join(person_dir, f"p{GALLERY_SAMPLE}.bmp")
        if not os.path.isfile(gallery_path):
            print(f"[dataset] WARNING: missing gallery image for p{person_id}, skipping person")
            continue

        probes = []
        for sample_id in PROBE_SAMPLES:
            probe_path = os.path.join(person_dir, f"p{sample_id}.bmp")
            if os.path.isfile(probe_path):
                probes.append(probe_path)
            else:
                print(f"[dataset] WARNING: missing probe p{person_id}/p{sample_id}.bmp")

        if not probes:
            print(f"[dataset] WARNING: no probe images for p{person_id}, skipping person")
            continue

        people[person_id] = {"gallery": gallery_path, "probes": probes}

    return people


if __name__ == "__main__":
    people = load_people()
    total_probes = sum(len(v["probes"]) for v in people.values())
    print(f"Loaded {len(people)} people, {total_probes} probe images, "
          f"{len(people)} gallery images")
