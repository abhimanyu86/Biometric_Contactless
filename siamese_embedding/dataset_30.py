"""
Loads the 30-person image_db (gallery, 1 photo/person) + test_db (2 probe
photos/person) dataset -- the "db + test" set restored from the git stash
(originally captured alongside the labtool testing, confirmed by a hash match
against lab_gallery.db's logged probe attempts). Unlike the HKPU 336-person
set, this is real, uncontrolled phone-camera data: different backgrounds,
framing, lighting per photo.
"""
import os

IMAGE_DB = r"D:\Contactless\image_db"
TEST_DB = r"D:\Contactless\test_db"

NUM_PEOPLE = 30


def _gallery_path(person_id):
    """image_db's naming is inconsistent: Person_01..Person_09 (zero-padded,
    underscore) for single digits, Person10..Person30 (no underscore) for the
    rest -- and mixed extensions (.png/.jpg/.jpeg). Try what actually exists."""
    candidates = []
    if person_id < 10:
        candidates.append(f"Person_{person_id:02d}")
    candidates.append(f"Person{person_id}")
    for stem in candidates:
        for ext in (".jpeg", ".jpg", ".png"):
            path = os.path.join(IMAGE_DB, stem + ext)
            if os.path.isfile(path):
                return path
    return None


def load_people():
    people = {}
    for person_id in range(1, NUM_PEOPLE + 1):
        gallery_path = _gallery_path(person_id)
        if gallery_path is None:
            print(f"[dataset_30] WARNING: no gallery image for Person{person_id}, skipping")
            continue

        person_test_dir = os.path.join(TEST_DB, f"Person{person_id}")
        probes = []
        for stem in ("image1", "image2"):
            found = None
            for ext in (".jpeg", ".jpg", ".png"):
                p = os.path.join(person_test_dir, stem + ext)
                if os.path.isfile(p):
                    found = p
                    break
            if found:
                probes.append(found)
            else:
                print(f"[dataset_30] WARNING: missing {stem}.* in {person_test_dir}")

        if not probes:
            print(f"[dataset_30] WARNING: no probes for Person{person_id}, skipping")
            continue

        people[person_id] = {"gallery": gallery_path, "probes": probes}

    return people


if __name__ == "__main__":
    people = load_people()
    total_probes = sum(len(v["probes"]) for v in people.values())
    print(f"Loaded {len(people)} people, {total_probes} probe images, "
          f"{len(people)} gallery images")
