"""
Run the same 3-way extractor pilot (compare_extractors_pilot.py) on the 7
photos the user shared directly in chat (saved to user_photos/), instead of
the image_db sample set. Reuses every function unchanged -- only the input
file list changes.
"""
import glob
import os

import compare_extractors_pilot as cep

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
USER_PHOTOS_DIR = os.path.join(BASE_DIR, "user_photos")

cep.SAMPLE_IMAGES = sorted(glob.glob(os.path.join(USER_PHOTOS_DIR, "*.jpg")))

if __name__ == "__main__":
    cep.main()
