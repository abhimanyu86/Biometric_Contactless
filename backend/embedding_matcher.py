"""
CNN/Siamese embedding matcher (Phase A) — a fixed, off-the-shelf ImageNet-pretrained
ResNet18 used as a feature extractor, compared via cosine similarity. No fingerprint-
specific training; see Contactless_Fingerprint_Matching_Next_Plan.pdf for why this was
tried, and siamese_embedding/ (branch siamese-embedding) for the full validation.

Validated result: 92.5% accuracy at EER=7.5% (FAR 7.05%, FRR 8.04%), threshold ~=0.94,
measured on the 336-person HKPU contactless fingerprint dataset (lab photos, known
preprocessing). That threshold has NOT been validated against this app's actual camera
captures -- different device, lighting and preprocessing than the lab dataset it was
calibrated on. Treat EMBEDDING_THRESHOLD as a starting point, not a production-ready
number, until it's re-measured against real enroll/authenticate traffic.

This runs alongside matching.py's minutiae matcher, not instead of it -- see cloud_app.py's
/v2/enroll_embedding, /v2/authenticate_embedding, /v2/verify_embedding, which are separate
endpoints against a separate embedding_templates table. Nothing about the existing
minutiae-based /v2/enroll, /v2/authenticate, /v2/verify flow changes.
"""
import os

import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights

EMBEDDING_THRESHOLD = float(os.environ.get("EMBEDDING_THRESHOLD", "0.94"))
INPUT_SIZE = 224
DEVICE = "cpu"

_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

_model = None  # lazy-loaded singleton, see get_embedding_model()


def get_embedding_model():
    """Lazy-loaded singleton — downloads/loads pretrained ResNet18 once, on first use.
    Requires outbound internet on first call unless the weights are already cached
    (see torch's TORCH_HOME) — bake the cache into the deploy image so a production
    boot doesn't depend on runtime internet access."""
    global _model
    if _model is None:
        m = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        m.fc = nn.Identity()  # 512-D pooled feature instead of 1000-way class logits
        m.eval()
        m.to(DEVICE)
        _model = m
    return _model


def preprocess_image(img_bgr_or_gray):
    """Matches siamese_embedding/embed.py's preprocessing exactly (same model, same
    validated numbers): grayscale -> 3-channel -> 224x224 -> ImageNet normalization."""
    if img_bgr_or_gray.ndim == 3:
        gray = cv2.cvtColor(img_bgr_or_gray, cv2.COLOR_BGR2GRAY)
    else:
        gray = img_bgr_or_gray
    resized = cv2.resize(gray, (INPUT_SIZE, INPUT_SIZE), interpolation=cv2.INTER_AREA)
    rgb = np.stack([resized] * 3, axis=-1).astype(np.float32) / 255.0
    normalized = (rgb - _IMAGENET_MEAN) / _IMAGENET_STD
    chw = np.transpose(normalized, (2, 0, 1))
    return torch.from_numpy(chw).float()


def embed_image(img_bgr_or_gray):
    """Single BGR/grayscale image (as loaded by cv2.imread/imdecode) -> 512-D np.ndarray."""
    model = get_embedding_model()
    tensor = preprocess_image(img_bgr_or_gray).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        out = model(tensor).cpu().numpy()[0]
    return out


def cosine_similarity(a, b):
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0.0:
        return 0.0
    return float(np.dot(a, b) / denom)
