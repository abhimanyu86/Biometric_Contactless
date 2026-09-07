"""
Phase A CNN/Siamese embedding extractor: a fixed, off-the-shelf ImageNet-pretrained
ResNet18 used purely as a texture/structure feature extractor. No fingerprint-specific
training happens here -- see CLAUDE.md/plan notes for why that's the deliberate scope
of this first pass.

Two images through the identical (weight-sharing) network -> two 512-D embeddings ->
compared later via cosine similarity in evaluate.py. That weight-sharing is what makes
this a "Siamese" setup even though nothing is being trained.
"""
import os
import json

import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights

from dataset import load_people, DATASET_ROOT

CACHE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "embeddings_cache.json")
INPUT_SIZE = 224

_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def build_embedder(device="cpu"):
    """Pretrained ResNet18 with its final FC layer stripped, so forward() returns
    the 512-D pooled feature vector instead of 1000-way ImageNet class logits."""
    model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    model.fc = nn.Identity()
    model.eval()
    model.to(device)
    return model


def preprocess_image(path):
    """Grayscale bmp -> 3-channel -> resize to 224x224 -> ImageNet-normalized tensor."""
    gray = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if gray is None:
        raise ValueError(f"could not read image: {path}")
    resized = cv2.resize(gray, (INPUT_SIZE, INPUT_SIZE), interpolation=cv2.INTER_AREA)
    rgb = np.stack([resized] * 3, axis=-1).astype(np.float32) / 255.0
    normalized = (rgb - _IMAGENET_MEAN) / _IMAGENET_STD
    chw = np.transpose(normalized, (2, 0, 1))
    return torch.from_numpy(chw).float()


def embed_paths(model, paths, device="cpu", batch_size=32):
    """Batched forward pass. Returns {path: np.ndarray(512,)}; unreadable images are
    logged and skipped rather than crashing the whole run."""
    embeddings = {}
    batch_tensors = []
    batch_paths = []

    def flush():
        if not batch_tensors:
            return
        batch = torch.stack(batch_tensors).to(device)
        with torch.no_grad():
            out = model(batch).cpu().numpy()
        for p, vec in zip(batch_paths, out):
            embeddings[p] = vec
        batch_tensors.clear()
        batch_paths.clear()

    for path in paths:
        try:
            tensor = preprocess_image(path)
        except ValueError as exc:
            print(f"[embed] WARNING: {exc}, skipping")
            continue
        batch_tensors.append(tensor)
        batch_paths.append(path)
        if len(batch_tensors) >= batch_size:
            flush()
    flush()

    return embeddings


def compute_all_embeddings(people, device="cpu"):
    model = build_embedder(device)
    all_paths = []
    for info in people.values():
        all_paths.append(info["gallery"])
        all_paths.extend(info["probes"])
    all_paths = list(dict.fromkeys(all_paths))  # de-dupe, preserve order

    print(f"[embed] computing embeddings for {len(all_paths)} images on {device}...")
    embeddings = embed_paths(model, all_paths, device=device)
    print(f"[embed] done, {len(embeddings)} embeddings computed")
    return embeddings


def save_cache(embeddings, cache_path=CACHE_PATH):
    serializable = {path: vec.tolist() for path, vec in embeddings.items()}
    with open(cache_path, "w", encoding="utf-8") as handle:
        json.dump(serializable, handle)


def load_cache(cache_path=CACHE_PATH):
    if not os.path.isfile(cache_path):
        return None
    with open(cache_path, "r", encoding="utf-8") as handle:
        raw = json.load(handle)
    return {path: np.array(vec, dtype=np.float32) for path, vec in raw.items()}


def get_embeddings(force_recompute=False, device="cpu"):
    """Cache-aware entry point used by evaluate.py."""
    if not force_recompute:
        cached = load_cache()
        if cached is not None:
            print(f"[embed] loaded {len(cached)} embeddings from cache at {CACHE_PATH}")
            return cached

    people = load_people()
    embeddings = compute_all_embeddings(people, device=device)
    save_cache(embeddings)
    return embeddings


if __name__ == "__main__":
    get_embeddings(force_recompute=True)
