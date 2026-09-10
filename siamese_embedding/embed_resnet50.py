"""
Same Phase A protocol as embed.py, swapped to a pretrained ResNet50 (2048-D
embeddings) instead of ResNet18 (512-D), for a direct architecture comparison
on the exact same 336-person gallery/probe set. No fingerprint-specific
training here either -- same "off-the-shelf feature extractor" premise as the
ResNet18 run, just a deeper/heavier backbone.
"""
import os
import json

import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights

from dataset import load_people

CACHE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "embeddings_cache_resnet50.json")
INPUT_SIZE = 224

_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def build_embedder(device="cpu"):
    model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
    model.fc = nn.Identity()  # 2048-D pooled feature instead of 1000-way class logits
    model.eval()
    model.to(device)
    return model


def preprocess_image(path):
    gray = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if gray is None:
        raise ValueError(f"could not read image: {path}")
    resized = cv2.resize(gray, (INPUT_SIZE, INPUT_SIZE), interpolation=cv2.INTER_AREA)
    rgb = np.stack([resized] * 3, axis=-1).astype(np.float32) / 255.0
    normalized = (rgb - _IMAGENET_MEAN) / _IMAGENET_STD
    chw = np.transpose(normalized, (2, 0, 1))
    return torch.from_numpy(chw).float()


def embed_paths(model, paths, device="cpu", batch_size=16):
    """Smaller default batch size than ResNet18's (32) -- ResNet50's bottleneck
    blocks make each forward pass heavier on CPU."""
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
            print(f"[embed_resnet50] WARNING: {exc}, skipping")
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
    all_paths = list(dict.fromkeys(all_paths))

    print(f"[embed_resnet50] computing embeddings for {len(all_paths)} images on {device}...")
    embeddings = embed_paths(model, all_paths, device=device)
    print(f"[embed_resnet50] done, {len(embeddings)} embeddings computed")
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
    if not force_recompute:
        cached = load_cache()
        if cached is not None:
            print(f"[embed_resnet50] loaded {len(cached)} embeddings from cache at {CACHE_PATH}")
            return cached

    people = load_people()
    embeddings = compute_all_embeddings(people, device=device)
    save_cache(embeddings)
    return embeddings


if __name__ == "__main__":
    get_embeddings(force_recompute=True)
