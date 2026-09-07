"""
Embedding extractor for the teammate's exp2 Siamese checkpoint (contrastive loss,
margin=1.0, trained on a separate dataset we don't have access to -- confirmed no
overlap with the 336-person HKPU gallery/probe set used here, so this is a clean
cross-dataset generalization test, not a leak).

Preprocessing mirrors experiments/siamese/preprocessing.py:get_eval_transform()
exactly: PIL grayscale, resize to 224x224, ToTensor, Normalize(mean=[0.5], std=[0.5]).
The model L2-normalizes its own output embeddings (SiameseResNet18.forward_once),
so cosine similarity and the Euclidean distance it was actually trained with are
monotonically equivalent on these vectors -- reusing evaluate.py's cosine-similarity
scoring is a fair comparison with Phase A, not an approximation.
"""
import os
import json

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from dataset import load_people
from model_v2 import SiameseResNet18

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CHECKPOINT_PATH = os.path.join(BASE_DIR, "checkpoints", "exp2_best_model.pth")
CACHE_PATH = os.path.join(BASE_DIR, "embeddings_cache_v2.json")
INPUT_SIZE = 224

_eval_transform = transforms.Compose([
    transforms.Resize((INPUT_SIZE, INPUT_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5]),
])


def load_checkpoint_metadata(checkpoint_path=CHECKPOINT_PATH):
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    return ckpt


def build_embedder(checkpoint_path=CHECKPOINT_PATH, device="cpu"):
    ckpt = load_checkpoint_metadata(checkpoint_path)
    embedding_dim = ckpt.get("embedding_dim", 512)
    model = SiameseResNet18(embedding_dim=embedding_dim)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    model.to(device)
    print(f"[embed_v2] loaded checkpoint: epoch={ckpt.get('epoch')} "
          f"val_loss={ckpt.get('val_loss'):.4f} embedding_dim={embedding_dim}")
    return model


def preprocess_image(path):
    img = Image.open(path).convert("L")
    return _eval_transform(img)


def embed_paths(model, paths, device="cpu", batch_size=32):
    embeddings = {}
    batch_tensors = []
    batch_paths = []

    def flush():
        if not batch_tensors:
            return
        batch = torch.stack(batch_tensors).to(device)
        with torch.no_grad():
            out = model.forward_once(batch).cpu().numpy()
        for p, vec in zip(batch_paths, out):
            embeddings[p] = vec
        batch_tensors.clear()
        batch_paths.clear()

    for path in paths:
        try:
            tensor = preprocess_image(path)
        except (OSError, ValueError) as exc:
            print(f"[embed_v2] WARNING: could not read {path}: {exc}, skipping")
            continue
        batch_tensors.append(tensor)
        batch_paths.append(path)
        if len(batch_tensors) >= batch_size:
            flush()
    flush()

    return embeddings


def compute_all_embeddings(people, device="cpu"):
    model = build_embedder(device=device)
    all_paths = []
    for info in people.values():
        all_paths.append(info["gallery"])
        all_paths.extend(info["probes"])
    all_paths = list(dict.fromkeys(all_paths))

    print(f"[embed_v2] computing embeddings for {len(all_paths)} images on {device}...")
    embeddings = embed_paths(model, all_paths, device=device)
    print(f"[embed_v2] done, {len(embeddings)} embeddings computed")
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
            print(f"[embed_v2] loaded {len(cached)} embeddings from cache at {CACHE_PATH}")
            return cached

    people = load_people()
    embeddings = compute_all_embeddings(people, device=device)
    save_cache(embeddings)
    return embeddings


if __name__ == "__main__":
    get_embeddings(force_recompute=True)
