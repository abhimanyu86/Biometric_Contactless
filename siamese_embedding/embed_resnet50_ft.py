"""
Embedding extractor for best_e01.pt -- a fine-tuned ResNet50 Siamese model
(checkpoint provided as a bare state_dict, no train.py/preprocessing.py this
time). Preprocessing ASSUMED identical to exp2's SiameseResNet18 recipe, since
the checkpoint's wrapper structure (grayscale conv1, encoder.fc -> 512-D,
L2-normalized output) is structurally identical, just with a ResNet50
backbone -- strongly suggesting the same training pipeline was reused with
only the architecture swapped. If results look off, this assumption is the
first thing to re-check against the actual training script.
"""
import os

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from model_resnet50_ft import SiameseResNet50

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CHECKPOINT_PATH = os.path.join(BASE_DIR, "checkpoints", "best_e01.pt")
INPUT_SIZE = 224

_eval_transform = transforms.Compose([
    transforms.Resize((INPUT_SIZE, INPUT_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5]),
])


def build_embedder(checkpoint_path=CHECKPOINT_PATH, device="cpu"):
    state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    model = SiameseResNet50(embedding_dim=512)
    model.load_state_dict(state_dict)
    model.eval()
    model.to(device)
    return model


def preprocess_image(path):
    img = Image.open(path).convert("L")
    return _eval_transform(img)


def embed_paths(model, paths, device="cpu", batch_size=16):
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
            print(f"[embed_resnet50_ft] WARNING: could not read {path}: {exc}, skipping")
            continue
        batch_tensors.append(tensor)
        batch_paths.append(path)
        if len(batch_tensors) >= batch_size:
            flush()
    flush()

    return embeddings


if __name__ == "__main__":
    m = build_embedder()
    print("loaded OK, embedding_dim:", m.encoder.fc.out_features)
