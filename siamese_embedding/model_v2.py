"""
Exact architecture copy of the teammate's SiameseResNet18 (experiments/siamese/model.py)
so exp2_best_model.pth's state_dict loads without key mismatches. Not modified.
"""
import torch
import torch.nn as nn
import torchvision.models as models


class SiameseResNet18(nn.Module):
    def __init__(self, embedding_dim=512):
        super().__init__()

        backbone = models.resnet18(weights=None)

        # Convert ResNet-18 input from RGB (3 channels) to grayscale (1 channel)
        backbone.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)

        # Replace ImageNet classification layer with embedding layer
        in_features = backbone.fc.in_features
        backbone.fc = nn.Linear(in_features, embedding_dim)

        self.encoder = backbone

    def forward_once(self, x):
        embedding = self.encoder(x)
        # L2-normalize embeddings for cosine similarity
        embedding = torch.nn.functional.normalize(embedding, p=2, dim=1)
        return embedding

    def forward(self, image1, image2):
        embedding1 = self.forward_once(image1)
        embedding2 = self.forward_once(image2)
        return embedding1, embedding2
