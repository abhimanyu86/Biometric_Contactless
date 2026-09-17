"""
Architecture for best_e01.pt -- a fine-tuned ResNet50 Siamese embedding model.
Inferred from the checkpoint's own state_dict keys/shapes (no train.py/model.py
was provided this time, unlike exp2's SiameseResNet18): Bottleneck blocks
([3,4,6,3] per layer -- confirmed via layer4 having 3 blocks), grayscale
1-channel conv1 (checkpoint shape (64,1,7,7)), final fc mapping 2048->512
(checkpoint shape (512,2048)) -- structurally identical wrapper style to
exp2's SiameseResNet18, just with a ResNet50 backbone.
"""
import torch
import torch.nn as nn
import torchvision.models as models


class SiameseResNet50(nn.Module):
    def __init__(self, embedding_dim=512):
        super().__init__()

        backbone = models.resnet50(weights=None)
        backbone.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)

        in_features = backbone.fc.in_features
        backbone.fc = nn.Linear(in_features, embedding_dim)

        self.encoder = backbone

    def forward_once(self, x):
        embedding = self.encoder(x)
        embedding = torch.nn.functional.normalize(embedding, p=2, dim=1)
        return embedding

    def forward(self, image1, image2):
        embedding1 = self.forward_once(image1)
        embedding2 = self.forward_once(image2)
        return embedding1, embedding2
