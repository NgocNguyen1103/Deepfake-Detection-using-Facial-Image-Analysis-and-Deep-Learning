import torch.nn as nn
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights


class EfficientNetB0Baseline(nn.Module):
    """
    Baseline model for frame-level deepfake detection.

    Architecture:
        Face image 224x224
        -> EfficientNet-B0 feature extractor
        -> Global Average Pooling
        -> Dropout
        -> Linear(1280 -> 1)

    Output:
        logit for binary classification
        0 -> REAL
        1 -> FAKE
    """

    def __init__(self, pretrained: bool = True):
        super().__init__()

        weights = EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
        base_model = efficientnet_b0(weights=weights)

        self.features = base_model.features
        self.pool = nn.AdaptiveAvgPool2d(1)

        self.classifier = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(1280, 1)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x)
        x = x.flatten(1)
        x = self.classifier(x)

        return x