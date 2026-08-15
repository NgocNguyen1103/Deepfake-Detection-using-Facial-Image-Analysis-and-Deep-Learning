"""
EfficientNet-B0 + CBAM Hybrid Model for Deepfake Detection

Architecture:
    Input Image (B×3×224×224)
    ↓
    EfficientNet-B0 Feature Extractor
    ↓
    Feature Map (B×1280×7×7)
    ↓
    CBAM (Channel Attention + Spatial Attention)
    ↓
    Refined Feature Map (B×1280×7×7)
    ↓
    Global Average Pooling + Flatten
    ↓
    Vector (B×1280)
    ↓
    Classifier (Dropout + Linear)
    ↓
    Logit (B×1)
    ↓
    Sigmoid
    ↓
    Fake Score (0-1)
    ↓
    REAL/FAKE (threshold=0.5)

Key Features:
    - CBAM focuses on important facial areas (eyes, nose, mouth, etc.)
    - Preserves spatial dimensions through attention
    - Compatible with pretrained EfficientNet weights
"""

import torch
import torch.nn as nn
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
from .cbam_modules import CBAM, CBAMConfig


class EfficientNetCBAM(nn.Module):
    def __init__(
        self,
        pretrained=True,
        cbam_config=None,
        num_classes=1,  # Binary classification: REAL vs FAKE
        dropout=0.5
    ):
        """
        Args:
            pretrained: Whether to use ImageNet pretrained weights
            cbam_config: Configuration dict for CBAM module
            num_classes: Number of output classes (1 for binary classification)
            dropout: Dropout rate before final classifier
        """
        super().__init__()

        # Load pretrained EfficientNet-B0
        weights = EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
        base_model = efficientnet_b0(weights=weights)

        # Extract feature extractor (excluding classifier head)
        self.features = base_model.features  # Output: (B, 1280, 7, 7)

        # CBAM configuration
        if cbam_config is None:
            cbam_config = CBAMConfig.DEFAULT_CONFIG

        # Create CBAM module
        self.cbam = CBAM(
            in_channels=cbam_config.get('in_channels', 1280),
            reduction_ratio=cbam_config.get('reduction_ratio', 16),
            kernel_size=cbam_config.get('kernel_size', 7)
        )

        # Classification head
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),      # Global Average Pooling: (B, 1280, 7, 7) → (B, 1280, 1, 1)
            nn.Flatten(),                   # Flatten: (B, 1280, 1, 1) → (B, 1280)
            nn.Dropout(p=dropout),         # Dropout for regularization
            nn.Linear(1280, num_classes)  # Linear: (B, 1280) → (B, 1)
        )

        # Initialize classifier weights
        self._initialize_weights()

        print(f"[OK] EfficientNet-CBAM initialized")
        print(f"   CBAM config: {cbam_config}")
        print(f"   Dropout: {dropout}")

    def _initialize_weights(self):
        """Initialize classifier weights using Xavier initialization"""
        for module in self.classifier.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_normal_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

    def forward(self, x):
        """
        Forward pass through the model.

        Args:
            x: Input images (B, 3, 224, 224)

        Returns:
            logits: Output logits (B, 1) - use sigmoid for probabilities
        """
        # Step 1: Feature extraction with EfficientNet
        features = self.features(x)  # (B, 3, 224, 224) → (B, 1280, 7, 7)

        # Step 2: Apply CBAM attention
        attended_features = self.cbam(features)  # (B, 1280, 7, 7) → (B, 1280, 7, 7)

        # Step 3: Classification
        logits = self.classifier(attended_features)  # (B, 1280, 7, 7) → (B, 1)

        return logits

    def get_attention_maps(self, x):

        self.eval()

        with torch.no_grad():
            # Extract features
            features = self.features(x)  # (B, 1280, 7, 7)

            # Get channel attention
            channel_attended = self.cbam.channel_att(features)  # (B, 1280, 7, 7)

            # Get spatial attention
            spatial_attended = self.cbam.spatial_att(channel_attended)  # (B, 1280, 7, 7)

            # Get spatial attention map only (for visualization)
            # Apply same pooling operations as in SpatialAttention
            avg_pool = torch.mean(channel_attended, dim=1, keepdim=True)  # (B, 1, 7, 7)
            max_pool, _ = torch.max(channel_attended, dim=1, keepdim=True)  # (B, 1, 7, 7)
            combined = torch.cat([avg_pool, max_pool], dim=1)  # (B, 2, 7, 7)
            spatial_map = torch.sigmoid(self.cbam.spatial_att.conv(combined))  # (B, 1, 7, 7)

            # Full CBAM
            attended_features = self.cbam(features)  # (B, 1280, 7, 7)

            # Classification
            logits = self.classifier(attended_features)  # (B, 1)

        return {
            'input': x.cpu(),
            'features': features.cpu(),
            'channel_attention': channel_attended.cpu(),
            'spatial_attention_map': spatial_map.cpu(),
            'attended_features': attended_features.cpu(),
            'logits': logits.cpu()
        }

    def load_from_baseline(self, baseline_model_path, device='cpu'):

        print(f"Loading EfficientNet weights from {baseline_model_path}")

        # Load checkpoint
        checkpoint = torch.load(baseline_model_path, map_location=device)

        # Extract EfficientNet weights if available
        if isinstance(checkpoint, dict):
            if 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            elif 'features' in checkpoint:
                # Extract only features part
                state_dict = {'features.' + k: v for k, v in checkpoint['features'].items()}
            else:
                # Assume full model state dict
                state_dict = checkpoint
        else:
            state_dict = checkpoint

        # Load EfficientNet features
        features_dict = {}
        for k, v in state_dict.items():
            if k.startswith('features.') or k.startswith('.'):
                features_dict[k] = v

        if features_dict:
            # Load state dict for features
            result = self.features.load_state_dict(features_dict, strict=False)
            print(f"Loaded EfficientNet features: {result}")
        else:
            print("No EfficientNet features found in checkpoint, using ImageNet pretrained")

    def count_parameters(self):
        """Count total and trainable parameters"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

        # CBAM parameters
        cbam_params = sum(p.numel() for p in self.cbam.parameters())

        # EfficientNet parameters (frozen if using pretrained)
        efficientnet_params = sum(p.numel() for p in self.features.parameters())

        print(f"Parameter Count:")
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")
        print(f"EfficientNet parameters: {efficientnet_params:,}")
        print(f"CBAM parameters: {cbam_params:,}")
        print(f"CBAM ratio: {cbam_params / total_params * 100:.2f}%")

        return {
            'total': total_params,
            'trainable': trainable_params,
            'efficientnet': efficientnet_params,
            'cbam': cbam_params
        }


def create_efficientnet_cbam(pretrained=True, cbam_variant='default', **kwargs):

    # Select CBAM configuration
    if cbam_variant == 'default':
        cbam_config = CBAMConfig.DEFAULT_CONFIG
    elif cbam_variant == 'lightweight':
        cbam_config = CBAMConfig.LIGHTWEIGHT_CONFIG
    elif cbam_variant == 'aggressive':
        cbam_config = CBAMConfig.AGGRESSIVE_CONFIG
    else:
        cbam_config = CBAMConfig.DEFAULT_CONFIG

    # Create model
    model = EfficientNetCBAM(
        pretrained=pretrained,
        cbam_config=cbam_config,
        **kwargs
    )

    return model


def test_efficientnet_cbam():
    """Test EfficientNet-CBAM model"""

    print("Testing EfficientNet-CBAM model...")

    # Create sample input: Batch=2, Channels=3, Height=224, Width=224
    x = torch.randn(2, 3, 224, 224)

    print(f"Input shape: {x.shape}")  # Should be: (2, 3, 224, 224)

    # Create model
    model = create_efficientnet_cbam(pretrained=False, cbam_variant='default')
    model.eval()

    # Forward pass
    logits = model(x)

    print(f"Output shape: {logits.shape}")  # Should be: (2, 1)

    # Count parameters
    params = model.count_parameters()

    # Test attention extraction
    print("\nTesting attention map extraction...")
    attention_data = model.get_attention_maps(x)
    print(f"Attention keys: {list(attention_data.keys())}")

    assert logits.shape == (2, 1), "Output shape should be (batch_size, 1)"
    print("EfficientNet-CBAM test passed!")

    return model, logits


if __name__ == "__main__":
    test_efficientnet_cbam()
