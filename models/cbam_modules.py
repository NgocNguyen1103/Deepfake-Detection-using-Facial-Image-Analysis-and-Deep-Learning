import torch
import torch.nn as nn
import torch.nn.functional as F


class ChannelAttention(nn.Module):

    def __init__(self, in_channels, reduction_ratio=16):

        super().__init__()

        self.avg_pool = nn.AdaptiveAvgPool2d(1)  # Global average pooling
        self.max_pool = nn.AdaptiveMaxPool2d(1)  # Global max pooling

        # Shared MLP for both pooling paths
        hidden_channels = max(in_channels // reduction_ratio, 1)

        self.fc = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, in_channels, 1, bias=False)
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Average pooling path
        avg_out = self.fc(self.avg_pool(x))  # (B, C, 1, 1) → (B, C, 1, 1)
        # Max pooling path
        max_out = self.fc(self.max_pool(x))  # (B, C, 1, 1) → (B, C, 1, 1)
        # Combine both paths 
        channel_attention = self.sigmoid(avg_out + max_out)  # (B, C, 1, 1)
        # Scale original input
        return x * channel_attention  # (B, C, H, W)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()

        # Convolution to compute spatial attention
        self.conv = nn.Conv2d(
            2, 1, kernel_size,
            padding=kernel_size // 2,
            bias=False
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):

        # Channel-wise average pooling along channel dimension
        avg_out = torch.mean(x, dim=1, keepdim=True)  # (B, 1, H, W)

        # Channel-wise max pooling along channel dimension
        max_out, _ = torch.max(x, dim=1, keepdim=True)  # (B, 1, H, W)

        # Concatenate along channel dimension
        combined = torch.cat([avg_out, max_out], dim=1)  # (B, 2, H, W)

        # Compute spatial attention map
        spatial_attention = self.sigmoid(self.conv(combined))  # (B, 1, H, W)

        # Scale original input
        return x * spatial_attention  # (B, C, H, W)


class CBAM(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16, kernel_size=7):

        super().__init__()

        self.channel_att = ChannelAttention(in_channels, reduction_ratio)
        self.spatial_att = SpatialAttention(kernel_size)

    def forward(self, x):

        # Apply channel attention first
        x = self.channel_att(x)

        # Then apply spatial attention
        x = self.spatial_att(x)

        return x


class CBAMConfig:
    """Configuration for CBAM modules"""

    # Default configuration for EfficientNet-B0
    DEFAULT_CONFIG = {
        'in_channels': 1280,        # EfficientNet-B0 final feature channels
        'reduction_ratio': 16,      # Channel attention reduction
        'kernel_size': 7            # Spatial attention kernel size
    }

    # Alternative configurations
    LIGHTWEIGHT_CONFIG = {
        'in_channels': 1280,
        'reduction_ratio': 32,      # More reduction, fewer parameters
        'kernel_size': 5            # Smaller kernel
    }

    AGGRESSIVE_CONFIG = {
        'in_channels': 1280,
        'reduction_ratio': 8,       # Less reduction, more capacity
        'kernel_size': 9            # Larger kernel
    }


def test_cbam():
    """Test CBAM module with sample input"""

    # Create sample input: Batch=2, Channels=1280, Height=7, Width=7
    x = torch.randn(2, 1280, 7, 7)

    print(f"Input shape: {x.shape}")  # Should be: (2, 1280, 7, 7)

    # Create CBAM module
    cbam = CBAM(in_channels=1280, reduction_ratio=16, kernel_size=7)

    # Forward pass
    output = cbam(x)

    print(f"Output shape: {output.shape}")  # Should be: (2, 1280, 7, 7)

    # Count parameters
    total_params = sum(p.numel() for p in cbam.parameters())
    print(f"Total CBAM parameters: {total_params:,}")

    assert output.shape == x.shape, "Output shape should match input shape"
    print("CBAM test passed!")

    return output


if __name__ == "__main__":
    test_cbam()
