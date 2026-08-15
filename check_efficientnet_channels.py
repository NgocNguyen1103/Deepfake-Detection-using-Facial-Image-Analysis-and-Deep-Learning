import torch
from torchvision.models import efficientnet_b0

model = efficientnet_b0()
x = torch.randn(1, 3, 224, 224)

features = model.features
print('EfficientNet-B0 Feature Extraction Stages:')
print('=' * 60)

for i, block in enumerate(features):
    print(f'\nBlock {i}: {block}')
    x = block(x)
    print(f'Output shape: {x.shape}')
    print(f'Channels: {x.shape[1]}')