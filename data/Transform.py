# data/transforms.py

from torchvision import transforms


def get_train_transform(image_size: int = 224):
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),

        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=5),

        transforms.RandomAffine(
            degrees=0,
            translate=(0.03, 0.03),
            scale=(0.95, 1.05),
        ),

        transforms.ColorJitter(
            brightness=0.10,
            contrast=0.10,
            saturation=0.10,
            hue=0.02,
        ),

        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])


def get_eval_transform(image_size: int = 224):
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),

        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])


def get_train_transform_fixed(image_size: int = 224):
    """
    FIXED training transforms for deepfake detection.

    Key: MINIMAL augmentation to preserve manipulation artifacts.
    Removed destructive transforms that destroy subtle manipulation features.

    NO: horizontal flip, rotation, color jitter (these destroy artifacts)
    YES: very mild geometric transforms only
    """
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        # NO horizontal flip - destroys spatial artifact locations
        # NO rotation - disrupts manipulation patterns
        # NO color jitter - removes color manipulation traces

        # ONLY safe augmentations:
        transforms.RandomResizedCrop(
            image_size,
            scale=(0.9, 1.0),  # Very mild zooming (90-100%)
            ratio=(0.95, 1.05)  # Minimal aspect ratio change
        ),

        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])