# data/dataloader.py

from pathlib import Path

from torch.utils.data import DataLoader

from data.FaceDatasetFrame import FaceDatasetFrame
from data.Transform import get_train_transform, get_eval_transform


def build_dataloaders(
    root_dir: str,
    batch_size: int = 32,
    image_size: int = 224,
    num_workers: int = 0,
):
    root_dir = Path(root_dir)

    train_dataset = FaceDatasetFrame(
        root_dir=root_dir,
        csv_path=root_dir / "train.csv",
        transform=get_train_transform(image_size),
    )

    val_dataset = FaceDatasetFrame(
        root_dir=root_dir,
        csv_path=root_dir / "val.csv",
        transform=get_eval_transform(image_size),
    )

    test_dataset = FaceDatasetFrame(
        root_dir=root_dir,
        csv_path=root_dir / "test.csv",
        transform=get_eval_transform(image_size),
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader