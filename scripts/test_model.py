import torch
import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

import pandas as pd
from data.DataLoader import build_dataloaders
from efficientnet_baseline import EfficientNetB0Baseline


def main():
    root_dir = r"D:\M1\M1 Internship\preprocessed_dataset"

    train_loader, val_loader, test_loader = build_dataloaders(
        root_dir=root_dir,
        batch_size=8,
        image_size=224,
        num_workers=0,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = EfficientNetB0Baseline(pretrained=True)
    model = model.to(device)

    batch = next(iter(train_loader))

    images = batch["image"].to(device)
    labels = batch["label"].to(device)

    logits = model(images)

    print("Images shape:", images.shape)
    print("Labels shape:", labels.shape)
    print("Logits shape:", logits.shape)
    print("Logits:", logits[:5])


if __name__ == "__main__":
    main()