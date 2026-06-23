# model/train_baseline.py

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

import torch
import torch.nn as nn
from tqdm import tqdm
import pandas as pd
from data.DataLoader import build_dataloaders
from model.efficientnet_baseline import EfficientNetB0Baseline


def train_one_epoch(model, train_loader, criterion, optimizer, device):
    model.train()

    total_loss = 0.0
    correct = 0
    total = 0

    for batch in tqdm(train_loader, desc="Training"):
        images = batch["image"].to(device)
        labels = batch["label"].to(device)

        optimizer.zero_grad()

        logits = model(images).squeeze(1)
        loss = criterion(logits, labels)

        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)

        probs = torch.sigmoid(logits)
        preds = (probs >= 0.5).float()

        correct += (preds == labels).sum().item()
        total += labels.size(0)

    avg_loss = total_loss / total
    acc = correct / total

    return avg_loss, acc


def validate(model, val_loader, criterion, device):
    model.eval()

    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Validation"):
            images = batch["image"].to(device)
            labels = batch["label"].to(device)

            logits = model(images).squeeze(1)
            loss = criterion(logits, labels)

            total_loss += loss.item() * images.size(0)

            probs = torch.sigmoid(logits)
            preds = (probs >= 0.5).float()

            correct += (preds == labels).sum().item()
            total += labels.size(0)

    avg_loss = total_loss / total
    acc = correct / total

    return avg_loss, acc


def save_checkpoint(
    checkpoint_path,
    model,
    optimizer,
    epoch,
    best_val_acc,
    train_loss,
    train_acc,
    val_loss,
    val_acc,
):
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "best_val_acc": best_val_acc,
        "train_loss": train_loss,
        "train_acc": train_acc,
        "val_loss": val_loss,
        "val_acc": val_acc,
    }

    torch.save(checkpoint, checkpoint_path)


def load_checkpoint(checkpoint_path, model, optimizer, device):
    checkpoint = torch.load(checkpoint_path, map_location=device)

    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    start_epoch = int(checkpoint["epoch"])
    best_val_acc = float(checkpoint["best_val_acc"])

    print(f"Loaded checkpoint from: {checkpoint_path}")
    print(f"Resume from epoch: {start_epoch + 1}")
    print(f"Best Val Acc: {best_val_acc:.4f}")

    return start_epoch, best_val_acc


def main():
    root_dir = r"D:\M1\M1 Internship\preprocessed_dataset"

    batch_size = 16
    num_epochs = 10
    learning_rate = 1e-4
    num_workers = 0
    resume = False

    checkpoint_dir = PROJECT_ROOT / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    history = []
    latest_checkpoint_path = checkpoint_dir / "efficientnet_b0_baseline_latest.pth"
    best_checkpoint_path = checkpoint_dir / "efficientnet_b0_baseline_best.pth"
    history_path = checkpoint_dir / "baseline_training_history.csv"

    if history_path.exists():
        history = pd.read_csv(history_path).to_dict("records")
    else:
        history = []
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    train_loader, val_loader, test_loader = build_dataloaders(
        root_dir=root_dir,
        batch_size=batch_size,
        image_size=224,
        num_workers=num_workers,
    )
    print("\nDataset size:")
    print("Train images:", len(train_loader.dataset))
    print("Val images:", len(val_loader.dataset))
    print("Test images:", len(test_loader.dataset))

    print("\nNumber of batches:")
    print("Train batches:", len(train_loader))
    print("Val batches:", len(val_loader))
    print("Test batches:", len(test_loader))

    print("\nTrain label distribution:")
    print(train_loader.dataset.df["label_id"].value_counts())

    print("\nVal label distribution:")
    print(val_loader.dataset.df["label_id"].value_counts())

    print("\nTest label distribution:")
    print(test_loader.dataset.df["label_id"].value_counts())

    model = EfficientNetB0Baseline(pretrained=True)
    model = model.to(device)

    criterion = nn.BCEWithLogitsLoss()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=1e-4,
    )

    start_epoch = 0
    best_val_acc = -1.0

    if resume and latest_checkpoint_path.exists():
        start_epoch, best_val_acc = load_checkpoint(
            checkpoint_path=latest_checkpoint_path,
            model=model,
            optimizer=optimizer,
            device=device,
        )
    else:
        print("No checkpoint found. Training from scratch.")
    for epoch in range(start_epoch, num_epochs):
        print(f"\nEpoch [{epoch + 1}/{num_epochs}]")

        train_loss, train_acc = train_one_epoch(
            model=model,
            train_loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
        )

        val_loss, val_acc = validate(
            model=model,
            val_loader=val_loader,
            criterion=criterion,
            device=device,
        )

        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
        print(f"Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.4f}")

        history.append({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
        })

        pd.DataFrame(history).to_csv(history_path, index=False)
        is_best = val_acc > best_val_acc

        if is_best:
            best_val_acc = val_acc

        save_checkpoint(
            checkpoint_path=latest_checkpoint_path,
            model=model,
            optimizer=optimizer,
            epoch=epoch + 1,
            best_val_acc=best_val_acc,
            train_loss=train_loss,
            train_acc=train_acc,
            val_loss=val_loss,
            val_acc=val_acc,
        )

        print(f"Saved latest checkpoint to: {latest_checkpoint_path}")

        if is_best:
            save_checkpoint(
                checkpoint_path=best_checkpoint_path,
                model=model,
                optimizer=optimizer,
                epoch=epoch + 1,
                best_val_acc=best_val_acc,
                train_loss=train_loss,
                train_acc=train_acc,
                val_loss=val_loss,
                val_acc=val_acc,
            )

            print(f"Saved best checkpoint to: {best_checkpoint_path}")

    print("\nTraining finished.")
    print(f"Best Val Acc: {best_val_acc:.4f}")


if __name__ == "__main__":
    main()