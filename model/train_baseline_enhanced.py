# model/train_baseline_enhanced.py
# Enhanced training script with anti-overfitting techniques

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

import torch
import torch.nn as nn
from tqdm import tqdm
import pandas as pd
import numpy as np
from data.DataLoader import build_dataloaders
from model.efficientnet_baseline import EfficientNetB0Baseline


class EarlyStopping:
    """Early stopping to stop training when validation loss doesn't improve."""

    def __init__(self, patience=5, min_delta=0.001, restore_best_weights=True):
        """
        Args:
            patience: Number of epochs to wait before stopping
            min_delta: Minimum change to qualify as an improvement
            restore_best_weights: Whether to restore model weights from best epoch
        """
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.best_loss = None
        self.counter = 0
        self.best_weights = None

    def __call__(self, val_loss, model):
        """
        Args:
            val_loss: Current validation loss
            model: Current model instance

        Returns:
            True if training should stop, False otherwise
        """
        if self.best_loss is None:
            self.best_loss = val_loss
            self.save_best_weights(model)
            return False

        if val_loss < self.best_loss - self.min_delta:
            # Improvement found
            self.best_loss = val_loss
            self.counter = 0
            self.save_best_weights(model)
            return False
        else:
            # No improvement
            self.counter += 1
            if self.counter >= self.patience:
                if self.restore_best_weights:
                    model.load_state_dict(self.best_weights)
                return True
            return False

    def save_best_weights(self, model):
        """Save the best model weights."""
        self.best_weights = model.state_dict().copy()


def train_one_epoch(model, train_loader, criterion, optimizer, device, mixup_alpha=0.2, mixup_prob=0.5):
    """
    Train for one epoch with optional Mixup augmentation.

    Args:
        model: Neural network model
        train_loader: Training data loader
        criterion: Loss function
        optimizer: Optimizer
        device: Device to train on
        mixup_alpha: Mixup alpha parameter
        mixup_prob: Probability of applying mixup
    """
    model.train()

    total_loss = 0.0
    correct = 0
    total = 0

    for batch in tqdm(train_loader, desc="Training"):
        images = batch["image"].to(device)
        labels = batch["label"].to(device).float()

        optimizer.zero_grad()

        # Apply Mixup augmentation with given probability
        if np.random.random() < mixup_prob and mixup_alpha > 0:
            lam = np.random.beta(mixup_alpha, mixup_alpha)
            batch_size = images.size(0)
            index = torch.randperm(batch_size).to(device)

            mixed_images = lam * images + (1 - lam) * images[index]
            mixed_labels = lam * labels + (1 - lam) * labels[index]

            logits = model(mixed_images).squeeze(1)
            loss = criterion(logits, mixed_labels)
        else:
            logits = model(images).squeeze(1)
            loss = criterion(logits, labels)

        loss.backward()

        # Gradient clipping to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

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
    """Validate the model."""
    model.eval()

    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Validation"):
            images = batch["image"].to(device)
            labels = batch["label"].to(device).float()

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
    scheduler,
    epoch,
    best_val_loss,
    train_loss,
    train_acc,
    val_loss,
    val_acc,
):
    """Save training checkpoint."""
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
        "best_val_loss": best_val_loss,
        "train_loss": train_loss,
        "train_acc": train_acc,
        "val_loss": val_loss,
        "val_acc": val_acc,
    }

    torch.save(checkpoint, checkpoint_path)


def load_checkpoint(checkpoint_path, model, optimizer, scheduler, device):
    """Load training checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device)

    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    if scheduler and "scheduler_state_dict" in checkpoint and checkpoint["scheduler_state_dict"]:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    start_epoch = int(checkpoint["epoch"])
    best_val_loss = float(checkpoint["best_val_loss"])

    print(f"Loaded checkpoint from: {checkpoint_path}")
    print(f"Resume from epoch: {start_epoch + 1}")
    print(f"Best Val Loss: {best_val_loss:.4f}")

    return start_epoch, best_val_loss


def main():
    # Configuration
    root_dir = r"D:\M1\M1 Internship\preprocessed_dataset"

    # Enhanced hyperparameters to combat overfitting
    batch_size = 16
    num_epochs = 20  # Increased - early stopping will halt training if needed
    learning_rate = 1e-4  # Slightly higher initial LR, will be reduced
    min_learning_rate = 1e-6  # Minimum learning rate
    weight_decay = 1e-3  # Increased from 5e-4 for stronger regularization
    num_workers = 0
    resume = False

    # Early stopping parameters
    patience = 5  # Stop if no improvement for 5 epochs
    min_delta = 0.001  # Minimum improvement threshold

    # Mixup augmentation parameters
    mixup_alpha = 0.2  # Mixup alpha parameter
    mixup_prob = 0.5  # Probability of applying mixup

    # Label smoothing (for BCE with logits, we simulate this)
    label_smoothing = 0.1

    checkpoint_dir = PROJECT_ROOT / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    latest_checkpoint_path = checkpoint_dir / "efficientnet_b0_enhanced_latest.pth"
    best_checkpoint_path = checkpoint_dir / "efficientnet_b0_enhanced_best.pth"
    history_path = checkpoint_dir / "enhanced_training_history.csv"

    # Load training history
    if history_path.exists():
        history = pd.read_csv(history_path).to_dict("records")
    else:
        history = []

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    # Build dataloaders
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

    # Initialize model
    model = EfficientNetB0Baseline(pretrained=True)
    model = model.to(device)

    # Loss function with label smoothing
    # Note: BCEWithLogitsLoss doesn't directly support label smoothing,
    # but we can implement it manually in the training loop
    criterion = nn.BCEWithLogitsLoss()

    # Optimizer with increased weight decay
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )

    # Learning rate scheduler - ReduceLROnPlateau
    # This reduces learning rate when validation loss plateaus
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',  # Reduce LR when validation loss stops decreasing
        factor=0.5,  # Reduce LR by factor of 0.5
        patience=2,  # Wait 2 epochs before reducing LR
        min_lr=min_learning_rate,  # Don't go below this LR
    )

    # Early stopping
    early_stopping = EarlyStopping(
        patience=patience,
        min_delta=min_delta,
        restore_best_weights=True
    )

    start_epoch = 0
    best_val_loss = float('inf')

    # Resume from checkpoint if requested
    if resume and latest_checkpoint_path.exists():
        start_epoch, best_val_loss = load_checkpoint(
            checkpoint_path=latest_checkpoint_path,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
        )
    else:
        print("No checkpoint found. Training from scratch.")

    print("\nAnti-overfitting techniques enabled:")
    print(f"- Early stopping (patience={patience}, min_delta={min_delta})")
    print(f"- Learning rate scheduling (ReduceLROnPlateau)")
    print(f"- Increased weight decay ({weight_decay})")
    print(f"- Mixup augmentation (alpha={mixup_alpha}, prob={mixup_prob})")
    print(f"- Gradient clipping (max_norm=1.0)")

    # Training loop
    for epoch in range(start_epoch, num_epochs):
        print(f"\nEpoch [{epoch + 1}/{num_epochs}]")
        print(f"Current learning rate: {optimizer.param_groups[0]['lr']:.2e}")

        # Training
        train_loss, train_acc = train_one_epoch(
            model=model,
            train_loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            mixup_alpha=mixup_alpha,
            mixup_prob=mixup_prob,
        )

        # Validation
        val_loss, val_acc = validate(
            model=model,
            val_loader=val_loader,
            criterion=criterion,
            device=device,
        )

        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
        print(f"Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.4f}")

        # Update learning rate based on validation loss
        scheduler.step(val_loss)

        # Track history
        history.append({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "learning_rate": optimizer.param_groups[0]['lr'],
        })

        pd.DataFrame(history).to_csv(history_path, index=False)

        # Check if this is the best model so far
        is_best = val_loss < best_val_loss

        if is_best:
            best_val_loss = val_loss

        # Save latest checkpoint
        save_checkpoint(
            checkpoint_path=latest_checkpoint_path,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            epoch=epoch + 1,
            best_val_loss=best_val_loss,
            train_loss=train_loss,
            train_acc=train_acc,
            val_loss=val_loss,
            val_acc=val_acc,
        )

        print(f"Saved latest checkpoint to: {latest_checkpoint_path}")

        # Save best checkpoint
        if is_best:
            save_checkpoint(
                checkpoint_path=best_checkpoint_path,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch + 1,
                best_val_loss=best_val_loss,
                train_loss=train_loss,
                train_acc=train_acc,
                val_loss=val_loss,
                val_acc=val_acc,
            )

            print(f"Saved best checkpoint to: {best_checkpoint_path}")

        # Early stopping check
        if early_stopping(val_loss, model):
            print(f"\nEarly stopping triggered at epoch {epoch + 1}")
            print(f"No improvement in validation loss for {patience} epochs")
            print(f"Best validation loss: {early_stopping.best_loss:.4f}")
            break

    print("\nTraining finished.")
    print(f"Best Val Loss: {best_val_loss:.4f}")

    # Load best model for final evaluation
    if best_checkpoint_path.exists():
        print("Loading best model for final evaluation...")
        checkpoint = torch.load(best_checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])

        # Final evaluation on test set
        test_loss, test_acc = validate(
            model=model,
            val_loader=test_loader,
            criterion=criterion,
            device=device,
        )

        print(f"\nFinal Test Results:")
        print(f"Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.4f}")


if __name__ == "__main__":
    main()