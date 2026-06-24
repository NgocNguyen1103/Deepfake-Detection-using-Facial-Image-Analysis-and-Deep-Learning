# model/evaluate_model.py
# Model evaluation script with comprehensive metrics

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)
import matplotlib.pyplot as plt
import seaborn as sns
from data.DataLoader import build_dataloaders
from model.efficientnet_baseline import EfficientNetB0Baseline


def evaluate_model(model, test_loader, device):
    """
    Evaluate model on test dataset and return predictions and true labels.

    Args:
        model: Trained model
        test_loader: Test data loader
        device: Device to run evaluation on

    Returns:
        Tuple of (true_labels, predictions, probabilities)
    """
    model.eval()

    all_labels = []
    all_preds = []
    all_probs = []

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating"):
            images = batch["image"].to(device)
            labels = batch["label"].to(device).float()

            logits = model(images).squeeze(1)
            probs = torch.sigmoid(logits)
            preds = (probs >= 0.5).float()

            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    return np.array(all_labels), np.array(all_preds), np.array(all_probs)


def calculate_metrics(y_true, y_pred, y_prob):
    """
    Calculate comprehensive evaluation metrics.

    Args:
        y_true: True labels
        y_pred: Binary predictions (0 or 1)
        y_prob: Prediction probabilities

    Returns:
        Dictionary of metrics
    """
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1_score": f1_score(y_true, y_pred, zero_division=0),
        "auc_roc": roc_auc_score(y_true, y_prob),
    }

    return metrics


def plot_confusion_matrix(cm, save_path):
    """
    Plot and save confusion matrix.

    Args:
        cm: Confusion matrix
        save_path: Path to save the plot
    """
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Real", "Fake"],
        yticklabels=["Real", "Fake"],
    )
    plt.title("Confusion Matrix")
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Confusion matrix saved to: {save_path}")


def save_evaluation_results(metrics, cm, output_csv_path, confusion_matrix_image_path):
    """
    Save evaluation results to CSV.

    Args:
        metrics: Dictionary of metrics
        cm: Confusion matrix
        output_csv_path: Path to save CSV
        confusion_matrix_image_path: Path where confusion matrix image is saved
    """
    # Create results dictionary
    results = {
        "metric": ["accuracy", "precision", "recall", "f1_score", "auc_roc"],
        "value": [
            metrics["accuracy"],
            metrics["precision"],
            metrics["recall"],
            metrics["f1_score"],
            metrics["auc_roc"],
        ],
    }

    df = pd.DataFrame(results)

    # Save to CSV
    df.to_csv(output_csv_path, index=False)
    print(f"\nEvaluation results saved to: {output_csv_path}")

    # Print results
    print("\n" + "=" * 50)
    print("EVALUATION RESULTS ON TEST SET")
    print("=" * 50)
    for _, row in df.iterrows():
        print(f"{row['metric']:15}: {row['value']:.4f}")
    print("=" * 50)

    # Print confusion matrix
    print("\nConfusion Matrix:")
    print("                 Predicted")
    print("              Real    Fake")
    print(f"Actual Real    {cm[0, 0]:4d}    {cm[0, 1]:4d}")
    print(f"       Fake    {cm[1, 0]:4d}    {cm[1, 1]:4d}")
    print()

    # Calculate and display additional metrics
    tn, fp, fn, tp = cm.ravel()
    print(f"True Negatives (Real correctly identified): {tn}")
    print(f"False Positives (Real incorrectly as Fake): {fp}")
    print(f"False Negatives (Fake incorrectly as Real): {fn}")
    print(f"True Positives (Fake correctly identified): {tp}")
    print(f"Confusion matrix plot saved to: {confusion_matrix_image_path}")


def main():
    # Configuration
    root_dir = r"D:\M1\M1 Internship\preprocessed_dataset"
    checkpoint_path = PROJECT_ROOT / "checkpoints" / "efficientnet_b0_enhanced_best.pth"
    batch_size = 16
    image_size = 224
    num_workers = 0

    # Output paths
    output_dir = PROJECT_ROOT / "output"
    output_dir.mkdir(parents=True, exist_ok=True)

    output_csv_path = output_dir / "baseline_evaluation.csv"
    confusion_matrix_image_path = output_dir / "baseline_confusion_matrix.png"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    # Load checkpoint
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    print(f"\nLoading checkpoint from: {checkpoint_path}")

    # Build test dataloader
    print("\nBuilding test dataloader...")
    _, _, test_loader = build_dataloaders(
        root_dir=root_dir,
        batch_size=batch_size,
        image_size=image_size,
        num_workers=num_workers,
    )

    print(f"Test dataset size: {len(test_loader.dataset)}")
    print(f"Number of test batches: {len(test_loader)}")

    print("\nTest label distribution:")
    print(test_loader.dataset.df["label_id"].value_counts())

    # Initialize model
    print("\nInitializing EfficientNet-B0 model...")
    model = EfficientNetB0Baseline(pretrained=False)
    model = model.to(device)

    # Load checkpoint weights
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    print("Checkpoint loaded successfully")

    # Evaluate model
    print("\nEvaluating model on test set...")
    y_true, y_pred, y_prob = evaluate_model(model, test_loader, device)

    # Calculate metrics
    print("\nCalculating metrics...")
    metrics = calculate_metrics(y_true, y_pred, y_prob)

    # Calculate confusion matrix
    cm = confusion_matrix(y_true, y_pred)

    # Plot confusion matrix
    plot_confusion_matrix(cm, confusion_matrix_image_path)

    # Save and display results
    save_evaluation_results(metrics, cm, output_csv_path, confusion_matrix_image_path)

    print("\nEvaluation completed successfully!")


if __name__ == "__main__":
    main()
