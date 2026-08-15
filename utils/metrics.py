# utils/metrics.py
# Utility functions for model evaluation metrics

import torch
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_auc_score
)
import matplotlib.pyplot as plt
import seaborn as sns


def calculate_metrics(y_true, y_pred, y_prob=None):
    """
    Calculate comprehensive evaluation metrics.

    Args:
        y_true: True labels (numpy array or tensor)
        y_pred: Binary predictions (0 or 1)
        y_prob: Prediction probabilities (optional, for AUC-ROC)

    Returns:
        Dictionary of metrics
    """
    # Convert to numpy if tensors
    if torch.is_tensor(y_true):
        y_true = y_true.cpu().numpy()
    if torch.is_tensor(y_pred):
        y_pred = y_pred.cpu().numpy()
    if y_prob is not None and torch.is_tensor(y_prob):
        y_prob = y_prob.cpu().numpy()

    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1_score": f1_score(y_true, y_pred, zero_division=0),
    }

    # Add AUC-ROC if probabilities provided
    if y_prob is not None:
        try:
            metrics["auc_roc"] = roc_auc_score(y_true, y_prob)
        except:
            metrics["auc_roc"] = 0.5

    # Add confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    metrics["confusion_matrix"] = cm

    return metrics


def print_metrics(metrics, title="Evaluation Metrics"):
    """Print metrics in a formatted way."""
    print(f"\n{'='*60}")
    print(f"{title}")
    print(f"{'='*60}")

    print(f"Accuracy:  {metrics.get('accuracy', 0):.4f}")
    print(f"Precision: {metrics.get('precision', 0):.4f}")
    print(f"Recall:    {metrics.get('recall', 0):.4f}")
    print(f"F1-Score:  {metrics.get('f1_score', 0):.4f}")

    if 'auc_roc' in metrics:
        print(f"AUC-ROC:   {metrics['auc_roc']:.4f}")

    if 'confusion_matrix' in metrics:
        cm = metrics['confusion_matrix']
        print(f"\nConfusion Matrix:")
        print("                 Predicted")
        print("              Real    Fake")
        print(f"Actual Real    {cm[0, 0]:4d}    {cm[0, 1]:4d}")
        print(f"       Fake    {cm[1, 0]:4d}    {cm[1, 1]:4d}")

    print(f"{'='*60}\n")


def plot_confusion_matrix(cm, save_path, title="Confusion Matrix"):
    """
    Plot and save confusion matrix.

    Args:
        cm: Confusion matrix
        save_path: Path to save the plot
        title: Plot title
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
    plt.title(title)
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def calculate_epoch_metrics(outputs, labels, threshold=0.5):
    """
    Calculate metrics for a single epoch.

    Args:
        outputs: Model outputs (logits)
        labels: Ground truth labels
        threshold: Classification threshold

    Returns:
        Dictionary of metrics
    """
    # Convert logits to probabilities
    probs = torch.sigmoid(outputs)

    # Convert to binary predictions
    preds = (probs >= threshold).float()

    # Calculate accuracy
    correct = (preds == labels).sum().item()
    total = labels.size(0)
    accuracy = correct / total if total > 0 else 0.0

    return {
        'accuracy': accuracy,
        'correct': correct,
        'total': total,
        'predictions': preds.cpu().numpy(),
        'probabilities': probs.cpu().numpy(),
        'labels': labels.cpu().numpy()
    }


def average_metrics(metrics_list):
    """
    Average metrics from multiple batches.

    Args:
        metrics_list: List of metric dictionaries

    Returns:
        Dictionary of averaged metrics
    """
    if not metrics_list:
        return {}

    # Calculate averages
    avg_metrics = {}
    for key in metrics_list[0].keys():
        if key not in ['predictions', 'probabilities', 'labels']:
            values = [m[key] for m in metrics_list if key in m]
            if values:
                avg_metrics[key] = sum(values) / len(values)

    return avg_metrics
