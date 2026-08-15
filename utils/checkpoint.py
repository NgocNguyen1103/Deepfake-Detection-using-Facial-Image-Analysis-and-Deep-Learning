# utils/checkpoint.py
# Utility functions for model checkpointing

import torch
import os
from pathlib import Path


def save_checkpoint(model, optimizer, epoch, val_acc, val_loss, checkpoint_path, **kwargs):
    """
    Save model checkpoint.

    Args:
        model: The model to save
        optimizer: The optimizer state
        epoch: Current epoch number
        val_acc: Validation accuracy
        val_loss: Validation loss
        checkpoint_path: Path to save checkpoint
        **kwargs: Additional information to save
    """
    # Create directory if it doesn't exist
    checkpoint_path = Path(checkpoint_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    # Prepare checkpoint dictionary
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_acc': val_acc,
        'val_loss': val_loss,
        **kwargs
    }

    # Save checkpoint
    torch.save(checkpoint, checkpoint_path)
    print(f"✅ Checkpoint saved: {checkpoint_path}")


def load_checkpoint(checkpoint_path, model, optimizer=None, device='cpu'):
    """
    Load model checkpoint.

    Args:
        checkpoint_path: Path to checkpoint file
        model: The model to load weights into
        optimizer: The optimizer to load state into (optional)
        device: Device to load checkpoint on

    Returns:
        Dictionary containing checkpoint information
    """
    checkpoint_path = Path(checkpoint_path)

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Load model state
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)

    # Load optimizer state if provided
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    # Extract additional information
    info = {
        'epoch': checkpoint.get('epoch', 0),
        'val_acc': checkpoint.get('val_acc', 0.0),
        'val_loss': checkpoint.get('val_loss', float('inf'))
    }

    # Remove keys that are already stored in info
    for key in ['epoch', 'model_state_dict', 'optimizer_state_dict', 'val_acc', 'val_loss']:
        checkpoint.pop(key, None)

    info.update(checkpoint)

    print(f"✅ Checkpoint loaded: {checkpoint_path}")
    print(f"   Epoch: {info['epoch']}, Val Acc: {info['val_acc']:.2f}%, Val Loss: {info['val_loss']:.4f}")

    return info


def save_best_model(model, val_acc, val_loss, output_dir, model_name='best_model.pth'):
    """
    Save best model based on validation accuracy.

    Args:
        model: The model to save
        val_acc: Current validation accuracy
        val_loss: Current validation loss
        output_dir: Directory to save model
        model_name: Name of the model file

    Returns:
        bool: True if this is the best model so far
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    best_model_path = output_dir / model_name

    # Check if this is the best model
    is_best = False
    if best_model_path.exists():
        try:
            checkpoint = torch.load(best_model_path, map_location='cpu')
            best_val_acc = checkpoint.get('val_acc', 0.0)

            if val_acc > best_val_acc:
                is_best = True
                print(f"🎉 New best model! Previous best: {best_val_acc:.2f}%, New: {val_acc:.2f}%")
        except Exception as e:
            print(f"⚠️ Could not load previous best model: {e}")
            is_best = True
    else:
        is_best = True
        print(f"🎉 First model saved! Val Acc: {val_acc:.2f}%")

    if is_best:
        save_checkpoint(
            model, None, 0, val_acc, val_loss,
            best_model_path,
            model_name=model_name
        )

    return is_best


def load_latest_checkpoint(checkpoint_dir, model, optimizer=None, device='cpu'):
    """
    Load the latest checkpoint from a directory.

    Args:
        checkpoint_dir: Directory containing checkpoints
        model: The model to load weights into
        optimizer: The optimizer to load state into (optional)
        device: Device to load checkpoint on

    Returns:
        Dictionary containing checkpoint information, or None if no checkpoints found
    """
    checkpoint_dir = Path(checkpoint_dir)

    if not checkpoint_dir.exists():
        return None

    # Find all checkpoint files
    checkpoint_files = list(checkpoint_dir.glob("*.pth"))

    if not checkpoint_files:
        return None

    # Sort by modification time and get the latest
    checkpoint_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
    latest_checkpoint = checkpoint_files[0]

    print(f"Loading latest checkpoint: {latest_checkpoint}")

    return load_checkpoint(latest_checkpoint, model, optimizer, device)


def get_checkpoint_info(checkpoint_path):
    """
    Get information about a checkpoint without loading the full model.

    Args:
        checkpoint_path: Path to checkpoint file

    Returns:
        Dictionary containing checkpoint information
    """
    checkpoint_path = Path(checkpoint_path)

    if not checkpoint_path.exists():
        return None

    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu')

        info = {
            'epoch': checkpoint.get('epoch', 0),
            'val_acc': checkpoint.get('val_acc', 0.0),
            'val_loss': checkpoint.get('val_loss', float('inf')),
            'file_size': checkpoint_path.stat().st_size,
            'file_path': str(checkpoint_path)
        }

        return info
    except Exception as e:
        print(f"⚠️ Could not load checkpoint info: {e}")
        return None
