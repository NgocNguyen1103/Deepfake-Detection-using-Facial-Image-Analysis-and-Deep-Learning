"""
Training Script for EfficientNet-CBAM Model

Features:
- Transfer learning from baseline model
- Differential learning rates (higher LR for CBAM modules)
- Early stopping and learning rate scheduling
- Training history tracking and visualization

Usage:
    # Standard training
    python training/train_cbam.py --baseline checkpoints/baselines/efficientnet_b0_enhanced_best.pth

    # Staged training (recommended)
    python training/train_cbam.py --baseline checkpoints/baselines/efficientnet_b0_enhanced_best.pth --staged_training

    # Custom staged training
    python training/train_cbam.py --baseline checkpoints/baselines/efficientnet_b0_enhanced_best.pth --staged_training --stage1_epochs 15 --stage2_epochs 35
"""

import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
from datetime import datetime
import argparse

from models.efficientnet_cbam import EfficientNetCBAM, create_efficientnet_cbam
from data.DataLoader import build_dataloaders


class EarlyStopping:

    def __init__(self, patience=7, min_delta=0.001, restore_best_weights=True):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.best_loss = None
        self.counter = 0
        self.best_weights = None

    def __call__(self, val_loss, model):
        if self.best_loss is None:
            self.best_loss = val_loss
            self.save_best_weights(model)
            return False

        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            self.save_best_weights(model)
            return False
        else:
            self.counter += 1
            if self.counter >= self.patience:
                if self.restore_best_weights:
                    model.load_state_dict(self.best_weights)
                return True
            return False

    def save_best_weights(self, model):
        self.best_weights = model.state_dict().copy()


class CBAMTrainer:
    def __init__(
        self,
        model,
        train_loader,
        val_loader,
        config,
        device='cuda',
        output_dir=None
    ):
        """
        Args:
            model: EfficientNetCBAM model
            train_loader: Training data loader
            val_loader: Validation data loader
            config: Training configuration dict
            device: Device to train on
            output_dir: Directory to save results
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = device

        # Loss function
        self.criterion = nn.BCEWithLogitsLoss()

        # Optimizer with differential learning rates
        self.optimizer = self._create_optimizer()

        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=config.get('lr_factor', 0.5),
            patience=config.get('lr_patience', 3),
            min_lr=1e-7
        )

        # Early stopping
        self.early_stopping = EarlyStopping(
            patience=config.get('early_stopping_patience', 10),
            min_delta=0.001
        )

        # Training history
        self.history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'learning_rate': []
        }

        # Output directory
        if output_dir is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_dir = Path(f'checkpoints/experiments/cbam_{timestamp}')
        else:
            output_dir = Path(output_dir)

        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Save configuration
        with open(self.output_dir / 'config.json', 'w') as f:
            json.dump(config, f, indent=2)

        # Model info
        print(f"Model initialized:")
        params = model.count_parameters()

    def _create_optimizer(self):
        base_lr = self.config.get('learning_rate', 1e-4)
        # Separate parameter groups
        efficientnet_params = []
        cbam_params = []
        classifier_params = []

        for name, param in self.model.named_parameters():
            if 'cbam' in name:
                cbam_params.append(param)
            elif 'classifier' in name:
                classifier_params.append(param)
            else:
                efficientnet_params.append(param)

        cbam_lr = base_lr * self.config.get('cbam_lr_multiplier', 2.0)
        classifier_lr = base_lr * self.config.get('classifier_lr_multiplier', 1.5)

        print(f"Learning rates:")
        print(f"EfficientNet: {base_lr:.6f}")
        print(f"CBAM: {cbam_lr:.6f} (×{self.config.get('cbam_lr_multiplier', 2.0)})")
        print(f"Classifier: {classifier_lr:.6f} (×{self.config.get('classifier_lr_multiplier', 1.5)})")

        return optim.Adam([
            {'params': efficientnet_params, 'lr': base_lr, 'name': 'efficientnet'},
            {'params': cbam_params, 'lr': cbam_lr, 'name': 'cbam'},
            {'params': classifier_params, 'lr': classifier_lr, 'name': 'classifier'}
        ], weight_decay=self.config.get('weight_decay', 1e-4))

    def train_one_epoch(self, epoch, stage="Standard"):
        """Train for one epoch"""

        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        pbar = tqdm(self.train_loader, desc=f'{stage} - Epoch {epoch+1}')
        for batch in pbar:
            images = batch["image"].to(self.device)
            labels = batch["label"].to(self.device).float()

            self.optimizer.zero_grad()

            # Forward pass
            outputs = self.model(images).squeeze(1)
            loss = self.criterion(outputs, labels)

            # Backward pass
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                max_norm=self.config.get('gradient_clip_max_norm', 1.0)
            )

            self.optimizer.step()

            # Metrics
            total_loss += loss.item()
            probs = torch.sigmoid(outputs)
            preds = (probs >= 0.5).float()
            correct += (preds == labels).sum().item()
            total += labels.size(0)

            # Update progress bar
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{100.*correct/total:.2f}%'
            })

        avg_loss = total_loss / len(self.train_loader)
        accuracy = 100. * correct / total

        return avg_loss, accuracy

    def validate(self):

        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc='Validation'):
                images = batch["image"].to(self.device)
                labels = batch["label"].to(self.device).float()

                outputs = self.model(images).squeeze(1)
                loss = self.criterion(outputs, labels)

                total_loss += loss.item()
                probs = torch.sigmoid(outputs)
                preds = (probs >= 0.5).float()
                correct += (preds == labels).sum().item()
                total += labels.size(0)

        avg_loss = total_loss / len(self.val_loader)
        accuracy = 100. * correct / total

        return avg_loss, accuracy

    def train(self):
        """Full training loop"""

        num_epochs = self.config.get('epochs', 50)
        best_val_loss = float('inf')

        print(f"\nStarting training for {num_epochs} epochs...")

        for epoch in range(num_epochs):
            # Train
            train_loss, train_acc = self.train_one_epoch(epoch, stage="Training")

            # Validate
            val_loss, val_acc = self.validate()

            # Update learning rate
            self.scheduler.step(val_loss)
            current_lr = self.optimizer.param_groups[0]['lr']

            # Record history
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)
            self.history['learning_rate'].append(current_lr)

            print(f"\nEpoch {epoch+1}/{num_epochs}:")
            print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
            print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
            print(f"LR: {current_lr:.6f}")

            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                self._save_checkpoint(epoch, val_loss, val_acc, 'best')
                print(f"Best model saved! (Val Loss: {val_loss:.4f})")

            # Early stopping check
            if self.early_stopping(val_loss, self.model):
                print(f"\nEarly stopping triggered at epoch {epoch+1}")
                break

            # Save periodic checkpoint
            if (epoch + 1) % 5 == 0:
                self._save_checkpoint(epoch, val_loss, val_acc, f'epoch_{epoch+1}')

        # Save training history
        self._save_history()

        print(f"\n{'='*60}")
        print("Training complete!")
        print(f"Best Val Loss: {best_val_loss:.4f}")
        print(f"Results saved to: {self.output_dir}")
        print(f"{'='*60}")

        return self.model, self.history

    def freeze_efficientnet(self):
        """Freeze EfficientNet parameters, keep CBAM and classifier trainable"""
        print("Freezing EfficientNet features...")
        for param in self.model.features.parameters():
            param.requires_grad = False

        # Count trainable parameters
        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.model.parameters())
        print(f"   Trainable: {trainable:,} / {total:,} ({trainable/total*100:.1f}%)")

    def unfreeze_efficientnet(self):
        """Unfreeze EfficientNet parameters for fine-tuning"""
        print("Unfreezing EfficientNet features...")
        for param in self.model.features.parameters():
            param.requires_grad = True

        # Count trainable parameters
        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.model.parameters())
        print(f"   Trainable: {trainable:,} / {total:,} ({trainable/total*100:.1f}%)")

    def recreate_optimizer_for_stage(self, stage):
        """Recreate optimizer with stage-specific learning rates"""

        if stage == 1:
            # Stage 1: High LR for CBAM and classifier
            base_lr = self.config.get('stage1_lr', 1e-3)
            cbam_lr = base_lr * 2.0
            classifier_lr = base_lr * 2.0
            efficientnet_lr = 0.0  # Frozen
        else:
            # Stage 2: Low LR for fine-tuning
            base_lr = self.config.get('stage2_lr', 1e-5)
            cbam_lr = base_lr * 2.0
            classifier_lr = base_lr * 2.0
            efficientnet_lr = base_lr

        print(f"Stage {stage} Learning rates:")
        print(f"EfficientNet: {efficientnet_lr:.6f}")
        print(f"CBAM: {cbam_lr:.6f}")
        print(f"Classifier: {classifier_lr:.6f}")

        # Separate parameter groups
        efficientnet_params = []
        cbam_params = []
        classifier_params = []

        for name, param in self.model.named_parameters():
            if 'cbam' in name:
                cbam_params.append(param)
            elif 'classifier' in name:
                classifier_params.append(param)
            else:
                efficientnet_params.append(param)

        # Create new optimizer
        if stage == 1:
            self.optimizer = optim.Adam([
                {'params': cbam_params, 'lr': cbam_lr, 'name': 'cbam'},
                {'params': classifier_params, 'lr': classifier_lr, 'name': 'classifier'}
            ], weight_decay=self.config.get('weight_decay', 1e-4))
        else:
            self.optimizer = optim.Adam([
                {'params': efficientnet_params, 'lr': efficientnet_lr, 'name': 'efficientnet'},
                {'params': cbam_params, 'lr': cbam_lr, 'name': 'cbam'},
                {'params': classifier_params, 'lr': classifier_lr, 'name': 'classifier'}
            ], weight_decay=self.config.get('weight_decay', 1e-4))

        # Recreate scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=self.config.get('lr_factor', 0.5),
            patience=self.config.get('lr_patience', 3),
            min_lr=1e-7
        )

    def train_staged(self, stage1_epochs=15, stage2_epochs=35):

        print(f"\nSTARTING STAGED TRAINING")
        print(f"{'='*60}")
        print(f"Stage 1: Train CBAM + Classifier (EfficientNet frozen) - {stage1_epochs} epochs")
        print(f"Stage 2: Fine-tune all layers - {stage2_epochs} epochs")
        print(f"{'='*60}\n")

        # ===== STAGE 1: Freeze EfficientNet, Train CBAM + Classifier =====
        print("\nSTAGE 1: Training CBAM + Classifier")
        print(f"{'='*60}")

        self.freeze_efficientnet()
        self.recreate_optimizer_for_stage(stage=1)

        # Reset early stopping for stage 1
        self.early_stopping = EarlyStopping(patience=self.config.get('early_stopping_patience', 7))

        stage1_history = {
            'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': [], 'learning_rate': []
        }

        for epoch in range(stage1_epochs):
            # Train
            train_loss, train_acc = self.train_one_epoch(epoch, stage="Stage 1")

            # Validate
            val_loss, val_acc = self.validate()

            # Update learning rate
            self.scheduler.step(val_loss)
            current_lr = self.optimizer.param_groups[0]['lr']

            # Record history
            stage1_history['train_loss'].append(train_loss)
            stage1_history['train_acc'].append(train_acc)
            stage1_history['val_loss'].append(val_loss)
            stage1_history['val_acc'].append(val_acc)
            stage1_history['learning_rate'].append(current_lr)

            print(f"\nStage 1 - Epoch {epoch+1}/{stage1_epochs}:")
            print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
            print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
            print(f"LR: {current_lr:.6f}")

            # Early stopping for stage 1
            if self.early_stopping(val_loss, self.model):
                print(f"\nStage 1 early stopping at epoch {epoch+1}")
                break

        # Save stage 1 checkpoint
        self._save_checkpoint(epoch, val_loss, val_acc, 'stage1_final')
        print(f"Stage 1 complete! Best Val Acc: {max(stage1_history['val_acc']):.2f}%")

        # ===== STAGE 2: Unfreeze EfficientNet, Fine-tune All =====
        print(f"\nSTAGE 2: Fine-tuning All Layers")
        print(f"{'='*60}")

        self.unfreeze_efficientnet()
        self.recreate_optimizer_for_stage(stage=2)

        # Reset early stopping for stage 2
        self.early_stopping = EarlyStopping(patience=self.config.get('early_stopping_patience', 10))
        best_val_loss = float('inf')

        for epoch in range(stage2_epochs):
            # Train
            train_loss, train_acc = self.train_one_epoch(epoch, stage="Stage 2")

            # Validate
            val_loss, val_acc = self.validate()

            # Update learning rate
            self.scheduler.step(val_loss)
            current_lr = self.optimizer.param_groups[0]['lr']

            # Record history (append to stage 1 history)
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)
            self.history['learning_rate'].append(current_lr)

            print(f"\nStage 2 - Epoch {epoch+1}/{stage2_epochs}:")
            print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
            print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
            print(f"LR: {current_lr:.6f}")

            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                self._save_checkpoint(epoch, val_loss, val_acc, 'best')
                print(f"Best model saved! (Val Loss: {val_loss:.4f})")

            # Early stopping for stage 2
            if self.early_stopping(val_loss, self.model):
                print(f"\nStage 2 early stopping at epoch {epoch+1}")
                break

            # Save periodic checkpoint
            if (epoch + 1) % 5 == 0:
                self._save_checkpoint(epoch, val_loss, val_acc, f'stage2_epoch_{epoch+1}')

        # Combine histories
        self.history['train_loss'] = stage1_history['train_loss'] + self.history['train_loss']
        self.history['train_acc'] = stage1_history['train_acc'] + self.history['train_acc']
        self.history['val_loss'] = stage1_history['val_loss'] + self.history['val_loss']
        self.history['val_acc'] = stage1_history['val_acc'] + self.history['val_acc']
        self.history['learning_rate'] = stage1_history['learning_rate'] + self.history['learning_rate']

        # Save training history
        self._save_history()

        print(f"\n{'='*60}")
        print("STAGED TRAINING COMPLETE!")
        print(f"Stage 1 epochs: {len(stage1_history['train_loss'])}")
        print(f"Stage 2 epochs: {len(self.history['train_acc']) - len(stage1_history['train_acc'])}")
        print(f"Best Val Loss: {best_val_loss:.4f}")
        print(f"Results saved to: {self.output_dir}")
        print(f"{'='*60}")

        return self.model, self.history

    def _save_checkpoint(self, epoch, val_loss, val_acc, filename):
        """Save model checkpoint"""

        checkpoint_path = self.output_dir / f'efficientnet_cbam_{filename}.pth'

        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'val_loss': val_loss,
            'val_acc': val_acc,
            'history': self.history,
            'config': self.config
        }, checkpoint_path)

        print(f"Checkpoint saved: {checkpoint_path.name}")

    def _save_history(self):
        """Save training history to CSV and plots"""

        # Save CSV
        history_df = pd.DataFrame(self.history)
        history_path = self.output_dir / 'cbam_training_history.csv'
        history_df.to_csv(history_path, index=False)
        print(f"Training history saved: {history_path.name}")

        # Plot training curves
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        # Loss
        axes[0].plot(self.history['train_loss'], label='Train')
        axes[0].plot(self.history['val_loss'], label='Val')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].set_title('Training and Validation Loss')
        axes[0].legend()
        axes[0].grid(True)

        # Accuracy
        axes[1].plot(self.history['train_acc'], label='Train')
        axes[1].plot(self.history['val_acc'], label='Val')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Accuracy (%)')
        axes[1].set_title('Training and Validation Accuracy')
        axes[1].legend()
        axes[1].grid(True)

        # Learning rate
        axes[2].plot(self.history['learning_rate'])
        axes[2].set_xlabel('Epoch')
        axes[2].set_ylabel('Learning Rate')
        axes[2].set_title('Learning Rate Schedule')
        axes[2].set_yscale('log')
        axes[2].grid(True)

        plt.tight_layout()
        plot_path = self.output_dir / 'cbam_training_curves.png'
        plt.savefig(plot_path, dpi=150)
        print(f"Training curves saved: {plot_path.name}")
        plt.close()


def main():
    """Main training function"""

    parser = argparse.ArgumentParser(description='Train EfficientNet-CBAM model')
    parser.add_argument('--baseline', type=str, default=None,
                        help='Path to baseline model for transfer learning')
    parser.add_argument('--config', type=str, default=None,
                        help='Path to config JSON file')
    parser.add_argument('--data_dir', type=str, default='data/processed',
                        help='Path to processed data directory')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of training epochs')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use (cuda/cpu)')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Output directory for checkpoints')
    parser.add_argument('--staged_training', action='store_true',
                        help='Use staged training (Stage 1: freeze EfficientNet, Stage 2: fine-tune all)')
    parser.add_argument('--stage1_epochs', type=int, default=15,
                        help='Number of epochs for stage 1 (CBAM+Classifier training)')
    parser.add_argument('--stage2_epochs', type=int, default=35,
                        help='Number of epochs for stage 2 (full network fine-tuning)')
    parser.add_argument('--stage1_lr', type=float, default=1e-3,
                        help='Learning rate for stage 1')
    parser.add_argument('--stage2_lr', type=float, default=1e-5,
                        help='Learning rate for stage 2')

    args = parser.parse_args()

    # Device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load or create config
    if args.config and Path(args.config).exists():
        with open(args.config, 'r') as f:
            config = json.load(f)
        print(f"Loaded config from {args.config}")
    else:
        config = {
            'learning_rate': 1e-4,
            'cbam_lr_multiplier': 2.0,
            'classifier_lr_multiplier': 1.5,
            'weight_decay': 1e-4,
            'epochs': args.epochs,
            'batch_size': args.batch_size,
            'early_stopping_patience': 10,
            'lr_factor': 0.5,
            'lr_patience': 3,
            'gradient_clip_max_norm': 1.0
        }
        print("Using default config")

    # Override with command line arguments
    config['epochs'] = args.epochs
    config['batch_size'] = args.batch_size

    # Add staged training parameters if using staged training
    if args.staged_training:
        config['stage1_lr'] = args.stage1_lr
        config['stage2_lr'] = args.stage2_lr
        config['stage1_epochs'] = args.stage1_epochs
        config['stage2_epochs'] = args.stage2_epochs
        print(f"Staged training enabled: Stage 1 ({args.stage1_epochs} epochs) → Stage 2 ({args.stage2_epochs} epochs)")

    # Create model
    print("Creating EfficientNet-CBAM model...")
    model = create_efficientnet_cbam(pretrained=True, cbam_variant='default')
    model.to(device)

    # Load baseline weights if provided
    if args.baseline and Path(args.baseline).exists():
        print(f"Loading baseline weights from {args.baseline}")
        model.load_from_baseline(args.baseline, device=device)

    # Create data loaders
    print("Creating data loaders...")
    train_loader, val_loader, test_loader = build_dataloaders(
        root_dir=args.data_dir,
        batch_size=config['batch_size'],
        image_size=224,
        num_workers=4
    )

    # Create trainer
    print("Creating trainer...")
    trainer = CBAMTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        device=device,
        output_dir=args.output_dir
    )

    # Train
    if args.staged_training:
        print("\nStarting STAGED TRAINING...")
        model, history = trainer.train_staged(
            stage1_epochs=args.stage1_epochs,
            stage2_epochs=args.stage2_epochs
        )
    else:
        print("\nStarting STANDARD TRAINING...")
        model, history = trainer.train()

    print(f"\nTraining complete!")
    print(f"Results saved to: {trainer.output_dir}")


if __name__ == '__main__':
    main()
