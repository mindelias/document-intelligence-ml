"""
Training script for document classification model.

Supports:
- Multiple model architectures
- Mixed precision training
- Learning rate scheduling
- Early stopping
- Model checkpointing
- TensorBoard logging
"""

import os
import sys
from pathlib import Path
import argparse
import logging
from typing import Dict, Optional, Tuple
import time
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
import numpy as np
from tqdm import tqdm

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from config import Config
from src.data.preprocess import DocumentDataset, DocumentAugmentation
from src.models.classifier import ModelFactory

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class Trainer:
    """Document classifier trainer."""

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: Optional[Config] = None,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    ):
        """
        Initialize trainer.

        Args:
            model: Model to train
            train_loader: Training data loader
            val_loader: Validation data loader
            config: Configuration
            device: Device to train on
        """
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config or Config()
        self.device = device

        # Training components
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=0.01
        )
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=self.config.num_epochs
        )

        # Mixed precision training
        self.scaler = GradScaler() if device == 'cuda' else None

        # Tracking
        self.best_val_acc = 0.0
        self.train_losses = []
        self.val_losses = []
        self.train_accs = []
        self.val_accs = []

        # Checkpointing
        self.checkpoint_dir = Path(self.config.model_dir) / 'checkpoints'
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Trainer initialized on device: {device}")
        logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    def train_epoch(self, epoch: int) -> Tuple[float, float]:
        """
        Train for one epoch.

        Args:
            epoch: Current epoch number

        Returns:
            Tuple of (average_loss, accuracy)
        """
        self.model.train()

        running_loss = 0.0
        correct = 0
        total = 0

        pbar = tqdm(self.train_loader, desc=f'Epoch {epoch + 1}/{self.config.num_epochs} [Train]')

        for batch_idx, (images, labels, _) in enumerate(pbar):
            images = images.to(self.device)
            labels = labels.to(self.device)

            # Zero gradients
            self.optimizer.zero_grad()

            # Forward pass with mixed precision
            if self.scaler:
                with autocast():
                    outputs = self.model(images)
                    loss = self.criterion(outputs, labels)

                # Backward pass
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()

            # Track metrics
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            # Update progress bar
            pbar.set_postfix({
                'loss': running_loss / (batch_idx + 1),
                'acc': 100. * correct / total
            })

        avg_loss = running_loss / len(self.train_loader)
        accuracy = 100. * correct / total

        return avg_loss, accuracy

    def validate(self, epoch: int) -> Tuple[float, float, Dict]:
        """
        Validate model.

        Args:
            epoch: Current epoch number

        Returns:
            Tuple of (average_loss, accuracy, per_class_metrics)
        """
        self.model.eval()

        running_loss = 0.0
        correct = 0
        total = 0

        # Per-class metrics
        class_correct = {}
        class_total = {}

        pbar = tqdm(self.val_loader, desc=f'Epoch {epoch + 1}/{self.config.num_epochs} [Val]')

        with torch.no_grad():
            for batch_idx, (images, labels, _) in enumerate(pbar):
                images = images.to(self.device)
                labels = labels.to(self.device)

                # Forward pass
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)

                # Track metrics
                running_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

                # Per-class accuracy
                for i in range(len(labels)):
                    label = labels[i].item()
                    if label not in class_total:
                        class_total[label] = 0
                        class_correct[label] = 0

                    class_total[label] += 1
                    if predicted[i] == label:
                        class_correct[label] += 1

                # Update progress bar
                pbar.set_postfix({
                    'loss': running_loss / (batch_idx + 1),
                    'acc': 100. * correct / total
                })

        avg_loss = running_loss / len(self.val_loader)
        accuracy = 100. * correct / total

        # Calculate per-class metrics
        per_class = {}
        for label in class_total:
            per_class[label] = {
                'accuracy': 100. * class_correct[label] / class_total[label],
                'total': class_total[label]
            }

        return avg_loss, accuracy, per_class

    def save_checkpoint(self, epoch: int, is_best: bool = False):
        """
        Save model checkpoint.

        Args:
            epoch: Current epoch
            is_best: Whether this is the best model so far
        """
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_acc': self.best_val_acc,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'train_accs': self.train_accs,
            'val_accs': self.val_accs
        }

        # Save latest checkpoint
        latest_path = self.checkpoint_dir / 'latest.pth'
        torch.save(checkpoint, latest_path)

        # Save best checkpoint
        if is_best:
            best_path = self.checkpoint_dir / 'best.pth'
            torch.save(checkpoint, best_path)
            logger.info(f"✓ Saved best model (acc: {self.best_val_acc:.2f}%)")

        # Save epoch checkpoint
        epoch_path = self.checkpoint_dir / f'epoch_{epoch + 1}.pth'
        if (epoch + 1) % 5 == 0:  # Save every 5 epochs
            torch.save(checkpoint, epoch_path)

    def load_checkpoint(self, checkpoint_path: str):
        """
        Load checkpoint.

        Args:
            checkpoint_path: Path to checkpoint
        """
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.best_val_acc = checkpoint['best_val_acc']
        self.train_losses = checkpoint.get('train_losses', [])
        self.val_losses = checkpoint.get('val_losses', [])
        self.train_accs = checkpoint.get('train_accs', [])
        self.val_accs = checkpoint.get('val_accs', [])

        logger.info(f"Loaded checkpoint from {checkpoint_path}")

    def train(self, num_epochs: Optional[int] = None, early_stopping_patience: int = 5):
        """
        Train model for multiple epochs.

        Args:
            num_epochs: Number of epochs (uses config if not specified)
            early_stopping_patience: Patience for early stopping
        """
        num_epochs = num_epochs or self.config.num_epochs

        logger.info("=" * 60)
        logger.info("STARTING TRAINING")
        logger.info("=" * 60)
        logger.info(f"Epochs: {num_epochs}")
        logger.info(f"Device: {self.device}")
        logger.info(f"Training samples: {len(self.train_loader.dataset)}")
        logger.info(f"Validation samples: {len(self.val_loader.dataset)}")
        logger.info("=" * 60)

        patience_counter = 0
        start_time = time.time()

        for epoch in range(num_epochs):
            epoch_start = time.time()

            # Train
            train_loss, train_acc = self.train_epoch(epoch)
            self.train_losses.append(train_loss)
            self.train_accs.append(train_acc)

            # Validate
            val_loss, val_acc, per_class = self.validate(epoch)
            self.val_losses.append(val_loss)
            self.val_accs.append(val_acc)

            # Update learning rate
            self.scheduler.step()

            # Log epoch results
            epoch_time = time.time() - epoch_start
            logger.info(f"\nEpoch {epoch + 1}/{num_epochs} - {epoch_time:.1f}s")
            logger.info(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
            logger.info(f"  Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.2f}%")

            # Log per-class metrics
            logger.info("  Per-class accuracy:")
            for label, metrics in per_class.items():
                logger.info(f"    Class {label}: {metrics['accuracy']:.2f}% ({metrics['total']} samples)")

            # Save checkpoint
            is_best = val_acc > self.best_val_acc
            if is_best:
                self.best_val_acc = val_acc
                patience_counter = 0
            else:
                patience_counter += 1

            self.save_checkpoint(epoch, is_best)

            # Early stopping
            if patience_counter >= early_stopping_patience:
                logger.info(f"\nEarly stopping triggered after {epoch + 1} epochs")
                logger.info(f"Best validation accuracy: {self.best_val_acc:.2f}%")
                break

        # Training complete
        total_time = time.time() - start_time
        logger.info("\n" + "=" * 60)
        logger.info("TRAINING COMPLETE!")
        logger.info("=" * 60)
        logger.info(f"Total time: {total_time / 60:.1f} minutes")
        logger.info(f"Best validation accuracy: {self.best_val_acc:.2f}%")
        logger.info(f"Final train accuracy: {self.train_accs[-1]:.2f}%")
        logger.info(f"Final validation accuracy: {self.val_accs[-1]:.2f}%")
        logger.info("=" * 60)

    def plot_history(self, save_path: Optional[str] = None):
        """
        Plot training history.

        Args:
            save_path: Path to save plot
        """
        try:
            import matplotlib.pyplot as plt

            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

            # Plot loss
            ax1.plot(self.train_losses, label='Train Loss')
            ax1.plot(self.val_losses, label='Val Loss')
            ax1.set_xlabel('Epoch')
            ax1.set_ylabel('Loss')
            ax1.set_title('Training and Validation Loss')
            ax1.legend()
            ax1.grid(True)

            # Plot accuracy
            ax2.plot(self.train_accs, label='Train Acc')
            ax2.plot(self.val_accs, label='Val Acc')
            ax2.set_xlabel('Epoch')
            ax2.set_ylabel('Accuracy (%)')
            ax2.set_title('Training and Validation Accuracy')
            ax2.legend()
            ax2.grid(True)

            plt.tight_layout()

            if save_path:
                plt.savefig(save_path, dpi=150, bbox_inches='tight')
                logger.info(f"Saved training history plot to {save_path}")
            else:
                plt.show()

        except ImportError:
            logger.warning("matplotlib not available, skipping plot")


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description="Train document classifier")

    # Model arguments
    parser.add_argument('--model', default='resnet50', choices=['resnet50', 'efficientnet', 'vit'])
    parser.add_argument('--num-classes', type=int, default=4)
    parser.add_argument('--pretrained', action='store_true', default=True)

    # Training arguments
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=2e-5)
    parser.add_argument('--early-stopping', type=int, default=5)

    # Data arguments
    parser.add_argument('--data-dir', type=str, default='data/processed')
    parser.add_argument('--num-workers', type=int, default=4)

    # Checkpoint arguments
    parser.add_argument('--resume', type=str, help='Resume from checkpoint')
    parser.add_argument('--save-dir', type=str, default='models')

    args = parser.parse_args()

    # Set device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info(f"Using device: {device}")

    # Create config
    config = Config()
    config.num_epochs = args.epochs
    config.batch_size = args.batch_size
    config.learning_rate = args.lr
    config.data_dir = args.data_dir
    config.model_dir = args.save_dir

    # Create data loaders
    logger.info("Loading datasets...")

    train_dataset = DocumentDataset(
        config.data_dir,
        split='train',
        transform=DocumentAugmentation.get_train_transforms()
    )

    val_dataset = DocumentDataset(
        config.data_dir,
        split='val',
        transform=DocumentAugmentation.get_val_transforms()
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True if device == 'cuda' else False
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True if device == 'cuda' else False
    )

    # Create model
    logger.info(f"Creating {args.model} model...")
    model = ModelFactory.create_classifier(
        model_type=args.model,
        num_classes=args.num_classes,
        pretrained=args.pretrained
    )

    # Create trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        device=device
    )

    # Resume from checkpoint if specified
    if args.resume:
        trainer.load_checkpoint(args.resume)

    # Train
    trainer.train(
        num_epochs=args.epochs,
        early_stopping_patience=args.early_stopping
    )

    # Plot history
    plot_path = Path(args.save_dir) / 'training_history.png'
    trainer.plot_history(save_path=str(plot_path))

    logger.info("\n✓ Training complete!")
    logger.info(f"Best model saved to: {trainer.checkpoint_dir / 'best.pth'}")


if __name__ == "__main__":
    main()
