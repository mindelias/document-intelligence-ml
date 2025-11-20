"""
Data preprocessing pipeline for document intelligence.

Handles:
- Image loading and normalization
- Data augmentation
- Train/val/test splitting
- Dataset creation for PyTorch
"""

import os
import sys
from pathlib import Path
import argparse
import logging
from typing import Dict, List, Tuple, Optional, Callable
import json
import random
import shutil

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import cv2

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from config import Config

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DocumentDataset(Dataset):
    """PyTorch Dataset for document images."""

    def __init__(
        self,
        data_dir: Path,
        split: str = 'train',
        transform: Optional[Callable] = None,
        target_size: Tuple[int, int] = (224, 224)
    ):
        """
        Initialize dataset.

        Args:
            data_dir: Root directory containing processed data
            split: Data split ('train', 'val', 'test')
            transform: Optional transform to apply
            target_size: Target image size (width, height)
        """
        self.data_dir = Path(data_dir) / split
        self.split = split
        self.transform = transform
        self.target_size = target_size

        # Load samples
        self.samples = self._load_samples()

        # Create label mapping
        self.classes = sorted(set(s['label'] for s in self.samples))
        self.label_to_idx = {label: idx for idx, label in enumerate(self.classes)}
        self.idx_to_label = {idx: label for label, idx in self.label_to_idx.items()}

        logger.info(f"Loaded {len(self.samples)} {split} samples")
        logger.info(f"Classes: {self.classes}")

    def _load_samples(self) -> List[Dict]:
        """Load all samples from data directory."""
        samples = []

        if not self.data_dir.exists():
            logger.warning(f"Data directory not found: {self.data_dir}")
            return samples

        # Walk through class directories
        for class_dir in self.data_dir.iterdir():
            if not class_dir.is_dir():
                continue

            label = class_dir.name

            # Find all images in class directory
            for img_path in class_dir.glob('*.png'):
                # Look for corresponding metadata
                metadata_path = img_path.with_suffix('.json')
                metadata = {}
                if metadata_path.exists():
                    with open(metadata_path) as f:
                        metadata = json.load(f)

                samples.append({
                    'image_path': img_path,
                    'label': label,
                    'metadata': metadata
                })

        return samples

    def __len__(self) -> int:
        """Return number of samples."""
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, Dict]:
        """
        Get a sample.

        Returns:
            Tuple of (image_tensor, label_idx, metadata)
        """
        sample = self.samples[idx]

        # Load image
        image = Image.open(sample['image_path']).convert('RGB')

        # Resize
        image = image.resize(self.target_size, Image.LANCZOS)

        # Apply transforms
        if self.transform:
            image = self.transform(image)
        else:
            # Default: convert to tensor and normalize
            image = transforms.ToTensor()(image)

        # Get label index
        label_idx = self.label_to_idx[sample['label']]

        return image, label_idx, sample['metadata']


class DocumentAugmentation:
    """Data augmentation for document images."""

    @staticmethod
    def get_train_transforms(target_size: Tuple[int, int] = (224, 224)) -> transforms.Compose:
        """
        Get training data augmentation pipeline.

        Includes realistic document variations:
        - Slight rotation (scanning artifacts)
        - Color jitter (lighting variations)
        - Random cropping (partial documents)
        - Gaussian blur (poor focus)
        """
        return transforms.Compose([
            transforms.Resize(target_size),
            transforms.RandomRotation(degrees=3),  # Slight rotation
            transforms.ColorJitter(
                brightness=0.2,
                contrast=0.2,
                saturation=0.1
            ),
            transforms.RandomApply([
                transforms.GaussianBlur(kernel_size=3)
            ], p=0.1),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

    @staticmethod
    def get_val_transforms(target_size: Tuple[int, int] = (224, 224)) -> transforms.Compose:
        """
        Get validation/test data transforms (no augmentation).
        """
        return transforms.Compose([
            transforms.Resize(target_size),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])


class DataPreprocessor:
    """Preprocess and organize data for training."""

    def __init__(self, config: Optional[Config] = None):
        """Initialize preprocessor."""
        self.config = config or Config()
        self.data_dir = Path(self.config.data_dir)
        self.raw_dir = self.data_dir / 'raw'
        self.synthetic_dir = self.data_dir / 'synthetic'
        self.processed_dir = self.data_dir / 'processed'

    def split_data(
        self,
        source_dirs: List[Path],
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
        seed: int = 42
    ) -> Dict[str, int]:
        """
        Split data into train/val/test sets.

        Args:
            source_dirs: List of directories containing images
            train_ratio: Fraction of data for training
            val_ratio: Fraction of data for validation
            test_ratio: Fraction of data for testing
            seed: Random seed for reproducibility

        Returns:
            Dict with counts per split
        """
        random.seed(seed)
        np.random.seed(seed)

        # Validate ratios
        assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, \
            "Ratios must sum to 1.0"

        logger.info("Splitting data into train/val/test sets...")
        logger.info(f"Ratios - Train: {train_ratio}, Val: {val_ratio}, Test: {test_ratio}")

        counts = {'train': 0, 'val': 0, 'test': 0}

        # Process each class
        for class_name in self.config.document_classes:
            logger.info(f"\nProcessing class: {class_name}")

            # Collect all images for this class
            all_images = []
            for source_dir in source_dirs:
                class_dir = source_dir / class_name
                if class_dir.exists():
                    images = list(class_dir.glob('*.png')) + list(class_dir.glob('*.jpg'))
                    all_images.extend(images)

            if not all_images:
                logger.warning(f"  No images found for class {class_name}")
                continue

            logger.info(f"  Found {len(all_images)} images")

            # Shuffle
            random.shuffle(all_images)

            # Calculate split indices
            n_total = len(all_images)
            n_train = int(n_total * train_ratio)
            n_val = int(n_total * val_ratio)

            # Split
            train_images = all_images[:n_train]
            val_images = all_images[n_train:n_train + n_val]
            test_images = all_images[n_train + n_val:]

            # Copy to processed directories
            for split, images in [
                ('train', train_images),
                ('val', val_images),
                ('test', test_images)
            ]:
                split_dir = self.processed_dir / split / class_name
                split_dir.mkdir(parents=True, exist_ok=True)

                for img_path in images:
                    # Copy image
                    dest_path = split_dir / img_path.name
                    shutil.copy2(img_path, dest_path)

                    # Copy metadata if exists
                    metadata_path = img_path.with_suffix('.json')
                    if metadata_path.exists():
                        dest_metadata = dest_path.with_suffix('.json')
                        shutil.copy2(metadata_path, dest_metadata)

                counts[split] += len(images)

            logger.info(f"  Split: train={len(train_images)}, val={len(val_images)}, test={len(test_images)}")

        logger.info("\n" + "=" * 60)
        logger.info("Data splitting complete!")
        logger.info(f"Total counts - Train: {counts['train']}, Val: {counts['val']}, Test: {counts['test']}")
        logger.info("=" * 60)

        return counts

    def create_dataloaders(
        self,
        batch_size: int = 16,
        num_workers: int = 4,
        target_size: Tuple[int, int] = (224, 224)
    ) -> Dict[str, DataLoader]:
        """
        Create PyTorch DataLoaders for train/val/test.

        Args:
            batch_size: Batch size
            num_workers: Number of data loading workers
            target_size: Target image size

        Returns:
            Dict of DataLoaders
        """
        logger.info("Creating DataLoaders...")

        # Create datasets with appropriate transforms
        train_dataset = DocumentDataset(
            self.processed_dir,
            split='train',
            transform=DocumentAugmentation.get_train_transforms(target_size),
            target_size=target_size
        )

        val_dataset = DocumentDataset(
            self.processed_dir,
            split='val',
            transform=DocumentAugmentation.get_val_transforms(target_size),
            target_size=target_size
        )

        test_dataset = DocumentDataset(
            self.processed_dir,
            split='test',
            transform=DocumentAugmentation.get_val_transforms(target_size),
            target_size=target_size
        )

        # Create dataloaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True
        )

        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True
        )

        logger.info(f"Train batches: {len(train_loader)}")
        logger.info(f"Val batches: {len(val_loader)}")
        logger.info(f"Test batches: {len(test_loader)}")

        return {
            'train': train_loader,
            'val': val_loader,
            'test': test_loader
        }

    def analyze_dataset(self) -> Dict:
        """
        Analyze processed dataset.

        Returns:
            Dict with dataset statistics
        """
        logger.info("Analyzing dataset...")

        stats = {
            'splits': {},
            'class_distribution': {},
            'total_samples': 0
        }

        for split in ['train', 'val', 'test']:
            split_dir = self.processed_dir / split

            if not split_dir.exists():
                continue

            split_stats = {'total': 0, 'classes': {}}

            for class_dir in split_dir.iterdir():
                if not class_dir.is_dir():
                    continue

                class_name = class_dir.name
                count = len(list(class_dir.glob('*.png'))) + len(list(class_dir.glob('*.jpg')))

                split_stats['classes'][class_name] = count
                split_stats['total'] += count

            stats['splits'][split] = split_stats
            stats['total_samples'] += split_stats['total']

        # Calculate class distribution across all splits
        for split_data in stats['splits'].values():
            for class_name, count in split_data['classes'].items():
                if class_name not in stats['class_distribution']:
                    stats['class_distribution'][class_name] = 0
                stats['class_distribution'][class_name] += count

        # Print statistics
        logger.info("\n" + "=" * 60)
        logger.info("DATASET STATISTICS")
        logger.info("=" * 60)

        for split, split_data in stats['splits'].items():
            logger.info(f"\n{split.upper()}:")
            logger.info(f"  Total: {split_data['total']}")
            for class_name, count in sorted(split_data['classes'].items()):
                logger.info(f"    {class_name}: {count}")

        logger.info(f"\nTOTAL SAMPLES: {stats['total_samples']}")
        logger.info("\nCLASS DISTRIBUTION:")
        for class_name, count in sorted(stats['class_distribution'].items()):
            pct = (count / stats['total_samples'] * 100) if stats['total_samples'] > 0 else 0
            logger.info(f"  {class_name}: {count} ({pct:.1f}%)")

        logger.info("=" * 60)

        return stats


def main():
    """Main function for CLI usage."""
    parser = argparse.ArgumentParser(
        description="Preprocess data for document intelligence"
    )
    parser.add_argument(
        '--split',
        action='store_true',
        help='Split data into train/val/test sets'
    )
    parser.add_argument(
        '--analyze',
        action='store_true',
        help='Analyze dataset statistics'
    )
    parser.add_argument(
        '--use-synthetic',
        action='store_true',
        help='Include synthetic data'
    )
    parser.add_argument(
        '--train-ratio',
        type=float,
        default=0.7,
        help='Training set ratio'
    )
    parser.add_argument(
        '--val-ratio',
        type=float,
        default=0.15,
        help='Validation set ratio'
    )
    parser.add_argument(
        '--test-ratio',
        type=float,
        default=0.15,
        help='Test set ratio'
    )

    args = parser.parse_args()

    # Create preprocessor
    preprocessor = DataPreprocessor()

    # Determine source directories
    source_dirs = []
    if args.use_synthetic:
        source_dirs.append(preprocessor.synthetic_dir)
    # Add raw data directory if it has data
    if preprocessor.raw_dir.exists():
        source_dirs.append(preprocessor.raw_dir)

    if not source_dirs:
        logger.error("No data sources found! Generate synthetic data first.")
        sys.exit(1)

    # Split data if requested
    if args.split:
        preprocessor.split_data(
            source_dirs,
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
            test_ratio=args.test_ratio
        )

    # Analyze if requested
    if args.analyze:
        preprocessor.analyze_dataset()


if __name__ == "__main__":
    main()
