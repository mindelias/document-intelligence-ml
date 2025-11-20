#!/usr/bin/env python3
"""
Quick Demo - Document Intelligence System

This script demonstrates the complete pipeline:
1. Generate synthetic documents
2. Train a classifier
3. Run inference and extract information

Perfect for testing and demonstrations!
"""

import sys
from pathlib import Path
import logging

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from src.data.generate_synthetic import SyntheticDocumentGenerator
from src.data.preprocess import DataPreprocessor
from src.models.classifier import ModelFactory
from src.training.train import Trainer
from src.inference.predict import DocumentIntelligence
from torch.utils.data import DataLoader
from src.data.preprocess import DocumentDataset, DocumentAugmentation

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    """Run quick demo."""

    print("\n" + "=" * 70)
    print("DOCUMENT INTELLIGENCE SYSTEM - QUICK DEMO")
    print("=" * 70)

    # Step 1: Generate synthetic data
    print("\n[Step 1/5] Generating synthetic documents...")
    print("-" * 70)

    generator = SyntheticDocumentGenerator()

    for doc_type in ['invoice', 'receipt', 'resume', 'contract']:
        logger.info(f"Generating {doc_type}s...")
        generator.generate_documents(doc_type, count=50, add_noise=True)

    print("✓ Generated 200 synthetic documents (50 per class)")

    # Step 2: Prepare data
    print("\n[Step 2/5] Preparing training data...")
    print("-" * 70)

    preprocessor = DataPreprocessor()
    source_dirs = [Path('data/synthetic')]

    counts = preprocessor.split_data(
        source_dirs,
        train_ratio=0.7,
        val_ratio=0.15,
        test_ratio=0.15
    )

    print(f"✓ Data split: Train={counts['train']}, Val={counts['val']}, Test={counts['test']}")

    # Step 3: Train classifier
    print("\n[Step 3/5] Training document classifier...")
    print("-" * 70)
    print("NOTE: Training with small dataset for demo (5 epochs)")
    print("For production, use more data and epochs (see QUICKSTART.md)")

    # Create model
    model = ModelFactory.create_classifier(
        model_type='resnet50',
        num_classes=4,
        pretrained=True
    )

    # Create data loaders
    train_dataset = DocumentDataset(
        'data/processed',
        split='train',
        transform=DocumentAugmentation.get_train_transforms()
    )

    val_dataset = DocumentDataset(
        'data/processed',
        split='val',
        transform=DocumentAugmentation.get_val_transforms()
    )

    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)

    # Train
    from config import Config
    config = Config()
    config.num_epochs = 5  # Quick demo

    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config
    )

    trainer.train(num_epochs=5, early_stopping_patience=3)

    print(f"✓ Training complete! Best accuracy: {trainer.best_val_acc:.2f}%")

    # Step 4: Run inference
    print("\n[Step 4/5] Testing inference...")
    print("-" * 70)

    # Create pipeline
    pipeline = DocumentIntelligence(
        classifier_path='models/checkpoints/best.pth'
    )

    # Test on a synthetic document
    test_image = Path('data/synthetic/invoice/invoice_0000.png')

    if test_image.exists():
        result = pipeline.process(test_image)

        print(f"\nTest Document: {test_image.name}")
        print(f"  Type: {result['classification']['type']}")
        print(f"  Confidence: {result['classification']['confidence']:.2%}")
        print(f"  Summary: {result['summary']}")
        print(f"  Processing Time: {result['processing_time']}")

    print("\n✓ Inference successful!")

    # Step 5: Summary
    print("\n[Step 5/5] Demo Complete!")
    print("=" * 70)
    print("\nWhat you just did:")
    print("  ✓ Generated 200 synthetic documents")
    print("  ✓ Split data into train/val/test sets")
    print("  ✓ Trained a ResNet-50 classifier")
    print("  ✓ Ran end-to-end inference")
    print("\nNext steps:")
    print("  → Generate more data: python src/data/generate_synthetic.py --count 500")
    print("  → Train longer: python src/training/train.py --epochs 20")
    print("  → Process documents: python src/inference/predict.py <image_path>")
    print("  → See QUICKSTART.md for more options")
    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
