"""
Basic tests for document intelligence components.
"""

import sys
from pathlib import Path
import pytest

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from config import Config
from src.models.classifier import ModelFactory
from src.models.extractor import DocumentExtractor
from src.data.generate_synthetic import SyntheticDocumentGenerator


class TestConfig:
    """Test configuration."""

    def test_config_creation(self):
        """Test that config can be created."""
        config = Config()
        assert config is not None
        assert len(config.document_classes) == 4

    def test_document_classes(self):
        """Test document classes are correct."""
        config = Config()
        assert 'invoice' in config.document_classes
        assert 'receipt' in config.document_classes
        assert 'resume' in config.document_classes
        assert 'contract' in config.document_classes


class TestModels:
    """Test model creation."""

    def test_create_resnet(self):
        """Test ResNet-50 creation."""
        model = ModelFactory.create_classifier(
            model_type='resnet50',
            num_classes=4,
            pretrained=False
        )
        assert model is not None
        assert model.num_classes == 4

    def test_create_efficientnet(self):
        """Test EfficientNet creation."""
        model = ModelFactory.create_classifier(
            model_type='efficientnet',
            num_classes=4,
            pretrained=False
        )
        assert model is not None
        assert model.num_classes == 4

    def test_model_parameters(self):
        """Test model has parameters."""
        model = ModelFactory.create_classifier(
            model_type='resnet50',
            num_classes=4,
            pretrained=False
        )
        num_params = model.get_num_parameters()
        assert num_params > 0


class TestSyntheticGeneration:
    """Test synthetic document generation."""

    def test_generator_creation(self):
        """Test generator can be created."""
        generator = SyntheticDocumentGenerator()
        assert generator is not None

    def test_generate_invoice(self):
        """Test invoice generation."""
        generator = SyntheticDocumentGenerator()
        image, metadata = generator.generate_invoice(1)

        assert image is not None
        assert metadata['type'] == 'invoice'
        assert 'invoice_number' in metadata

    def test_generate_receipt(self):
        """Test receipt generation."""
        generator = SyntheticDocumentGenerator()
        image, metadata = generator.generate_receipt(1)

        assert image is not None
        assert metadata['type'] == 'receipt'
        assert 'total' in metadata

    def test_generate_resume(self):
        """Test resume generation."""
        generator = SyntheticDocumentGenerator()
        image, metadata = generator.generate_resume(1)

        assert image is not None
        assert metadata['type'] == 'resume'
        assert 'name' in metadata

    def test_generate_contract(self):
        """Test contract generation."""
        generator = SyntheticDocumentGenerator()
        image, metadata = generator.generate_contract(1)

        assert image is not None
        assert metadata['type'] == 'contract'
        assert 'parties' in metadata


class TestExtractor:
    """Test information extraction."""

    def test_extractor_creation(self):
        """Test extractor can be created."""
        extractor = DocumentExtractor()
        assert extractor is not None

    def test_extract_date(self):
        """Test date extraction."""
        extractor = DocumentExtractor()

        text1 = "Date: 01/15/2024"
        date1 = extractor._extract_date(text1)
        assert date1 is not None

        text2 = "Effective Date: Jan 15, 2024"
        date2 = extractor._extract_date(text2)
        assert date2 is not None


def test_imports():
    """Test that all modules can be imported."""
    try:
        import torch
        import numpy as np
        import PIL
        import cv2
        import pytesseract
        assert True
    except ImportError as e:
        pytest.fail(f"Import failed: {e}")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
