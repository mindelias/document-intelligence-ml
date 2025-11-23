"""
Document classification models.

Supports multiple architectures:
- ResNet-50 (fast, good baseline)
- EfficientNet (efficient, accurate)
- LayoutLMv3 (state-of-the-art for documents)
"""

import os
import sys
from pathlib import Path
from typing import Optional, Dict, List, Tuple
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import timm

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


class DocumentClassifier(nn.Module):
    """Base document classifier class."""

    def __init__(
        self,
        num_classes: int = 4,
        model_name: str = 'resnet50',
        pretrained: bool = True,
        dropout: float = 0.5
    ):
        """
        Initialize classifier.

        Args:
            num_classes: Number of document classes
            model_name: Backbone model ('resnet50', 'efficientnet', 'vit')
            pretrained: Whether to use pretrained weights
            dropout: Dropout rate
        """
        super().__init__()

        self.num_classes = num_classes
        self.model_name = model_name

        # Create backbone
        if model_name == 'resnet50':
            self.backbone = self._create_resnet50(pretrained)
            feature_dim = 2048
        elif model_name == 'efficientnet':
            self.backbone = self._create_efficientnet(pretrained)
            feature_dim = 1280
        elif model_name == 'vit':
            self.backbone = self._create_vit(pretrained)
            feature_dim = 768
        else:
            raise ValueError(f"Unknown model: {model_name}")

        # Classification head
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(feature_dim, 512),
            nn.ReLU(),
            nn.Dropout(dropout / 2),
            nn.Linear(512, num_classes)
        )

        logger.info(f"Created {model_name} classifier with {num_classes} classes")

    def _create_resnet50(self, pretrained: bool) -> nn.Module:
        """Create ResNet-50 backbone."""
        # Load pretrained ResNet
        weights = models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        resnet = models.resnet50(weights=weights)

        # Remove final FC layer
        backbone = nn.Sequential(*list(resnet.children())[:-1])

        return backbone

    def _create_efficientnet(self, pretrained: bool) -> nn.Module:
        """Create EfficientNet backbone."""
        model = timm.create_model(
            'efficientnet_b0',
            pretrained=pretrained,
            num_classes=0,  # Remove classification head
            global_pool=''  # We'll add our own pooling
        )

        # Add global average pooling
        return nn.Sequential(
            model,
            nn.AdaptiveAvgPool2d(1)
        )

    def _create_vit(self, pretrained: bool) -> nn.Module:
        """Create Vision Transformer backbone."""
        model = timm.create_model(
            'vit_base_patch16_224',
            pretrained=pretrained,
            num_classes=0  # Remove classification head
        )

        return model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input images [batch_size, 3, height, width]

        Returns:
            Logits [batch_size, num_classes]
        """
        # Extract features
        features = self.backbone(x)

        # Flatten if needed
        if len(features.shape) > 2:
            features = features.view(features.size(0), -1)

        # Classify
        logits = self.classifier(features)

        return logits

    def predict(self, x: torch.Tensor, return_probs: bool = True) -> Tuple:
        """
        Make predictions.

        Args:
            x: Input images
            return_probs: Whether to return probabilities

        Returns:
            Tuple of (predictions, probabilities) or just predictions
        """
        self.eval()
        with torch.no_grad():
            logits = self.forward(x)

            if return_probs:
                probs = F.softmax(logits, dim=1)
                preds = torch.argmax(probs, dim=1)
                return preds, probs
            else:
                preds = torch.argmax(logits, dim=1)
                return preds

    def get_num_parameters(self) -> int:
        """Get total number of parameters."""
        return sum(p.numel() for p in self.parameters())

    def get_trainable_parameters(self) -> int:
        """Get number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class DocumentClassifierLayoutLM(nn.Module):
    """
    Document classifier using LayoutLMv3.

    LayoutLMv3 is designed specifically for document understanding
    and achieves state-of-the-art results. However, it requires:
    - More computational resources
    - OCR preprocessing
    - Bounding box information

    For simpler use cases, use DocumentClassifier with ResNet.
    """

    def __init__(
        self,
        num_classes: int = 4,
        pretrained: bool = True,
        dropout: float = 0.1
    ):
        """
        Initialize LayoutLMv3 classifier.

        Args:
            num_classes: Number of document classes
            pretrained: Whether to use pretrained weights
            dropout: Dropout rate
        """
        super().__init__()

        self.num_classes = num_classes

        try:
            from transformers import LayoutLMv3Model, LayoutLMv3Config

            if pretrained:
                # Load pretrained model
                self.layoutlm = LayoutLMv3Model.from_pretrained(
                    'microsoft/layoutlmv3-base'
                )
            else:
                # Create from scratch
                config = LayoutLMv3Config(num_labels=num_classes)
                self.layoutlm = LayoutLMv3Model(config)

            hidden_size = self.layoutlm.config.hidden_size

            # Classification head
            self.classifier = nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(hidden_size, num_classes)
            )

            logger.info(f"Created LayoutLMv3 classifier with {num_classes} classes")

        except ImportError:
            raise ImportError(
                "LayoutLMv3 requires transformers library. "
                "Install with: pip install transformers"
            )

    def forward(
        self,
        input_ids: torch.Tensor,
        bbox: torch.Tensor,
        pixel_values: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            input_ids: Token IDs [batch_size, seq_len]
            bbox: Bounding boxes [batch_size, seq_len, 4]
            pixel_values: Image pixels [batch_size, 3, 224, 224]
            attention_mask: Attention mask [batch_size, seq_len]

        Returns:
            Logits [batch_size, num_classes]
        """
        # Get LayoutLM outputs
        outputs = self.layoutlm(
            input_ids=input_ids,
            bbox=bbox,
            pixel_values=pixel_values,
            attention_mask=attention_mask
        )

        # Use [CLS] token representation
        sequence_output = outputs.last_hidden_state[:, 0, :]

        # Classify
        logits = self.classifier(sequence_output)

        return logits


class ModelFactory:
    """Factory for creating document classification models."""

    @staticmethod
    def create_classifier(
        model_type: str = 'resnet50',
        num_classes: int = 4,
        pretrained: bool = True,
        **kwargs
    ) -> nn.Module:
        """
        Create a document classifier.

        Args:
            model_type: Type of model ('resnet50', 'efficientnet', 'vit', 'layoutlm')
            num_classes: Number of document classes
            pretrained: Whether to use pretrained weights
            **kwargs: Additional arguments for model

        Returns:
            Document classifier model
        """
        if model_type == 'layoutlm':
            return DocumentClassifierLayoutLM(
                num_classes=num_classes,
                pretrained=pretrained,
                **kwargs
            )
        else:
            return DocumentClassifier(
                num_classes=num_classes,
                model_name=model_type,
                pretrained=pretrained,
                **kwargs
            )

    @staticmethod
    def load_checkpoint(
        checkpoint_path: str,
        model_type: str = 'resnet50',
        num_classes: int = 4,
        device: str = 'cpu'
    ) -> nn.Module:
        """
        Load model from checkpoint.

        Args:
            checkpoint_path: Path to checkpoint file
            model_type: Type of model
            num_classes: Number of classes
            device: Device to load model on

        Returns:
            Loaded model
        """
        # Create model
        model = ModelFactory.create_classifier(
            model_type=model_type,
            num_classes=num_classes,
            pretrained=False
        )

        # Load weights
        checkpoint = torch.load(checkpoint_path, map_location=device)

        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)

        model.to(device)
        model.eval()

        logger.info(f"Loaded model from {checkpoint_path}")

        return model


def main():
    """Test model creation."""
    import argparse

    parser = argparse.ArgumentParser(description="Test model creation")
    parser.add_argument('--model', default='resnet50', choices=['resnet50', 'efficientnet', 'vit', 'layoutlm'])
    parser.add_argument('--num-classes', type=int, default=4)

    args = parser.parse_args()

    # Create model
    model = ModelFactory.create_classifier(
        model_type=args.model,
        num_classes=args.num_classes
    )

    print(f"\nModel: {args.model}")
    print(f"Total parameters: {model.get_num_parameters():,}")
    print(f"Trainable parameters: {model.get_trainable_parameters():,}")

    # Test forward pass
    if args.model == 'layoutlm':
        print("\nLayoutLM requires special inputs (tokens + bbox + image)")
        print("Use with LayoutLM processor for real inference")
    else:
        dummy_input = torch.randn(2, 3, 224, 224)
        output = model(dummy_input)
        print(f"\nInput shape: {dummy_input.shape}")
        print(f"Output shape: {output.shape}")
        print("✓ Model created successfully!")


if __name__ == "__main__":
    main()
