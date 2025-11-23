"""
End-to-end document intelligence inference.

Pipeline:
1. Classify document type
2. Extract key information
3. Summarize content
4. Return structured results
"""

import os
import sys
from pathlib import Path
import argparse
import logging
from typing import Dict, Optional, Union
import json
import time

import torch
import numpy as np
from PIL import Image
from torchvision import transforms

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from config import Config
from src.models.classifier import ModelFactory
from src.models.extractor import DocumentExtractor, SmartDocumentExtractor
from src.data.preprocess import DocumentAugmentation

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DocumentIntelligence:
    """
    Complete document intelligence pipeline.

    Combines classification, extraction, and summarization.
    """

    def __init__(
        self,
        classifier_path: Optional[str] = None,
        extractor_path: Optional[str] = None,
        config: Optional[Config] = None,
        device: str = 'cpu'
    ):
        """
        Initialize document intelligence pipeline.

        Args:
            classifier_path: Path to trained classifier
            extractor_path: Path to trained extractor (optional)
            config: Configuration
            device: Device to run on
        """
        self.config = config or Config()
        self.device = device

        # Class labels
        self.classes = self.config.document_classes
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        self.idx_to_class = {idx: cls for cls, idx in self.class_to_idx.items()}

        # Load classifier
        if classifier_path and Path(classifier_path).exists():
            logger.info(f"Loading classifier from {classifier_path}")
            self.classifier = self._load_classifier(classifier_path)
        else:
            logger.warning("No classifier loaded - will use dummy predictions")
            self.classifier = None

        # Load extractor
        if extractor_path:
            self.extractor = SmartDocumentExtractor(extractor_path, config)
        else:
            self.extractor = DocumentExtractor(config)

        # Image preprocessing
        self.transform = DocumentAugmentation.get_val_transforms()

        logger.info("Document Intelligence pipeline initialized")

    def _load_classifier(self, checkpoint_path: str) -> torch.nn.Module:
        """Load classifier from checkpoint."""
        try:
            checkpoint = torch.load(checkpoint_path, map_location=self.device)

            # Create model
            model = ModelFactory.create_classifier(
                model_type='resnet50',  # Default, should be in checkpoint metadata
                num_classes=len(self.classes),
                pretrained=False
            )

            # Load weights
            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
            else:
                model.load_state_dict(checkpoint)

            model.to(self.device)
            model.eval()

            return model

        except Exception as e:
            logger.error(f"Failed to load classifier: {e}")
            return None

    def classify(self, image: Image.Image) -> Dict:
        """
        Classify document type.

        Args:
            image: Document image

        Returns:
            Dict with type, confidence, probabilities
        """
        if self.classifier is None:
            # Return dummy classification for testing
            return {
                'type': 'invoice',
                'confidence': 0.95,
                'probabilities': {cls: 0.25 for cls in self.classes}
            }

        # Preprocess image
        img_tensor = self.transform(image).unsqueeze(0).to(self.device)

        # Predict
        with torch.no_grad():
            logits = self.classifier(img_tensor)
            probs = torch.softmax(logits, dim=1).cpu().numpy()[0]

        # Get prediction
        pred_idx = np.argmax(probs)
        pred_class = self.idx_to_class[pred_idx]
        confidence = float(probs[pred_idx])

        # Build probabilities dict
        probabilities = {
            self.idx_to_class[i]: float(probs[i])
            for i in range(len(probs))
        }

        return {
            'type': pred_class,
            'confidence': confidence,
            'probabilities': probabilities
        }

    def extract_information(self, image: Image.Image, doc_type: str) -> Dict:
        """
        Extract information from document.

        Args:
            image: Document image
            doc_type: Document type

        Returns:
            Extracted information
        """
        return self.extractor.extract(image, doc_type)

    def summarize(self, extracted_info: Dict) -> str:
        """
        Generate summary from extracted information.

        Args:
            extracted_info: Extracted document information

        Returns:
            Summary text
        """
        doc_type = extracted_info.get('type', 'document')

        if doc_type == 'invoice':
            return self._summarize_invoice(extracted_info)
        elif doc_type == 'receipt':
            return self._summarize_receipt(extracted_info)
        elif doc_type == 'resume':
            return self._summarize_resume(extracted_info)
        elif doc_type == 'contract':
            return self._summarize_contract(extracted_info)
        else:
            # Use generic summarization
            raw_text = extracted_info.get('raw_text', '')
            return self.extractor.summarize(raw_text) if raw_text else "No summary available."

    def _summarize_invoice(self, info: Dict) -> str:
        """Summarize invoice."""
        parts = []

        if info.get('company'):
            parts.append(f"Invoice from {info['company']}")

        if info.get('invoice_number'):
            parts.append(f"(#{info['invoice_number']})")

        if info.get('customer'):
            parts.append(f"to {info['customer']}")

        if info.get('total'):
            parts.append(f"for {info['total']}")

        if info.get('date'):
            parts.append(f"dated {info['date']}")

        if parts:
            return ' '.join(parts) + '.'
        return "Invoice document."

    def _summarize_receipt(self, info: Dict) -> str:
        """Summarize receipt."""
        parts = []

        if info.get('store'):
            parts.append(f"Receipt from {info['store']}")

        if info.get('total'):
            parts.append(f"for {info['total']}")

        if info.get('date'):
            parts.append(f"on {info['date']}")

        num_items = len(info.get('items', []))
        if num_items > 0:
            parts.append(f"({num_items} items)")

        if parts:
            return ' '.join(parts) + '.'
        return "Receipt document."

    def _summarize_resume(self, info: Dict) -> str:
        """Summarize resume."""
        parts = []

        if info.get('name'):
            parts.append(f"Resume for {info['name']}")

        skills = info.get('skills', [])
        if skills:
            parts.append(f"with skills in {', '.join(skills[:3])}")

        experience = info.get('experience', [])
        if experience:
            parts.append(f"({len(experience)} positions)")

        if parts:
            return ' '.join(parts) + '.'
        return "Resume document."

    def _summarize_contract(self, info: Dict) -> str:
        """Summarize contract."""
        parts = []

        parties = info.get('parties', [])
        if parties:
            parts.append(f"Contract between {' and '.join(parties[:2])}")

        if info.get('value'):
            parts.append(f"worth {info['value']}")

        if info.get('term_months'):
            parts.append(f"for {info['term_months']} months")

        if info.get('date'):
            parts.append(f"effective {info['date']}")

        if parts:
            return ' '.join(parts) + '.'
        return "Contract document."

    def process(
        self,
        image_path: Union[str, Path],
        return_image: bool = False
    ) -> Dict:
        """
        Process document end-to-end.

        Args:
            image_path: Path to document image
            return_image: Whether to return image in results

        Returns:
            Dict with classification, extraction, summary results
        """
        start_time = time.time()

        # Load image
        image = Image.open(image_path).convert('RGB')
        logger.info(f"Processing document: {image_path}")

        # Step 1: Classify
        logger.info("Step 1: Classifying document type...")
        classification = self.classify(image)
        logger.info(f"  → Type: {classification['type']} (confidence: {classification['confidence']:.2%})")

        # Step 2: Extract information
        logger.info("Step 2: Extracting information...")
        extraction = self.extract_information(image, classification['type'])
        logger.info(f"  → Extracted {len(extraction)} fields")

        # Step 3: Generate summary
        logger.info("Step 3: Generating summary...")
        summary = self.summarize(extraction)
        logger.info(f"  → {summary}")

        # Build result
        processing_time = time.time() - start_time

        result = {
            'file': str(image_path),
            'classification': classification,
            'extraction': extraction,
            'summary': summary,
            'processing_time': f"{processing_time:.2f}s"
        }

        if return_image:
            result['image'] = image

        logger.info(f"✓ Processing complete ({processing_time:.2f}s)")

        return result

    def process_batch(
        self,
        image_paths: list,
        output_path: Optional[str] = None
    ) -> list:
        """
        Process multiple documents.

        Args:
            image_paths: List of image paths
            output_path: Optional path to save results JSON

        Returns:
            List of results
        """
        logger.info(f"Processing {len(image_paths)} documents...")

        results = []
        for i, image_path in enumerate(image_paths, 1):
            logger.info(f"\n[{i}/{len(image_paths)}] {image_path}")

            try:
                result = self.process(image_path)
                results.append(result)
            except Exception as e:
                logger.error(f"  ✗ Error: {e}")
                results.append({
                    'file': str(image_path),
                    'error': str(e)
                })

        # Save results if requested
        if output_path:
            with open(output_path, 'w') as f:
                json.dump(results, f, indent=2)
            logger.info(f"\n✓ Results saved to {output_path}")

        return results


def main():
    """CLI interface for document intelligence."""
    parser = argparse.ArgumentParser(
        description="Document Intelligence - Classify, Extract, Summarize"
    )

    parser.add_argument(
        'image_path',
        help='Path to document image or directory'
    )
    parser.add_argument(
        '--classifier',
        help='Path to trained classifier checkpoint'
    )
    parser.add_argument(
        '--extractor',
        help='Path to trained extractor model'
    )
    parser.add_argument(
        '--output',
        help='Path to save JSON results'
    )
    parser.add_argument(
        '--device',
        default='cpu',
        choices=['cpu', 'cuda'],
        help='Device to run on'
    )

    args = parser.parse_args()

    # Create pipeline
    pipeline = DocumentIntelligence(
        classifier_path=args.classifier,
        extractor_path=args.extractor,
        device=args.device
    )

    # Check if path is directory or file
    input_path = Path(args.image_path)

    if input_path.is_dir():
        # Process all images in directory
        image_paths = []
        for ext in ['*.png', '*.jpg', '*.jpeg']:
            image_paths.extend(input_path.glob(ext))

        results = pipeline.process_batch(
            image_paths,
            output_path=args.output
        )

        # Print summary
        print(f"\n{'=' * 60}")
        print(f"PROCESSED {len(results)} DOCUMENTS")
        print('=' * 60)

        for result in results:
            if 'error' in result:
                print(f"\n✗ {result['file']}: ERROR")
            else:
                print(f"\n✓ {result['file']}")
                print(f"  Type: {result['classification']['type']}")
                print(f"  Summary: {result['summary']}")

    else:
        # Process single image
        result = pipeline.process(input_path)

        # Print results
        print(f"\n{'=' * 60}")
        print("DOCUMENT INTELLIGENCE RESULTS")
        print('=' * 60)
        print(f"\nFile: {result['file']}")
        print(f"\nClassification:")
        print(f"  Type: {result['classification']['type']}")
        print(f"  Confidence: {result['classification']['confidence']:.2%}")

        print(f"\nExtracted Information:")
        for key, value in result['extraction'].items():
            if key != 'raw_text' and value:
                print(f"  {key}: {value}")

        print(f"\nSummary:")
        print(f"  {result['summary']}")

        print(f"\nProcessing Time: {result['processing_time']}")
        print('=' * 60)

        # Save if requested
        if args.output:
            with open(args.output, 'w') as f:
                # Remove image if present
                result_to_save = {k: v for k, v in result.items() if k != 'image'}
                json.dump(result_to_save, f, indent=2)
            print(f"\n✓ Results saved to {args.output}")


if __name__ == "__main__":
    main()
