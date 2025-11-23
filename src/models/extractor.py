"""
Document information extraction models.

Extracts key information from documents:
- Invoice: invoice number, total, date, company
- Receipt: store name, total, date, items
- Resume: name, email, phone, skills
- Contract: parties, date, terms, value
"""

import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import logging
import re
from datetime import datetime

import torch
import torch.nn as nn
import pytesseract
from PIL import Image
import numpy as np
import spacy

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


class DocumentExtractor:
    """Base class for document information extraction."""

    def __init__(self, config: Optional[Config] = None):
        """Initialize extractor."""
        self.config = config or Config()

        # Try to load spaCy model
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            logger.warning(
                "spaCy model not found. Install with: "
                "python -m spacy download en_core_web_sm"
            )
            self.nlp = None

    def extract_text_from_image(self, image: Image.Image) -> str:
        """
        Extract text from image using OCR.

        Args:
            image: PIL Image

        Returns:
            Extracted text
        """
        try:
            text = pytesseract.image_to_string(image)
            return text
        except Exception as e:
            logger.error(f"OCR failed: {e}")
            return ""

    def extract_entities(self, text: str) -> List[Dict]:
        """
        Extract named entities from text.

        Args:
            text: Input text

        Returns:
            List of entities
        """
        if self.nlp is None:
            return []

        doc = self.nlp(text)
        entities = []

        for ent in doc.ents:
            entities.append({
                'text': ent.text,
                'label': ent.label_,
                'start': ent.start_char,
                'end': ent.end_char
            })

        return entities

    def extract(self, image: Image.Image, doc_type: str) -> Dict:
        """
        Extract information from document.

        Args:
            image: Document image
            doc_type: Type of document

        Returns:
            Extracted information
        """
        # Extract text
        text = self.extract_text_from_image(image)

        # Route to specific extractor
        if doc_type == 'invoice':
            return self.extract_invoice(text, image)
        elif doc_type == 'receipt':
            return self.extract_receipt(text, image)
        elif doc_type == 'resume':
            return self.extract_resume(text, image)
        elif doc_type == 'contract':
            return self.extract_contract(text, image)
        else:
            return {'error': f'Unknown document type: {doc_type}'}

    def extract_invoice(self, text: str, image: Image.Image) -> Dict:
        """
        Extract key information from invoice.

        Returns:
            Dict with invoice_number, total, date, company, customer
        """
        result = {
            'type': 'invoice',
            'raw_text': text,
            'invoice_number': None,
            'total': None,
            'date': None,
            'company': None,
            'customer': None
        }

        # Extract invoice number
        invoice_patterns = [
            r'Invoice\s*#?\s*:?\s*(\d+)',
            r'INV-?(\d+)',
            r'Invoice\s+Number\s*:?\s*(\d+)'
        ]
        for pattern in invoice_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                result['invoice_number'] = match.group(1)
                break

        # Extract total
        total_patterns = [
            r'Total\s*:?\s*\$?([\d,]+\.?\d*)',
            r'Amount\s+Due\s*:?\s*\$?([\d,]+\.?\d*)',
            r'TOTAL\s*:?\s*\$?([\d,]+\.?\d*)'
        ]
        for pattern in total_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                result['total'] = f"${match.group(1)}"
                break

        # Extract date
        result['date'] = self._extract_date(text)

        # Extract entities for company/customer names
        entities = self.extract_entities(text)
        orgs = [e['text'] for e in entities if e['label'] == 'ORG']
        persons = [e['text'] for e in entities if e['label'] == 'PERSON']

        if orgs:
            result['company'] = orgs[0]
        if persons:
            result['customer'] = persons[0]

        return result

    def extract_receipt(self, text: str, image: Image.Image) -> Dict:
        """
        Extract key information from receipt.

        Returns:
            Dict with store, total, date, items
        """
        result = {
            'type': 'receipt',
            'raw_text': text,
            'store': None,
            'total': None,
            'date': None,
            'items': []
        }

        # Extract store name (usually first line or in header)
        entities = self.extract_entities(text)
        orgs = [e['text'] for e in entities if e['label'] == 'ORG']
        if orgs:
            result['store'] = orgs[0]

        # Extract total
        total_patterns = [
            r'TOTAL\s*:?\s*\$?([\d,]+\.?\d*)',
            r'Total\s*:?\s*\$?([\d,]+\.?\d*)',
            r'Amount\s*:?\s*\$?([\d,]+\.?\d*)'
        ]
        for pattern in total_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                result['total'] = f"${match.group(1)}"
                break

        # Extract date
        result['date'] = self._extract_date(text)

        # Extract line items (simplified)
        item_pattern = r'(.+?)\s+\$?([\d,]+\.?\d{2})'
        items = re.findall(item_pattern, text)
        result['items'] = [
            {'name': name.strip(), 'price': f"${price}"}
            for name, price in items[:10]  # Limit to 10 items
        ]

        return result

    def extract_resume(self, text: str, image: Image.Image) -> Dict:
        """
        Extract key information from resume.

        Returns:
            Dict with name, email, phone, skills, experience
        """
        result = {
            'type': 'resume',
            'raw_text': text,
            'name': None,
            'email': None,
            'phone': None,
            'skills': [],
            'experience': []
        }

        # Extract email
        email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        email_match = re.search(email_pattern, text)
        if email_match:
            result['email'] = email_match.group(0)

        # Extract phone
        phone_pattern = r'[\+\(]?[1-9][0-9 .\-\(\)]{8,}[0-9]'
        phone_match = re.search(phone_pattern, text)
        if phone_match:
            result['phone'] = phone_match.group(0).strip()

        # Extract name (usually first PERSON entity)
        entities = self.extract_entities(text)
        persons = [e['text'] for e in entities if e['label'] == 'PERSON']
        if persons:
            result['name'] = persons[0]

        # Extract skills (common technical skills)
        skill_keywords = [
            'Python', 'Java', 'JavaScript', 'C\\+\\+', 'SQL', 'React', 'Node',
            'AWS', 'Docker', 'Kubernetes', 'Machine Learning', 'Data Science',
            'Project Management', 'Agile', 'Leadership'
        ]

        found_skills = []
        for skill in skill_keywords:
            if re.search(skill, text, re.IGNORECASE):
                found_skills.append(skill)

        result['skills'] = found_skills

        # Extract work experience (simplified - look for years)
        year_pattern = r'(\d{4})\s*-\s*(\d{4}|Present)'
        years = re.findall(year_pattern, text, re.IGNORECASE)
        result['experience'] = [f"{start}-{end}" for start, end in years]

        return result

    def extract_contract(self, text: str, image: Image.Image) -> Dict:
        """
        Extract key information from contract.

        Returns:
            Dict with contract_number, parties, date, terms
        """
        result = {
            'type': 'contract',
            'raw_text': text,
            'contract_number': None,
            'parties': [],
            'date': None,
            'term_months': None,
            'value': None,
            'key_terms': []
        }

        # Extract contract number
        contract_patterns = [
            r'Contract\s*#?\s*:?\s*(\d+)',
            r'Agreement\s+No\.\s*:?\s*(\d+)'
        ]
        for pattern in contract_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                result['contract_number'] = match.group(1)
                break

        # Extract parties
        entities = self.extract_entities(text)
        orgs = [e['text'] for e in entities if e['label'] == 'ORG']
        persons = [e['text'] for e in entities if e['label'] == 'PERSON']
        result['parties'] = orgs + persons

        # Extract date
        result['date'] = self._extract_date(text)

        # Extract term duration
        term_pattern = r'(\d+)\s*months?'
        term_match = re.search(term_pattern, text, re.IGNORECASE)
        if term_match:
            result['term_months'] = int(term_match.group(1))

        # Extract value
        value_pattern = r'\$\s?([\d,]+(?:\.\d{2})?)'
        values = re.findall(value_pattern, text)
        if values:
            # Take the largest value (likely total contract value)
            result['value'] = max([f"${v}" for v in values], key=lambda x: float(x.replace('$', '').replace(',', '')))

        # Extract key terms (sections)
        section_pattern = r'\d+\.\s+([A-Z][A-Za-z\s]+)'
        sections = re.findall(section_pattern, text)
        result['key_terms'] = sections[:5]  # First 5 sections

        return result

    def _extract_date(self, text: str) -> Optional[str]:
        """
        Extract date from text.

        Args:
            text: Input text

        Returns:
            Date string or None
        """
        # Common date patterns
        date_patterns = [
            r'(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})',  # MM/DD/YYYY or MM-DD-YYYY
            r'(\d{4}[/-]\d{1,2}[/-]\d{1,2})',    # YYYY-MM-DD
            r'((?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{1,2},?\s+\d{4})',  # Month DD, YYYY
        ]

        for pattern in date_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(1)

        return None

    def summarize(self, text: str, max_sentences: int = 3) -> str:
        """
        Summarize document text.

        Args:
            text: Input text
            max_sentences: Maximum number of sentences in summary

        Returns:
            Summary text
        """
        # Simple extractive summarization
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if len(s.strip()) > 20]

        if len(sentences) <= max_sentences:
            return '. '.join(sentences) + '.'

        # Take first and last sentences (simple heuristic)
        summary_sentences = [sentences[0]]

        if len(sentences) > 2:
            summary_sentences.append(sentences[len(sentences) // 2])

        if len(sentences) > 1:
            summary_sentences.append(sentences[-1])

        return '. '.join(summary_sentences[:max_sentences]) + '.'


class SmartDocumentExtractor(DocumentExtractor):
    """
    Enhanced extractor with ML-based entity recognition.

    Uses LayoutLM or similar models for better accuracy.
    Falls back to rule-based extraction if ML model unavailable.
    """

    def __init__(self, model_path: Optional[str] = None, config: Optional[Config] = None):
        """
        Initialize smart extractor.

        Args:
            model_path: Path to trained extraction model
            config: Configuration
        """
        super().__init__(config)

        self.model = None
        if model_path and Path(model_path).exists():
            try:
                self.model = torch.load(model_path)
                self.model.eval()
                logger.info(f"Loaded extraction model from {model_path}")
            except Exception as e:
                logger.warning(f"Could not load model from {model_path}: {e}")
                logger.info("Falling back to rule-based extraction")

    def extract(self, image: Image.Image, doc_type: str) -> Dict:
        """
        Extract information using ML model if available, else use rules.

        Args:
            image: Document image
            doc_type: Type of document

        Returns:
            Extracted information
        """
        if self.model is not None:
            # Use ML model for extraction
            try:
                result = self._extract_with_model(image, doc_type)
                return result
            except Exception as e:
                logger.warning(f"ML extraction failed: {e}")
                logger.info("Falling back to rule-based extraction")

        # Fall back to rule-based extraction
        return super().extract(image, doc_type)

    def _extract_with_model(self, image: Image.Image, doc_type: str) -> Dict:
        """
        Extract using ML model (LayoutLM, etc.).

        Args:
            image: Document image
            doc_type: Document type

        Returns:
            Extracted information
        """
        # This would use LayoutLM or similar for extraction
        # For now, falls back to rule-based
        raise NotImplementedError("ML-based extraction not yet implemented")


def main():
    """Test extraction."""
    import argparse

    parser = argparse.ArgumentParser(description="Test document extraction")
    parser.add_argument('image_path', help='Path to document image')
    parser.add_argument('--type', required=True, choices=['invoice', 'receipt', 'resume', 'contract'])

    args = parser.parse_args()

    # Create extractor
    extractor = DocumentExtractor()

    # Load image
    image = Image.open(args.image_path)

    # Extract information
    result = extractor.extract(image, args.type)

    # Print results
    print(f"\n{'=' * 60}")
    print(f"EXTRACTED INFORMATION - {args.type.upper()}")
    print('=' * 60)

    for key, value in result.items():
        if key != 'raw_text':
            print(f"{key}: {value}")

    print('=' * 60)


if __name__ == "__main__":
    main()
