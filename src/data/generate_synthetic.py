"""
Generate synthetic documents for training and testing.

Creates realistic-looking documents:
- Invoices
- Receipts
- Resumes
- Contracts

Perfect for prototyping without needing real data.
"""

import os
import sys
from pathlib import Path
import argparse
import logging
from typing import Dict, List, Tuple, Optional
import random
from datetime import datetime, timedelta
from PIL import Image, ImageDraw, ImageFont
import numpy as np

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


class SyntheticDocumentGenerator:
    """Generate synthetic documents for training."""

    # Sample data for generation
    COMPANIES = [
        "TechCorp Solutions", "Global Industries", "Acme Corporation",
        "Smith & Associates", "Premier Services", "Innovation Labs",
        "Elite Consulting", "Quantum Systems", "Vista Enterprises"
    ]

    NAMES = [
        "John Smith", "Sarah Johnson", "Michael Brown", "Emily Davis",
        "David Wilson", "Jessica Martinez", "James Anderson", "Jennifer Taylor",
        "Robert Thomas", "Linda Garcia", "William Rodriguez", "Maria Lopez"
    ]

    CITIES = [
        ("New York", "NY"), ("Los Angeles", "CA"), ("Chicago", "IL"),
        ("Houston", "TX"), ("Phoenix", "AZ"), ("Philadelphia", "PA"),
        ("San Antonio", "TX"), ("San Diego", "CA"), ("Dallas", "TX")
    ]

    ITEMS = [
        ("Software License", 99.99, 599.99),
        ("Consulting Services", 150.00, 500.00),
        ("Cloud Storage", 29.99, 199.99),
        ("Support Package", 49.99, 299.99),
        ("Training Program", 199.99, 999.99),
        ("API Access", 79.99, 399.99),
        ("Premium Features", 39.99, 199.99),
        ("Custom Development", 500.00, 5000.00)
    ]

    SKILLS = [
        "Python", "JavaScript", "Java", "C++", "SQL", "React", "Node.js",
        "AWS", "Docker", "Kubernetes", "Machine Learning", "Data Analysis",
        "Project Management", "Agile", "Leadership", "Communication"
    ]

    JOBS = [
        ("Software Engineer", "TechCorp", "2020-2023"),
        ("Senior Developer", "Global Industries", "2018-2020"),
        ("Data Scientist", "Innovation Labs", "2017-2018"),
        ("Product Manager", "Elite Consulting", "2015-2017"),
        ("Team Lead", "Quantum Systems", "2013-2015")
    ]

    def __init__(self, config: Optional[Config] = None):
        """Initialize generator with configuration."""
        self.config = config or Config()
        self.synthetic_dir = Path(self.config.data_dir) / "synthetic"
        self.synthetic_dir.mkdir(parents=True, exist_ok=True)

        # Default font (will try to use system fonts)
        self.default_font_size = 20
        self.title_font_size = 32
        self.small_font_size = 16

    def get_font(self, size: int = 20):
        """Get font for text rendering."""
        try:
            # Try to use a nice font
            return ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", size)
        except:
            # Fall back to default
            return ImageFont.load_default()

    def random_date(self, start_year: int = 2020, end_year: int = 2024) -> str:
        """Generate random date."""
        start = datetime(start_year, 1, 1)
        end = datetime(end_year, 12, 31)
        delta = end - start
        random_days = random.randint(0, delta.days)
        date = start + timedelta(days=random_days)
        return date.strftime("%m/%d/%Y")

    def add_noise(self, image: Image.Image, noise_level: float = 0.02) -> Image.Image:
        """Add realistic scan noise to image."""
        img_array = np.array(image)

        # Add Gaussian noise
        noise = np.random.normal(0, noise_level * 255, img_array.shape)
        noisy = np.clip(img_array + noise, 0, 255).astype(np.uint8)

        # Add slight rotation
        angle = random.uniform(-1, 1)
        noisy_img = Image.fromarray(noisy)
        rotated = noisy_img.rotate(angle, fillcolor=(255, 255, 255))

        return rotated

    def generate_invoice(self, invoice_id: int = None) -> Tuple[Image.Image, Dict]:
        """
        Generate synthetic invoice.

        Returns:
            Tuple of (image, metadata)
        """
        # Create canvas
        width, height = 800, 1000
        image = Image.new('RGB', (width, height), color='white')
        draw = ImageDraw.Draw(image)

        # Fonts
        title_font = self.get_font(self.title_font_size)
        normal_font = self.get_font(self.default_font_size)
        small_font = self.get_font(self.small_font_size)

        # Generate data
        invoice_num = invoice_id or random.randint(10000, 99999)
        company = random.choice(self.COMPANIES)
        customer = random.choice(self.NAMES)
        date = self.random_date()
        city, state = random.choice(self.CITIES)

        # Draw invoice
        y = 50

        # Title
        draw.text((50, y), "INVOICE", font=title_font, fill='black')
        y += 60

        # Invoice details
        draw.text((50, y), f"Invoice #: {invoice_num}", font=normal_font, fill='black')
        draw.text((500, y), f"Date: {date}", font=normal_font, fill='black')
        y += 40

        # Company info
        draw.text((50, y), "From:", font=normal_font, fill='black')
        y += 30
        draw.text((50, y), company, font=normal_font, fill='black')
        y += 25
        draw.text((50, y), f"{city}, {state}", font=normal_font, fill='black')
        y += 50

        # Customer info
        draw.text((50, y), "Bill To:", font=normal_font, fill='black')
        y += 30
        draw.text((50, y), customer, font=normal_font, fill='black')
        y += 50

        # Line items header
        draw.line([(50, y), (750, y)], fill='black', width=2)
        y += 10
        draw.text((50, y), "Description", font=normal_font, fill='black')
        draw.text((500, y), "Quantity", font=normal_font, fill='black')
        draw.text((650, y), "Amount", font=normal_font, fill='black')
        y += 30
        draw.line([(50, y), (750, y)], fill='black', width=1)
        y += 20

        # Line items
        num_items = random.randint(2, 5)
        total = 0.0

        for _ in range(num_items):
            item, min_price, max_price = random.choice(self.ITEMS)
            quantity = random.randint(1, 5)
            price = random.uniform(min_price, max_price)
            amount = quantity * price
            total += amount

            draw.text((50, y), item, font=small_font, fill='black')
            draw.text((500, y), str(quantity), font=small_font, fill='black')
            draw.text((650, y), f"${amount:.2f}", font=small_font, fill='black')
            y += 25

        # Total
        y += 20
        draw.line([(500, y), (750, y)], fill='black', width=2)
        y += 15
        draw.text((500, y), "TOTAL:", font=normal_font, fill='black')
        draw.text((650, y), f"${total:.2f}", font=normal_font, fill='black')

        # Footer
        y = height - 100
        draw.text((50, y), "Thank you for your business!", font=small_font, fill='gray')

        # Metadata
        metadata = {
            'type': 'invoice',
            'invoice_number': str(invoice_num),
            'company': company,
            'customer': customer,
            'date': date,
            'total': f"${total:.2f}",
            'num_items': num_items
        }

        return image, metadata

    def generate_receipt(self, receipt_id: int = None) -> Tuple[Image.Image, Dict]:
        """
        Generate synthetic receipt.

        Returns:
            Tuple of (image, metadata)
        """
        # Create canvas (receipts are narrower)
        width, height = 400, 600
        image = Image.new('RGB', (width, height), color='white')
        draw = ImageDraw.Draw(image)

        # Fonts
        title_font = self.get_font(24)
        normal_font = self.get_font(16)
        small_font = self.get_font(14)

        # Generate data
        receipt_num = receipt_id or random.randint(1000, 9999)
        store = random.choice(self.COMPANIES)
        date = self.random_date()
        time = f"{random.randint(8, 20)}:{random.randint(0, 59):02d}"

        # Draw receipt
        y = 30

        # Store name
        draw.text((width // 2 - 80, y), store, font=title_font, fill='black')
        y += 40

        # Date and time
        draw.text((width // 2 - 60, y), f"{date} {time}", font=small_font, fill='black')
        y += 25
        draw.text((width // 2 - 60, y), f"Receipt #: {receipt_num}", font=small_font, fill='black')
        y += 40

        # Line items
        draw.line([(20, y), (380, y)], fill='black', width=1)
        y += 15

        num_items = random.randint(3, 8)
        subtotal = 0.0

        for _ in range(num_items):
            item, min_price, max_price = random.choice(self.ITEMS)
            price = random.uniform(min_price / 10, max_price / 10)  # Lower prices for receipts
            subtotal += price

            draw.text((20, y), item[:20], font=normal_font, fill='black')
            draw.text((320, y), f"${price:.2f}", font=normal_font, fill='black')
            y += 22

        # Totals
        y += 10
        draw.line([(20, y), (380, y)], fill='black', width=1)
        y += 15

        tax = subtotal * 0.08
        total = subtotal + tax

        draw.text((20, y), "Subtotal:", font=normal_font, fill='black')
        draw.text((320, y), f"${subtotal:.2f}", font=normal_font, fill='black')
        y += 25

        draw.text((20, y), "Tax (8%):", font=normal_font, fill='black')
        draw.text((320, y), f"${tax:.2f}", font=normal_font, fill='black')
        y += 25

        draw.line([(200, y), (380, y)], fill='black', width=2)
        y += 10

        draw.text((20, y), "TOTAL:", font=title_font, fill='black')
        draw.text((320, y), f"${total:.2f}", font=title_font, fill='black')

        # Footer
        y = height - 60
        draw.text((width // 2 - 80, y), "Thank you!", font=small_font, fill='gray')

        # Metadata
        metadata = {
            'type': 'receipt',
            'receipt_number': str(receipt_num),
            'store': store,
            'date': date,
            'time': time,
            'subtotal': f"${subtotal:.2f}",
            'tax': f"${tax:.2f}",
            'total': f"${total:.2f}",
            'num_items': num_items
        }

        return image, metadata

    def generate_resume(self, resume_id: int = None) -> Tuple[Image.Image, Dict]:
        """
        Generate synthetic resume.

        Returns:
            Tuple of (image, metadata)
        """
        # Create canvas
        width, height = 800, 1000
        image = Image.new('RGB', (width, height), color='white')
        draw = ImageDraw.Draw(image)

        # Fonts
        title_font = self.get_font(self.title_font_size)
        section_font = self.get_font(24)
        normal_font = self.get_font(self.default_font_size)
        small_font = self.get_font(self.small_font_size)

        # Generate data
        name = random.choice(self.NAMES)
        email = f"{name.lower().replace(' ', '.')}@email.com"
        phone = f"({random.randint(200, 999)}) {random.randint(200, 999)}-{random.randint(1000, 9999)}"
        city, state = random.choice(self.CITIES)

        # Draw resume
        y = 50

        # Name
        draw.text((50, y), name, font=title_font, fill='black')
        y += 50

        # Contact info
        draw.text((50, y), f"{email} | {phone}", font=small_font, fill='gray')
        y += 25
        draw.text((50, y), f"{city}, {state}", font=small_font, fill='gray')
        y += 50

        # Summary
        draw.text((50, y), "PROFESSIONAL SUMMARY", font=section_font, fill='black')
        y += 35
        draw.text((50, y), "Experienced professional with expertise in software development,", font=normal_font, fill='black')
        y += 25
        draw.text((50, y), "data analysis, and project management.", font=normal_font, fill='black')
        y += 50

        # Skills
        draw.text((50, y), "SKILLS", font=section_font, fill='black')
        y += 35
        skills = random.sample(self.SKILLS, 6)
        draw.text((50, y), " • " + " • ".join(skills[:3]), font=normal_font, fill='black')
        y += 25
        draw.text((50, y), " • " + " • ".join(skills[3:]), font=normal_font, fill='black')
        y += 50

        # Experience
        draw.text((50, y), "WORK EXPERIENCE", font=section_font, fill='black')
        y += 35

        num_jobs = random.randint(2, 4)
        for job, company, dates in random.sample(self.JOBS, num_jobs):
            draw.text((50, y), job, font=normal_font, fill='black')
            y += 25
            draw.text((50, y), f"{company} | {dates}", font=small_font, fill='gray')
            y += 35

        # Education
        y += 20
        draw.text((50, y), "EDUCATION", font=section_font, fill='black')
        y += 35
        draw.text((50, y), "Bachelor of Science in Computer Science", font=normal_font, fill='black')
        y += 25
        draw.text((50, y), f"{random.choice(self.COMPANIES)} University | 2012-2016", font=small_font, fill='gray')

        # Metadata
        metadata = {
            'type': 'resume',
            'name': name,
            'email': email,
            'phone': phone,
            'location': f"{city}, {state}",
            'skills': skills,
            'num_jobs': num_jobs
        }

        return image, metadata

    def generate_contract(self, contract_id: int = None) -> Tuple[Image.Image, Dict]:
        """
        Generate synthetic contract.

        Returns:
            Tuple of (image, metadata)
        """
        # Create canvas
        width, height = 800, 1000
        image = Image.new('RGB', (width, height), color='white')
        draw = ImageDraw.Draw(image)

        # Fonts
        title_font = self.get_font(28)
        normal_font = self.get_font(self.default_font_size)
        small_font = self.get_font(self.small_font_size)

        # Generate data
        contract_num = contract_id or random.randint(1000, 9999)
        party1 = random.choice(self.COMPANIES)
        party2 = random.choice(self.NAMES)
        date = self.random_date()
        term_months = random.choice([6, 12, 24, 36])
        value = random.randint(10000, 100000)

        # Draw contract
        y = 50

        # Title
        draw.text((width // 2 - 150, y), "SERVICE AGREEMENT", font=title_font, fill='black')
        y += 60

        # Contract number and date
        draw.text((50, y), f"Contract #: {contract_num}", font=normal_font, fill='black')
        y += 30
        draw.text((50, y), f"Effective Date: {date}", font=normal_font, fill='black')
        y += 50

        # Parties
        draw.text((50, y), "This Agreement is entered into between:", font=normal_font, fill='black')
        y += 35

        draw.text((50, y), f"Party A: {party1} (\"Provider\")", font=normal_font, fill='black')
        y += 30
        draw.text((50, y), f"Party B: {party2} (\"Client\")", font=normal_font, fill='black')
        y += 50

        # Terms
        draw.text((50, y), "TERMS AND CONDITIONS", font=section_font, fill='black')
        y += 40

        # Term 1
        draw.text((50, y), "1. Scope of Services", font=normal_font, fill='black')
        y += 30
        draw.text((70, y), "Provider agrees to deliver professional services as outlined", font=small_font, fill='black')
        y += 22
        draw.text((70, y), "in Exhibit A, subject to the terms of this Agreement.", font=small_font, fill='black')
        y += 40

        # Term 2
        draw.text((50, y), "2. Term", font=normal_font, fill='black')
        y += 30
        draw.text((70, y), f"This Agreement shall remain in effect for {term_months} months", font=small_font, fill='black')
        y += 22
        draw.text((70, y), f"from the Effective Date, unless terminated earlier.", font=small_font, fill='black')
        y += 40

        # Term 3
        draw.text((50, y), "3. Compensation", font=normal_font, fill='black')
        y += 30
        draw.text((70, y), f"Client agrees to pay Provider ${value:,} for services", font=small_font, fill='black')
        y += 22
        draw.text((70, y), "rendered under this Agreement.", font=small_font, fill='black')
        y += 40

        # Term 4
        draw.text((50, y), "4. Termination", font=normal_font, fill='black')
        y += 30
        draw.text((70, y), "Either party may terminate this Agreement with 30 days", font=small_font, fill='black')
        y += 22
        draw.text((70, y), "written notice to the other party.", font=small_font, fill='black')
        y += 60

        # Signature lines
        y = height - 150
        draw.line([(50, y), (300, y)], fill='black', width=1)
        draw.line([(500, y), (750, y)], fill='black', width=1)
        y += 10
        draw.text((50, y), party1, font=small_font, fill='black')
        draw.text((500, y), party2, font=small_font, fill='black')
        y += 25
        draw.text((50, y), "Provider", font=small_font, fill='gray')
        draw.text((500, y), "Client", font=small_font, fill='gray')

        # Metadata
        metadata = {
            'type': 'contract',
            'contract_number': str(contract_num),
            'party1': party1,
            'party2': party2,
            'date': date,
            'term_months': term_months,
            'value': f"${value:,}",
        }

        return image, metadata

    def generate_documents(
        self,
        doc_type: str,
        count: int,
        add_noise: bool = True
    ) -> List[Tuple[Path, Dict]]:
        """
        Generate multiple documents of specified type.

        Args:
            doc_type: Type of document (invoice, receipt, resume, contract)
            count: Number of documents to generate
            add_noise: Whether to add realistic noise/artifacts

        Returns:
            List of (file_path, metadata) tuples
        """
        logger.info(f"Generating {count} synthetic {doc_type}s...")

        # Get generator function
        generators = {
            'invoice': self.generate_invoice,
            'receipt': self.generate_receipt,
            'resume': self.generate_resume,
            'contract': self.generate_contract
        }

        if doc_type not in generators:
            raise ValueError(f"Unknown document type: {doc_type}")

        generator = generators[doc_type]

        # Create output directory
        output_dir = self.synthetic_dir / doc_type
        output_dir.mkdir(parents=True, exist_ok=True)

        results = []

        for i in range(count):
            # Generate document
            image, metadata = generator(i)

            # Add noise if requested
            if add_noise:
                image = self.add_noise(image)

            # Save image
            filename = f"{doc_type}_{i:04d}.png"
            filepath = output_dir / filename
            image.save(filepath)

            # Save metadata
            metadata_file = output_dir / f"{doc_type}_{i:04d}.json"
            import json
            with open(metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2)

            results.append((filepath, metadata))

            if (i + 1) % 10 == 0:
                logger.info(f"  Generated {i + 1}/{count} {doc_type}s")

        logger.info(f"Successfully generated {count} {doc_type}s in {output_dir}")
        return results


def main():
    """Main function for CLI usage."""
    parser = argparse.ArgumentParser(
        description="Generate synthetic documents for training"
    )
    parser.add_argument(
        '--type',
        choices=['invoice', 'receipt', 'resume', 'contract', 'all'],
        default='all',
        help='Type of document to generate'
    )
    parser.add_argument(
        '--count',
        type=int,
        default=100,
        help='Number of documents to generate per type'
    )
    parser.add_argument(
        '--no-noise',
        action='store_true',
        help='Disable realistic noise/artifacts'
    )

    args = parser.parse_args()

    # Create generator
    generator = SyntheticDocumentGenerator()

    # Generate documents
    if args.type == 'all':
        for doc_type in ['invoice', 'receipt', 'resume', 'contract']:
            generator.generate_documents(
                doc_type,
                args.count,
                add_noise=not args.no_noise
            )
    else:
        generator.generate_documents(
            args.type,
            args.count,
            add_noise=not args.no_noise
        )

    logger.info("Document generation complete!")


if __name__ == "__main__":
    main()
