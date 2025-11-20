"""
Download public datasets for document intelligence training.

Supports:
- RVL-CDIP: Document classification (400K images)
- SROIE: Receipt understanding (1K receipts)
- CUAD: Contract understanding (510 contracts)
- Kaggle Resume Dataset
"""

import os
import sys
from pathlib import Path
import argparse
import logging
from typing import Optional, List
import requests
from tqdm import tqdm
import zipfile
import tarfile

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


class DatasetDownloader:
    """Download and prepare public datasets for document intelligence."""

    def __init__(self, config: Optional[Config] = None):
        """Initialize downloader with configuration."""
        self.config = config or Config()
        self.raw_data_dir = Path(self.config.data_dir) / "raw"
        self.raw_data_dir.mkdir(parents=True, exist_ok=True)

    def download_file(self, url: str, destination: Path, chunk_size: int = 8192) -> bool:
        """
        Download a file from URL with progress bar.

        Args:
            url: URL to download from
            destination: Local path to save file
            chunk_size: Download chunk size in bytes

        Returns:
            bool: True if successful, False otherwise
        """
        try:
            logger.info(f"Downloading from {url}")
            response = requests.get(url, stream=True, timeout=30)
            response.raise_for_status()

            total_size = int(response.headers.get('content-length', 0))

            destination.parent.mkdir(parents=True, exist_ok=True)

            with open(destination, 'wb') as f, tqdm(
                desc=destination.name,
                total=total_size,
                unit='iB',
                unit_scale=True,
                unit_divisor=1024,
            ) as pbar:
                for chunk in response.iter_content(chunk_size=chunk_size):
                    size = f.write(chunk)
                    pbar.update(size)

            logger.info(f"Successfully downloaded to {destination}")
            return True

        except Exception as e:
            logger.error(f"Error downloading {url}: {e}")
            return False

    def extract_archive(self, archive_path: Path, extract_to: Path) -> bool:
        """
        Extract zip or tar archive.

        Args:
            archive_path: Path to archive file
            extract_to: Directory to extract to

        Returns:
            bool: True if successful, False otherwise
        """
        try:
            logger.info(f"Extracting {archive_path.name}")
            extract_to.mkdir(parents=True, exist_ok=True)

            if archive_path.suffix == '.zip':
                with zipfile.ZipFile(archive_path, 'r') as zip_ref:
                    zip_ref.extractall(extract_to)
            elif archive_path.suffix in ['.tar', '.gz', '.tgz', '.bz2']:
                with tarfile.open(archive_path, 'r:*') as tar_ref:
                    tar_ref.extractall(extract_to)
            else:
                logger.error(f"Unsupported archive format: {archive_path.suffix}")
                return False

            logger.info(f"Successfully extracted to {extract_to}")
            return True

        except Exception as e:
            logger.error(f"Error extracting {archive_path}: {e}")
            return False

    def download_rvl_cdip_subset(self, num_samples_per_class: int = 250) -> bool:
        """
        Download RVL-CDIP dataset subset using HuggingFace datasets.

        RVL-CDIP contains 400,000 grayscale images in 16 classes:
        letter, form, email, handwritten, advertisement, scientific report,
        scientific publication, specification, file folder, news article,
        budget, invoice, presentation, questionnaire, resume, memo

        We'll download a subset focusing on our 4 target classes:
        - invoice
        - resume
        - email (proxy for contracts)
        - advertisement (proxy for general documents)

        Args:
            num_samples_per_class: Number of samples to download per class

        Returns:
            bool: True if successful, False otherwise
        """
        try:
            logger.info("Downloading RVL-CDIP subset using HuggingFace datasets...")

            from datasets import load_dataset

            # Load dataset (streaming to avoid downloading everything)
            dataset = load_dataset(
                "aharley/rvl_cdip",
                split="train",
                streaming=True
            )

            # Map class indices to names
            class_names = [
                'letter', 'form', 'email', 'handwritten', 'advertisement',
                'scientific_report', 'scientific_publication', 'specification',
                'file_folder', 'news_article', 'budget', 'invoice',
                'presentation', 'questionnaire', 'resume', 'memo'
            ]

            # Our target classes
            target_classes = {
                'invoice': 11,
                'resume': 14,
                'email': 2,  # Can be adapted for contracts
                'advertisement': 4
            }

            # Create directories
            rvl_dir = self.raw_data_dir / "rvl_cdip"
            rvl_dir.mkdir(parents=True, exist_ok=True)

            # Download samples
            class_counts = {cls: 0 for cls in target_classes.keys()}

            logger.info(f"Downloading {num_samples_per_class} samples per class...")

            for idx, example in enumerate(dataset):
                # Check if we've collected enough samples
                if all(count >= num_samples_per_class for count in class_counts.values()):
                    break

                label = example['label']

                # Check if this is a target class
                for class_name, class_idx in target_classes.items():
                    if label == class_idx and class_counts[class_name] < num_samples_per_class:
                        # Create class directory
                        class_dir = rvl_dir / class_name
                        class_dir.mkdir(parents=True, exist_ok=True)

                        # Save image
                        img_path = class_dir / f"{class_name}_{class_counts[class_name]:04d}.png"
                        example['image'].save(img_path)

                        class_counts[class_name] += 1

                        if class_counts[class_name] % 50 == 0:
                            logger.info(f"  {class_name}: {class_counts[class_name]}/{num_samples_per_class}")

                        break

            logger.info("RVL-CDIP download complete!")
            logger.info(f"Downloaded samples: {class_counts}")

            return True

        except Exception as e:
            logger.error(f"Error downloading RVL-CDIP: {e}")
            logger.info("You may need to install: pip install datasets")
            return False

    def download_sroie(self) -> bool:
        """
        Download SROIE 2019 receipt dataset.

        SROIE (Scanned Receipts OCR and Information Extraction)
        - 1,000 receipt images with annotations
        - Task 1: OCR
        - Task 2: Key information extraction (company, date, address, total)

        Returns:
            bool: True if successful, False otherwise
        """
        try:
            logger.info("Setting up SROIE dataset instructions...")

            sroie_dir = self.raw_data_dir / "sroie"
            sroie_dir.mkdir(parents=True, exist_ok=True)

            # Create instructions file
            instructions = """
# SROIE Dataset Download Instructions

The SROIE dataset requires manual download from the official source.

## Steps:

1. Visit: https://rrc.cvc.uab.es/?ch=13&com=downloads
2. Register for an account (free)
3. Download the following files:
   - Task 1: 0325updated.task1train(626p).zip
   - Task 2: task2train(626p).zip
   - Test data (optional)

4. Place the downloaded files in this directory:
   {sroie_dir}

5. Run the extraction script:
   python src/data/download_datasets.py --extract-sroie

## Alternative: Use sample receipts

If you just want to test the system, you can use the synthetic
receipt generator instead:

   python src/data/generate_synthetic.py --type receipt --count 100

## Dataset Details:

- 1,000 scanned receipt images
- Annotations for: company, date, address, total amount
- Train/Test split provided
- File formats: JPG images + TXT annotations
"""

            with open(sroie_dir / "README.md", 'w') as f:
                f.write(instructions.format(sroie_dir=sroie_dir))

            logger.info(f"SROIE instructions created at: {sroie_dir / 'README.md'}")
            logger.info("SROIE requires manual download. See instructions above.")

            return True

        except Exception as e:
            logger.error(f"Error setting up SROIE: {e}")
            return False

    def download_cuad(self) -> bool:
        """
        Download CUAD (Contract Understanding Atticus Dataset).

        CUAD contains 510 legal contracts with 41,000+ expert annotations
        for 41 types of clauses (e.g., termination, liability, pricing).

        Returns:
            bool: True if successful, False otherwise
        """
        try:
            logger.info("Downloading CUAD dataset...")

            from datasets import load_dataset

            # Load CUAD dataset
            dataset = load_dataset("cuad", split="train")

            # Create directory
            cuad_dir = self.raw_data_dir / "cuad"
            cuad_dir.mkdir(parents=True, exist_ok=True)

            # Save contracts
            logger.info(f"Saving {len(dataset)} contracts...")

            for idx, example in enumerate(tqdm(dataset, desc="Saving contracts")):
                # Save contract text
                contract_file = cuad_dir / f"contract_{idx:04d}.txt"
                with open(contract_file, 'w', encoding='utf-8') as f:
                    f.write(example['context'])

                # Save metadata
                if idx == 0:
                    logger.info(f"Sample contract saved to: {contract_file}")

            logger.info(f"CUAD dataset downloaded: {len(dataset)} contracts")
            logger.info(f"Location: {cuad_dir}")

            return True

        except Exception as e:
            logger.error(f"Error downloading CUAD: {e}")
            logger.info("You may need to install: pip install datasets")
            return False

    def download_sample_documents(self) -> bool:
        """
        Download sample documents for quick testing.

        Returns:
            bool: True if successful, False otherwise
        """
        try:
            logger.info("Creating sample document links...")

            samples_dir = self.raw_data_dir.parent / "samples"
            samples_dir.mkdir(parents=True, exist_ok=True)

            # Sample document sources
            samples = {
                'invoice': [
                    'https://templates.invoicehome.com/invoice-template-us-neat-750px.png',
                ],
                'receipt': [
                    'https://www.merchantmaverick.com/wp-content/uploads/2019/02/Sample-Receipt-Template.jpg',
                ],
                'resume': [
                    'https://www.resume.com/hs-fs/hubfs/resume-sample.png',
                ],
                'contract': [
                    # Contracts are text-based, will use synthetic generation
                ]
            }

            # Download samples
            for doc_type, urls in samples.items():
                if not urls:
                    continue

                type_dir = samples_dir / doc_type
                type_dir.mkdir(parents=True, exist_ok=True)

                for idx, url in enumerate(urls):
                    filename = f"sample_{idx:02d}.{url.split('.')[-1]}"
                    destination = type_dir / filename

                    logger.info(f"Downloading {doc_type} sample {idx + 1}...")
                    self.download_file(url, destination)

            logger.info("Sample documents downloaded!")
            return True

        except Exception as e:
            logger.error(f"Error downloading samples: {e}")
            return False

    def download_all(self, include_large: bool = False) -> bool:
        """
        Download all available datasets.

        Args:
            include_large: Whether to download large datasets (RVL-CDIP)

        Returns:
            bool: True if all successful, False otherwise
        """
        results = []

        logger.info("Starting dataset download process...")
        logger.info("=" * 60)

        # Download sample documents (quick)
        logger.info("\n[1/4] Downloading sample documents...")
        results.append(self.download_sample_documents())

        # Download RVL-CDIP subset (medium - ~250MB)
        if include_large:
            logger.info("\n[2/4] Downloading RVL-CDIP subset...")
            results.append(self.download_rvl_cdip_subset(num_samples_per_class=250))
        else:
            logger.info("\n[2/4] Skipping RVL-CDIP (use --include-large to download)")

        # Setup SROIE instructions
        logger.info("\n[3/4] Setting up SROIE instructions...")
        results.append(self.download_sroie())

        # Download CUAD (contracts)
        logger.info("\n[4/4] Downloading CUAD contracts...")
        results.append(self.download_cuad())

        logger.info("=" * 60)
        success_count = sum(results)
        total_count = len(results)
        logger.info(f"Download complete: {success_count}/{total_count} successful")

        return all(results)


def main():
    """Main function for CLI usage."""
    parser = argparse.ArgumentParser(
        description="Download public datasets for document intelligence"
    )
    parser.add_argument(
        '--dataset',
        choices=['rvl-cdip', 'sroie', 'cuad', 'samples', 'all'],
        default='all',
        help='Which dataset to download'
    )
    parser.add_argument(
        '--include-large',
        action='store_true',
        help='Include large datasets (RVL-CDIP)'
    )
    parser.add_argument(
        '--samples-per-class',
        type=int,
        default=250,
        help='Number of samples per class for RVL-CDIP'
    )

    args = parser.parse_args()

    # Create downloader
    downloader = DatasetDownloader()

    # Download requested dataset
    if args.dataset == 'all':
        success = downloader.download_all(include_large=args.include_large)
    elif args.dataset == 'rvl-cdip':
        success = downloader.download_rvl_cdip_subset(args.samples_per_class)
    elif args.dataset == 'sroie':
        success = downloader.download_sroie()
    elif args.dataset == 'cuad':
        success = downloader.download_cuad()
    elif args.dataset == 'samples':
        success = downloader.download_sample_documents()

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
