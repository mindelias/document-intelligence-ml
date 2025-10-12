"""
config.py - Environment-Agnostic Configuration System

This configuration system allows the project to run in ANY environment:
- Udacity AWS Account
- Personal AWS Account  
- Local Development Machine
- SageMaker Notebook Instance
- Any other environment

Just change .env file and everything works!
"""

import os
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, Dict, Any
import json

# ============================================================================
# PROJECT PATHS (Works locally and on any cloud environment)
# ============================================================================

PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
LOGS_DIR = PROJECT_ROOT / "logs"
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"

# Create directories if they don't exist
for directory in [DATA_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR, 
                  MODELS_DIR, LOGS_DIR, NOTEBOOKS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)


# ============================================================================
# ENVIRONMENT DETECTION
# ============================================================================

def detect_environment() -> str:
    """
    Automatically detect which environment we're running in.
    Returns: 'local', 'sagemaker', or 'cloud'
    """
    # Check if running on SageMaker
    if os.path.exists('/opt/ml/metadata/'):
        return 'sagemaker'
    
    # Check if AWS credentials exist
    if os.getenv('AWS_ACCESS_KEY_ID') or os.path.exists(os.path.expanduser('~/.aws/credentials')):
        return 'cloud'
    
    return 'local'


# ============================================================================
# CONFIGURATION CLASS
# ============================================================================

@dataclass
class Config:
    """
    Main configuration class. All settings in one place.
    Override with environment variables or .env file.
    """
    
    # Environment
    environment: str = detect_environment()
    
    # AWS Configuration (Can be changed per account)
    aws_region: str = os.getenv("AWS_REGION", "us-east-1")
    s3_bucket: str = os.getenv("S3_BUCKET", None)  # MUST be set in .env
    sagemaker_role: str = os.getenv("SAGEMAKER_ROLE", None)
    aws_account_id: str = os.getenv("AWS_ACCOUNT_ID", None)
    
    # Data Configuration
    use_s3: bool = os.getenv("USE_S3", "true").lower() == "true"
    local_data_dir: Path = DATA_DIR
    s3_data_prefix: str = "document-intelligence/data"
    
    # Model Configuration
    document_classes: list = None
    image_size: tuple = (224, 224)
    batch_size: int = int(os.getenv("BATCH_SIZE", "16"))
    learning_rate: float = float(os.getenv("LEARNING_RATE", "2e-5"))
    num_epochs: int = int(os.getenv("NUM_EPOCHS", "5"))
    
    # Model Selection
    model_name: str = os.getenv("MODEL_NAME", "microsoft/layoutlmv3-base")
    use_pretrained: bool = os.getenv("USE_PRETRAINED", "true").lower() == "true"
    
    # SageMaker Configuration (Optional - only if using SageMaker)
    sagemaker_instance_type: str = os.getenv("SAGEMAKER_INSTANCE_TYPE", "ml.m5.xlarge")
    sagemaker_instance_count: int = int(os.getenv("SAGEMAKER_INSTANCE_COUNT", "1"))
    endpoint_name: str = os.getenv("ENDPOINT_NAME", "doc-classifier-endpoint")
    
    # Extraction Configuration
    extraction_config: Dict[str, list] = None
    
    def __post_init__(self):
        """Initialize default values"""
        if self.document_classes is None:
            self.document_classes = ["invoice", "receipt", "resume", "contract"]
        
        if self.extraction_config is None:
            self.extraction_config = {
                "invoice": ["invoice_number", "date", "vendor", "total_amount"],
                "receipt": ["merchant", "date", "total", "items"],
                "resume": ["name", "email", "phone", "skills", "experience"],
                "contract": ["parties", "effective_date", "expiration_date", "terms"]
            }
        
        # Validate required fields when using AWS
        if self.use_s3 and not self.s3_bucket:
            raise ValueError(
                "S3_BUCKET must be set in environment variables when USE_S3=true. "
                "Add it to your .env file or set it as environment variable."
            )
    
    def get_data_path(self, filename: str, use_processed: bool = False) -> str:
        """
        Get path to data file - works for both local and S3.
        
        Args:
            filename: Name of the file
            use_processed: If True, use processed data directory
            
        Returns:
            Full path (local or S3 URI)
        """
        if self.use_s3:
            subdir = "processed" if use_processed else "raw"
            return f"s3://{self.s3_bucket}/{self.s3_data_prefix}/{subdir}/{filename}"
        else:
            data_dir = PROCESSED_DATA_DIR if use_processed else RAW_DATA_DIR
            return str(data_dir / filename)
    
    def get_model_path(self, model_name: str) -> str:
        """Get path to model - works for both local and S3"""
        if self.use_s3:
            return f"s3://{self.s3_bucket}/document-intelligence/models/{model_name}"
        else:
            return str(MODELS_DIR / model_name)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary"""
        return {
            'environment': self.environment,
            'aws_region': self.aws_region,
            's3_bucket': self.s3_bucket,
            'use_s3': self.use_s3,
            'document_classes': self.document_classes,
            'batch_size': self.batch_size,
            'learning_rate': self.learning_rate,
            'num_epochs': self.num_epochs,
            'model_name': self.model_name,
        }
    
    def save(self, filepath: str):
        """Save configuration to JSON file"""
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def load(cls, filepath: str):
        """Load configuration from JSON file"""
        with open(filepath, 'r') as f:
            data = json.load(f)
        return cls(**data)


# ============================================================================
# HELPER FUNCTIONS FOR DATA ACCESS
# ============================================================================

def get_storage_client(config: Config):
    """
    Get appropriate storage client based on environment.
    Returns S3 client or local file system handler.
    """
    if config.use_s3:
        import boto3
        return boto3.client('s3', region_name=config.aws_region)
    else:
        return None  # Use regular file operations


def upload_file(local_path: str, remote_path: str, config: Config):
    """
    Upload file to storage (S3 or local copy).
    Works regardless of environment.
    """
    if config.use_s3:
        import boto3
        s3 = boto3.client('s3', region_name=config.aws_region)
        
        # Parse S3 URI
        if remote_path.startswith('s3://'):
            remote_path = remote_path[5:]  # Remove s3://
        
        parts = remote_path.split('/', 1)
        bucket = parts[0]
        key = parts[1] if len(parts) > 1 else ''
        
        s3.upload_file(local_path, bucket, key)
        print(f"✓ Uploaded {local_path} to s3://{bucket}/{key}")
    else:
        import shutil
        Path(remote_path).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(local_path, remote_path)
        print(f"✓ Copied {local_path} to {remote_path}")


def download_file(remote_path: str, local_path: str, config: Config):
    """
    Download file from storage (S3 or local copy).
    Works regardless of environment.
    """
    if config.use_s3:
        import boto3
        s3 = boto3.client('s3', region_name=config.aws_region)
        
        # Parse S3 URI
        if remote_path.startswith('s3://'):
            remote_path = remote_path[5:]
        
        parts = remote_path.split('/', 1)
        bucket = parts[0]
        key = parts[1] if len(parts) > 1 else ''
        
        Path(local_path).parent.mkdir(parents=True, exist_ok=True)
        s3.download_file(bucket, key, local_path)
        print(f"✓ Downloaded s3://{bucket}/{key} to {local_path}")
    else:
        import shutil
        Path(local_path).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(remote_path, local_path)
        print(f"✓ Copied {remote_path} to {local_path}")


def list_files(remote_path: str, config: Config) -> list:
    """
    List files in storage location (S3 or local).
    Works regardless of environment.
    """
    if config.use_s3:
        import boto3
        s3 = boto3.client('s3', region_name=config.aws_region)
        
        # Parse S3 URI
        if remote_path.startswith('s3://'):
            remote_path = remote_path[5:]
        
        parts = remote_path.split('/', 1)
        bucket = parts[0]
        prefix = parts[1] if len(parts) > 1 else ''
        
        response = s3.list_objects_v2(Bucket=bucket, Prefix=prefix)
        files = [obj['Key'] for obj in response.get('Contents', [])]
        return files
    else:
        path = Path(remote_path)
        if path.exists():
            return [str(f) for f in path.rglob('*') if f.is_file()]
        return []


# ============================================================================
# GLOBAL CONFIG INSTANCE
# ============================================================================

# Load configuration from .env if it exists
try:
    from dotenv import load_dotenv
    load_dotenv()
    print("✓ Loaded configuration from .env file")
except ImportError:
    print("⚠ python-dotenv not installed. Using environment variables only.")

# Create global config instance
config = Config()

print(f"""
╔════════════════════════════════════════════════════════════╗
║          DOCUMENT INTELLIGENCE - CONFIGURATION             ║
╚════════════════════════════════════════════════════════════╝

Environment: {config.environment}
AWS Region: {config.aws_region}
S3 Bucket: {config.s3_bucket or 'Not configured (using local storage)'}
Use S3: {config.use_s3}
Data Directory: {config.local_data_dir}
Model: {config.model_name}

Document Classes: {', '.join(config.document_classes)}
Batch Size: {config.batch_size}
Learning Rate: {config.learning_rate}

✓ Configuration loaded successfully!
""")


# ============================================================================
# USAGE EXAMPLES
# ============================================================================

if __name__ == "__main__":
    print("\n" + "="*60)
    print("CONFIGURATION EXAMPLES")
    print("="*60)
    
    # Example 1: Get data path
    print("\n1. Data Paths:")
    print(f"   Raw data: {config.get_data_path('sample.pdf')}")
    print(f"   Processed: {config.get_data_path('sample.pdf', use_processed=True)}")
    
    # Example 2: Get model path
    print("\n2. Model Path:")
    print(f"   Model: {config.get_model_path('classifier_v1.pth')}")
    
    # Example 3: Save config
    print("\n3. Save Configuration:")
    config.save('config.json')
    print("   ✓ Saved to config.json")
    
    # Example 4: Environment detection
    print(f"\n4. Detected Environment: {config.environment}")
    print(f"   Using S3: {config.use_s3}")