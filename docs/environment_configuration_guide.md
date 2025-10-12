# Environment Configuration Guide

## 📁 Configuration Files

Your project will have these configuration files:

```
document-intelligence-ml/
│
├── config.py                    # ✅ Main configuration (ROOT!)
├── .env                         # Environment variables
├── .env.example                 # Template
├── requirements.txt             # Dependencies
├── setup.sh                     # Setup script
├── README.md                    # Documentation
├── .gitignore                   # Git ignore rules
│
├── data/                        # Data directory
│   ├── raw/
│   ├── processed/
│   └── samples/
│
├── models/                      # Saved models
│
├── logs/                        # Training logs
│
├── src/                         # Source code
│   ├── __init__.py
│   │
│   ├── data/                    # Data processing
│   │   ├── __init__.py
│   │   ├── download.py
│   │   └── preprocess.py
│   │
│   ├── models/                  # Model definitions
│   │   ├── __init__.py
│   │   ├── classifier.py
│   │   └── extractor.py
│   │
│   ├── training/                # Training scripts
│   │   ├── __init__.py
│   │   ├── train.py
│   │   └── evaluate.py
│   │
│   ├── pipeline/                # ML pipelines
│   │   ├── __init__.py
│   │   └── sagemaker_pipeline.py
│   │
│   └── deployment/              # Deployment code
│       ├── __init__.py
│       └── inference.py
│
├── notebooks/                   # Jupyter notebooks
│   ├── sprint1_eda.ipynb
│   ├── sprint2_training.ipynb
│   └── ...
│
├── scripts/                     # Utility scripts
│   ├── test_config.py
│   ├── migrate_data.py
│   └── ...
│
├── web_app/                     # Web application
│   ├── app.py
│   └── ...
│
├── tests/                       # Unit tests
│   ├── test_data.py
│   └── ...
│
└── docs/                        # Documentation
    ├── project_plan.md
    └── blog_post.md
```

---

## 🔧 Setup Instructions

### **OPTION 1: Using Udacity AWS Account** (Current)

Create `.env.udacity`:

```bash
# ============================================
# UDACITY AWS ACCOUNT CONFIGURATION
# ============================================

# AWS Credentials (already configured in SageMaker)
AWS_REGION=us-east-1

# S3 Configuration - CHANGE THIS TO YOUR BUCKET
S3_BUCKET=your-udacity-bucket-name
USE_S3=true

# SageMaker Configuration (optional - only if deploying)
SAGEMAKER_ROLE=arn:aws:iam::YOUR_ACCOUNT:role/service-role/AmazonSageMaker-ExecutionRole
SAGEMAKER_INSTANCE_TYPE=ml.m5.xlarge
SAGEMAKER_INSTANCE_COUNT=1
ENDPOINT_NAME=doc-classifier-endpoint

# Model Configuration
MODEL_NAME=microsoft/layoutlmv3-base
USE_PRETRAINED=true
BATCH_SIZE=16
LEARNING_RATE=2e-5
NUM_EPOCHS=5

# Environment
ENVIRONMENT=sagemaker
```

**Activate it:**
```bash
cp .env.udacity .env
```

---

### **OPTION 2: Using Personal AWS Account**

Create `.env.personal`:

```bash
# ============================================
# PERSONAL AWS ACCOUNT CONFIGURATION
# ============================================

# AWS Credentials (if not using SageMaker)
# Option A: Use AWS CLI configured credentials (recommended)
# Option B: Set explicit credentials (not recommended for security)
# AWS_ACCESS_KEY_ID=your_access_key
# AWS_SECRET_ACCESS_KEY=your_secret_key

AWS_REGION=us-east-1

# S3 Configuration - YOUR PERSONAL BUCKET
S3_BUCKET=your-personal-bucket-name
USE_S3=true

# SageMaker Configuration (if using)
SAGEMAKER_ROLE=arn:aws:iam::YOUR_PERSONAL_ACCOUNT:role/SageMakerRole
SAGEMAKER_INSTANCE_TYPE=ml.m5.xlarge
SAGEMAKER_INSTANCE_COUNT=1
ENDPOINT_NAME=doc-classifier-personal-endpoint

# Model Configuration
MODEL_NAME=microsoft/layoutlmv3-base
USE_PRETRAINED=true
BATCH_SIZE=16
LEARNING_RATE=2e-5
NUM_EPOCHS=5

# Environment
ENVIRONMENT=cloud
```

**Activate it:**
```bash
cp .env.personal .env
```

---

### **OPTION 3: Local Development (No AWS)**

Create `.env.local`:

```bash
# ============================================
# LOCAL DEVELOPMENT CONFIGURATION
# ============================================

# No AWS - everything runs locally
USE_S3=false
AWS_REGION=us-east-1

# All data stored locally in ./data/
# Models stored locally in ./models/

# Model Configuration
MODEL_NAME=microsoft/layoutlmv3-base
USE_PRETRAINED=true
BATCH_SIZE=8  # Smaller for local machine
LEARNING_RATE=2e-5
NUM_EPOCHS=3  # Fewer epochs for testing

# Environment
ENVIRONMENT=local
```

**Activate it:**
```bash
cp .env.local .env
```

---

## 🔄 Switching Between Accounts (3 Simple Steps)

### **Step 1: Switch Configuration File**

```bash
# From Udacity to Personal
cp .env.personal .env

# From Personal to Local
cp .env.local .env

# From Local to Udacity
cp .env.udacity .env
```

### **Step 2: Update S3 Bucket (if using S3)**

If switching AWS accounts, sync your data to the new bucket:

```bash
# One-time data migration
python scripts/migrate_data.py --from-bucket udacity-bucket --to-bucket personal-bucket
```

Or manually using AWS CLI:
```bash
aws s3 sync s3://old-bucket/document-intelligence s3://new-bucket/document-intelligence
```

### **Step 3: Test Configuration**

```python
python -c "from src.config import config; print(config.to_dict())"
```

**That's it!** ✅ Everything else works automatically.

---

## 📋 Quick Reference: What Changes When You Switch

| Component | Udacity Account | Personal Account | Local Dev |
|-----------|----------------|------------------|-----------|
| Data Storage | S3 (Udacity bucket) | S3 (Your bucket) | Local `./data/` |
| Models | S3 (Udacity bucket) | S3 (Your bucket) | Local `./models/` |
| Credentials | Pre-configured | AWS CLI or Keys | Not needed |
| SageMaker | Udacity role | Your role | Not used |
| Cost | Uses Udacity credits | Your money | Free |

---

## 🚀 Initial Setup for Each Environment

### **On Udacity SageMaker Notebook:**

```bash
# 1. Clone your repo
git clone  https://github.com/mindelias/document-intelligence-ml.git
cd document-intelligence-ml

# 2. Create virtual environment (optional, notebook has most packages)
# pip install -r requirements.txt

# 3. Copy Udacity config
cp .env.udacity .env

# 4. Edit .env and set your S3 bucket name
nano .env  # or vim .env
# Change: S3_BUCKET=your-udacity-bucket-name

# 5. Test configuration
python -c "from src.config import config; print(config.to_dict())"

# 6. You're ready to go!
```

### **On Personal AWS Account:**

```bash
# 1. Setup AWS CLI
aws configure
# Enter your AWS Access Key ID
# Enter your AWS Secret Access Key
# Enter your default region: us-east-1

# 2. Create S3 bucket
aws s3 mb s3://your-personal-bucket-name

# 3. Clone repo
git clone https://github.com/YOUR_USERNAME/document-intelligence-ml.git
cd document-intelligence-ml

# 4. Install dependencies
pip install -r requirements.txt

# 5. Copy personal config
cp .env.personal .env

# 6. Edit .env with your bucket name
nano .env

# 7. Test
python -c "from src.config import config; print(config.to_dict())"
```

### **On Local Machine (No AWS):**

```bash
# 1. Clone repo
git clone https://github.com/YOUR_USERNAME/document-intelligence-ml.git
cd document-intelligence-ml

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Copy local config
cp .env.local .env

# 5. Test
python -c "from src.config import config; print(config.to_dict())"

# 6. Download datasets locally
python src/data/download.py --output ./data/raw/
```

---

## 🛠️ Helper Scripts

### **migrate_data.py** (For switching accounts)

```python
"""
Script to migrate data between S3 buckets or download to local.
Usage: python scripts/migrate_data.py --from s3://old-bucket --to s3://new-bucket
"""

import argparse
import boto3
from pathlib import Path

def migrate_s3_to_s3(from_bucket, to_bucket):
    """Copy data between S3 buckets"""
    s3 = boto3.client('s3')
    
    # List all objects in source bucket
    response = s3.list_objects_v2(Bucket=from_bucket, Prefix='document-intelligence/')
    
    for obj in response.get('Contents', []):
        key = obj['Key']
        copy_source = {'Bucket': from_bucket, 'Key': key}
        print(f"Copying {key}...")
        s3.copy_object(CopySource=copy_source, Bucket=to_bucket, Key=key)
    
    print(f"✓ Migration complete: {from_bucket} -> {to_bucket}")

def migrate_s3_to_local(bucket, local_dir):
    """Download data from S3 to local"""
    s3 = boto3.client('s3')
    local_path = Path(local_dir)
    local_path.mkdir(parents=True, exist_ok=True)
    
    response = s3.list_objects_v2(Bucket=bucket, Prefix='document-intelligence/')
    
    for obj in response.get('Contents', []):
        key = obj['Key']
        filename = key.split('/')[-1]
        local_file = local_path / filename
        
        print(f"Downloading {key}...")
        s3.download_file(bucket, key, str(local_file))
    
    print(f"✓ Download complete: s3://{bucket} -> {local_dir}")

def migrate_local_to_s3(local_dir, bucket):
    """Upload local data to S3"""
    s3 = boto3.client('s3')
    local_path = Path(local_dir)
    
    for file in local_path.rglob('*'):
        if file.is_file():
            key = f"document-intelligence/data/{file.name}"
            print(f"Uploading {file.name}...")
            s3.upload_file(str(file), bucket, key)
    
    print(f"✓ Upload complete: {local_dir} -> s3://{bucket}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--from-bucket', help='Source S3 bucket')
    parser.add_argument('--to-bucket', help='Destination S3 bucket')
    parser.add_argument('--to-local', help='Download to local directory')
    parser.add_argument('--from-local', help='Upload from local directory')
    
    args = parser.parse_args()
    
    if args.from_bucket and args.to_bucket:
        migrate_s3_to_s3(args.from_bucket, args.to_bucket)
    elif args.from_bucket and args.to_local:
        migrate_s3_to_local(args.from_bucket, args.to_local)
    elif args.from_local and args.to_bucket:
        migrate_local_to_s3(args.from_local, args.to_bucket)
    else:
        print("Usage examples:")
        print("  S3 to S3:    python migrate_data.py --from-bucket old-bucket --to-bucket new-bucket")
        print("  S3 to local: python migrate_data.py --from-bucket my-bucket --to-local ./data/")
        print("  Local to S3: python migrate_data.py --from-local ./data/ --to-bucket my-bucket")
```

### **test_config.py** (Verify your setup)

```python
"""
Test configuration and environment setup.
Usage: python scripts/test_config.py
"""

from src.config import config
import sys

def test_configuration():
    """Test if configuration is valid"""
    print("="*60)
    print("CONFIGURATION TEST")
    print("="*60)
    
    errors = []
    warnings = []
    
    # Test 1: Environment detection
    print(f"\n✓ Environment: {config.environment}")
    
    # Test 2: S3 configuration
    if config.use_s3:
        if not config.s3_bucket:
            errors.append("S3_BUCKET is not set but USE_S3=true")
        else:
            print(f"✓ S3 Bucket: {config.s3_bucket}")
            
            # Try to access S3
            try:
                import boto3
                s3 = boto3.client('s3', region_name=config.aws_region)
                s3.head_bucket(Bucket=config.s3_bucket)
                print(f"✓ S3 bucket is accessible")
            except Exception as e:
                warnings.append(f"Cannot access S3 bucket: {e}")
    else:
        print("✓ Using local storage (S3 disabled)")
    
    # Test 3: Data directories
    if config.local_data_dir.exists():
        print(f"✓ Data directory exists: {config.local_data_dir}")
    else:
        warnings.append(f"Data directory doesn't exist: {config.local_data_dir}")
    
    # Test 4: Model configuration
    print(f"✓ Model: {config.model_name}")
    print(f"✓ Document classes: {', '.join(config.document_classes)}")
    
    # Summary
    print("\n" + "="*60)
    if errors:
        print("❌ ERRORS:")
        for error in errors:
            print(f"  - {error}")
        print("\nConfiguration has errors. Please fix them in .env file.")
        sys.exit(1)
    elif warnings:
        print("⚠ WARNINGS:")
        for warning in warnings:
            print(f"  - {warning}")
        print("\nConfiguration works but has warnings.")
    else:
        print("✅ ALL TESTS PASSED!")
        print("Configuration is valid and ready to use.")
    
    print("="*60)

if __name__ == "__main__":
    test_configuration()
```

---

## 📝 .gitignore (IMPORTANT!)

Add this to your `.gitignore`:

```
# Environment files (keep templates only)
.env
.env.udacity
.env.personal
.env.local

# Don't commit credentials
*.pem
*.key

# AWS credentials
.aws/

# Data (too large for git)
data/raw/*
data/processed/*
!data/raw/.gitkeep
!data/processed/.gitkeep

# Models (too large for git)
models/*.pth
models/*.pt
models/*.bin
!models/.gitkeep

# Logs
logs/*
!logs/.gitkeep

# Cache
__pycache__/
*.pyc
.ipynb_checkpoints/
```

---

## ✅ Checklist: Ready to Switch Accounts?

Before switching from Udacity to Personal account:

- [ ] All code uses `config.get_data_path()` and `config.get_model_path()`
- [ ] No hard-coded bucket names in code
- [ ] No hard-coded AWS account IDs
- [ ] No hard-coded regions (uses `config.aws_region`)
- [ ] Created `.env.personal` with your bucket name
- [ ] Tested with `python scripts/test_config.py`
- [ ] (Optional) Migrated data: `python scripts/migrate_data.py`

**Then just:**
```bash
cp .env.personal .env
```

**And you're done!** 🎉

---

## 🎯 Example: How Code Works in Any Environment

```python
# ❌ BAD (Hard-coded, won't work when switching)
data_path = "s3://udacity-bucket/data/train.csv"
model_path = "s3://udacity-bucket/models/classifier.pth"

# ✅ GOOD (Environment-agnostic, works anywhere)
from src.config import config

data_path = config.get_data_path("train.csv")
model_path = config.get_model_path("classifier.pth")

# Automatically gives you:
# - Udacity account: s3://udacity-bucket/document-intelligence/data/raw/train.csv
# - Personal account: s3://personal-bucket/document-intelligence/data/raw/train.csv  
# - Local: ./data/raw/train.csv
```

---

## 💡 Tips

1. **Keep .env.example in git** - Template for others
2. **Never commit .env** - Contains your credentials
3. **Keep multiple .env files** - Easy switching
4. **Test after switching** - Run `test_config.py`
5. **Sync data once** - Use `migrate_data.py`

**You're now completely portable!** 🚀