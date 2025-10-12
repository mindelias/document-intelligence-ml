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