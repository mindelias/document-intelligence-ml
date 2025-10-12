"""
Script to migrate data between S3 buckets or download to local.
Usage: 
  python scripts/migrate_data.py --from-bucket old-bucket --to-bucket new-bucket
  python scripts/migrate_data.py --from-bucket my-bucket --to-local ./data/
"""

import argparse
import boto3
from pathlib import Path
from tqdm import tqdm

def migrate_s3_to_s3(from_bucket, to_bucket, prefix='document-intelligence/'):
    """Copy data between S3 buckets"""
    s3 = boto3.client('s3')
    
    print(f"Migrating from s3://{from_bucket} to s3://{to_bucket}")
    
    # List all objects in source bucket
    paginator = s3.get_paginator('list_objects_v2')
    pages = paginator.paginate(Bucket=from_bucket, Prefix=prefix)
    
    total_copied = 0
    for page in pages:
        for obj in page.get('Contents', []):
            key = obj['Key']
            copy_source = {'Bucket': from_bucket, 'Key': key}
            
            print(f"Copying {key}...")
            s3.copy_object(CopySource=copy_source, Bucket=to_bucket, Key=key)
            total_copied += 1
    
    print(f"✓ Migration complete! Copied {total_copied} files")

def migrate_s3_to_local(bucket, local_dir, prefix='document-intelligence/'):
    """Download data from S3 to local"""
    s3 = boto3.client('s3')
    local_path = Path(local_dir)
    local_path.mkdir(parents=True, exist_ok=True)
    
    print(f"Downloading from s3://{bucket} to {local_dir}")
    
    paginator = s3.get_paginator('list_objects_v2')
    pages = paginator.paginate(Bucket=bucket, Prefix=prefix)
    
    files = []
    for page in pages:
        files.extend(page.get('Contents', []))
    
    for obj in tqdm(files, desc="Downloading"):
        key = obj['Key']
        filename = key.replace(prefix, '')
        local_file = local_path / filename
        local_file.parent.mkdir(parents=True, exist_ok=True)
        
        s3.download_file(bucket, key, str(local_file))
    
    print(f"✓ Download complete! {len(files)} files")

def migrate_local_to_s3(local_dir, bucket, prefix='document-intelligence/'):
    """Upload local data to S3"""
    s3 = boto3.client('s3')
    local_path = Path(local_dir)
    
    print(f"Uploading from {local_dir} to s3://{bucket}")
    
    files = list(local_path.rglob('*'))
    files = [f for f in files if f.is_file()]
    
    for file in tqdm(files, desc="Uploading"):
        relative_path = file.relative_to(local_path)
        key = f"{prefix}data/{relative_path}"
        
        s3.upload_file(str(file), bucket, key)
    
    print(f"✓ Upload complete! {len(files)} files")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Migrate data between S3 buckets or local storage'
    )
    parser.add_argument('--from-bucket', help='Source S3 bucket')
    parser.add_argument('--to-bucket', help='Destination S3 bucket')
    parser.add_argument('--to-local', help='Download to local directory')
    parser.add_argument('--from-local', help='Upload from local directory')
    parser.add_argument('--prefix', default='document-intelligence/',
                       help='S3 prefix (default: document-intelligence/)')
    
    args = parser.parse_args()
    
    try:
        if args.from_bucket and args.to_bucket:
            migrate_s3_to_s3(args.from_bucket, args.to_bucket, args.prefix)
        elif args.from_bucket and args.to_local:
            migrate_s3_to_local(args.from_bucket, args.to_local, args.prefix)
        elif args.from_local and args.to_bucket:
            migrate_local_to_s3(args.from_local, args.to_bucket, args.prefix)
        else:
            print("Usage examples:")
            print("  S3 to S3:    python migrate_data.py --from-bucket old --to-bucket new")
            print("  S3 to local: python migrate_data.py --from-bucket my-bucket --to-local ./data/")
            print("  Local to S3: python migrate_data.py --from-local ./data/ --to-bucket my-bucket")
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)