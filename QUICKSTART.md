# Quick Start Guide

Get started with Document Intelligence in 5 minutes!

## Prerequisites

```bash
# Install dependencies
pip install -r requirement.txt

# Download spaCy model
python -m spacy download en_core_web_sm
```

## Option 1: Quick Test with Synthetic Data (Recommended)

Perfect for testing without downloading large datasets.

### Step 1: Generate Synthetic Documents

```bash
# Generate 100 documents of each type
python src/data/generate_synthetic.py --type all --count 100
```

This creates realistic-looking documents in `data/synthetic/`:
- `invoice/` - Business invoices
- `receipt/` - Store receipts
- `resume/` - Professional resumes
- `contract/` - Legal contracts

### Step 2: Prepare Training Data

```bash
# Split data into train/val/test sets
python src/data/preprocess.py --split --use-synthetic --analyze
```

This creates:
- `data/processed/train/` - 70% of data
- `data/processed/val/` - 15% of data
- `data/processed/test/` - 15% of data

### Step 3: Train the Classifier

```bash
# Train with ResNet-50 (recommended for quick start)
python src/training/train.py \
    --model resnet50 \
    --epochs 10 \
    --batch-size 16 \
    --early-stopping 3
```

Training takes ~5-10 minutes on GPU, ~30-60 minutes on CPU.

The model will be saved to `models/checkpoints/best.pth`.

### Step 4: Test Inference

```bash
# Process a single document
python src/inference/predict.py \
    data/synthetic/invoice/invoice_0001.png \
    --classifier models/checkpoints/best.pth
```

Expected output:
```
Classification:
  Type: invoice
  Confidence: 95.23%

Extracted Information:
  invoice_number: 12345
  total: $1,234.56
  company: TechCorp Solutions
  date: 01/15/2024

Summary:
  Invoice from TechCorp Solutions (#12345) for $1,234.56 dated 01/15/2024.
```

## Option 2: Using Public Datasets

For production-quality models, use real datasets.

### Step 1: Download Public Datasets

```bash
# Download sample documents (quick, ~10MB)
python src/data/download_datasets.py --dataset samples

# Download RVL-CDIP subset (medium, ~250MB, 1000 images)
python src/data/download_datasets.py --dataset rvl-cdip --samples-per-class 250

# Download CUAD contracts (small, ~50MB)
python src/data/download_datasets.py --dataset cuad

# Download ALL datasets
python src/data/download_datasets.py --dataset all --include-large
```

### Step 2: Combine with Synthetic Data

```bash
# Prepare mixed dataset (real + synthetic)
python src/data/preprocess.py --split --use-synthetic --analyze
```

### Step 3: Train

Same as Option 1, Step 3.

## Option 3: Advanced - Full Pipeline

For best results:

```bash
# 1. Generate synthetic data
python src/data/generate_synthetic.py --type all --count 500

# 2. Download public datasets
python src/data/download_datasets.py --dataset all --include-large

# 3. Prepare data
python src/data/preprocess.py --split --use-synthetic --analyze

# 4. Train with EfficientNet (better accuracy)
python src/training/train.py \
    --model efficientnet \
    --epochs 30 \
    --batch-size 32 \
    --lr 1e-4 \
    --early-stopping 5

# 5. Process multiple documents
python src/inference/predict.py \
    data/samples/ \
    --classifier models/checkpoints/best.pth \
    --output results.json
```

## Testing Individual Components

### Test Synthetic Generation

```bash
# Generate just invoices
python src/data/generate_synthetic.py --type invoice --count 10

# Output: data/synthetic/invoice/
```

### Test Model Creation

```bash
# Test ResNet-50
python src/models/classifier.py --model resnet50

# Test EfficientNet
python src/models/classifier.py --model efficientnet
```

### Test Extraction

```bash
# Extract from a document
python src/models/extractor.py \
    data/synthetic/invoice/invoice_0001.png \
    --type invoice
```

## Common Issues

### Issue: CUDA out of memory

**Solution:** Reduce batch size
```bash
python src/training/train.py --batch-size 8
```

### Issue: "No module named 'transformers'"

**Solution:** Transformers is optional for LayoutLM only
```bash
# If you want to use LayoutLM:
pip install transformers

# Otherwise, use ResNet/EfficientNet (recommended)
```

### Issue: "spaCy model not found"

**Solution:** Download the model
```bash
python -m spacy download en_core_web_sm
```

### Issue: Slow training

**Solution:** Use smaller model or fewer epochs
```bash
python src/training/train.py --model resnet50 --epochs 5
```

## Next Steps

- **Improve accuracy:** Train longer with more data
- **Deploy model:** See `docs/deployment_guide.md` (coming soon)
- **Custom extraction:** Modify `src/models/extractor.py` for your use case
- **Web interface:** Run Streamlit app (coming soon)

## Performance Expectations

### With Synthetic Data (100 per class)
- Training time: 5-10 mins (GPU) / 30-60 mins (CPU)
- Accuracy: 85-90%
- Good for: Testing, prototyping, demos

### With Public Datasets (1000+ per class)
- Training time: 30-60 mins (GPU) / 2-4 hours (CPU)
- Accuracy: 90-95%
- Good for: Production use

### With LayoutLM (Advanced)
- Training time: 2-4 hours (GPU required)
- Accuracy: 95-98%
- Good for: State-of-the-art results

## Resources

- **README.md** - Full documentation
- **docs/** - Detailed guides
- **requirement.txt** - All dependencies
- **config.py** - Configuration settings

## Getting Help

- Check existing code documentation
- Review error messages carefully
- Start with synthetic data for quick testing
- Use smaller models (ResNet-50) before trying larger ones

Happy document processing! 🚀
