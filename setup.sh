#!/bin/bash

# ============================================================================
# Document Intelligence ML - Automated Setup Script
# ============================================================================
# This script sets up the project in ANY environment:
# - Udacity SageMaker Notebook
# - Personal AWS Account
# - Local Machine
# 
# Usage: bash setup.sh
# ============================================================================

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Helper functions
print_header() {
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${BLUE}  $1${NC}"
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
}

print_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠ $1${NC}"
}

print_error() {
    echo -e "${RED}✗ $1${NC}"
}

print_info() {
    echo -e "${BLUE}ℹ $1${NC}"
}

# ============================================================================
# STEP 1: Detect Environment
# ============================================================================

print_header "STEP 1: Environment Detection"

if [ -d "/opt/ml" ]; then
    ENV_TYPE="sagemaker"
    print_success "Detected: AWS SageMaker Notebook Instance"
elif command -v aws &> /dev/null; then
    ENV_TYPE="cloud"
    print_success "Detected: Cloud environment (AWS CLI installed)"
else
    ENV_TYPE="local"
    print_success "Detected: Local development environment"
fi

echo ""

# ============================================================================
# STEP 2: Create Directory Structure
# ============================================================================

print_header "STEP 2: Creating Project Structure"

# Create main directories
directories=(
    "data/raw"
    "data/processed"
    "data/samples"
    "models"
    "logs"
    "notebooks"
    "scripts"
    "tests"
    "web_app"
    "docs"
    "src/data"
    "src/models"
    "src/training"
    "src/pipeline"
    "src/deployment"
)

for dir in "${directories[@]}"; do
    if [ ! -d "$dir" ]; then
        mkdir -p "$dir"
        touch "$dir/.gitkeep"
        print_success "Created: $dir"
    else
        print_info "Exists: $dir"
    fi
done

echo ""

# ============================================================================
# STEP 3: Python Environment Setup
# ============================================================================

print_header "STEP 3: Python Environment Setup"

# Check Python version
PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
print_info "Python version: $PYTHON_VERSION"

# Create virtual environment (skip on SageMaker)
if [ "$ENV_TYPE" != "sagemaker" ]; then
    if [ ! -d "venv" ]; then
        print_info "Creating virtual environment..."
        python3 -m venv venv
        print_success "Virtual environment created"
        
        # Activate virtual environment
        if [ -f "venv/bin/activate" ]; then
            source venv/bin/activate
            print_success "Virtual environment activated"
        fi
    else
        print_info "Virtual environment already exists"
        source venv/bin/activate
    fi
else
    print_info "Skipping virtual environment (SageMaker has built-in environment)"
fi

# Install/upgrade pip
print_info "Upgrading pip..."
pip install --upgrade pip -q
print_success "Pip upgraded"

echo ""

# ============================================================================
# STEP 4: Install Dependencies
# ============================================================================

print_header "STEP 4: Installing Dependencies"

if [ -f "requirements.txt" ]; then
    print_info "Installing from requirements.txt..."
    pip install -r requirements.txt -q
    print_success "Dependencies installed"
else
    print_warning "requirements.txt not found, skipping package installation"
fi

echo ""

# ============================================================================
# STEP 5: Configuration Setup
# ============================================================================

print_header "STEP 5: Configuration Setup"

# Check if .env exists
if [ ! -f ".env" ]; then
    if [ -f ".env.example" ]; then
        print_info "Creating .env from template..."
        cp .env.example .env
        print_success "Created .env file"
        
        # Prompt for S3 bucket
        echo ""
        print_info "Configuration needed!"
        read -p "Enter your S3 bucket name (or press Enter to skip): " s3_bucket
        
        if [ ! -z "$s3_bucket" ]; then
            # Update .env file with bucket name
            if [[ "$OSTYPE" == "darwin"* ]]; then
                # macOS
                sed -i '' "s/S3_BUCKET=your-bucket-name-here/S3_BUCKET=$s3_bucket/" .env
            else
                # Linux
                sed -i "s/S3_BUCKET=your-bucket-name-here/S3_BUCKET=$s3_bucket/" .env
            fi
            print_success "S3 bucket configured: $s3_bucket"
        else
            print_warning "S3 bucket not configured. Using local storage."
            if [[ "$OSTYPE" == "darwin"* ]]; then
                sed -i '' "s/USE_S3=true/USE_S3=false/" .env
            else
                sed -i "s/USE_S3=true/USE_S3=false/" .env
            fi
        fi
    else
        print_error ".env.example not found. Cannot create configuration."
    fi
else
    print_info ".env file already exists"
fi

echo ""

# ============================================================================
# STEP 6: Test Configuration
# ============================================================================

print_header "STEP 6: Testing Configuration"

if [ -f "scripts/test_config.py" ] && [ -f "src/config.py" ]; then
    print_info "Running configuration tests..."
    if python scripts/test_config.py; then
        print_success "Configuration test passed!"
    else
        print_error "Configuration test failed. Please check your .env file."
    fi
else
    print_warning "Configuration test script not found, skipping tests"
fi

echo ""

# ============================================================================
# STEP 7: Git Setup
# ============================================================================

print_header "STEP 7: Git Configuration"

# Initialize git if not already done
if [ ! -d ".git" ]; then
    print_info "Initializing git repository..."
    git init
    print_success "Git repository initialized"
else
    print_info "Git repository already initialized"
fi

# Check for .gitignore
if [ ! -f ".gitignore" ]; then
    print_warning ".gitignore not found. Environment files may be committed!"
else
    print_success ".gitignore configured"
fi

echo ""

# ============================================================================
# STEP 8: Download Sample Data (Optional)
# ============================================================================

print_header "STEP 8: Sample Data Setup"

echo ""
read -p "Download sample documents for testing? (y/N): " download_samples

if [[ $download_samples =~ ^[Yy]$ ]]; then
    print_info "Downloading sample documents..."
    
    # Create samples directory
    mkdir -p data/samples
    
    # Download some sample documents (you can customize these URLs)
    # For now, just create placeholder files
    touch data/samples/sample_invoice.pdf
    touch data/samples/sample_receipt.jpg
    touch data/samples/sample_resume.pdf
    
    print_success "Sample data setup complete"
else
    print_info "Skipping sample data download"
fi

echo ""

# ============================================================================
# STEP 9: Final Summary
# ============================================================================

print_header "Setup Complete! 🎉"

echo ""
echo "Environment Type: $ENV_TYPE"
echo ""
echo "Project Structure Created:"
echo "  ✓ Data directories (data/raw, data/processed)"
echo "  ✓ Model directory (models/)"
echo "  ✓ Source code directories (src/)"
echo "  ✓ Scripts directory (scripts/)"
echo "  ✓ Notebooks directory (notebooks/)"
echo ""

if [ "$ENV_TYPE" != "sagemaker" ]; then
    echo "Python Environment:"
    echo "  ✓ Virtual environment created (venv/)"
    echo "  ✓ To activate: source venv/bin/activate"
    echo ""
fi

echo "Configuration:"
if [ -f ".env" ]; then
    echo "  ✓ .env file created"
    echo "  ℹ Remember to update .env with your settings"
else
    echo "  ⚠ .env file not created - please create manually"
fi
echo ""

echo "Next Steps:"
echo "  1. Review and update .env file with your configuration"
echo "  2. Test configuration: python scripts/test_config.py"
echo "  3. Start Sprint 1: jupyter lab notebooks/sprint1_eda.ipynb"
echo ""

print_header "Quick Commands"
echo ""
echo "  Test Config:     python scripts/test_config.py"
echo "  Start Jupyter:   jupyter lab"
echo "  Run Training:    python src/training/train.py"
echo "  Migrate Data:    python scripts/migrate_data.py --help"
echo ""

if [ "$ENV_TYPE" = "local" ]; then
    echo "  Activate venv:   source venv/bin/activate"
    echo ""
fi

print_success "You're ready to start building! 🚀"
echo ""
echo "For detailed instructions, see: docs/project_plan.md"
echo ""

# ============================================================================
# End of setup script
# ============================================================================