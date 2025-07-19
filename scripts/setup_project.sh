#!/bin/bash
# Complete setup script for XAI Lung Segmentation project

set -e

echo "========================================================"
echo "    XAI LUNG SEGMENTATION PROJECT - COMPLETE SETUP     "
echo "========================================================"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}✓${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}⚠${NC} $1"
}

print_error() {
    echo -e "${RED}✗${NC} $1"
}

print_info() {
    echo -e "${BLUE}ℹ${NC} $1"
}

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Check system requirements
echo "Checking system requirements..."

# Check Python
if ! command_exists python3; then
    print_error "Python 3 is not installed"
    exit 1
fi
print_status "Python 3 found: $(python3 --version)"

# Check pip
if ! command_exists pip3; then
    print_error "pip3 is not installed"
    exit 1
fi
print_status "pip3 found"

# Check CUDA availability
if command_exists nvidia-smi; then
    print_status "CUDA GPU detected"
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader,nounits | head -1
else
    print_warning "No CUDA GPU detected - will use CPU (slower training)"
fi

# Check disk space
available_space=$(df . | tail -1 | awk '{print $4}')
if [ "$available_space" -lt 5000000 ]; then  # Less than 5GB
    print_warning "Low disk space detected. You may need more space for datasets and outputs."
else
    print_status "Sufficient disk space available"
fi

echo ""
echo "Setting up Python environment..."

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    print_info "Creating virtual environment..."
    python3 -m venv venv
    print_status "Virtual environment created"
else
    print_info "Virtual environment already exists"
fi

# Activate virtual environment
print_info "Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
print_info "Upgrading pip..."
pip install --upgrade pip > /dev/null 2>&1
print_status "pip upgraded"

# Install requirements
print_info "Installing Python packages..."
if [ -f "requirements.txt" ]; then
    pip install -r requirements.txt > /dev/null 2>&1
    print_status "Requirements installed successfully"
else
    print_warning "requirements.txt not found, installing basic packages..."
    pip install torch torchvision pandas numpy matplotlib seaborn tqdm Pillow PyYAML captum streamlit > /dev/null 2>&1
    print_status "Basic packages installed"
fi

echo ""
echo "Setting up project structure..."

# Make all scripts executable
print_info "Making scripts executable..."
chmod +x *.sh 2>/dev/null || true
print_status "Scripts made executable"

# Create output directory structure
print_info "Creating directory structure..."
mkdir -p outputs/{logs,organized_models,extended_training_logs}
print_status "Directory structure created"

# Generate data split manifests if script exists
if [ -f "generate_split_manifest.py" ]; then
    print_info "Generating data split manifests..."
    python3 generate_split_manifest.py
    print_status "Data split manifests generated"
fi

echo ""
echo "Validating configuration..."

# Check config file
if [ -f "config.yaml" ]; then
    print_status "config.yaml found"

    # Validate dataset paths
    print_info "Checking dataset paths..."
    python3 -c "
import yaml
import os
import sys

try:
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    datasets = config.get('datasets', {})
    all_valid = True

    for name, info in datasets.items():
        path = info.get('path', '')
        if os.path.exists(path):
            print(f'  ✓ {name}: {path}')
        else:
            print(f'  ✗ {name}: {path} (NOT FOUND)')
            all_valid = False

    if not all_valid:
        print('\n  Warning: Some dataset paths are invalid.')
        print('  Please update config.yaml with correct paths.')

except Exception as e:
    print(f'Error validating config: {e}')
    sys.exit(1)
"
else
    print_error "config.yaml not found"
    echo "Please create config.yaml with your dataset configurations"
    exit 1
fi

echo ""
echo "Testing installation..."

# Test imports
print_info "Testing Python imports..."
python3 -c "
import torch
import torchvision
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import yaml
import streamlit
import PIL
print('All imports successful')
" 2>/dev/null && print_status "All Python packages imported successfully" || print_error "Some imports failed"

# Test CUDA if available
if command_exists nvidia-smi; then
    print_info "Testing CUDA availability in PyTorch..."
    python3 -c "
import torch
if torch.cuda.is_available():
    print(f'CUDA available: {torch.cuda.get_device_name(0)}')
    print(f'PyTorch CUDA version: {torch.version.cuda}')
else:
    print('CUDA not available in PyTorch')
"
fi

echo ""
echo "========================================================"
echo "    SETUP COMPLETE!                                     "
echo "========================================================"
echo ""
echo "📋 Next Steps:"
echo ""
echo "1. 📊 Verify dataset paths in config.yaml"
echo "   - Edit config.yaml to point to your datasets"
echo "   - Ensure Montgomery and JSRT datasets are accessible"
echo ""
echo "2. 🚀 Run initial training:"
echo "   ./run_training.sh"
echo ""
echo "3. 📈 Run extended training for overfitting analysis:"
echo "   ./run_extended_training.sh"
echo ""
echo "4. 🔬 Run evaluation with XAI analysis:"
echo "   ./run_evaluation.sh"
echo "   ./run_extended_evaluation.sh"
echo ""
echo "5. 🎨 Launch the analysis dashboard:"
echo "   streamlit run app.py"
echo ""
echo "📁 Project Structure:"
echo "   ├── 🔧 Configuration: config.yaml"
echo "   ├── 🐍 Training: train.py, train_extended.py"
echo "   ├── 📊 Evaluation: evaluate.py"
echo "   ├── 🎨 Visualization: app.py"
echo "   ├── 🧰 Utilities: utils.py, data_loader.py, model.py"
echo "   └── 📤 Outputs: outputs/"
echo ""
echo "💡 Tips:"
echo "   - Monitor GPU usage with: nvidia-smi"
echo "   - Check training progress: tail -f outputs/[run_name]/train.log"
echo "   - View logs: ls outputs/logs/"
echo ""
echo "🔗 For help, run the app and go to the Help section"
echo ""
echo "Happy analyzing! 🫁✨"