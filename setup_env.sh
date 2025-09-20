#!/bin/bash
# Environment setup script for Reddit Mod Collection Pipeline

set -e  # Exit on any error

echo "🚀 Setting up Reddit Mod Collection Pipeline environment..."

# Check if conda is available
if command -v conda &> /dev/null; then
    echo "🐍 Found conda, creating conda environment..."

    # Create conda environment with Python 3.10
    conda create -n reddit-mod-pipeline python=3.10 -y

    # Activate conda environment
    echo "🔄 Activating conda environment..."
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate reddit-mod-pipeline

    # Install dependencies using pip (within conda env)
    echo "📥 Installing dependencies from requirements.txt..."
    pip install -r requirements.txt

    echo "✅ Conda environment setup complete!"
    echo ""
    echo "To activate the environment:"
    echo "  conda activate reddit-mod-pipeline"
    echo ""
    echo "To run the pipeline:"
    echo "  python run_pipeline.py"
    echo ""
    echo "To deactivate when done:"
    echo "  conda deactivate"

elif command -v python3 &> /dev/null; then
    echo "🐍 Conda not found, using Python venv..."

    PYTHON_VERSION=$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
    echo "🐍 Found Python $PYTHON_VERSION"

    # Create virtual environment
    echo "📦 Creating virtual environment..."
    python3 -m venv venv

    # Activate virtual environment
    echo "🔄 Activating virtual environment..."
    source venv/bin/activate

    # Upgrade pip
    echo "⬆️  Upgrading pip..."
    pip install --upgrade pip

    # Install dependencies
    echo "📥 Installing dependencies from requirements.txt..."
    pip install -r requirements.txt

    echo "✅ Virtual environment setup complete!"
    echo ""
    echo "To activate the environment:"
    echo "  source venv/bin/activate"
    echo ""
    echo "To run the pipeline:"
    echo "  python run_pipeline.py"
    echo ""
    echo "To deactivate when done:"
    echo "  deactivate"

else
    echo "❌ Neither conda nor python3 found. Please install Python 3.8+ or Anaconda/Miniconda."
    exit 1
fi