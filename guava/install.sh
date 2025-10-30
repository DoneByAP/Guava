#!/bin/bash
# Quick installation and testing script for distrib-train-net

set -e

echo "=================================="
echo "distrib-train-net Installation"
echo "=================================="
echo

# Check Python version
echo "Checking Python version..."
python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo "✓ Python $python_version"

# Check for CUDA
echo
echo "Checking CUDA availability..."
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi --query-gpu=name --format=csv,noheader | nl
    echo "✓ CUDA available"
else
    echo "⚠️  CUDA not detected (CPU-only mode)"
fi

# Install package
echo
echo "Installing distrib-train-net..."
pip install -e . --quiet
echo "✓ Package installed"

# Install optional dependencies
echo
echo "Installing optional dependencies..."
pip install pytest black flake8 --quiet
echo "✓ Dev dependencies installed"

# Run quick test
echo
echo "Running quick test..."
python3 << 'EOF'
import torch
from distrib_train_net import DistributedConfig, __version__

print(f"✓ distrib-train-net v{__version__} imported successfully")
print(f"  - PyTorch: {torch.__version__}")
print(f"  - CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"  - CUDA devices: {torch.cuda.device_count()}")

# Test basic config
config = DistributedConfig()
print(f"✓ Configuration created")
print(f"  - Strategy: {config.get_parallelism_strategy().value}")
EOF

echo
echo "=================================="
echo "Installation Complete!"
echo "=================================="
echo
echo "Next steps:"
echo "  1. Check examples: cd src/distrib_train_net/examples"
echo "  2. Read migration guide: cat MIGRATION.md"
echo "  3. Run example: python -m distrib_train_net.examples.transformer_example --mode local"
echo
echo "For multi-machine setup:"
echo "  Orchestrator: python your_script.py --mode orchestrator"
echo "  Workers: python your_script.py --mode worker --master-ip <ORCHESTRATOR_IP>"
echo
