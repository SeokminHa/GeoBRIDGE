#!/bin/bash
# Setup script for Velorama comparison benchmark on CSAIL GPU server (charybdis)
#
# Usage:
#   ssh seokmin@charybdis.csail.mit.edu
#   cd /data/cb/scratch/seokmin/geobridge
#   bash benchmark/setup_velorama_server.sh

set -e

echo "========================================="
echo "Setting up Velorama benchmark environment"
echo "========================================="

WORKDIR="/data/cb/scratch/seokmin/geobridge"
cd "$WORKDIR"

# Check GPU
echo ""
echo "[1] Checking GPU..."
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
else
    echo "WARNING: nvidia-smi not found. GPU may not be available."
fi

# Check Python
echo ""
echo "[2] Python version..."
python3 --version

# Create/update venv with GPU support
echo ""
echo "[3] Setting up virtual environment..."
if [ ! -d "venv_gpu" ]; then
    python3 -m venv venv_gpu
    echo "Created new venv_gpu"
else
    echo "venv_gpu already exists"
fi
source venv_gpu/bin/activate

# Install PyTorch with CUDA
echo ""
echo "[4] Installing PyTorch with CUDA support..."
pip install --upgrade pip
pip install torch --index-url https://download.pytorch.org/whl/cu118 2>/dev/null || \
    pip install torch  # fallback to default

# Install GeoBRIDGE dependencies
echo ""
echo "[5] Installing GeoBRIDGE dependencies..."
pip install -r requirements.txt

# Install Velorama
echo ""
echo "[6] Installing Velorama..."
if [ ! -d "velorama_repo" ]; then
    git clone https://github.com/rs239/velorama.git velorama_repo
fi
cd velorama_repo
export SKLEARN_ALLOW_DEPRECATED_SKLEARN_PACKAGE_INSTALL=True
pip install .
cd "$WORKDIR"

# Verify installations
echo ""
echo "[7] Verifying installations..."
python3 -c "import torch; print(f'PyTorch {torch.__version__}, CUDA available: {torch.cuda.is_available()}')"
python3 -c "import velorama; print('Velorama imported OK')"
python3 -c "import geobridge_model; print('GeoBRIDGE imported OK')"

# Download synthetic datasets
echo ""
echo "[8] Downloading Velorama synthetic datasets..."
python3 -u -m benchmark.bm_velorama_inspect --outdir results/bm_velorama

echo ""
echo "========================================="
echo "Setup complete!"
echo ""
echo "Next steps:"
echo "  source venv_gpu/bin/activate"
echo ""
echo "  # Phase 1: Run GeoBRIDGE (can also run on dragon)"
echo "  nohup python -u -m benchmark.bm_velorama_comparison \\"
echo "    --phase geobridge \\"
echo "    --data-dir results/bm_velorama/velorama_data \\"
echo "    --outdir results/bm_velorama \\"
echo "    > velorama_geobridge_log.txt 2>&1 &"
echo ""
echo "  # Phase 2: Run Velorama (GPU)"
echo "  nohup python -u -m benchmark.bm_velorama_comparison \\"
echo "    --phase velorama \\"
echo "    --data-dir results/bm_velorama/velorama_data \\"
echo "    --outdir results/bm_velorama \\"
echo "    --device cuda \\"
echo "    > velorama_velorama_log.txt 2>&1 &"
echo ""
echo "  # Phase 3: Compare (after both complete)"
echo "  python -u -m benchmark.bm_velorama_comparison \\"
echo "    --phase compare --outdir results/bm_velorama"
echo "========================================="
