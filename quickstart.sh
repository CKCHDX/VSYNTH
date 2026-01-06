#!/bin/bash

# VSYNTH Quick Start Script
# Automates environment setup and initial tests

set -e

echo ""
echo "================================="
echo "🎙️  VSYNTH Quick Start Script"
echo "================================="
echo ""

# Color codes
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Check if Python 3.12 is available
echo -e "${YELLOW}[1/6]${NC} Checking Python version..."
if ! command -v python3.12 &> /dev/null; then
    echo -e "${RED}⚠️  Python 3.12 not found${NC}"
    echo "Please install Python 3.12 first."
    echo "Falling back to python3..."
    PYTHON_CMD="python3"
else
    PYTHON_CMD="python3.12"
fi

$PYTHON_CMD --version
echo -e "${GREEN}✓ Python OK${NC}"

# Create virtual environment if it doesn't exist
echo ""
echo -e "${YELLOW}[2/6]${NC} Setting up virtual environment..."
if [ ! -d "venv" ]; then
    $PYTHON_CMD -m venv venv
    echo -e "${GREEN}✓ Created virtual environment${NC}"
else
    echo -e "${GREEN}✓ Virtual environment already exists${NC}"
fi

# Activate virtual environment
echo ""
echo -e "${YELLOW}[3/6]${NC} Activating virtual environment..."
source venv/bin/activate
echo -e "${GREEN}✓ Virtual environment activated${NC}"

# Upgrade pip and install requirements
echo ""
echo -e "${YELLOW}[4/6]${NC} Installing dependencies..."
pip install --upgrade pip setuptools wheel > /dev/null 2>&1

# Check if requirements.txt exists
if [ -f "requirements.txt" ]; then
    pip install -r requirements.txt
    echo -e "${GREEN}✓ Dependencies installed${NC}"
else
    echo -e "${RED}⚠️  requirements.txt not found${NC}"
fi

# Create required directories
echo ""
echo -e "${YELLOW}[5/6]${NC} Creating project directories..."
mkdir -p voice_reference/cyclops_original
mkdir -p voice_reference/voice_profiles
mkdir -p generated_tts/{raw,normalized,final}
mkdir -p wav_input/original_cyclops
mkdir -p logs/{training,synthesis,errors}
mkdir -p tests
echo -e "${GREEN}✓ Directories created${NC}"

# Run diagnostics
echo ""
echo -e "${YELLOW}[6/6]${NC} Running system diagnostics..."
echo ""

$PYTHON_CMD << 'EOF'
import sys
from pathlib import Path

print("System Diagnostics:")
print("-" * 40)

# Check Python
print(f"Python: {sys.version.split()[0]}")

# Check libraries
try:
    import torch
    print(f"PyTorch: {torch.__version__}")
    print(f"  Device: {'CUDA' if torch.cuda.is_available() else 'CPU'}")
except ImportError:
    print("PyTorch: NOT INSTALLED")

try:
    import torchaudio
    print(f"TorchAudio: {torchaudio.__version__}")
except ImportError:
    print("TorchAudio: NOT INSTALLED")

try:
    import librosa
    print("Librosa: OK")
except ImportError:
    print("Librosa: NOT INSTALLED")

try:
    import soundfile
    print("SoundFile: OK")
except ImportError:
    print("SoundFile: NOT INSTALLED")

try:
    from TTS.api import TTS
    print("Tortoise-TTS: OK")
except ImportError:
    print("Tortoise-TTS: NOT INSTALLED (optional fallback)")

try:
    from f5_tts.api import F5TTS
    print("F5-TTS: OK")
except ImportError:
    print("F5-TTS: NOT INSTALLED (primary engine)")

print("-" * 40)

# Check directories
print("\nProject Structure:")
print("-" * 40)

required_dirs = [
    "voice_reference/cyclops_original",
    "voice_reference/voice_profiles",
    "generated_tts",
    "scripts",
    "config",
    "logs"
]

for dir_path in required_dirs:
    exists = "✓" if Path(dir_path).exists() else "✗"
    print(f"{exists} {dir_path}")

print("-" * 40)
EOF

echo ""
echo "================================="
echo -e "${GREEN}✓ Quick Start Complete!${NC}"
echo "================================="
echo ""
echo "Next steps:"
echo ""
echo "1. Copy Cyclops WAV files:"
echo "   cp /path/to/cyclops/*.wav voice_reference/cyclops_original/"
echo ""
echo "2. Analyze voice reference:"
echo "   python scripts/voice_reference_analyzer.py"
echo ""
echo "3. Check the logs:"
echo "   tail logs/*.log"
echo ""
echo "4. Read the documentation:"
echo "   - SETUP.md (detailed installation)"
echo "   - PROJECT_README.md (project overview)"
echo ""
echo "Virtual environment ready at: $(pwd)/venv"
echo "Activate it with: source venv/bin/activate"
echo ""
