# VSYNTH Setup Guide (Windows WSL2)

## Phase 1: WSL2 Installation & Configuration

### Step 1.1: Install WSL2 on Windows 11
```powershell
# Run PowerShell as Administrator
wsl --install

# Enable WSL2 as default
wsl --set-default-version 2

# Restart your computer
```

### Step 1.2: Install Ubuntu 24.04 LTS in WSL2
```powershell
# In PowerShell
wsl --install -d Ubuntu-24.04

# Launch Ubuntu and complete initial setup
ubuntu2404
```

### Step 1.3: Initial Ubuntu Setup
```bash
# Update package lists
sudo apt update && sudo apt upgrade -y

# Install system dependencies
sudo apt install -y \
    build-essential \
    libssl-dev \
    libffi-dev \
    python3-dev \
    python3-pip \
    python3-venv \
    git \
    curl \
    wget \
    ffmpeg \
    libsndfile1 \
    sox \
    espeak-ng

# Verify ffmpeg installation
ffmpeg -version
```

---

## Phase 2: Python Environment Setup

### Step 2.1: Python 3.12 Installation
```bash
# Check default Python version
python3 --version  # Should be 3.10+, ideally 3.12

# If you need Python 3.12
sudo apt install -y python3.12 python3.12-venv python3.12-dev

# Set as default (optional)
sudo update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.12 100
```

### Step 2.2: Create VSYNTH Project Directory
```bash
# Navigate to your projects
cd ~/projects  # or create it
mkdir -p VSYNTH && cd VSYNTH

# Create Python virtual environment (Python 3.12)
python3.12 -m venv venv

# Activate virtual environment
source venv/bin/activate

# Verify Python version
python --version  # Should show 3.12.x
```

### Step 2.3: Upgrade pip and Install Core Dependencies
```bash
# Upgrade pip, setuptools, wheel
pip install --upgrade pip setuptools wheel

# Install PyTorch with CPU support (for WSL2 - no GPU acceleration needed initially)
# For GPU support: replace cpu with cu118 or cu121 based on your NVIDIA driver
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu

# Verify PyTorch
python -c "import torch; print(torch.__version__)"
```

---

## Phase 3: VSYNTH Dependencies Installation

### Step 3.1: Core TTS & Voice Cloning Packages
```bash
# Ensure venv is activated
source ~/projects/VSYNTH/venv/bin/activate

# Install F5-TTS (primary: state-of-the-art zero-shot voice cloning)
pip install f5-tts

# Install Tortoise-TTS (fallback: mature voice cloning)
pip install TTS

# Install additional audio processing libraries
pip install librosa soundfile scipy numpy matplotlib pydub
```

### Step 3.2: Audio Conversion & Processing
```bash
# For OGA/WAV conversion and audio analysis
pip install pydub==0.25.1
pip install scipy==1.13.1
pip install librosa==0.10.2
```

### Step 3.3: Utility & Development Tools
```bash
# Configuration management
pip install pyyaml

# Data processing
pip install pandas

# Progress bars and logging
pip install tqdm

# Testing and validation
pip install pytest

# Install all at once
pip install pyyaml pandas tqdm pytest
```

---

## Phase 4: Project Structure Creation

### Step 4.1: Create Directory Layout
```bash
# From VSYNTH project root
mkdir -p {voice_reference,wav_input,wav_output,generated_tts,config,scripts,tests,logs}

# Create subdirectories for voice samples
mkdir -p voice_reference/cyclops_original
mkdir -p voice_reference/voice_profiles
mkdir -p generated_tts/{raw,normalized,final}

# Create data directories
mkdir -p wav_input/original_cyclops
mkdir -p wav_output/processed
mkdir -p logs/{training,synthesis,errors}
```

### Step 4.2: Verify Structure
```bash
# List the created structure
tree -L 2 .  # If tree is not installed: sudo apt install tree

# Or use ls
ls -la
```

---

## Phase 5: Configuration Files

### Step 5.1: Create config/vsynth_config.yaml
```yaml
# VSYNTH Configuration File

project:
  name: "VSYNTH - Cyclops Voice Synthesis"
  version: "1.0.0"
  description: "Local voice cloning and TTS for Subnautica Cyclops AI"

paths:
  voice_reference: "./voice_reference"
  wav_input: "./wav_input"
  generated_tts: "./generated_tts"
  config: "./config"
  logs: "./logs"

audio:
  sample_rate: 24000  # Subnautica standard
  bit_depth: 16
  channels: 1
  audio_format: "wav"

tts:
  primary_engine: "f5-tts"  # f5-tts, tortoise
  backup_engine: "tortoise"
  
  f5_tts:
    model: "F5-TTS"
    device: "cpu"  # 'cuda' if GPU available
    
  tortoise:
    model: "tts_models/en/multi-dataset/tortoise-v2"
    preset: "fast"  # fast, normal, high_quality
    device: "cpu"

voice_cloning:
  reference_audio_min_duration: 3  # seconds
  reference_audio_max_duration: 30
  voice_similarity_threshold: 0.75

synthesis:
  batch_size: 4
  max_text_length: 500
  speed: 1.0  # 0.5 - 2.0
  
logging:
  level: "INFO"  # DEBUG, INFO, WARNING, ERROR
  format: "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
```

---

## Phase 6: Verification & Testing

### Step 6.1: Verify Installation
```bash
# Activate venv
source venv/bin/activate

# Test PyTorch
python -c "
import torch
import torchaudio
import librosa
print('✓ PyTorch:', torch.__version__)
print('✓ TorchAudio:', torchaudio.__version__)
print('✓ Librosa installed')
"

# Test TTS libraries
python -c "
try:
    from TTS.api import TTS
    print('✓ Tortoise TTS available')
except:
    print('⚠ Tortoise TTS import check')

try:
    from f5_tts.api import F5TTS
    print('✓ F5-TTS available')
except:
    print('⚠ F5-TTS import check')
"
```

### Step 6.2: Quick Audio Test
```bash
# Create simple test script
cat > test_audio.py << 'EOF'
#!/usr/bin/env python3
import librosa
import soundfile as sf
import numpy as np

print("Testing audio capabilities...")

# Generate test sine wave
duration = 2  # seconds
sr = 24000
freq = 440  # Hz (A note)

t = np.linspace(0, duration, int(sr * duration))
y = 0.3 * np.sin(2 * np.pi * freq * t)

# Save test audio
test_file = "./test_audio.wav"
sf.write(test_file, y, sr)
print(f"✓ Generated test audio: {test_file}")

# Load and verify
audio, sr_loaded = librosa.load(test_file, sr=sr)
print(f"✓ Loaded audio: {len(audio)} samples at {sr_loaded} Hz")
print(f"✓ Duration: {len(audio) / sr_loaded:.2f} seconds")
print("\n✓ All audio tests passed!")
EOF

python test_audio.py
```

---

## Phase 7: Cyclops WAV File Integration

### Step 7.1: Copy Your WAV Files
```bash
# Copy all Cyclops WAV files to voice_reference/cyclops_original/
cp /path/to/your/cyclops/wavs/* voice_reference/cyclops_original/

# Verify files
ls -lh voice_reference/cyclops_original/
echo "Total WAV files: $(ls voice_reference/cyclops_original/*.wav | wc -l)"
```

### Step 7.2: Analyze Voice Reference
```bash
# Create analysis script
cat > scripts/analyze_voice_reference.py << 'EOF'
#!/usr/bin/env python3
import os
import librosa
import soundfile as sf
from pathlib import Path

voice_dir = Path("voice_reference/cyclops_original")
wav_files = sorted(voice_dir.glob("*.wav"))

print(f"\n{'Filename':<30} {'Duration':<12} {'Sample Rate':<12} {'Channels':<10}")
print("=" * 64)

total_duration = 0
for wav_file in wav_files:
    audio, sr = librosa.load(str(wav_file), sr=None)
    duration = len(audio) / sr
    info = sf.info(str(wav_file))
    
    print(f"{wav_file.name:<30} {duration:>6.2f}s{'':<5} {sr:>6} Hz{'':<5} {info.channels:>2}")
    total_duration += duration

print("=" * 64)
print(f"Total audio duration: {total_duration:.2f} seconds ({total_duration/60:.2f} minutes)")
print(f"Total files: {len(wav_files)}")

# Save reference profile
import json
ref_profile = {
    "files_count": len(wav_files),
    "total_duration": total_duration,
    "average_duration": total_duration / len(wav_files) if wav_files else 0,
    "sample_rate": sr,
    "voice_type": "Cyclops AI"
}

with open("voice_reference/voice_profiles/cyclops_profile.json", "w") as f:
    json.dump(ref_profile, f, indent=2)

print(f"\n✓ Voice profile saved: voice_reference/voice_profiles/cyclops_profile.json")
EOF

python scripts/analyze_voice_reference.py
```

---

## Phase 8: Quick Start - Your First Synthesis

### Step 8.1: Create Basic Synthesis Script
```bash
cat > scripts/simple_synthesis.py << 'EOF'
#!/usr/bin/env python3
"""
Simple voice synthesis test using F5-TTS
"""
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import librosa
import soundfile as sf
import numpy as np

print("VSYNTH - Simple Synthesis Test")
print("=" * 50)

# For now, just verify setup
voice_ref = Path("voice_reference/cyclops_original")
wav_files = list(voice_ref.glob("*.wav"))

if wav_files:
    print(f"\n✓ Found {len(wav_files)} reference audio files")
    print(f"✓ Primary reference: {wav_files[0].name}")
    
    # Load first file for verification
    audio, sr = librosa.load(str(wav_files[0]), sr=None)
    print(f"✓ Sample rate: {sr} Hz")
    print(f"✓ Duration: {len(audio)/sr:.2f} seconds")
    print("\n✓ System ready for voice cloning!")
else:
    print("\n⚠ No WAV files found in voice_reference/cyclops_original/")
    print("Please copy your Cyclops WAV files there first.")
EOF

python scripts/simple_synthesis.py
```

---

## Troubleshooting

### WSL2 Audio/FFmpeg Issues
```bash
# If ffmpeg not found
sudo apt install -y ffmpeg

# Verify
ffmpeg -version
```

### Python Virtual Environment Issues
```bash
# If venv doesn't work
python3.12 -m venv --clear venv
source venv/bin/activate
```

### PyTorch Installation Issues
```bash
# For CPU (recommended for WSL2)
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu

# Verify installation
python -c "import torch; print(torch.version.cuda)"  # Should print None for CPU
```

### Memory/Performance
```bash
# WSL2 uses shared memory with Windows
# If experiencing slow performance, modify .wslconfig in Windows:
# [wsl2]
# memory=8GB
# processors=4
# swap=2GB
```

---

## Next Steps

1. ✅ **Setup complete**: You now have a working VSYNTH environment
2. 🎵 **Copy Cyclops WAV files** to `voice_reference/cyclops_original/`
3. 📝 **Run voice analysis** to understand your reference audio
4. 🔊 **Test text-to-speech** with sample Cyclops text
5. 🎛️ **Fine-tune voice parameters** for consistency

---

## Summary

- **WSL2**: Ubuntu 24.04 LTS running on Windows 11
- **Python**: 3.12.x in isolated virtual environment
- **Core Packages**:
  - `torch` + `torchaudio` (audio processing)
  - `f5-tts` (zero-shot voice cloning - primary)
  - `TTS` (Tortoise - fallback)
  - `librosa` (audio analysis)
  - `soundfile` (audio I/O)
  - `pydub` (format conversion)

Your environment is now ready for local voice cloning and synthesis! 🚀
