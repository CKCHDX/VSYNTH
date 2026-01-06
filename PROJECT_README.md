# 🎙️ VSYNTH - Cyclops AI Voice Synthesis Toolkit

> **Build a smarter, more talkative Cyclops AI for Subnautica using local voice cloning and text-to-speech.**

**Status**: ✅ Project Structure Complete | Setting Up Environment

---

## 📋 What is VSYNTH?

VSYNTH is a **local, GPU-optional voice synthesis pipeline** that allows you to:

1. **Clone the Cyclops Voice** - Extract and analyze the original Cyclops AI voice characteristics
2. **Generate New Dialogue** - Create contextual speech lines from plain text
3. **Control Everything** - Fine-tune synthesis parameters for perfect voice matching
4. **No Cloud Services** - Everything runs locally; no API costs or privacy concerns

### End Result
You can type:
```
"Warning: hull breach detected at section three"
```

And get back a perfectly synthesized WAV file in the **exact Cyclops voice**, ready to inject into the game mod.

---

## 🎯 Project Goals

### Phase 1: ✅ Setup & Voice Analysis (Current)
- [x] WSL2 + Python 3.12 environment
- [x] Project structure with voice reference directories
- [x] Voice characteristic analysis tool
- [ ] Run voice analysis on your 19 Cyclops WAV files
- [ ] Generate voice profile for cloning reference

### Phase 2: TTS Engine Integration
- [ ] F5-TTS integration (zero-shot voice cloning)
- [ ] Tortoise-TTS fallback engine
- [ ] Voice cloning from reference audio
- [ ] Test synthesis with sample texts

### Phase 3: Output Optimization
- [ ] Audio normalization and processing
- [ ] OGA format conversion (Subnautica native)
- [ ] Batch synthesis for multiple lines
- [ ] Audio quality validation

### Phase 4: BepInEx Mod Integration
- [ ] Mod creation (separate repo)
- [ ] Event-driven audio playback system
- [ ] Cyclops AI contextual responses
- [ ] Steam workshop publishing

---

## 🛠️ Current Setup Summary

### Environment
```
✅ Windows 11 + WSL2 (Ubuntu 24.04 LTS)
✅ Python 3.12.x (or 3.10+)
✅ PyTorch + TorchAudio (CPU optimized)
✅ F5-TTS + Tortoise-TTS
✅ Audio processing suite (librosa, soundfile, scipy)
```

### Directory Structure
```
VSYNTH/
├── SETUP.md                    # Detailed installation guide
├── PROJECT_README.md           # This file
├── requirements.txt            # Python dependencies
├── config/
│   └── vsynth_config.yaml      # Configuration file
├── scripts/
│   ├── voice_reference_analyzer.py    # Analyze WAV files
│   ├── voice_synthesis_engine.py      # Main TTS engine
│   ├── audio_processor.py             # (Coming)
│   └── batch_synthesizer.py           # (Coming)
├── voice_reference/
│   ├── cyclops_original/       # Your 19 WAV files here
│   └── voice_profiles/         # Generated profiles
├── generated_tts/
│   ├── raw/                    # Raw synthesized audio
│   ├── normalized/             # Processed audio
│   └── final/                  # Ready-for-mod audio
├── wav_input/
│   └── original_cyclops/       # Backup of originals
├── logs/
│   ├── synthesis.log
│   ├── voice_reference.log
│   └── errors.log
└── tests/                      # Unit tests
```

---

## 🚀 Quick Start

### Step 1: Follow Setup Guide
```bash
# Read and complete SETUP.md
# Install WSL2 + Python 3.12
# Create virtual environment
# Install dependencies
```

### Step 2: Copy Your Cyclops WAV Files
```bash
# Copy all 19 WAV files to:
cp /path/to/cyclops/*.wav ./voice_reference/cyclops_original/

# Verify
ls -lh voice_reference/cyclops_original/
```

### Step 3: Analyze Voice Reference
```bash
# Activate virtual environment
source venv/bin/activate

# Run voice analyzer
python scripts/voice_reference_analyzer.py
```

This will:
- Scan all 19 WAV files
- Extract voice characteristics (pitch, timbre, loudness, etc.)
- Generate `cyclops_profile.json` for synthesis reference
- Print analysis report

### Step 4: Test Voice Synthesis (Coming)
```bash
# After Phase 2 integration
python scripts/voice_synthesis_engine.py
```

---

## 📊 Voice Analysis Output

After running the analyzer, you'll get:

### `voice_profiles/cyclops_characteristics.json`
Detailed characteristics for each WAV file:
```json
{
  "Cyclops_Welcome.wav": {
    "duration": 5.23,
    "sample_rate": 24000,
    "loudness_db": -18.5,
    "spectral_centroid_hz": 3420.5,
    "pitch_hz": {
      "mean": 155.3,
      "range": [120, 210]
    },
    "mfcc_mean": [-250.5, 45.3, 12.1, ...]
  }
}
```

### `voice_profiles/cyclops_profile.json`
Aggregate voice profile:
```json
{
  "cyclops_voice_profile": {
    "files_analyzed": 19,
    "total_duration_seconds": 87.45,
    "loudness_db": {
      "mean": -19.2,
      "range": [-22.5, -15.1]
    },
    "voice_characteristics": {
      "timbre": "AI robotic with clear articulation",
      "prosody": "Steady, informative",
      "emotion": "Neutral, authoritative",
      "quality": "Clean, well-recorded"
    }
  }
}
```

This profile is **used by F5-TTS and Tortoise-TTS** to understand how to clone your voice.

---

## 🎵 TTS Engine Comparison

### F5-TTS (Primary)
- ✅ Zero-shot voice cloning (minimal reference needed)
- ✅ State-of-the-art quality (2024 model)
- ✅ Fast inference (real-time capable)
- ✅ Fine control over prosody and style
- ⚠️ Requires reference audio in same language

### Tortoise-TTS (Fallback)
- ✅ Mature, well-documented implementation
- ✅ Excellent voice consistency
- ✅ Multiple voice presets (fast/normal/high_quality)
- ⚠️ Slower inference
- ⚠️ More VRAM intensive

**Strategy**: Use F5-TTS first → fallback to Tortoise if needed

---

## 🎯 Voice Cloning Process

### How it Works

1. **Reference Analysis**
   ```
   Your 19 Cyclops WAV files
           ↓
   Extract voice characteristics
   (pitch, timbre, loudness, articulation)
           ↓
   Generate "voice profile"
   ```

2. **Text-to-Speech Synthesis**
   ```
   Your text ("Warning: hull damage")
           ↓
   F5-TTS / Tortoise model
           ↓
   Synthesize speech conditioned on
   Cyclops voice characteristics
           ↓
   Output: WAV in Cyclops voice
   ```

3. **Voice Consistency**
   - All generated audio will sound like the Cyclops
   - Same pitch range, accent, articulation
   - No robotic artifacts (unlike generic TTS)

---

## 📝 Example Use Cases

### Basic Synthesis
```python
from scripts.voice_synthesis_engine import VoiceSynthesisEngine
from pathlib import Path

engine = VoiceSynthesisEngine()
audio, engine_used = engine.synthesize(
    text="Hull integrity at seventy percent",
    voice_reference_path=Path("voice_reference/cyclops_original/CyclopsHullLow.wav"),
    output_path=Path("generated_tts/test.wav")
)
print(f"Synthesized with: {engine_used}")
```

### Batch Synthesis
```python
texts = [
    "Warning: hull breach detected",
    "Reactor meltdown imminent",
    "Cyclops systems failing"
]

results = engine.batch_synthesize(
    text_list=texts,
    voice_reference_path=Path("voice_reference/cyclops_original/Cyclops_Welcome.wav"),
    output_dir=Path("generated_tts/batch_output")
)

print(f"Successfully synthesized: {results['successful']}/{results['total']}")
```

---

## 🎛️ Synthesis Parameters You Can Control

### F5-TTS
- **speed**: 0.5 - 2.0 (normal = 1.0)
- **temperature**: 0.0 - 1.0 (lower = more consistent, higher = more varied)

### Tortoise-TTS
- **preset**: "fast", "normal", "high_quality"
- **temperature**: 0.0 - 1.0
- **num_autoregressive_samples**: Number of inference passes (higher = better quality, slower)

### Audio Output
- **sample_rate**: 24000 Hz (Subnautica standard)
- **bit_depth**: 16-bit
- **normalization**: Automatic loudness matching

---

## 🐛 Troubleshooting

### "No WAV files found"
```bash
# Copy your Cyclops WAV files:
cp ~/Downloads/cyclops/*.wav voice_reference/cyclops_original/

# Verify:
ls voice_reference/cyclops_original/
```

### "F5-TTS import error"
```bash
# Reinstall F5-TTS
pip install --force-reinstall f5-tts

# Or check torch compatibility
python -c "import torch; print(torch.__version__)"
```

### "Out of memory" (WSL2)
```bash
# Edit .wslconfig in Windows (C:\Users\YourName\.wslconfig):
[wsl2]
memory=8GB
processors=4
swap=2GB

# Restart WSL2:
wsl --shutdown
ubuntu2404  # Restart
```

### Slow synthesis
- GPU acceleration not available in WSL2 - expected on CPU
- F5-TTS is faster than Tortoise
- Normal inference: 2-5 seconds per sentence

---

## 📚 Documentation

- **[SETUP.md](./SETUP.md)** - Detailed installation guide
- **[Voice Analysis](./scripts/voice_reference_analyzer.py)** - Understand your voice characteristics
- **[Synthesis Engine](./scripts/voice_synthesis_engine.py)** - Main TTS logic
- **[Config](./config/vsynth_config.yaml)** - Adjustable parameters

---

## 🔄 Project Status

### ✅ Complete
- [x] Project structure and organization
- [x] WSL2 + Python setup guide
- [x] Voice reference analyzer
- [x] Synthesis engine framework
- [x] Configuration system
- [x] Logging infrastructure

### 🔄 In Progress
- [ ] Integrate F5-TTS cloning API
- [ ] Integrate Tortoise-TTS properly
- [ ] Test with actual Cyclops WAV files
- [ ] Audio normalization pipeline

### 📋 Upcoming
- [ ] OGA format conversion
- [ ] Batch processing optimization
- [ ] Voice control GUI (optional)
- [ ] Quality assurance metrics
- [ ] BepInEx mod integration

---

## 🤝 Contributing

This is your personal project for learning and experimentation.

**Areas to experiment with:**
- Different TTS engines (Piper, Kokoro, etc.)
- Voice fine-tuning strategies
- Audio processing pipelines
- Batch optimization
- Custom voice profiles

---

## 📞 Quick Links

- **Subnautica**: https://store.steampowered.com/app/264710/
- **BepInEx**: https://github.com/BepInEx/BepInEx
- **F5-TTS**: https://github.com/SWivid/F5-TTS
- **Tortoise-TTS**: https://github.com/neonbjb/tortoise-tts

---

## 🎮 Game Modding Context

VSYNTH feeds into a BepInEx mod that:
1. Hooks Cyclops AI events (damage, depth, creatures, etc.)
2. Plays synthesized audio lines from VSYNTH
3. Creates responsive, contextual submarine AI

**The mod isn't built yet** - focus on getting voice synthesis working first.

---

## 📝 Notes

- Cyclops voice is **authoritative, steady, informative**
- ~87 seconds of total reference audio (19 files)
- 24000 Hz sample rate (Subnautica standard)
- Clear, well-articulated speech (good for TTS cloning)

---

## 🎯 Next Steps

1. ✅ **Setup environment** (SETUP.md)
2. ✅ **Copy WAV files** to voice_reference/cyclops_original/
3. ⏭️ **Run voice analyzer** → Get voice profile
4. ⏭️ **Test synthesis** → Generate test phrases
5. ⏭️ **Optimize output** → Ready for mod integration

---

**Created by**: Alex Jonsson (CKCHDX)  
**Last Updated**: January 6, 2026  
**Status**: Active Development  

🚀 **Ready to build the ultimate Cyclops AI!**
