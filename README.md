# VSYNTH — Subnautica Cyclops AI Voice Enhancement System

**VSYNTH** is a comprehensive voice synthesis and enhancement toolkit for Subnautica. It allows you to clone the Cyclops AI voice and generate contextual dialogue to create an enhanced, more responsive submarine AI experience through BepInEx modding.

## Overview

VSYNTH combines two major components:

1. **Voice Cloning & Synthesis Pipeline** — Extract, analyze, and synthesize the Cyclops AI voice
2. **BepInEx Mod Framework** — Inject synthesized voices into the game with event-triggered dialogue

The result: A Cyclops AI that responds contextually to game events (hull damage, creature detection, depth warnings, etc.) with dynamically generated speech in the original voice.

## Features

- ✅ **Voice Reference Extraction** — Scan and verify Cyclops OGA audio files
- ✅ **Transcript Management** — Map audio clips to dialogue text
- ✅ **Text-to-Speech Synthesis** — Generate new Cyclops voice lines from custom text
- ✅ **Batch Processing** — Create multiple new lines in one operation
- ✅ **OGA Format Conversion** — Automatic conversion to Subnautica-compatible audio format
- ✅ **BepInEx Integration** — Event-driven audio playback system
- ✅ **Cross-Platform** — Works on Windows (WSL), Linux, and macOS

## Project Structure

```
VSYNTH/
├── cyclops_voice_piper.py      # Main voice synthesis script
├── cyclops_transcripts.json     # Transcript mappings (auto-generated)
├── conversion_report.json       # Audio verification report
│
├── cyclops_raw/                 # Original Subnautica Cyclops OGA files (input)
├── cyclops_wav/                 # Converted WAV files (reference)
├── cyclops_generated/           # Generated WAV files
├── cyclops_oga/                 # Final synthesized OGA files (output for mod)
│
├── mod/                         # BepInEx mod source code
│   ├── CyclopsEnhancer.cs       # C# event hooks and audio system
│   └── StreamingAssets/
│       └── Cyclops_Audio/       # Synthesized OGA files go here
│
└── README.md                    # This file
```

## Installation

### Prerequisites

- **WSL2** (Windows) or Linux/macOS
- **Python 3.12+**
- **ffmpeg** (system package)
- **Subnautica** (Steam version recommended)
- **BepInEx 5.4+** (installed in Subnautica folder)

### Setup

#### 1. Install System Dependencies

**WSL/Ubuntu:**
```bash
sudo apt update
sudo apt install python3-pip python3-venv ffmpeg -y
```

**macOS:**
```bash
brew install python3 ffmpeg
```

#### 2. Create Virtual Environment

```bash
python3 -m venv vsynth_env
source vsynth_env/bin/activate  # On Windows (WSL): source vsynth_env/bin/activate
```

#### 3. Install Python Dependencies

```bash
pip install --upgrade pip
pip install piper-tts librosa soundfile numpy
```

#### 4. Download Piper Voice Model

```bash
piper --download-voice en_US-lessac-medium
```

## Usage

### Phase 1: Setup & Verification

Extract and verify your Cyclops audio files:

```bash
python3 cyclops_voice_piper.py --mode setup
```

This will:
- Scan `cyclops_raw/` for OGA files
- Convert to WAV format (22050 Hz, mono)
- Verify audio integrity
- Create `cyclops_transcripts.json` template

### Phase 2: Transcription

Edit `cyclops_transcripts.json` and fill in the exact dialogue for each Cyclops line. Listen to each OGA file and match the text exactly.

Example:
```json
{
  "CyclopsAbandon": {
    "text": "Abandoning ship. Hull integrity compromised.",
    "verified": true
  },
  "CyclopsCreatureAttack": {
    "text": "Creature attack detected. Initiating evasive maneuvers.",
    "verified": true
  }
}
```

### Phase 3: Generate New Lines

Generate synthesized Cyclops voice lines from custom text:

**Single line:**
```bash
python3 cyclops_voice_piper.py --mode single --text "Leviathan detected on sonar"
```

**Batch from file** (one line per line):
```bash
# Create contextual_lines.txt with your dialogue
python3 cyclops_voice_piper.py --mode batch --file contextual_lines.txt
```

**Full generation** (regenerate all transcripts):
```bash
python3 cyclops_voice_piper.py --mode generate
```

### Phase 4: Integration with BepInEx Mod

Copy generated OGA files to your mod:

```bash
cp cyclops_oga/*.oga mod/StreamingAssets/Cyclops_Audio/
```

Then load the mod in BepInEx. The C# code will:
- Hook into Cyclops game events (hull damage, creature nearby, depth warnings)
- Load synthesized audio from the StreamingAssets folder
- Play appropriate audio based on game state

## Contextual Dialogue Examples

Here are example new lines to enhance the Cyclops AI:

**Creature Warnings:**
- "Leviathan class creature detected on sonar"
- "Unknown biological signature at bearing two seven zero"
- "Creature attack detected. Defensive measures recommended"

**Hull Damage:**
- "Hull breach in compartment three"
- "Structural integrity compromised. Recommend immediate ascent"
- "Multiple hull fractures detected. Emergency repairs required"

**Power & Thermal:**
- "Reactor temperature exceeding safe limits"
- "Power levels critical. Recommend shutting down auxiliary systems"
- "Thermal vent detected ahead. Recommend course correction"

**Navigation:**
- "Depth approaching safety limit. Prepare for ascent"
- "Unknown terrain configuration ahead. Recommend reduced speed"
- "Navigation beacon detected two hundred meters forward"

## Audio Format Specifications

- **Format:** OGA (Ogg Vorbis)
- **Sample Rate:** 22050 Hz
- **Channels:** Mono
- **Bitrate:** Quality 9 (high quality)
- **Duration:** 0.5 - 10 seconds per clip

## BepInEx Mod Integration

The mod listens to Cyclops game events and triggers audio playback. Key events include:

```csharp
OnHullDamage(location, severity)
OnCreatureDetected(type, distance)
OnDepthWarning(currentDepth, safeDepth)
OnThermalWarning(temperature)
OnPowerWarning(percentage)
OnEngineStateChange(state)
```

See `mod/CyclopsEnhancer.cs` for implementation details.

## Troubleshooting

**Issue: `piper --download-voice` fails**

Solution: Download manually from Piper GitHub releases and place in `~/.local/share/piper_tts/voices/`

**Issue: OGA conversion fails**

Solution: Ensure ffmpeg is installed: `which ffmpeg`

**Issue: Audio quality is poor**

Solution: Check original OGA files are clean (no background noise). Re-record if needed.

**Issue: Mod audio doesn't play in-game**

Solution: 
1. Verify BepInEx is properly installed
2. Check `StreamingAssets/Cyclops_Audio/` folder exists
3. Ensure OGA files are in correct format (22050 Hz, mono)
4. Check game console for error messages

## Performance Notes

- **Voice synthesis:** ~2-5 seconds per line on standard hardware
- **Batch processing:** ~50 lines per minute
- **In-game playback:** Negligible performance impact
- **Memory footprint:** ~200 MB for Piper model + audio library

## Advanced Usage

### Custom Voice Models

To use a different Piper voice:

```bash
# List available voices
piper --list-voices

# Download alternative voice
piper --download-voice en_US-amy-medium

# Modify cyclops_voice_piper.py, line 38:
# piper_voice: str = "en_US-amy-medium"
```

### Audio Post-Processing

Enhance synthesized audio with normalization, EQ, or reverb using your favorite audio editor before importing into the mod.

### Extended Dialogue System

Create response trees for multi-part warnings:

```
Event: Hull Breach
├── Initial: "Hull breach in compartment three"
├── Escalation: "Structural failure imminent"
└── Critical: "All personnel abandon ship"
```

## Contributing

This is an active development project. Contributions welcome:

- New contextual dialogue
- Audio quality improvements
- BepInEx integration enhancements
- Documentation & examples

## License

VSYNTH is released under the **MIT License**. Game assets (Subnautica audio) remain property of Unknown Worlds Entertainment.

## Credits

- **Voice Synthesis:** Piper TTS (Mozilla)
- **Audio Processing:** librosa, soundfile
- **Game Modding:** BepInEx community
- **Original Cyclops Voice:** Unknown Worlds Entertainment

## Roadmap

- [ ] Aurora AI voice enhancement (harder due to limited source audio)
- [ ] Real-time voice synthesis for dynamic dialogue
- [ ] Machine learning fine-tuning on reference clips
- [ ] Multi-language support
- [ ] Web UI for easy dialogue generation
- [ ] Community dialogue library

## Support

For issues, questions, or feature requests:
- GitHub Issues: [VSYNTH Issues]
- Documentation: See `/docs/` folder
- Community Discord: [Subnautica Modding]

---

**VSYNTH v1.0** — Bring the Cyclops AI to life.
