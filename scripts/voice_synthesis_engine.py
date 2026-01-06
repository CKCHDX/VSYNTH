#!/usr/bin/env python3
"""
Voice Synthesis Engine - Generate speech from text using voice cloning
Supports F5-TTS (primary) and Tortoise-TTS (fallback)
"""

import json
import logging
from pathlib import Path
from typing import Optional, Dict, List, Tuple
from dataclasses import dataclass
import numpy as np

import torch
import librosa
import soundfile as sf
from scipy.io.wavfile import read as wav_read

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('logs/synthesis.log')
    ]
)
logger = logging.getLogger(__name__)


@dataclass
class SynthesisConfig:
    """Configuration for TTS synthesis"""
    primary_engine: str = "f5-tts"  # f5-tts, tortoise
    backup_engine: str = "tortoise"
    device: str = "cpu"
    sample_rate: int = 24000
    
    # F5-TTS settings
    f5_speed: float = 1.0
    f5_temperature: float = 0.7
    
    # Tortoise settings
    tortoise_preset: str = "fast"  # fast, normal, high_quality
    tortoise_num_autoregressive_samples: int = 16
    tortoise_temperature: float = 0.75
    
    def __post_init__(self):
        """Validate configuration"""
        if self.primary_engine not in ["f5-tts", "tortoise"]:
            raise ValueError(f"Unknown primary engine: {self.primary_engine}")
        if self.tortoise_preset not in ["fast", "normal", "high_quality"]:
            raise ValueError(f"Unknown Tortoise preset: {self.tortoise_preset}")


class F5TTSEngine:
    """F5-TTS Voice Cloning Engine (Primary)"""
    
    def __init__(self, device: str = "cpu"):
        self.device = device
        self.model = None
        self.loaded = False
        self._init_model()
    
    def _init_model(self):
        """Initialize F5-TTS model"""
        try:
            logger.info("Initializing F5-TTS model...")
            # Import F5-TTS
            from f5_tts.api import F5TTS
            
            self.model = F5TTS()
            self.loaded = True
            logger.info("✓ F5-TTS model loaded successfully")
        except Exception as e:
            logger.error(f"✗ Failed to load F5-TTS: {e}")
            self.loaded = False
    
    def clone_voice(self, text: str, voice_reference_path: Path,
                   speed: float = 1.0, temperature: float = 0.7) -> Optional[np.ndarray]:
        """
        Synthesize speech using reference voice
        
        Args:
            text: Text to synthesize
            voice_reference_path: Path to reference WAV file
            speed: Synthesis speed (0.5-2.0)
            temperature: Synthesis temperature (0.0-1.0) - controls randomness
            
        Returns:
            Audio waveform as numpy array or None on failure
        """
        if not self.loaded:
            logger.warning("F5-TTS model not loaded")
            return None
        
        try:
            logger.info(f"Synthesizing with F5-TTS: '{text[:50]}...'")
            
            # Load reference audio
            if not Path(voice_reference_path).exists():
                logger.error(f"Reference file not found: {voice_reference_path}")
                return None
            
            # Use F5-TTS API
            # Note: Actual API may differ based on F5-TTS version
            # This is a placeholder for the actual implementation
            audio = self.model.synthesize(
                text=text,
                reference_audio_path=str(voice_reference_path),
                speed=speed,
                temperature=temperature
            )
            
            logger.info("✓ F5-TTS synthesis completed")
            return audio
            
        except Exception as e:
            logger.error(f"✗ F5-TTS synthesis failed: {e}")
            return None


class TortoiseTTSEngine:
    """Tortoise-TTS Voice Cloning Engine (Fallback)"""
    
    def __init__(self, device: str = "cpu"):
        self.device = device
        self.model = None
        self.loaded = False
        self._init_model()
    
    def _init_model(self):
        """Initialize Tortoise-TTS model"""
        try:
            logger.info("Initializing Tortoise-TTS model...")
            from TTS.api import TTS
            
            # Initialize Tortoise model
            self.model = TTS(
                model_name="tts_models/en/multi-dataset/tortoise-v2",
                progress_bar=True,
                gpu=(self.device == "cuda")
            )
            self.loaded = True
            logger.info("✓ Tortoise-TTS model loaded successfully")
        except Exception as e:
            logger.error(f"✗ Failed to load Tortoise-TTS: {e}")
            self.loaded = False
    
    def clone_voice(self, text: str, voice_reference_path: Path,
                   preset: str = "fast", temperature: float = 0.75) -> Optional[np.ndarray]:
        """
        Synthesize speech using reference voice
        
        Args:
            text: Text to synthesize
            voice_reference_path: Path to reference WAV file
            preset: Quality preset (fast, normal, high_quality)
            temperature: Synthesis temperature (0.0-1.0)
            
        Returns:
            Audio waveform as numpy array or None on failure
        """
        if not self.loaded:
            logger.warning("Tortoise-TTS model not loaded")
            return None
        
        try:
            logger.info(f"Synthesizing with Tortoise-TTS: '{text[:50]}...' (preset: {preset})")
            
            # Load reference audio
            ref_path = Path(voice_reference_path)
            if not ref_path.exists():
                logger.error(f"Reference file not found: {voice_reference_path}")
                return None
            
            # Use Tortoise-TTS
            audio = self.model.tts(
                text=text,
                voice_dir=str(ref_path.parent),
                speaker_idx=ref_path.stem,
                preset=preset,
                temperature=temperature
            )
            
            logger.info("✓ Tortoise-TTS synthesis completed")
            return audio
            
        except Exception as e:
            logger.error(f"✗ Tortoise-TTS synthesis failed: {e}")
            return None


class VoiceSynthesisEngine:
    """Main voice synthesis engine with fallback support"""
    
    def __init__(self, config: Optional[SynthesisConfig] = None):
        self.config = config or SynthesisConfig()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.config.device = self.device
        
        logger.info(f"Using device: {self.device}")
        logger.info(f"Primary engine: {self.config.primary_engine}")
        logger.info(f"Backup engine: {self.config.backup_engine}")
        
        # Initialize engines
        self.f5_engine = F5TTSEngine(device=self.device) if self.config.primary_engine == "f5-tts" else None
        self.tortoise_engine = TortoiseTTSEngine(device=self.device)
    
    def synthesize(self, text: str, voice_reference_path: Path,
                  output_path: Optional[Path] = None,
                  use_primary: bool = True) -> Tuple[Optional[np.ndarray], str]:
        """
        Synthesize speech from text
        
        Args:
            text: Text to synthesize
            voice_reference_path: Path to reference WAV file
            output_path: Optional path to save output WAV
            use_primary: Try primary engine first before fallback
            
        Returns:
            Tuple of (audio_array, engine_used) or (None, error_message)
        """
        audio = None
        engine_used = None
        
        # Try primary engine first
        if use_primary and self.f5_engine and self.f5_engine.loaded:
            logger.info("Attempting synthesis with primary engine (F5-TTS)...")
            audio = self.f5_engine.clone_voice(
                text=text,
                voice_reference_path=voice_reference_path,
                speed=self.config.f5_speed,
                temperature=self.config.f5_temperature
            )
            if audio is not None:
                engine_used = "F5-TTS"
        
        # Fallback to Tortoise if primary failed
        if audio is None and self.tortoise_engine.loaded:
            logger.info("Falling back to Tortoise-TTS...")
            audio = self.tortoise_engine.clone_voice(
                text=text,
                voice_reference_path=voice_reference_path,
                preset=self.config.tortoise_preset,
                temperature=self.config.tortoise_temperature
            )
            if audio is not None:
                engine_used = "Tortoise-TTS"
        
        # Save output if requested
        if audio is not None and output_path:
            try:
                output_path = Path(output_path)
                output_path.parent.mkdir(parents=True, exist_ok=True)
                
                # Ensure proper shape
                if isinstance(audio, torch.Tensor):
                    audio = audio.cpu().numpy()
                
                # Save as WAV
                sf.write(str(output_path), audio, self.config.sample_rate)
                logger.info(f"✓ Saved output: {output_path}")
            except Exception as e:
                logger.error(f"✗ Failed to save output: {e}")
        
        if audio is None:
            return None, f"Failed to synthesize with any available engine"
        
        return audio, engine_used
    
    def batch_synthesize(self, text_list: List[str], voice_reference_path: Path,
                        output_dir: Path) -> Dict[str, Dict]:
        """
        Synthesize multiple texts in batch
        
        Args:
            text_list: List of texts to synthesize
            voice_reference_path: Path to reference WAV file
            output_dir: Directory to save outputs
            
        Returns:
            Dictionary with synthesis results
        """
        results = {
            "total": len(text_list),
            "successful": 0,
            "failed": 0,
            "files": []
        }
        
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        for i, text in enumerate(text_list, 1):
            output_file = output_dir / f"synthesis_{i:03d}.wav"
            
            logger.info(f"Batch [{i}/{len(text_list)}]: {text[:60]}...")
            
            audio, engine_used = self.synthesize(
                text=text,
                voice_reference_path=voice_reference_path,
                output_path=output_file
            )
            
            if audio is not None:
                results["successful"] += 1
                results["files"].append({
                    "index": i,
                    "text": text,
                    "output": str(output_file),
                    "engine": engine_used,
                    "duration": len(audio) / self.config.sample_rate
                })
            else:
                results["failed"] += 1
                results["files"].append({
                    "index": i,
                    "text": text,
                    "error": engine_used,
                    "status": "failed"
                })
        
        return results
    
    def normalize_audio(self, audio: np.ndarray, target_db: float = -20.0) -> np.ndarray:
        """
        Normalize audio to target loudness
        
        Args:
            audio: Audio waveform
            target_db: Target loudness in dB
            
        Returns:
            Normalized audio
        """
        # Calculate current loudness
        current_db = 20 * np.log10(np.sqrt(np.mean(audio ** 2)) + 1e-10)
        
        # Calculate gain needed
        gain = 10 ** ((target_db - current_db) / 20)
        
        # Apply gain with soft clipping
        normalized = audio * gain
        normalized = np.tanh(normalized)  # Soft clipping
        
        return normalized


def main():
    """Example usage"""
    import sys
    
    # Create logs directory
    Path("logs").mkdir(exist_ok=True)
    
    # Initialize engine
    config = SynthesisConfig(
        primary_engine="f5-tts",
        backup_engine="tortoise",
        device="cpu"
    )
    
    engine = VoiceSynthesisEngine(config=config)
    
    # Find reference voice
    voice_dir = Path("voice_reference/cyclops_original")
    wav_files = list(voice_dir.glob("*.wav"))
    
    if not wav_files:
        print("Error: No WAV files found in voice_reference/cyclops_original/")
        print("Please copy your Cyclops WAV files there first.")
        sys.exit(1)
    
    reference_voice = wav_files[0]
    print(f"\nUsing reference voice: {reference_voice.name}")
    
    # Test synthesis
    test_texts = [
        "Warning: hull integrity critical",
        "Submerged at depth of five hundred meters",
        "Engine offline"
    ]
    
    output_dir = Path("generated_tts/test")
    
    print(f"\nSynthesizing {len(test_texts)} test phrases...")
    results = engine.batch_synthesize(test_texts, reference_voice, output_dir)
    
    print(f"\n✓ Synthesis complete:")
    print(f"  Successful: {results['successful']}/{results['total']}")
    print(f"  Failed: {results['failed']}/{results['total']}")
    print(f"  Output directory: {output_dir}")


if __name__ == "__main__":
    main()
