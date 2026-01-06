#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Voice Reference Analyzer - Analyze and profile Cyclops WAV files
Extracts voice characteristics for cloning reference
"""

import json
import logging
import sys
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional

import numpy as np
import librosa
import soundfile as sf
from scipy.signal import spectrogram
from scipy.fftpack import fft

# Fix Windows console encoding
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('logs/voice_reference.log', encoding='utf-8')
    ]
)
logger = logging.getLogger(__name__)


@dataclass
class VoiceCharacteristics:
    """Voice profile characteristics"""
    file_path: str
    duration: float
    sample_rate: int
    channels: int
    rms_energy: float  # Loudness
    spectral_centroid: float  # Brightness
    zero_crossing_rate: float  # Articulation
    mfcc_mean: List[float]  # Mel-frequency cepstral coefficients
    pitch_mean: Optional[float]
    pitch_range: Optional[tuple]
    loudness_db: float
    

class VoiceReferenceAnalyzer:
    """Analyze WAV files to extract voice characteristics"""
    
    def __init__(self, voice_dir: Path = Path("voice_reference/cyclops_original"),
                 sample_rate: int = 24000):
        self.voice_dir = Path(voice_dir)
        self.target_sr = sample_rate
        self.characteristics: Dict[str, VoiceCharacteristics] = {}
        
        logger.info(f"Initialized analyzer for: {voice_dir}")
    
    def analyze_file(self, wav_path: Path) -> Optional[VoiceCharacteristics]:
        """Analyze single WAV file"""
        try:
            # Load audio
            audio, sr = librosa.load(str(wav_path), sr=None, mono=True)
            
            # Get file info
            info = sf.info(str(wav_path))
            duration = len(audio) / sr
            
            logger.info(f"Analyzing: {wav_path.name} ({duration:.2f}s @ {sr}Hz)")
            
            # Extract characteristics
            # RMS Energy (loudness) - direct calculation
            rms = np.sqrt(np.mean(audio ** 2))
            rms_mean = float(rms)
            
            # Loudness in dB
            loudness_db = 20 * np.log10(rms + 1e-10)
            
            # Spectral Centroid (brightness)
            spectral_centroid = librosa.feature.spectral_centroid(y=audio, sr=sr)[0]
            spectral_centroid_mean = np.mean(spectral_centroid)
            
            # Zero Crossing Rate (articulation)
            zcr = librosa.feature.zero_crossing_rate(audio)[0]
            zcr_mean = np.mean(zcr)
            
            # MFCC (Mel-Frequency Cepstral Coefficients) - voice timbre
            # Fixed: Use proper hop_length and n_fft for melspectrogram
            n_fft = 2048
            hop_length = 512
            n_mels = 128
            
            mfcc = librosa.feature.mfcc(
                y=audio, 
                sr=sr, 
                n_mfcc=13,
                n_fft=n_fft,
                hop_length=hop_length,
                n_mels=n_mels
            )
            mfcc_mean = np.mean(mfcc, axis=1).tolist()
            
            # Pitch estimation (fundamental frequency)
            pitch, mag = librosa.piptrack(
                y=audio, 
                sr=sr,
                n_fft=n_fft,
                hop_length=hop_length
            )
            
            # Get mean and range of detected pitch
            pitch_values = []
            for t in range(pitch.shape[1]):
                index = np.argmax(mag[:, t])
                if mag[index, t] > 0:
                    freq = pitch[index, t]
                    if freq > 0:  # Valid pitch
                        pitch_values.append(freq)
            
            pitch_mean = np.mean(pitch_values) if pitch_values else None
            pitch_range = (min(pitch_values), max(pitch_values)) if pitch_values else None
            
            char = VoiceCharacteristics(
                file_path=str(wav_path),
                duration=duration,
                sample_rate=sr,
                channels=info.channels,
                rms_energy=float(rms_mean),
                spectral_centroid=float(spectral_centroid_mean),
                zero_crossing_rate=float(zcr_mean),
                mfcc_mean=mfcc_mean,
                pitch_mean=float(pitch_mean) if pitch_mean else None,
                pitch_range=(float(pitch_range[0]), float(pitch_range[1])) if pitch_range else None,
                loudness_db=float(loudness_db)
            )
            
            logger.info(f"[OK] Extracted characteristics for {wav_path.name}")
            return char
            
        except Exception as e:
            logger.error(f"[ERROR] Error analyzing {wav_path.name}: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return None
    
    def analyze_all(self) -> Dict[str, VoiceCharacteristics]:
        """Analyze all WAV files in directory"""
        wav_files = sorted(self.voice_dir.glob("*.wav"))
        
        if not wav_files:
            logger.warning(f"No WAV files found in {self.voice_dir}")
            return {}
        
        logger.info(f"Found {len(wav_files)} WAV files to analyze")
        
        self.characteristics = {}
        for wav_file in wav_files:
            char = self.analyze_file(wav_file)
            if char:
                self.characteristics[wav_file.name] = char
        
        logger.info(f"Successfully analyzed {len(self.characteristics)} files")
        return self.characteristics
    
    def get_reference_profile(self) -> Dict:
        """Generate aggregate voice reference profile"""
        if not self.characteristics:
            logger.warning("No characteristics to aggregate")
            return {}
        
        # Aggregate statistics
        durations = [c.duration for c in self.characteristics.values()]
        loudness_values = [c.loudness_db for c in self.characteristics.values()]
        spectral_centroids = [c.spectral_centroid for c in self.characteristics.values()]
        pitch_means = [c.pitch_mean for c in self.characteristics.values() if c.pitch_mean]
        
        # Average MFCC across all files
        all_mfcc = np.array([c.mfcc_mean for c in self.characteristics.values()])
        mfcc_avg = np.mean(all_mfcc, axis=0).tolist()
        
        profile = {
            "cyclops_voice_profile": {
                "files_analyzed": len(self.characteristics),
                "total_duration_seconds": sum(durations),
                "average_duration_seconds": np.mean(durations),
                "loudness_db": {
                    "mean": float(np.mean(loudness_values)),
                    "min": float(np.min(loudness_values)),
                    "max": float(np.max(loudness_values)),
                    "std": float(np.std(loudness_values))
                },
                "spectral_centroid_hz": {
                    "mean": float(np.mean(spectral_centroids)),
                    "min": float(np.min(spectral_centroids)),
                    "max": float(np.max(spectral_centroids))
                },
                "pitch_hz": {
                    "mean": float(np.mean(pitch_means)) if pitch_means else None,
                    "range": [
                        float(np.min(pitch_means)) if pitch_means else None,
                        float(np.max(pitch_means)) if pitch_means else None
                    ]
                },
                "mfcc_coefficients": mfcc_avg,
                "voice_characteristics": {
                    "timbre": "AI robotic with clear articulation",
                    "prosody": "Steady, informative",
                    "emotion": "Neutral, authoritative",
                    "quality": "Clean, well-recorded"
                }
            }
        }
        
        return profile
    
    def save_analysis(self, output_path: Path = Path("voice_reference/voice_profiles")):
        """Save analysis results to JSON"""
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save individual file characteristics
        chars_dict = {}
        for filename, char in self.characteristics.items():
            chars_dict[filename] = asdict(char)
        
        chars_file = output_path / "cyclops_characteristics.json"
        with open(chars_file, 'w', encoding='utf-8') as f:
            json.dump(chars_dict, f, indent=2)
        logger.info(f"[OK] Saved characteristics: {chars_file}")
        
        # Save aggregate profile
        profile = self.get_reference_profile()
        profile_file = output_path / "cyclops_profile.json"
        with open(profile_file, 'w', encoding='utf-8') as f:
            json.dump(profile, f, indent=2)
        logger.info(f"[OK] Saved profile: {profile_file}")
        
        return chars_file, profile_file
    
    def print_report(self):
        """Print analysis report"""
        if not self.characteristics:
            print("No characteristics to report")
            return
        
        profile = self.get_reference_profile()
        p = profile["cyclops_voice_profile"]
        
        print("\n" + "="*70)
        print("CYCLOPS VOICE REFERENCE PROFILE")
        print("="*70)
        print(f"Files Analyzed: {p['files_analyzed']}")
        print(f"Total Duration: {p['total_duration_seconds']:.2f}s ({p['total_duration_seconds']/60:.2f}m)")
        print(f"Average Duration: {p['average_duration_seconds']:.2f}s")
        
        print(f"\nLoudness:")
        print(f"  Mean: {p['loudness_db']['mean']:.2f} dB")
        print(f"  Range: {p['loudness_db']['min']:.2f} - {p['loudness_db']['max']:.2f} dB")
        print(f"  Std Dev: {p['loudness_db']['std']:.2f} dB")
        
        print(f"\nSpectral Centroid (Brightness):")
        print(f"  Mean: {p['spectral_centroid_hz']['mean']:.0f} Hz")
        print(f"  Range: {p['spectral_centroid_hz']['min']:.0f} - {p['spectral_centroid_hz']['max']:.0f} Hz")
        
        if p['pitch_hz']['mean']:
            print(f"\nPitch (Fundamental Frequency):")
            print(f"  Mean: {p['pitch_hz']['mean']:.0f} Hz")
            print(f"  Range: {p['pitch_hz']['range'][0]:.0f} - {p['pitch_hz']['range'][1]:.0f} Hz")
        
        print(f"\nVoice Characteristics:")
        for key, value in p['voice_characteristics'].items():
            print(f"  {key.title()}: {value}")
        
        print("\nMFCC Coefficients (Timbre representation):")
        for i, coeff in enumerate(p['mfcc_coefficients'][:5]):
            print(f"  MFCC[{i}]: {coeff:.4f}")
        
        print("="*70 + "\n")


def main():
    """Main execution"""
    import sys
    from pathlib import Path
    
    # Ensure logs directory exists
    Path("logs").mkdir(exist_ok=True)
    
    logger.info("Starting Voice Reference Analysis")
    
    # Initialize analyzer
    analyzer = VoiceReferenceAnalyzer(
        voice_dir=Path("voice_reference/cyclops_original"),
        sample_rate=24000
    )
    
    # Analyze all files
    characteristics = analyzer.analyze_all()
    
    if characteristics:
        # Save results
        chars_file, profile_file = analyzer.save_analysis()
        
        # Print report
        analyzer.print_report()
        
        print(f"\n[OK] Analysis complete!")
        print(f"  Characteristics: {chars_file}")
        print(f"  Profile: {profile_file}")
    else:
        logger.error("Failed to analyze any files")
        print("\n[ERROR] No files were successfully analyzed.")
        print("Check logs/voice_reference.log for details.")
        sys.exit(1)


if __name__ == "__main__":
    main()
