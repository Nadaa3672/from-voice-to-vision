"""
Audio preprocessing for SER.

Pipeline for a single clip:
    load  ->  (optional denoise)  ->  trim silence  ->  fix length (pad/cut)

All clips end up as a fixed-length waveform at config.SAMPLE_RATE, so that
the downstream features (log-Mel / MFCC) have a constant shape.
"""
import numpy as np
import librosa

from src import config


def load_audio(path, sr: int = config.SAMPLE_RATE) -> np.ndarray:
    """Load a wav as mono float32 at the target sample rate."""
    y, _ = librosa.load(path, sr=sr, mono=True)
    return y.astype(np.float32)


def reduce_noise(y: np.ndarray, sr: int = config.SAMPLE_RATE) -> np.ndarray:
    """Spectral-gating denoising (course topic). Optional: RAVDESS is already clean."""
    import noisereduce as nr
    return nr.reduce_noise(y=y, sr=sr).astype(np.float32)


def trim_silence(y: np.ndarray, top_db: int = 25) -> np.ndarray:
    """Remove leading/trailing silence so emotion cues aren't drowned by dead air."""
    yt, _ = librosa.effects.trim(y, top_db=top_db)
    return yt if len(yt) > 0 else y


def fix_length(y: np.ndarray, sr: int = config.SAMPLE_RATE,
               duration: float = config.DURATION) -> np.ndarray:
    """Pad with zeros (centered) or center-crop to a fixed number of samples."""
    target = int(sr * duration)
    if len(y) < target:
        pad = target - len(y)
        left = pad // 2
        y = np.pad(y, (left, pad - left), mode="constant")
    elif len(y) > target:
        start = (len(y) - target) // 2
        y = y[start:start + target]
    return y.astype(np.float32)


def preprocess(path, denoise: bool = False, trim: bool = True,
               sr: int = config.SAMPLE_RATE, duration: float = config.DURATION) -> np.ndarray:
    """Full single-clip preprocessing -> fixed-length waveform."""
    y = load_audio(path, sr=sr)
    if denoise:
        y = reduce_noise(y, sr=sr)
    if trim:
        y = trim_silence(y)
    y = fix_length(y, sr=sr, duration=duration)
    return y
