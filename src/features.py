"""
Feature extraction for SER.

Two feature families:
  * log-Mel spectrogram  -> 2D image for the CNN            shape (N_MELS, T)
  * MFCC statistics       -> 1D vector for classical ML      shape (D,)

`build_dataset` runs the whole RAVDESS index through preprocessing + feature
extraction once, and caches the arrays as .npy so later phases load instantly.
"""
import numpy as np
import librosa
from tqdm.auto import tqdm

from src import config, preprocessing


# ----------------------------------------------------------------------
# Single-clip features
# ----------------------------------------------------------------------
def log_mel(y: np.ndarray, sr: int = config.SAMPLE_RATE) -> np.ndarray:
    """Log-Mel spectrogram in dB. Shape (N_MELS, T)."""
    S = librosa.feature.melspectrogram(
        y=y, sr=sr, n_mels=config.N_MELS,
        n_fft=config.N_FFT, hop_length=config.HOP_LENGTH)
    return librosa.power_to_db(S, ref=np.max).astype(np.float32)


def mfcc(y: np.ndarray, sr: int = config.SAMPLE_RATE) -> np.ndarray:
    """MFCC matrix. Shape (N_MFCC, T)."""
    return librosa.feature.mfcc(
        y=y, sr=sr, n_mfcc=config.N_MFCC,
        n_fft=config.N_FFT, hop_length=config.HOP_LENGTH).astype(np.float32)


def mfcc_vector(y: np.ndarray, sr: int = config.SAMPLE_RATE) -> np.ndarray:
    """
    Summary vector for classical ML: mean + std of MFCC and its 1st delta.
    Shape (4 * N_MFCC,) = (160,) with default config.
    """
    m = mfcc(y, sr=sr)
    d = librosa.feature.delta(m)
    return np.concatenate([m.mean(1), m.std(1), d.mean(1), d.std(1)]).astype(np.float32)


# ----------------------------------------------------------------------
# Whole-dataset extraction (+ caching)
# ----------------------------------------------------------------------
def build_dataset(df, denoise: bool = False, cache: bool = True):
    """
    Extract features for every clip in the index `df`.
    Returns dict with:
        X_mel  (N, N_MELS, T)  log-Mel spectrograms
        X_vec  (N, D)          MFCC summary vectors
        y      (N,)            emotion ids
        split  (N,)            'train' / 'val' / 'test'
    """
    feat_dir = config.DATA_DIR / "features"
    feat_dir.mkdir(parents=True, exist_ok=True)
    tag = "denoise" if denoise else "raw"
    mel_path = feat_dir / f"X_mel_{tag}.npy"
    vec_path = feat_dir / f"X_vec_{tag}.npy"
    y_path = feat_dir / "y.npy"
    sp_path = feat_dir / "split.npy"

    if cache and mel_path.exists():
        print("✓ Loading cached features")
        return {
            "X_mel": np.load(mel_path), "X_vec": np.load(vec_path),
            "y": np.load(y_path), "split": np.load(sp_path, allow_pickle=True),
        }

    X_mel, X_vec, y, split = [], [], [], []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Extracting features"):
        wav = preprocessing.preprocess(row["path"], denoise=denoise)
        X_mel.append(log_mel(wav))
        X_vec.append(mfcc_vector(wav))
        y.append(int(row["emotion_id"]))
        split.append(row["split"])

    out = {
        "X_mel": np.stack(X_mel).astype(np.float32),
        "X_vec": np.stack(X_vec).astype(np.float32),
        "y": np.array(y, dtype=np.int64),
        "split": np.array(split, dtype=object),
    }
    if cache:
        np.save(mel_path, out["X_mel"]); np.save(vec_path, out["X_vec"])
        np.save(y_path, out["y"]); np.save(sp_path, out["split"])
        print(f"✓ Cached features to {feat_dir}")
    return out


def split_arrays(data: dict):
    """Convenience: split the cached arrays into train/val/test."""
    s = data["split"]
    out = {}
    for name in ["train", "val", "test"]:
        m = s == name
        out[name] = {"X_mel": data["X_mel"][m], "X_vec": data["X_vec"][m], "y": data["y"][m]}
    return out
