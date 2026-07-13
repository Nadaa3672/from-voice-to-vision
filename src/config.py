"""
Global configuration for the From Voice to Vision project.
Central place for paths, the emotion map, and audio/feature parameters.
"""
from pathlib import Path

# ----------------------------------------------------------------------
# Paths
# ----------------------------------------------------------------------
# On Colab we typically clone the repo and download data next to it.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RAVDESS_DIR = DATA_DIR / "ravdess"                 # extracted .wav files live here
RESULTS_DIR = PROJECT_ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"

# RAVDESS speech audio (Zenodo, no login required)
RAVDESS_URL = "https://zenodo.org/record/1188976/files/Audio_Speech_Actors_01-24.zip"

# ----------------------------------------------------------------------
# RAVDESS filename decoding
# Filename: "03-01-06-01-02-01-12.wav"
#   [modality]-[vocal_channel]-[EMOTION]-[intensity]-[statement]-[repetition]-[ACTOR].wav
# ----------------------------------------------------------------------
EMOTION_MAP = {
    "01": "neutral",
    "02": "calm",
    "03": "happy",
    "04": "sad",
    "05": "angry",
    "06": "fearful",
    "07": "disgust",
    "08": "surprised",
}

# Ordered list of classes (index = label id used by the models)
EMOTIONS = ["neutral", "calm", "happy", "sad", "angry", "fearful", "disgust", "surprised"]
EMOTION_TO_ID = {e: i for i, e in enumerate(EMOTIONS)}
ID_TO_EMOTION = {i: e for e, i in EMOTION_TO_ID.items()}

# ----------------------------------------------------------------------
# Audio & feature parameters
# ----------------------------------------------------------------------
SAMPLE_RATE = 16000     # resample everything to 16 kHz
DURATION = 3.0          # seconds; pad/trim clips to this fixed length
N_MELS = 128            # mel bands for the log-mel spectrogram
N_MFCC = 40             # number of MFCC coefficients
N_FFT = 1024
HOP_LENGTH = 256

# ----------------------------------------------------------------------
# Reproducibility & splits
# ----------------------------------------------------------------------
SEED = 42
# Speaker-independent split: actors reserved for validation / test.
# (odd actor ids = male, even = female — kept balanced across splits)
# Larger validation (4 actors) makes the optimization fitness far less noisy
# and avoids over-fitting the hyper-parameters to just a couple of speakers.
VAL_ACTORS = [3, 4, 5, 6]
TEST_ACTORS = [1, 2, 21, 22]
# everyone else -> training (16 actors)
