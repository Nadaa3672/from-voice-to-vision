"""
Download RAVDESS and build a tidy index (DataFrame) of every clip with its
decoded metadata: emotion, intensity, actor, gender, and the split it belongs to.
"""
import os
import zipfile
import urllib.request
from pathlib import Path

import pandas as pd

from src import config


# ----------------------------------------------------------------------
# Download / extract
# ----------------------------------------------------------------------
def download_ravdess(force: bool = False) -> Path:
    """Download and extract the RAVDESS speech set into config.RAVDESS_DIR."""
    config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    config.RAVDESS_DIR.mkdir(parents=True, exist_ok=True)

    # already extracted?
    existing = list(config.RAVDESS_DIR.rglob("*.wav"))
    if existing and not force:
        print(f"✓ RAVDESS already present: {len(existing)} .wav files")
        return config.RAVDESS_DIR

    zip_path = config.DATA_DIR / "ravdess_speech.zip"
    if not zip_path.exists() or force:
        print("⬇️  Downloading RAVDESS (~200 MB)…")
        urllib.request.urlretrieve(config.RAVDESS_URL, zip_path)
        print("✓ Download complete")

    print("📦 Extracting…")
    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(config.RAVDESS_DIR)

    n = len(list(config.RAVDESS_DIR.rglob("*.wav")))
    print(f"✓ Extracted {n} .wav files into {config.RAVDESS_DIR}")
    return config.RAVDESS_DIR


# ----------------------------------------------------------------------
# Filename decoding -> index DataFrame
# ----------------------------------------------------------------------
def _decode_filename(path: Path) -> dict:
    parts = path.stem.split("-")
    if len(parts) != 7:
        return {}
    modality, vocal, emotion, intensity, statement, repetition, actor = parts
    actor_id = int(actor)
    return {
        "path": str(path),
        "emotion": config.EMOTION_MAP.get(emotion, "unknown"),
        "emotion_id": config.EMOTION_TO_ID.get(config.EMOTION_MAP.get(emotion, ""), -1),
        "intensity": "strong" if intensity == "02" else "normal",
        "statement": statement,
        "repetition": repetition,
        "actor": actor_id,
        "gender": "male" if actor_id % 2 == 1 else "female",
    }


def _assign_split(actor_id: int) -> str:
    if actor_id in config.TEST_ACTORS:
        return "test"
    if actor_id in config.VAL_ACTORS:
        return "val"
    return "train"


def build_index() -> pd.DataFrame:
    """Return a DataFrame with one row per clip, including the speaker-independent split."""
    wavs = sorted(config.RAVDESS_DIR.rglob("*.wav"))
    rows = [_decode_filename(p) for p in wavs]
    rows = [r for r in rows if r]  # drop malformed names
    df = pd.DataFrame(rows)
    df["split"] = df["actor"].apply(_assign_split)
    return df


if __name__ == "__main__":
    download_ravdess()
    df = build_index()
    print(df.head())
    print("\nClips per split:\n", df["split"].value_counts())
    print("\nClips per emotion:\n", df["emotion"].value_counts())
