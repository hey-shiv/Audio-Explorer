"""
core/loader.py

Responsible for loading audio files from disk and extracting
all essential metadata. This is the entry point for ALL audio
data into the Audio Explorer pipeline.

What this module does:
- Loads WAV, MP3, and FLAC files using librosa + soundfile
- Extracts metadata: sample rate, duration, channels, bit depth
- Returns a clean AudioFile dataclass for use throughout the app
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import librosa
import numpy as np
import soundfile as sf


# ---------------------------------------------------------------------------
# AudioFile: The central data structure for this entire project.
#
# A dataclass is like a struct — it holds data with type hints
# and gives you __repr__, __eq__ for free. Much cleaner than a dict.
# ---------------------------------------------------------------------------

@dataclass
class AudioFile:
    """
    Represents a loaded audio file and all its associated metadata.

    Attributes
    ----------
    file_path   : The absolute path to the source audio file.
    file_name   : Just the filename (e.g. "song.mp3").
    file_format : The file extension in uppercase (e.g. "MP3").
    file_size_mb: Size of the file on disk in megabytes.

    y           : The raw audio signal as a 1D NumPy array of float32 values.
                  Each value is an air pressure sample between -1.0 and +1.0.
                  This is the NUMBER that every algorithm in this project processes.

    sr          : Sample rate — how many samples exist per second (e.g. 22050 Hz).
    duration    : Length of the audio in seconds.
    num_channels: 1 = Mono, 2 = Stereo.
    num_samples : Total number of samples in the signal array.
    bit_depth   : Bit depth of the original file (e.g. 16, 24, 32).
                  Only available for WAV/FLAC files, None for MP3.
    """

    # --- File Info ---
    file_path: str
    file_name: str
    file_format: str
    file_size_mb: float

    # --- Audio Signal ---
    y: np.ndarray           # The raw waveform — a 1D array of float32 samples
    sr: int                 # Sample rate in Hz

    # --- Derived Metadata ---
    duration: float         # seconds
    num_channels: int       # 1 (Mono) or 2 (Stereo)
    num_samples: int        # Total number of samples
    bit_depth: Optional[int] = None  # Only available for WAV/FLAC


def load_audio(
    file_path: str,
    target_sr: int = 22050,
    mono: bool = True,
) -> AudioFile:
    """
    Load an audio file from disk and return a fully populated AudioFile.

    Parameters
    ----------
    file_path : str
        Path to the audio file (WAV, MP3, or FLAC).

    target_sr : int, optional
        The sample rate to resample the audio to.
        Default is 22050 Hz — the librosa standard, and common in AI research.
        WHY 22050? It covers the full range of human hearing (20 Hz–20 kHz)
        per the Nyquist theorem (requires at least 2× the max frequency),
        while keeping file sizes small.

    mono : bool, optional
        If True, convert stereo to mono by averaging the two channels.
        Most audio ML models expect mono input. Default is True.

    Returns
    -------
    AudioFile
        A dataclass containing the raw waveform and all metadata.

    Raises
    ------
    FileNotFoundError
        If the file path does not exist.
    ValueError
        If the file format is not supported.
    """

    # --- Step 1: Validate the file path ---
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Audio file not found: {file_path}")

    supported_formats = {".wav", ".mp3", ".flac", ".ogg", ".m4a"}
    if path.suffix.lower() not in supported_formats:
        raise ValueError(
            f"Unsupported format: '{path.suffix}'. "
            f"Supported formats: {supported_formats}"
        )

    # --- Step 2: Load the audio signal using librosa ---
    #
    # librosa.load() does the following internally:
    #   1. Detects file format (WAV/MP3/FLAC) and selects the right decoder.
    #   2. Decodes compressed audio (MP3) into raw PCM samples.
    #   3. Resamples to target_sr using a high-quality algorithm (soxr).
    #   4. Converts to float32, normalized between -1.0 and +1.0.
    #   5. If mono=True, mixes stereo L+R channels into one signal by averaging.
    #
    # Returns:
    #   y  — 1D NumPy array of shape (num_samples,)
    #   sr — The sample rate we requested (target_sr)
    #
    y, sr = librosa.load(file_path, sr=target_sr, mono=mono)

    # --- Step 3: Try to get extra metadata for lossless formats ---
    #
    # soundfile is faster and more precise than librosa for WAV/FLAC
    # because it reads metadata directly from the file header without decoding.
    # MP3 doesn't have a proper header for bit depth, so we skip it for MP3.
    #
    bit_depth = None
    original_channels = 1

    if path.suffix.lower() in {".wav", ".flac"}:
        try:
            info = sf.info(file_path)
            # info.subtype looks like "PCM_16", "PCM_24", "FLOAT", etc.
            # We parse the number out of it.
            subtype = info.subtype  # e.g. "PCM_16"
            if "PCM_" in subtype:
                bit_depth = int(subtype.split("_")[1])  # "PCM_16" → 16
            elif "FLOAT" in subtype:
                bit_depth = 32
            original_channels = info.channels
        except Exception:
            # If metadata reading fails, we just leave bit_depth as None.
            pass
    else:
        # For MP3: librosa doesn't expose channel info after mono conversion.
        # We estimate from the shape of y before mono conversion.
        # For simplicity in sprint mode, we default to 2 for MP3 (most are stereo).
        original_channels = 2

    # --- Step 4: Compute derived values ---
    num_samples = len(y)

    # duration (seconds) = total samples ÷ samples per second
    duration = num_samples / sr

    # File size: Path.stat().st_size returns bytes. Divide by 1024² for MB.
    file_size_mb = path.stat().st_size / (1024 * 1024)

    # --- Step 5: Assemble and return the AudioFile dataclass ---
    return AudioFile(
        file_path=str(path.resolve()),
        file_name=path.name,
        file_format=path.suffix.upper().lstrip("."),
        file_size_mb=round(file_size_mb, 2),
        y=y,
        sr=sr,
        duration=round(duration, 3),
        num_channels=original_channels,
        num_samples=num_samples,
        bit_depth=bit_depth,
    )
