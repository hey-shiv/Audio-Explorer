"""
visualization/mel_spectrogram.py

Generates Mel Spectrogram visualizations.

The Mel Spectrogram warps the frequency axis of a regular spectrogram
to match human hearing perception using the Mel Scale.

This is THE standard input representation for modern audio AI models
including Whisper, CLAP, AudioMAE, BEATs, and MusicGen.

PIPELINE:
    y (1D) → STFT → |magnitude|² → Mel filterbank → log scale → Plotly Heatmap
"""

import numpy as np
import plotly.graph_objects as go
import librosa

from audio_explorer.core.loader import AudioFile


def compute_mel_spectrogram(
    audio: AudioFile,
    n_fft: int = 2048,
    hop_length: int = 512,
    n_mels: int = 128,
) -> dict:
    """
    Compute a Mel Spectrogram from an AudioFile.

    Parameters
    ----------
    audio      : AudioFile — our loaded audio data.
    n_fft      : FFT window size (same as Day 3 spectrogram).
    hop_length : Window slide distance (same as Day 3).
    n_mels     : Number of Mel frequency bands. Default 128.
                 Whisper uses 80. Most research uses 64 or 128.
                 More bands = finer frequency detail but larger matrix.

    Returns
    -------
    dict with keys:
        S_db       : The Mel Spectrogram in decibels. Shape: (n_mels, n_frames)
        times      : Time value for each column (seconds).
        mel_freqs  : The center frequency (Hz) of each Mel band.
        n_mels     : Number of Mel bands used.
        n_fft      : FFT size used.
        hop_length : Hop size used.
    """

    # Step 1: Compute the Mel Spectrogram in one call.
    #
    # librosa.feature.melspectrogram() does THREE things internally:
    #   1. Computes the STFT (same as Day 3)
    #   2. Squares the magnitude to get POWER spectrum (energy)
    #   3. Multiplies by a "Mel filterbank" matrix that warps
    #      the linear frequency axis into the Mel scale
    #
    # The Mel filterbank is a set of n_mels triangular filters
    # spaced according to human perception — closely packed at
    # low frequencies, widely spaced at high frequencies.
    #
    # Output shape: (n_mels, n_frames) — e.g., (128, 9044)
    # Compare to Day 3 spectrogram: (1025, 9044)
    # We went from 1025 frequency bins to 128 Mel bands.

    S_mel = librosa.feature.melspectrogram(
        y=audio.y,
        sr=audio.sr,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels,
    )

    # Step 2: Convert power to decibels.
    #
    # On Day 3 we used amplitude_to_db (for magnitude).
    # Here we use power_to_db (for squared magnitude).
    # The difference: power_to_db uses 10*log10 instead of 20*log10
    # because power is already squared (magnitude²).

    S_db = librosa.power_to_db(S_mel, ref=np.max, top_db=80)

    # Step 3: Build the time axis (same method as Day 3).

    n_frames = S_db.shape[1]
    times = librosa.frames_to_time(
        np.arange(n_frames), sr=audio.sr, hop_length=hop_length
    )

    # Step 4: Get the center frequencies of each Mel band.
    #
    # librosa.mel_frequencies() returns n_mels+2 frequency values
    # (the +2 are the edges of the first and last filters).
    # We take only the first n_mels to label our Y-axis.

    mel_freqs = librosa.mel_frequencies(n_mels=n_mels + 2, fmin=0, fmax=audio.sr / 2)
    mel_freqs = mel_freqs[1:n_mels + 1]  # center frequencies only

    return {
        "S_db": S_db,
        "times": times,
        "mel_freqs": mel_freqs,
        "n_mels": n_mels,
        "n_fft": n_fft,
        "hop_length": hop_length,
    }


def plot_mel_spectrogram(
    mel_data: dict,
    color_scale: str = "Magma",
    title: str = "Mel Spectrogram",
) -> go.Figure:
    """
    Generate an interactive Mel Spectrogram heatmap.

    Parameters
    ----------
    mel_data    : dict from compute_mel_spectrogram().
    color_scale : Plotly colormap. "Magma" is warm dark-to-bright,
                  great for Mel spectrograms. Alternatives: "Inferno", "Viridis".
    title       : Chart title.

    Returns
    -------
    go.Figure ready for st.plotly_chart().
    """

    S_db = mel_data["S_db"]
    times = mel_data["times"]
    n_mels = mel_data["n_mels"]
    mel_freqs = mel_data["mel_freqs"]

    # ------------------------------------------------------------------
    # Downsample the time axis for browser performance.
    # Same technique as Day 3 spectrogram.
    # ------------------------------------------------------------------

    max_time_points = 2000
    n_time = S_db.shape[1]

    if n_time > max_time_points:
        time_step = n_time // max_time_points
        S_display = S_db[:, ::time_step]
        times_display = times[::time_step]
    else:
        S_display = S_db
        times_display = times

    # ------------------------------------------------------------------
    # Build the Y-axis using actual Mel center frequencies.
    #
    # This gives a meaningful Hz label on the Y-axis while preserving
    # the non-linear Mel spacing visually. Low frequencies are stretched
    # (more visual space), high frequencies are compressed.
    # ------------------------------------------------------------------

    fig = go.Figure()

    fig.add_trace(
        go.Heatmap(
            x=times_display,
            y=mel_freqs,
            z=S_display,
            colorscale=color_scale,
            colorbar=dict(
                title=dict(text="dB", font=dict(color="#94a3b8")),
                tickfont=dict(color="#94a3b8"),
                thickness=12,
            ),
            hovertemplate=(
                "<b>Time:</b> %{x:.2f}s<br>"
                "<b>Freq:</b> %{y:.0f} Hz (Mel)<br>"
                "<b>Power:</b> %{z:.1f} dB<br>"
                "<extra></extra>"
            ),
        )
    )

    fig.update_layout(
        title=dict(
            text=title,
            font=dict(size=16, color="#e2e8f0"),
            x=0.02,
        ),
        paper_bgcolor="#0f172a",
        plot_bgcolor="#0f172a",
        font=dict(family="Inter, sans-serif", color="#94a3b8"),
        margin=dict(l=70, r=30, t=60, b=60),
        height=400,
    )

    fig.update_xaxes(
        title_text="Time (seconds)",
        title_font=dict(size=13, color="#64748b"),
        tickfont=dict(size=11, color="#64748b"),
        gridcolor="#1e293b",
        zeroline=False,
    )

    fig.update_yaxes(
        title_text="Frequency (Hz — Mel Scale)",
        title_font=dict(size=13, color="#64748b"),
        tickfont=dict(size=11, color="#64748b"),
        gridcolor="#1e293b",
        zeroline=False,
    )

    return fig
