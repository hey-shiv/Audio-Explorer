"""
visualization/mfcc.py

Generates Mel-Frequency Cepstral Coefficients (MFCCs) and interactive visualizations.

WHY THIS MODULE EXISTS:
While the spectrogram shows frequency content over time, raw spectral bins
are highly correlated and include both source pitch (glottal vibrations)
and filter characteristics (vocal tract shape). MFCCs separate the vocal tract
filter from the excitation source by performing a Discrete Cosine Transform (DCT)
on the log-Mel spectrogram.

WHAT THIS MODULE DOES:
1. Computes MFCCs using librosa.feature.mfcc.
2. Computes Delta MFCCs (first derivative / velocity of spectral change).
3. Renders interactive Plotly heatmaps for Streamlit.

THE PROCESSING PIPELINE:
    y (1D signal) → STFT → Mel Filterbank → Log Scale → DCT Type-II → MFCCs (13 coefficients)
                                                                ↓
                                                         librosa.feature.delta → Delta MFCCs
"""

import librosa
import numpy as np
import plotly.graph_objects as go

from audio_explorer.core.loader import AudioFile


def compute_mfcc(
    audio: AudioFile,
    n_mfcc: int = 13,
    n_fft: int = 2048,
    hop_length: int = 512,
) -> dict:
    """
    Compute Mel-Frequency Cepstral Coefficients (MFCCs) and Delta MFCCs from an AudioFile.

    MFCCs represent the short-term power spectrum of a sound based on a linear cosine transform
    of a log power spectrum on a non-linear Mel scale of frequency.

    Parameters
    ----------
    audio : AudioFile
        The loaded audio container with audio.y (signal) and audio.sr (sample rate).

    n_mfcc : int, optional
        The number of MFCC coefficients to return. Default is 13.

        WHY 13?
        According to source-filter speech production models, the vocal tract acts as a filter
        shaping the sound source. The lower-order 12–13 DCT coefficients capture the smooth
        "spectral envelope" (vocal tract shape and formant positions). Higher coefficients represent
        fine harmonic ripples and excitation pitch (glottal source), which are usually discarded in
        speech recognition, timbral analysis, and speaker identification.

    n_fft : int, optional
        FFT window size in samples. Default is 2048.

    hop_length : int, optional
        Number of samples between successive frames. Default is 512.

    Returns
    -------
    dict with keys:
        mfccs       : np.ndarray, shape (n_mfcc, n_time_frames)
                      Static MFCC coefficient matrix.
        delta_mfccs : np.ndarray, shape (n_mfcc, n_time_frames)
                      First-order temporal derivative (velocity) of MFCCs.
        times       : np.ndarray, shape (n_time_frames,)
                      Time stamp array in seconds for each column frame.
        n_mfcc      : int, number of MFCCs computed.
        n_fft       : int, FFT window size used.
        hop_length  : int, hop size used.
    """

    # ------------------------------------------------------------------
    # Step 1: Compute MFCCs
    #
    # librosa.feature.mfcc() executes the full pipeline internally:
    #   1. Computes the Mel spectrogram (STFT -> magnitude squared -> Mel filterbank).
    #   2. Converts power to log-scale (dB).
    #   3. Performs a Discrete Cosine Transform (DCT Type-II) across the log-Mel energies.
    #   4. Retains the first n_mfcc coefficients.
    #
    # Result shape: (n_mfcc, n_frames) - e.g. (13, n_frames)
    # ------------------------------------------------------------------
    mfccs = librosa.feature.mfcc(
        y=audio.y,
        sr=audio.sr,
        n_mfcc=n_mfcc,
        n_fft=n_fft,
        hop_length=hop_length,
    )

    # ------------------------------------------------------------------
    # Step 2: Compute Delta MFCCs (First Derivative / Velocity)
    #
    # librosa.feature.delta() computes finite difference approximations across time:
    #   d[t] = sum(k * (mfcc[t+k] - mfcc[t-k])) / norm
    #
    # WHAT DELTA MFCCS REPRESENT:
    # Static MFCCs describe the spectral envelope shape at a single frozen moment.
    # Delta MFCCs describe how fast the spectral envelope is CHANGING over time
    # (velocity/trajectory of spectral shifts, such as transitioning between phonemes
    # or instrument dynamics).
    # ------------------------------------------------------------------
    delta_mfccs = librosa.feature.delta(mfccs)

    # ------------------------------------------------------------------
    # Step 3: Build the time axis
    #
    # librosa.frames_to_time() converts frame column indices [0, 1, 2, ...]
    # into time values in seconds: time = frame_idx * hop_length / sr
    # ------------------------------------------------------------------
    n_frames = mfccs.shape[1]
    times = librosa.frames_to_time(
        np.arange(n_frames), sr=audio.sr, hop_length=hop_length
    )

    return {
        "mfccs": mfccs,
        "delta_mfccs": delta_mfccs,
        "times": times,
        "n_mfcc": n_mfcc,
        "n_fft": n_fft,
        "hop_length": hop_length,
    }


def plot_mfcc(
    mfcc_data: dict,
    title: str = "MFCCs",
    color_scale: str = "RdBu_r",
) -> go.Figure:
    """
    Generate an interactive Plotly Heatmap of Mel-Frequency Cepstral Coefficients.

    Parameters
    ----------
    mfcc_data : dict
        Output dictionary from compute_mfcc().

    title : str, optional
        Title of the Plotly figure. Default is 'MFCCs'.

    color_scale : str, optional
        Plotly color scale. Default is 'RdBu_r' (Reversed Red-Blue).

        WHY RdBu_r COLORSCALE?
        MFCCs are calculated via a Discrete Cosine Transform on log-Mel values, resulting in
        both positive and negative values roughly centered around 0. A diverging colormap
        like RdBu_r (Red for positive values, Blue for negative values, and white/neutral near zero)
        clearly highlights symmetrical positive and negative cepstral variations and mean-normalized
        features.

    Returns
    -------
    go.Figure
        A styled Plotly figure ready for rendering in Streamlit via st.plotly_chart().
    """

    mfccs = mfcc_data["mfccs"]
    times = mfcc_data["times"]
    n_mfcc = mfcc_data["n_mfcc"]

    # ------------------------------------------------------------------
    # Downsample the time axis for browser performance
    #
    # Plotly Heatmaps perform best with up to 2000 columns in web browsers.
    # Downsampling takes every N-th frame across the time axis (same pattern
    # as spectrogram.py).
    # ------------------------------------------------------------------
    max_time_points = 2000
    n_time = mfccs.shape[1]

    if n_time > max_time_points:
        time_step = n_time // max_time_points
        mfccs_display = mfccs[:, ::time_step]
        times_display = times[::time_step]
    else:
        mfccs_display = mfccs
        times_display = times

    coeff_indices = np.arange(n_mfcc)

    # ------------------------------------------------------------------
    # Build the Plotly Heatmap
    #
    # x: Time in seconds
    # y: MFCC coefficient index (0 to n_mfcc - 1)
    # z: MFCC matrix values
    # ------------------------------------------------------------------
    fig = go.Figure()

    fig.add_trace(
        go.Heatmap(
            x=times_display,
            y=coeff_indices,
            z=mfccs_display,
            colorscale=color_scale,
            colorbar=dict(
                title=dict(text="Val", font=dict(color="#94a3b8")),
                tickfont=dict(color="#94a3b8"),
                thickness=12,
            ),
            hovertemplate=(
                "<b>Time:</b> %{x:.2f}s<br>"
                "<b>Coefficient:</b> MFCC %{y}<br>"
                "<b>Value:</b> %{z:.2f}<br>"
                "<extra></extra>"
            ),
        )
    )

    # ------------------------------------------------------------------
    # Dark Theme and Axis Styling
    # ------------------------------------------------------------------
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
        height=300,
    )

    fig.update_xaxes(
        title_text="Time (seconds)",
        title_font=dict(size=13, color="#64748b"),
        tickfont=dict(size=11, color="#64748b"),
        gridcolor="#1e293b",
        zeroline=False,
    )

    fig.update_yaxes(
        title_text="MFCC Coefficient",
        title_font=dict(size=13, color="#64748b"),
        tickfont=dict(size=11, color="#64748b"),
        gridcolor="#1e293b",
        zeroline=False,
        dtick=1,
    )

    return fig
