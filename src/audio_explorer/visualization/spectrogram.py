"""
visualization/spectrogram.py

Generates interactive spectrogram visualizations from a loaded AudioFile.

WHY THIS MODULE EXISTS:
The waveform (Day 2) shows WHEN sound is loud/quiet, but hides WHICH
frequencies are present. The spectrogram reveals the frequency content
over time — it is the 2D "image" that almost every audio AI model
(Whisper, AudioMAE, CLAP, BEATs, MusicGen) uses as input.

WHAT THIS MODULE DOES:
1. Computes the Short-Time Fourier Transform (STFT) using librosa.
2. Converts the complex-valued STFT output to a magnitude spectrogram.
3. Applies a decibel (log) scale for visual clarity.
4. Renders an interactive Plotly heatmap for the Streamlit UI.

THE PIPELINE:
    y (1D array) → STFT → Complex matrix → |magnitude| → dB scale → Plotly heatmap
"""

import numpy as np
import plotly.graph_objects as go
import librosa

from audio_explorer.core.loader import AudioFile


def compute_spectrogram(
    audio: AudioFile,
    n_fft: int = 2048,
    hop_length: int = 512,
) -> dict:
    """
    Compute a power spectrogram from an AudioFile.

    This function performs the complete STFT pipeline and returns
    all the data needed for visualization and analysis.

    Parameters
    ----------
    audio : AudioFile
        The loaded audio. We use audio.y (signal) and audio.sr (sample rate).

    n_fft : int, optional
        The FFT window size in samples. Default is 2048.

        WHY 2048?
        At sr=22050 Hz, each window covers 2048/22050 ≈ 0.093 seconds.
        This gives 1025 frequency bins (n_fft/2 + 1), each bin spanning
        about 10.8 Hz. This is a good balance between frequency resolution
        (distinguishing nearby notes) and time resolution (tracking fast changes).

        The result has (1 + n_fft/2) = 1025 frequency bins.
        Each bin covers sr/n_fft = 22050/2048 ≈ 10.77 Hz.

    hop_length : int, optional
        How many samples to slide the window forward between FFT computations.
        Default is 512 (which is n_fft / 4).

        WHY n_fft / 4?
        This means consecutive windows overlap by 75%. High overlap means
        smoother time resolution and fewer visual artifacts. Each time
        step covers hop_length/sr = 512/22050 ≈ 0.023 seconds.

    Returns
    -------
    dict with keys:
        S_db        : np.ndarray, shape (n_freq_bins, n_time_frames)
                      The spectrogram in decibels. This is the main result.
        freqs       : np.ndarray, shape (n_freq_bins,)
                      The frequency value (in Hz) for each row.
        times       : np.ndarray, shape (n_time_frames,)
                      The time value (in seconds) for each column.
        n_fft       : int, the FFT size used.
        hop_length  : int, the hop size used.
        freq_resolution : float, Hz per frequency bin.
        time_resolution : float, seconds per time frame.
    """

    # ------------------------------------------------------------------
    # Step 1: Compute the STFT
    #
    # librosa.stft() performs the complete Short-Time Fourier Transform:
    #   - Slices the signal into overlapping chunks of size n_fft
    #   - Applies a Hann window to each chunk (reduces spectral leakage)
    #   - Computes the FFT on each windowed chunk
    #
    # Returns a COMPLEX-VALUED matrix S_complex of shape:
    #   (1 + n_fft/2, num_frames)
    #   = (1025, num_frames) for n_fft=2048
    #
    # Each element S_complex[f, t] is a complex number a + bj where:
    #   - magnitude = sqrt(a² + b²) = how loud frequency f is at time t
    #   - phase = arctan(b/a) = when the wave started (we discard this)
    #
    # WHY COMPLEX?
    # The Fourier Transform naturally produces complex numbers because
    # it correlates the signal against both cosine (real) and sine
    # (imaginary) test waves. The magnitude combines both correlations.
    # ------------------------------------------------------------------

    S_complex = librosa.stft(audio.y, n_fft=n_fft, hop_length=hop_length)

    # ------------------------------------------------------------------
    # Step 2: Convert complex → magnitude
    #
    # np.abs() on a complex number returns its magnitude:
    #   |a + bj| = sqrt(a² + b²)
    #
    # This discards the phase information and keeps only "how loud
    # is each frequency at each moment" — which is what we want
    # for visualization.
    # ------------------------------------------------------------------

    S_magnitude = np.abs(S_complex)

    # ------------------------------------------------------------------
    # Step 3: Convert magnitude → decibels
    #
    # librosa.amplitude_to_db() computes:
    #   S_db = 20 * log10(S_magnitude / ref) where ref = max(S_magnitude)
    #
    # WHY DECIBELS?
    # Raw magnitude values have enormous dynamic range. A drum hit might
    # be 10,000× louder than a quiet background hum. On a linear scale,
    # the quiet sounds are invisible. The logarithmic dB scale compresses
    # this range so both loud and quiet content is visible.
    #
    # ref=np.max means the loudest point will be 0 dB, and everything
    # else will be negative dB values (e.g., -20 dB = much quieter).
    #
    # top_db=80 means anything quieter than 80 dB below the peak is
    # clipped to -80 dB. This prevents -infinity values from silent regions.
    # ------------------------------------------------------------------

    S_db = librosa.amplitude_to_db(S_magnitude, ref=np.max, top_db=80)

    # ------------------------------------------------------------------
    # Step 4: Build the frequency and time axis arrays
    #
    # librosa.fft_frequencies(sr, n_fft):
    #   Returns an array of the frequency (in Hz) for each row of the
    #   spectrogram. For sr=22050, n_fft=2048:
    #   [0.0, 10.77, 21.53, 32.30, ..., 11025.0]  (1025 values)
    #
    # librosa.frames_to_time(range(n_frames), sr, hop_length):
    #   Converts frame indices [0, 1, 2, ...] into time in seconds.
    #   frame i corresponds to time = i * hop_length / sr
    # ------------------------------------------------------------------

    freqs = librosa.fft_frequencies(sr=audio.sr, n_fft=n_fft)

    n_frames = S_db.shape[1]
    times = librosa.frames_to_time(
        np.arange(n_frames), sr=audio.sr, hop_length=hop_length
    )

    freq_resolution = audio.sr / n_fft      # Hz per bin
    time_resolution = hop_length / audio.sr  # seconds per frame

    return {
        "S_db": S_db,
        "freqs": freqs,
        "times": times,
        "n_fft": n_fft,
        "hop_length": hop_length,
        "freq_resolution": round(freq_resolution, 2),
        "time_resolution": round(time_resolution, 4),
    }


def plot_spectrogram(
    spec_data: dict,
    color_scale: str = "Viridis",
    title: str = "Spectrogram — Frequency vs Time",
    max_freq: float = 11025.0,
) -> go.Figure:
    """
    Generate an interactive spectrogram heatmap from computed spectrogram data.

    Parameters
    ----------
    spec_data : dict
        The output of compute_spectrogram().

    color_scale : str, optional
        The Plotly color scale for the heatmap. Default is "Viridis" —
        a perceptually uniform colormap designed for scientific visualization.
        It works well for color-blind users and prints well in grayscale.

        Other good options: "Inferno", "Magma", "Plasma", "Hot"

    title : str, optional
        Chart title.

    max_freq : float, optional
        Maximum frequency to display in Hz. Default 11025 (Nyquist limit
        for sr=22050). You can lower this to zoom into the bass range.

    Returns
    -------
    go.Figure
        A Plotly figure ready for st.plotly_chart().
    """

    S_db = spec_data["S_db"]
    freqs = spec_data["freqs"]
    times = spec_data["times"]

    # ------------------------------------------------------------------
    # Trim to max_freq
    #
    # Find the index where freqs exceeds max_freq.
    # np.searchsorted() performs binary search on the sorted freqs array.
    # We then slice the spectrogram to only include rows up to that index.
    # This is purely cosmetic — it removes the top portion of the chart
    # where there is often very little energy anyway.
    # ------------------------------------------------------------------

    freq_limit_idx = np.searchsorted(freqs, max_freq)
    S_display = S_db[:freq_limit_idx, :]
    freqs_display = freqs[:freq_limit_idx]

    # ------------------------------------------------------------------
    # Downsample the time axis for browser performance
    #
    # A 3.5-minute song produces ~9,000 time frames.
    # 1025 freq bins × 9,000 time frames = 9.2 million data points.
    # Plotly Heatmap can handle this, but it's slow. We downsample
    # the time axis to at most 2,000 columns.
    # ------------------------------------------------------------------

    max_time_points = 2000
    n_time = S_display.shape[1]

    if n_time > max_time_points:
        time_step = n_time // max_time_points
        S_display = S_display[:, ::time_step]
        times_display = times[::time_step]
    else:
        times_display = times

    # ------------------------------------------------------------------
    # Build the Plotly Heatmap
    #
    # go.Heatmap renders a 2D matrix as colored cells.
    # - x: time axis values
    # - y: frequency axis values
    # - z: the spectrogram matrix (each cell value maps to a color)
    # - colorscale: which colors to use
    # - colorbar: the legend showing dB-to-color mapping
    #
    # IMPORTANT: z rows correspond to y-axis values.
    # S_display[0, :] is the lowest frequency (0 Hz) → bottom of chart.
    # S_display[-1, :] is the highest frequency → top of chart.
    # Plotly Heatmap naturally renders row 0 at the bottom, which is
    # correct for us since row 0 = lowest frequency.
    # ------------------------------------------------------------------

    fig = go.Figure()

    fig.add_trace(
        go.Heatmap(
            x=times_display,
            y=freqs_display,
            z=S_display,
            colorscale=color_scale,
            colorbar=dict(
                title=dict(text="dB", font=dict(color="#94a3b8")),
                tickfont=dict(color="#94a3b8"),
                thickness=12,
            ),
            hovertemplate=(
                "<b>Time:</b> %{x:.2f}s<br>"
                "<b>Freq:</b> %{y:.0f} Hz<br>"
                "<b>Power:</b> %{z:.1f} dB<br>"
                "<extra></extra>"
            ),
        )
    )

    # ------------------------------------------------------------------
    # Style the figure
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
        title_text="Frequency (Hz)",
        title_font=dict(size=13, color="#64748b"),
        tickfont=dict(size=11, color="#64748b"),
        gridcolor="#1e293b",
        zeroline=False,
    )

    return fig
