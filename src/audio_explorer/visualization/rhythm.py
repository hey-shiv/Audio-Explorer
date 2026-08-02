"""
visualization/rhythm.py

Rhythm analysis, tempo (BPM) estimation, beat tracking, and interval analysis.

WHY THIS MODULE EXISTS:
Rhythm is one of the core pillars of music and audio analysis (alongside pitch and timbre).
Understanding tempo (the speed of the musical pulse) and tracking exact beat locations
allows applications to align audio tracks, estimate danceability, detect structural changes,
and synchronize visuals or effects to music.

CONCEPTUAL BREAKDOWN:
1. WHAT IS TEMPO (BPM)?
   Tempo represents the speed or pace of a musical piece, measured in Beats Per Minute (BPM).
   A tempo of 120 BPM means there are 120 musical pulses per minute (1 beat every 0.5 seconds).
   Tempo provides a global estimate of the speed of the audio track.

2. HOW BEAT TRACKING WORKS IN LIBROSA:
   Librosa's beat tracking algorithm (`librosa.beat.beat_track`) operates in two main stages:
     a) Onset Strength Envelope:
        First, it computes an onset strength envelope (novelty function). This tracks sudden
        increases in spectral energy across multiple frequency bands (e.g., drum transients,
        plucked strings, note attacks).
     b) Dynamic Programming Beat Tracking:
        Next, a dynamic programming algorithm searches for a sequence of beat frame locations
        that maximize the onset strength values at those frames while penalizing deviations
        from a steady periodic tempo (estimated from autocorrelation or tempogram).
        This balances sticking to high-energy transients with maintaining a realistic, consistent rhythm.

3. WHY BEAT INTERVALS MATTER:
   Beat intervals (the time difference between consecutive beats: delta_t_i = t_{i+1} - t_i)
   reveal the micro-timing and consistency of the rhythm:
     - Consistent, uniform intervals indicate a metronomic or tight tempo (e.g., electronic dance
       music or pop recorded to a click track).
     - Variable intervals indicate rubato (expressive timing shifts), human playing feel, live band
       fluctuations, or accelerando/ritardando (speeding up or slowing down).
"""

import numpy as np
import plotly.graph_objects as go
import librosa

from audio_explorer.core.loader import AudioFile


def compute_rhythm(
    audio: AudioFile,
    hop_length: int = 512,
) -> dict:
    """
    Compute tempo (BPM), beat locations, beat count, and beat intervals.

    Parameters
    ----------
    audio : AudioFile
        The loaded audio object containing audio.y (signal) and audio.sr (sample rate).

    hop_length : int, optional
        Number of samples between successive STFT frames used for onset detection.
        Default is 512 samples. At sr=22050 Hz, 512 samples corresponds to
        ~23 ms time resolution per frame.

    Returns
    -------
    dict
        Dictionary containing rhythm analysis results:
        - "tempo": float, estimated global tempo in beats per minute (BPM).
        - "beat_times": np.ndarray, timestamps (in seconds) of detected beats.
        - "beat_count": int, total number of beats detected.
        - "beat_intervals": np.ndarray, duration (in seconds) between consecutive beats.
        - "hop_length": int, hop length used for calculation.
    """
    # ------------------------------------------------------------------
    # Step 1: Compute global tempo (BPM)
    #
    # librosa.feature.tempo estimates global tempo in beats per minute (BPM)
    # from the audio signal's onset envelope.
    # It returns a 1D NumPy array (e.g. array([120.0])). We extract index [0]
    # and convert to a float.
    # ------------------------------------------------------------------
    tempo_arr = librosa.feature.tempo(
        y=audio.y,
        sr=audio.sr,
        hop_length=hop_length,
    )
    tempo = float(tempo_arr[0])

    # ------------------------------------------------------------------
    # Step 2: Perform beat tracking using dynamic programming
    #
    # librosa.beat.beat_track estimates the global tempo and finds frame
    # indices of detected beats.
    # Returns a tuple: (tempo, beat_frames)
    # ------------------------------------------------------------------
    _, beat_frames = librosa.beat.beat_track(
        y=audio.y,
        sr=audio.sr,
        hop_length=hop_length,
    )

    # ------------------------------------------------------------------
    # Step 3: Convert beat frame indices to time timestamps in seconds
    #
    # frame_index -> time = frame_index * hop_length / sample_rate
    # ------------------------------------------------------------------
    beat_times = librosa.frames_to_time(
        beat_frames,
        sr=audio.sr,
        hop_length=hop_length,
    )

    # ------------------------------------------------------------------
    # Step 4: Compute beat intervals (durations between adjacent beats)
    #
    # beat_intervals[i] = beat_times[i+1] - beat_times[i]
    # If fewer than 2 beats are detected, return an empty array.
    # ------------------------------------------------------------------
    if len(beat_times) > 1:
        beat_intervals = np.diff(beat_times)
    else:
        beat_intervals = np.array([], dtype=float)

    beat_count = int(len(beat_times))

    return {
        "tempo": tempo,
        "beat_times": beat_times,
        "beat_count": beat_count,
        "beat_intervals": beat_intervals,
        "hop_length": hop_length,
    }


def plot_beats_on_waveform(
    audio: AudioFile,
    rhythm_data: dict,
    max_points: int = 10000,
) -> go.Figure:
    """
    Plot waveform with vertical lines overlaying detected beat timestamps.

    Parameters
    ----------
    audio : AudioFile
        The loaded audio object containing audio.y and audio.sr.

    rhythm_data : dict
        Dictionary output from compute_rhythm(), containing "beat_times".

    max_points : int, optional
        Maximum number of points to render on the waveform line chart.
        Default 10,000.

    Returns
    -------
    go.Figure
        Plotly Figure showing the waveform with beat lines overlay.
    """
    # ------------------------------------------------------------------
    # Step 1: Downsample waveform signal for display
    # (Same stride-based downsampling logic as waveform.py)
    # ------------------------------------------------------------------
    y = audio.y
    sr = audio.sr
    n = len(y)

    if n > max_points:
        step = n // max_points
        y_display = y[::step]
    else:
        y_display = y
        step = 1

    t_display = np.arange(len(y_display)) * (step / sr)

    # ------------------------------------------------------------------
    # Step 2: Build the Plotly Figure
    # ------------------------------------------------------------------
    fig = go.Figure()

    # Add waveform trace in indigo (#818cf8)
    fig.add_trace(
        go.Scatter(
            x=t_display,
            y=y_display,
            mode="lines",
            name="Amplitude",
            line=dict(color="#818cf8", width=1),
            hovertemplate=(
                "<b>Time:</b> %{x:.3f}s<br>"
                "<b>Amplitude:</b> %{y:.4f}<br>"
                "<extra></extra>"
            ),
        )
    )

    # ------------------------------------------------------------------
    # Step 3: Add vertical dotted red lines at beat positions
    #
    # Performance safety: limit vertical lines to max 200 beat lines
    # to avoid overwhelming browser WebGL rendering.
    # ------------------------------------------------------------------
    beat_times = rhythm_data.get("beat_times", np.array([]))
    for beat_time in beat_times[:200]:
        fig.add_vline(
            x=beat_time,
            line_dash="dot",
            line_color="#f87171",
            line_width=1,
            opacity=0.7,
        )

    # ------------------------------------------------------------------
    # Step 4: Apply dark layout theme and formatting
    # ------------------------------------------------------------------
    fig.update_layout(
        title=dict(
            text="Beat Tracking — Waveform with Detected Beats",
            font=dict(size=16, color="#e2e8f0"),
            x=0.02,
        ),
        paper_bgcolor="#0f172a",
        plot_bgcolor="#0f172a",
        font=dict(family="Inter, sans-serif", color="#94a3b8"),
        margin=dict(l=60, r=30, t=60, b=60),
        height=300,
        showlegend=False,
        hovermode="x unified",
    )

    fig.update_xaxes(
        title_text="Time (seconds)",
        title_font=dict(size=13, color="#64748b"),
        tickfont=dict(size=11, color="#64748b"),
        gridcolor="#1e293b",
        zeroline=False,
    )

    fig.update_yaxes(
        title_text="Amplitude",
        title_font=dict(size=13, color="#64748b"),
        tickfont=dict(size=11, color="#64748b"),
        gridcolor="#1e293b",
        range=[-1.05, 1.05],
        zeroline=True,
        zerolinecolor="#334155",
        zerolinewidth=1,
    )

    return fig


def plot_beat_intervals(
    rhythm_data: dict,
) -> go.Figure:
    """
    Plot beat interval durations as a bar chart for rhythmic consistency analysis.

    Parameters
    ----------
    rhythm_data : dict
        Dictionary output from compute_rhythm(), containing "beat_intervals".

    Returns
    -------
    go.Figure
        Plotly Figure showing beat interval distribution and mean interval line.
    """
    intervals = rhythm_data.get("beat_intervals", np.array([]))

    # ------------------------------------------------------------------
    # Edge case: If no beat intervals are available (0 or 1 beat detected),
    # return an empty figure with annotation.
    # ------------------------------------------------------------------
    if len(intervals) == 0:
        fig = go.Figure()
        fig.update_layout(
            title=dict(
                text="Beat Intervals — Consistency Analysis",
                font=dict(size=16, color="#e2e8f0"),
                x=0.02,
            ),
            paper_bgcolor="#0f172a",
            plot_bgcolor="#0f172a",
            font=dict(family="Inter, sans-serif", color="#94a3b8"),
            margin=dict(l=60, r=30, t=60, b=60),
            height=250,
            showlegend=False,
            annotations=[
                dict(
                    text="Not enough beats detected",
                    xref="paper",
                    yref="paper",
                    x=0.5,
                    y=0.5,
                    showarrow=False,
                    font=dict(size=14, color="#94a3b8"),
                )
            ],
        )
        fig.update_xaxes(
            title_text="Beat Index",
            title_font=dict(size=13, color="#64748b"),
            tickfont=dict(size=11, color="#64748b"),
            gridcolor="#1e293b",
            zeroline=False,
        )
        fig.update_yaxes(
            title_text="Interval (seconds)",
            title_font=dict(size=13, color="#64748b"),
            tickfont=dict(size=11, color="#64748b"),
            gridcolor="#1e293b",
            zeroline=False,
        )
        return fig

    # ------------------------------------------------------------------
    # Step 1: Create bar chart of beat interval durations
    # ------------------------------------------------------------------
    beat_indices = np.arange(len(intervals))
    mean_interval = float(np.mean(intervals))

    fig = go.Figure()

    fig.add_trace(
        go.Bar(
            x=beat_indices,
            y=intervals,
            name="Beat Interval",
            marker=dict(color="#34d399"),  # Emerald green bar color
            hovertemplate=(
                "<b>Beat Index:</b> %{x}<br>"
                "<b>Interval:</b> %{y:.3f}s<br>"
                "<extra></extra>"
            ),
        )
    )

    # ------------------------------------------------------------------
    # Step 2: Add horizontal line at mean interval duration
    # ------------------------------------------------------------------
    fig.add_hline(
        y=mean_interval,
        line_dash="dash",
        line_color="#f59e0b",
        line_width=1.5,
        annotation_text=f"Mean: {mean_interval:.3f}s",
        annotation_position="top right",
        annotation_font=dict(size=11, color="#f59e0b"),
    )

    # ------------------------------------------------------------------
    # Step 3: Apply dark layout theme and formatting
    # ------------------------------------------------------------------
    fig.update_layout(
        title=dict(
            text="Beat Intervals — Consistency Analysis",
            font=dict(size=16, color="#e2e8f0"),
            x=0.02,
        ),
        paper_bgcolor="#0f172a",
        plot_bgcolor="#0f172a",
        font=dict(family="Inter, sans-serif", color="#94a3b8"),
        margin=dict(l=60, r=30, t=60, b=60),
        height=250,
        showlegend=False,
    )

    fig.update_xaxes(
        title_text="Beat Index",
        title_font=dict(size=13, color="#64748b"),
        tickfont=dict(size=11, color="#64748b"),
        gridcolor="#1e293b",
        zeroline=False,
    )

    fig.update_yaxes(
        title_text="Interval (seconds)",
        title_font=dict(size=13, color="#64748b"),
        tickfont=dict(size=11, color="#64748b"),
        gridcolor="#1e293b",
        zeroline=False,
    )

    return fig
