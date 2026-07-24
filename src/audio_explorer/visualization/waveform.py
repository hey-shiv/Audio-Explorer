"""
visualization/waveform.py

Generates interactive, publication-quality waveform (time-domain)
visualizations from a loaded AudioFile.

WHY THIS MODULE EXISTS:
The waveform is the most fundamental visual representation of audio.
It is the first thing every audio engineer, researcher, and AI system
checks when inspecting a new audio file.

WHAT IT DOES:
- Takes a raw NumPy audio array (y) and sample rate (sr)
- Converts sample indices into actual time values (seconds)
- Downsamples intelligently for browser performance
- Returns a styled Plotly Figure ready for st.plotly_chart()

PLOTLY vs MATPLOTLIB:
We use Plotly here (not Matplotlib) because Plotly charts are:
  - Interactive: you can zoom, pan, and hover in the browser
  - WebGL-accelerated: handles large arrays smoothly
  - Embeddable in Streamlit with a single function call
Matplotlib produces static images — good for papers, not for UIs.
"""

import numpy as np
import plotly.graph_objects as go

from audio_explorer.core.loader import AudioFile


def plot_waveform(
    audio: AudioFile,
    max_points: int = 10_000,
    color: str = "#818cf8",
    title: str = "Waveform — Time Domain",
) -> go.Figure:
    """
    Generate an interactive waveform plot from an AudioFile.

    The waveform plots amplitude (air pressure) on the Y-axis
    against time (seconds) on the X-axis.

    Parameters
    ----------
    audio : AudioFile
        The loaded audio object from loader.py.
        We use audio.y (the signal array) and audio.sr (sample rate).

    max_points : int, optional
        Maximum number of data points to render in the browser.
        Default 10,000. Above this, your browser will lag.
        Below ~5,000, the waveform starts to look too blocky.
        10,000 is the sweet spot for quality vs. performance.

    color : str, optional
        The line color in hex or CSS color format.
        Default is a soft indigo (#818cf8) matching the app theme.

    title : str, optional
        The chart title displayed at the top.

    Returns
    -------
    go.Figure
        A Plotly Figure object. Pass this directly to st.plotly_chart().
    """

    # ------------------------------------------------------------------
    # Step 1: Downsample the signal for display
    #
    # audio.y is a 1D NumPy array of shape (num_samples,).
    # For a 3.5-minute song at 22050 Hz, this is ~4.6 million values.
    # We cannot render 4.6 million points in a browser chart.
    #
    # STRATEGY: Stride-based downsampling.
    # Take every Nth sample where N = total_samples // max_points.
    # This preserves the visual shape of the waveform perfectly
    # because adjacent samples at 22050 Hz are virtually identical.
    # ------------------------------------------------------------------

    y = audio.y           # The raw audio signal array
    sr = audio.sr         # The sample rate (e.g., 22050)
    n = len(y)            # Total number of samples

    if n > max_points:
        # Calculate the step size
        # e.g., 4,630,500 samples // 10,000 points = step of 463
        # So we take samples at indices: 0, 463, 926, 1389, ...
        step = n // max_points

        # NumPy slice notation: array[start:stop:step]
        # y[::step] means: from start to end, take every `step`-th element
        y_display = y[::step]
    else:
        # If the file is short (fewer samples than max_points),
        # no downsampling needed — use the full array as-is.
        y_display = y
        step = 1

    # ------------------------------------------------------------------
    # Step 2: Build the time axis
    #
    # y_display has len(y_display) samples.
    # Each sample is `step` samples apart in the original signal.
    # Each original sample represents (1 / sr) seconds.
    # Therefore, each display sample represents (step / sr) seconds.
    #
    # np.arange(len(y_display)) creates: [0, 1, 2, 3, ..., n-1]
    # Multiply by (step / sr) to convert sample indices to seconds.
    #
    # Example:
    #   sr = 22050, step = 463, 10,000 display points
    #   t_display[0] = 0 * (463/22050) = 0.0 seconds
    #   t_display[1] = 1 * (463/22050) = 0.021 seconds
    #   t_display[9999] = 9999 * (463/22050) ≈ 210.0 seconds (end of 3.5 min)
    # ------------------------------------------------------------------

    t_display = np.arange(len(y_display)) * (step / sr)

    # ------------------------------------------------------------------
    # Step 3: Build the Plotly Figure
    #
    # Plotly works in layers. You create a Figure, then add "traces"
    # (data series) to it. Here we add one trace: a line chart.
    #
    # go.Figure() creates an empty canvas.
    # go.Scatter() defines a line chart trace.
    # fig.update_layout() customizes the overall figure appearance.
    # fig.update_xaxes() / fig.update_yaxes() customizes each axis.
    # ------------------------------------------------------------------

    fig = go.Figure()

    # Add the waveform trace
    fig.add_trace(
        go.Scatter(
            x=t_display,           # X-axis: time in seconds
            y=y_display,           # Y-axis: amplitude values (-1.0 to +1.0)
            mode="lines",          # Draw as a continuous line (not dots)
            name="Amplitude",      # Label in the legend and hover tooltip

            line=dict(
                color=color,       # Line color
                width=1,           # Thin line looks cleaner for dense data
            ),

            # Hover tooltip text when user hovers over the chart.
            # %{x:.3f} formats the x value (time) to 3 decimal places.
            # %{y:.4f} formats the y value (amplitude) to 4 decimal places.
            hovertemplate=(
                "<b>Time:</b> %{x:.3f}s<br>"
                "<b>Amplitude:</b> %{y:.4f}<br>"
                "<extra></extra>"  # Hides the trace name from tooltip
            ),
        )
    )

    # ------------------------------------------------------------------
    # Step 4: Style the figure
    #
    # We use a dark theme to match the app's overall design.
    # 'paper_bgcolor': the background color of the whole figure canvas.
    # 'plot_bgcolor':  the background color of the chart area only.
    # 'font':          global font settings for all text in the chart.
    # ------------------------------------------------------------------

    fig.update_layout(
        title=dict(
            text=title,
            font=dict(size=16, color="#e2e8f0"),
            x=0.02,                # Align title slightly to the left
        ),
        paper_bgcolor="#0f172a",   # Dark navy — matches app background
        plot_bgcolor="#0f172a",    # Same dark background for the chart area
        font=dict(
            family="Inter, sans-serif",
            color="#94a3b8",       # Soft grey for axis labels
        ),
        margin=dict(l=60, r=30, t=60, b=60),  # Padding around the chart
        height=300,                # Fixed height in pixels
        showlegend=False,          # Hide the legend (one trace, no need)
        hovermode="x unified",     # Show all trace values at the hovered x position
    )

    # Customize the X-axis (Time)
    fig.update_xaxes(
        title_text="Time (seconds)",
        title_font=dict(size=13, color="#64748b"),
        tickfont=dict(size=11, color="#64748b"),
        gridcolor="#1e293b",       # Subtle dark grid lines
        zeroline=False,            # Don't draw a bold line at x=0
        showspikes=True,           # Show a vertical line on hover
        spikecolor="#475569",
        spikethickness=1,
        spikedash="dot",
    )

    # Customize the Y-axis (Amplitude)
    fig.update_yaxes(
        title_text="Amplitude",
        title_font=dict(size=13, color="#64748b"),
        tickfont=dict(size=11, color="#64748b"),
        gridcolor="#1e293b",
        range=[-1.05, 1.05],       # Fix Y range so chart does not jump around
        zeroline=True,             # Draw a subtle line at y=0 (silence baseline)
        zerolinecolor="#334155",
        zerolinewidth=1,
    )

    return fig


def get_waveform_stats(audio: AudioFile) -> dict:
    """
    Compute descriptive statistics about the audio signal.

    These are simple but important numbers that every audio engineer
    checks when auditing a new file.

    Parameters
    ----------
    audio : AudioFile

    Returns
    -------
    dict with keys:
        peak_amplitude  : Maximum absolute amplitude (should be ≤ 1.0)
        rms_amplitude   : Root Mean Square amplitude — a better measure
                          of "loudness" than peak. Reflects average energy.
        dc_offset       : The mean of the signal. Should be ~0.0.
                          A non-zero DC offset means the signal is shifted
                          up or down, which causes problems in processing.
        is_clipped      : True if any sample reached exactly ±1.0.
                          Clipping means the microphone was too loud and
                          information was permanently lost.
        dynamic_range_db: Difference between peak and RMS in decibels.
                          Higher = more dynamic content. Lower = compressed.
    """

    y = audio.y

    peak = float(np.max(np.abs(y)))

    # RMS: Root Mean Square
    # Step 1: Square every sample:   y²
    # Step 2: Take the mean:         mean(y²)
    # Step 3: Take the square root:  sqrt(mean(y²))
    # This gives a single number representing the "average energy" of the signal.
    rms = float(np.sqrt(np.mean(y ** 2)))

    dc_offset = float(np.mean(y))

    # A sample is "clipped" if it exactly equals ±1.0.
    # In practice, we check if it's within floating-point tolerance of 1.0.
    is_clipped = bool(np.any(np.abs(y) >= 0.999))

    # Dynamic range in dB:
    # 20 * log10(peak / rms) converts the amplitude ratio to decibels.
    # We add a small epsilon (1e-10) to avoid log10(0) = -infinity.
    dynamic_range_db = float(
        20 * np.log10((peak + 1e-10) / (rms + 1e-10))
    )

    return {
        "peak_amplitude": round(peak, 4),
        "rms_amplitude": round(rms, 4),
        "dc_offset": round(dc_offset, 6),
        "is_clipped": is_clipped,
        "dynamic_range_db": round(dynamic_range_db, 2),
    }
