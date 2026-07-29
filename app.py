"""
app.py — The main Streamlit application for Audio Explorer.

Run with:
    streamlit run app.py

This is the entry point for the entire interactive interface.
We build features incrementally here — Day by Day throughout the sprint.

Day 1: Audio loading + metadata display
Day 2: Waveform visualization  ✅
Day 3: Spectrogram  ✅
Day 4: Mel Spectrogram (coming soon)
Day 5: MFCC (coming soon)
"""

import sys
from pathlib import Path

import streamlit as st

# ---------------------------------------------------------------------------
# Path Setup
#
# Because our code lives in src/audio_explorer/, and we're running app.py
# from the root of the project, we need to tell Python where to find it.
# sys.path.insert(0, "src") adds the src/ folder to Python's module search path.
# This is a standard pattern for the "src layout" in Python projects.
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).parent / "src"))

from audio_explorer.core.loader import load_audio, AudioFile
from audio_explorer.visualization.waveform import plot_waveform, get_waveform_stats
from audio_explorer.visualization.spectrogram import compute_spectrogram, plot_spectrogram


# ---------------------------------------------------------------------------
# Page Configuration
#
# st.set_page_config() MUST be the first Streamlit call in the script.
# - page_title: The text shown in the browser tab.
# - page_icon:  The emoji or image shown in the browser tab.
# - layout:     "wide" uses the full browser width instead of a narrow column.
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Audio Explorer",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ---------------------------------------------------------------------------
# Custom CSS Styling
#
# We inject CSS directly into the Streamlit page using st.markdown with
# unsafe_allow_html=True. This lets us override Streamlit's default styles
# and give the app a premium, dark, research-inspired look.
# ---------------------------------------------------------------------------
st.markdown(
    """
    <style>
        /* Import a clean, modern font from Google Fonts */
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

        /* Apply the font globally */
        html, body, [class*="css"] {
            font-family: 'Inter', sans-serif;
        }

        /* Hero header styling */
        .hero-title {
            font-size: 3rem;
            font-weight: 700;
            background: linear-gradient(135deg, #a78bfa, #60a5fa, #34d399);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 0.2rem;
        }

        .hero-subtitle {
            font-size: 1.1rem;
            color: #94a3b8;
            margin-bottom: 2rem;
        }

        /* Metadata card styling */
        .metadata-card {
            background: linear-gradient(135deg, #1e1b4b 0%, #0f172a 100%);
            border: 1px solid #312e81;
            border-radius: 12px;
            padding: 1.2rem 1.5rem;
            margin-bottom: 1rem;
        }

        .metadata-label {
            font-size: 0.75rem;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            color: #818cf8;
            margin-bottom: 0.25rem;
        }

        .metadata-value {
            font-size: 1.4rem;
            font-weight: 700;
            color: #e2e8f0;
        }

        .metadata-unit {
            font-size: 0.85rem;
            color: #64748b;
            margin-left: 0.2rem;
        }

        /* Status badge */
        .badge-success {
            background-color: #064e3b;
            color: #34d399;
            padding: 0.2rem 0.75rem;
            border-radius: 999px;
            font-size: 0.8rem;
            font-weight: 600;
        }

        /* Divider */
        hr { border-color: #1e293b; }

        /* Hide Streamlit branding */
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("## 🎵 Audio Explorer")
    st.markdown("*A research-grade audio analysis toolkit.*")
    st.markdown("---")

    st.markdown("### 📂 Load Audio")

    # st.file_uploader allows the user to drag-and-drop or browse for a file.
    # The 'type' parameter restricts accepted file types.
    # The uploaded object is a file-like object we can pass to our loader.
    uploaded_file = st.file_uploader(
        label="Upload an audio file",
        type=["wav", "mp3", "flac"],
        help="Supported formats: WAV, MP3, FLAC",
    )

    st.markdown("---")
    st.markdown("**Sprint Mode — Day 3**")
    st.markdown("""
    ✅ Audio loading  
    ✅ Metadata display  
    ✅ Waveform  
    ✅ Spectrogram  
    ⏳ Mel Spectrogram (Day 4)  
    ⏳ MFCC (Day 5)  
    ⏳ Beat Tracking (Day 6)  
    """)


# ---------------------------------------------------------------------------
# Main Area — Hero Header
# ---------------------------------------------------------------------------
st.markdown('<div class="hero-title">Audio Explorer 🎵</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="hero-subtitle">Foundations of Computational Musicology — '
    'a research-grade audio analysis toolkit built from first principles.</div>',
    unsafe_allow_html=True,
)
st.markdown("---")


# ---------------------------------------------------------------------------
# Main Area — Content
# ---------------------------------------------------------------------------

if uploaded_file is None:
    # Landing state: No file uploaded yet. Show a clear call-to-action.
    st.markdown("### 👈 Upload an audio file from the sidebar to get started.")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.info("🔊 **Waveform**\nTime-domain visualization of the raw audio signal.")
    with col2:
        st.info("📊 **Spectrogram**\nFrequency content across time. The language of audio AI.")
    with col3:
        st.info("🥁 **Beat Tracking**\nAutomatic tempo and beat detection using librosa.")

else:
    # ---------------------------------------------------------------------------
    # File has been uploaded. Save it to a temp location and load it.
    #
    # Streamlit's uploaded_file is a BytesIO object (in-memory bytes).
    # librosa.load() needs an actual file path, so we write it to a temp file.
    # We use Python's built-in tempfile module for this.
    # ---------------------------------------------------------------------------

    import tempfile
    import os

    # Save the uploaded bytes to a temporary file on disk
    # NamedTemporaryFile creates a file with a random name.
    # delete=False means the file persists after the with block closes —
    # necessary so librosa can read it.
    # suffix preserves the original file extension so librosa detects the format.
    suffix = Path(uploaded_file.name).suffix
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
        tmp_file.write(uploaded_file.getvalue())  # Write the raw bytes
        tmp_path = tmp_file.name                  # Remember the path

    try:
        # Load audio using our custom loader module
        with st.spinner(f"Loading **{uploaded_file.name}**..."):
            audio = load_audio(tmp_path, target_sr=22050, mono=True)

        # ------------------------------------------------------------------
        # Success Banner
        # ------------------------------------------------------------------
        st.markdown(
            f'<span class="badge-success">✓ Loaded successfully</span>',
            unsafe_allow_html=True,
        )
        st.markdown(f"### {audio.file_name}")
        st.markdown("---")

        # ------------------------------------------------------------------
        # Metadata Section
        # Build a 4-column grid of metadata cards.
        # ------------------------------------------------------------------
        st.markdown("### 📋 File Metadata")

        col1, col2, col3, col4 = st.columns(4)

        def metric_card(label: str, value: str, unit: str = "") -> str:
            """Generate HTML for a styled metadata card."""
            return f"""
            <div class="metadata-card">
                <div class="metadata-label">{label}</div>
                <div class="metadata-value">{value}<span class="metadata-unit">{unit}</span></div>
            </div>
            """

        with col1:
            st.markdown(
                metric_card("Format", audio.file_format),
                unsafe_allow_html=True,
            )
            st.markdown(
                metric_card("File Size", str(audio.file_size_mb), "MB"),
                unsafe_allow_html=True,
            )

        with col2:
            st.markdown(
                metric_card("Duration", str(round(audio.duration, 2)), "sec"),
                unsafe_allow_html=True,
            )
            st.markdown(
                metric_card("Channels", "Mono" if audio.num_channels == 1 else "Stereo"),
                unsafe_allow_html=True,
            )

        with col3:
            st.markdown(
                metric_card("Sample Rate", f"{audio.sr:,}", "Hz"),
                unsafe_allow_html=True,
            )
            st.markdown(
                metric_card("Total Samples", f"{audio.num_samples:,}"),
                unsafe_allow_html=True,
            )

        with col4:
            bit_depth_val = str(audio.bit_depth) if audio.bit_depth else "N/A"
            st.markdown(
                metric_card("Bit Depth", bit_depth_val, "bit" if audio.bit_depth else ""),
                unsafe_allow_html=True,
            )
            # Show the signal range (min/max amplitude values)
            sig_range = f"{audio.y.min():.3f} to {audio.y.max():.3f}"
            st.markdown(
                metric_card("Signal Range", sig_range),
                unsafe_allow_html=True,
            )

        st.markdown("---")

        # ------------------------------------------------------------------
        # Technical Details Expander
        # Provides a deeper view for advanced users / debugging.
        # ------------------------------------------------------------------
        with st.expander("🔬 Technical Details"):
            st.markdown(f"**Full path:** `{audio.file_path}`")
            st.markdown(f"**NumPy array shape:** `{audio.y.shape}` (1D array of {audio.num_samples:,} float32 values)")
            st.markdown(f"**NumPy dtype:** `{audio.y.dtype}`")
            st.markdown(f"**First 10 samples:** `{audio.y[:10]}`")
            st.markdown(
                "*These raw float32 values ARE your audio signal. "
                "Every spectrogram, MFCC, and beat detection algorithm "
                "in this project is just mathematics applied to this array.*"
            )

        # ------------------------------------------------------------------
        # Audio Playback
        # st.audio() renders a native HTML5 audio player in the browser.
        # We pass the raw bytes from the uploaded file.
        # ------------------------------------------------------------------
        st.markdown("### ▶️ Audio Playback")
        st.audio(uploaded_file.getvalue(), format=f"audio/{suffix.lstrip('.')}")

        st.markdown("---")

        # ------------------------------------------------------------------
        # Day 2: Waveform Visualization
        #
        # We call plot_waveform() from our visualization module.
        # It returns a Plotly Figure object.
        # st.plotly_chart() renders it as an interactive chart in the browser.
        #   - use_container_width=True makes the chart stretch to fill the
        #     full width of the Streamlit column.
        # ------------------------------------------------------------------
        st.markdown("### 📈 Waveform — Time Domain")

        st.markdown(
            "The waveform shows **amplitude (air pressure)** on the Y-axis "
            "vs **time** on the X-axis. Each point is one decoded PCM sample "
            "from the `y` array. You can zoom, pan, and hover over any point."
        )

        # Render the interactive waveform chart
        waveform_fig = plot_waveform(audio)
        st.plotly_chart(waveform_fig, use_container_width=True)

        # ------------------------------------------------------------------
        # Signal Statistics Section
        #
        # get_waveform_stats() returns a dict of signal health metrics.
        # We display them in a 5-column row.
        # ------------------------------------------------------------------
        with st.expander("📊 Signal Statistics"):
            stats = get_waveform_stats(audio)

            s1, s2, s3, s4, s5 = st.columns(5)

            s1.metric(
                label="Peak Amplitude",
                value=stats["peak_amplitude"],
                help="Maximum absolute value in the signal. Should be ≤ 1.0.",
            )
            s2.metric(
                label="RMS Amplitude",
                value=stats["rms_amplitude"],
                help="Root Mean Square — a measure of average loudness.",
            )
            s3.metric(
                label="DC Offset",
                value=stats["dc_offset"],
                help="Mean of the signal. Ideally 0.0. Non-zero indicates a bias.",
            )
            s4.metric(
                label="Dynamic Range",
                value=f"{stats['dynamic_range_db']} dB",
                help="Difference between peak and RMS in decibels.",
            )

            clipped_label = "⚠️ YES — Data Lost!" if stats["is_clipped"] else "✅ No Clipping"
            s5.metric(
                label="Clipping Detected",
                value=clipped_label,
                help="Clipping means samples hit ±1.0, causing permanent distortion.",
            )

            if stats["is_clipped"]:
                st.warning(
                    "⚠️ **Clipping detected.** One or more samples reached ±1.0. "
                    "This means the original recording was too loud for the microphone, "
                    "causing permanent information loss. This audio may sound distorted."
                )

        st.markdown("---")

        # ==================================================================
        # Day 3: Spectrogram — Frequency vs Time
        #
        # This is the most important visualization in all of audio AI.
        # We convert the 1D waveform into a 2D heatmap showing which
        # frequencies are active at every moment in the audio.
        #
        # PIPELINE:
        #   y (1D) → STFT → |magnitude| → dB → Plotly Heatmap
        # ==================================================================
        st.markdown("### 🎨 Spectrogram — Frequency vs Time")

        st.markdown(
            "The spectrogram shows **frequency (Hz)** on the Y-axis vs "
            "**time (seconds)** on the X-axis. Brighter colors = more energy. "
            "This is the exact 2D representation that Whisper, AudioMAE, "
            "and CLAP use as input."
        )

        # Compute the spectrogram using our module
        spec_data = compute_spectrogram(audio, n_fft=2048, hop_length=512)

        # Render the interactive heatmap
        spec_fig = plot_spectrogram(spec_data)
        st.plotly_chart(spec_fig, use_container_width=True)

        # ------------------------------------------------------------------
        # STFT Technical Details Expander
        #
        # This section exposes the STFT parameters and computed metadata
        # so the user understands what happened mathematically.
        # ------------------------------------------------------------------
        with st.expander("🔬 STFT Parameters & Technical Details"):

            p1, p2, p3, p4 = st.columns(4)

            p1.metric(
                label="FFT Size (n_fft)",
                value=f"{spec_data['n_fft']:,}",
                help=(
                    "Number of samples per analysis window. "
                    "Controls frequency resolution: more samples = "
                    "finer frequency detail, but coarser time detail."
                ),
            )
            p2.metric(
                label="Hop Length",
                value=f"{spec_data['hop_length']:,}",
                help=(
                    "How far the window slides between frames. "
                    "Smaller hop = more overlap = smoother time resolution."
                ),
            )
            p3.metric(
                label="Freq Resolution",
                value=f"{spec_data['freq_resolution']} Hz/bin",
                help="Each row of the spectrogram spans this many Hz.",
            )
            p4.metric(
                label="Time Resolution",
                value=f"{spec_data['time_resolution']} sec/frame",
                help="Each column of the spectrogram spans this many seconds.",
            )

            st.markdown("---")

            st.markdown(
                f"**Spectrogram shape:** `{spec_data['S_db'].shape}` — "
                f"`{spec_data['S_db'].shape[0]}` frequency bins × "
                f"`{spec_data['S_db'].shape[1]}` time frames"
            )
            st.markdown(
                f"**Frequency range:** 0 Hz → {spec_data['freqs'][-1]:,.0f} Hz "
                f"(Nyquist limit = sample rate / 2 = {audio.sr} / 2)"
            )
            st.markdown(
                f"**dB range:** {spec_data['S_db'].min():.1f} dB (quietest) → "
                f"{spec_data['S_db'].max():.1f} dB (loudest)"
            )
            st.markdown(
                "*Each cell in the matrix tells you how much energy (in dB) "
                "exists at that specific frequency and time. This 2D matrix "
                "is the primary input format for modern audio neural networks.*"
            )

        st.markdown("---")
        st.markdown(
            "📌 **Next:** Mel Spectrogram comes in **Day 4**. "
            "We will warp the frequency axis to match human hearing perception — "
            "the Mel Scale — making the spectrogram even more useful for AI."
        )

    except Exception as e:
        # Friendly error display — never show raw tracebacks to users.
        st.error(f"❌ Failed to load audio file: {e}")
        st.markdown("**Possible causes:**")
        st.markdown("- File is corrupted or not a real audio file.")
        st.markdown("- File format is not supported (try WAV or MP3).")

    finally:
        # Always clean up the temp file, even if an error occurred.
        # This prevents leftover temp files from accumulating on disk.
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
