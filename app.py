"""
app.py — The main Streamlit application for Audio Explorer.

Run with:
    streamlit run app.py

This is the entry point for the entire interactive interface.
We build features incrementally here — Day by Day throughout the sprint.

Day 1: Audio loading + metadata display
Day 2: Waveform visualization (coming soon)
Day 3: Spectrogram (coming soon)
...
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
    st.markdown("**Sprint Mode — Day 1**")
    st.markdown("""
    ✅ Audio loading  
    ✅ Metadata display  
    ⏳ Waveform (Day 2)  
    ⏳ Spectrogram (Day 3)  
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
        st.markdown(
            "📌 **Next:** Waveform visualization comes in **Day 2**. "
            "The raw signal array above will be plotted as an interactive time-domain chart."
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
