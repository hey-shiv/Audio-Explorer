"""
app.py — Audio Explorer: The main Streamlit application.

Run with:
    streamlit run app.py

A research-grade audio analysis toolkit built from first principles.
Features: Waveform, Spectrogram, Mel Spectrogram, MFCC, Beat Tracking.
"""

import sys
import tempfile
import os
from pathlib import Path

import streamlit as st
import numpy as np

# ---------------------------------------------------------------------------
# Path Setup — tell Python where to find our src/audio_explorer package.
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).parent / "src"))

from audio_explorer.core.loader import load_audio, AudioFile
from audio_explorer.visualization.waveform import plot_waveform, get_waveform_stats
from audio_explorer.visualization.spectrogram import compute_spectrogram, plot_spectrogram
from audio_explorer.visualization.mel_spectrogram import compute_mel_spectrogram, plot_mel_spectrogram
from audio_explorer.visualization.mfcc import compute_mfcc, plot_mfcc
from audio_explorer.visualization.rhythm import compute_rhythm, plot_beats_on_waveform, plot_beat_intervals


# ---------------------------------------------------------------------------
# Page Configuration — MUST be the first Streamlit call.
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Audio Explorer",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ---------------------------------------------------------------------------
# Custom CSS — Premium dark theme with glassmorphism cards.
# ---------------------------------------------------------------------------
st.markdown(
    """
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

        html, body, [class*="css"] {
            font-family: 'Inter', sans-serif;
        }

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

        .badge-success {
            background-color: #064e3b;
            color: #34d399;
            padding: 0.2rem 0.75rem;
            border-radius: 999px;
            font-size: 0.8rem;
            font-weight: 600;
        }

        .section-divider {
            border: none;
            height: 1px;
            background: linear-gradient(90deg, transparent, #312e81, transparent);
            margin: 2rem 0;
        }

        hr { border-color: #1e293b; }

        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# Helper function for metadata cards
# ---------------------------------------------------------------------------
def metric_card(label: str, value: str, unit: str = "") -> str:
    return f"""
    <div class="metadata-card">
        <div class="metadata-label">{label}</div>
        <div class="metadata-value">{value}<span class="metadata-unit">{unit}</span></div>
    </div>
    """


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("## 🎵 Audio Explorer")
    st.markdown("*A research-grade audio analysis toolkit.*")
    st.markdown("---")

    st.markdown("### 📂 Load Audio")
    uploaded_file = st.file_uploader(
        label="Upload an audio file",
        type=["wav", "mp3", "flac"],
        help="Supported formats: WAV, MP3, FLAC",
    )

    st.markdown("---")
    st.markdown("**Features**")
    st.markdown("""
    ✅ Audio Loading & Metadata  
    ✅ Waveform Visualization  
    ✅ Spectrogram (STFT)  
    ✅ Mel Spectrogram  
    ✅ MFCC Visualization  
    ✅ Tempo & Beat Tracking  
    """)

    st.markdown("---")
    st.markdown(
        "Built by [Shivashant](https://github.com/hey-shiv) "
        "as part of an AI + Music research curriculum."
    )


# ---------------------------------------------------------------------------
# Main Area — Hero Header
# ---------------------------------------------------------------------------
st.markdown('<div class="hero-title">Audio Explorer 🎵</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="hero-subtitle">A research-grade audio analysis toolkit — '
    'Waveform · Spectrogram · Mel Spectrogram · MFCC · Beat Tracking</div>',
    unsafe_allow_html=True,
)
st.markdown("---")


# ---------------------------------------------------------------------------
# Main Content
# ---------------------------------------------------------------------------
if uploaded_file is None:
    st.markdown("### 👈 Upload an audio file from the sidebar to get started.")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.info("🔊 **Waveform & Signal Stats**\nTime-domain visualization with RMS, peak, clipping detection.")
    with col2:
        st.info("📊 **Spectrogram & Mel Spectrogram**\nSTFT-based frequency analysis with Mel-scale warping.")
    with col3:
        st.info("🥁 **MFCC & Beat Tracking**\nCepstral features and automatic tempo/beat detection.")

else:
    # -------------------------------------------------------------------
    # Load the audio file
    # -------------------------------------------------------------------
    suffix = Path(uploaded_file.name).suffix
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        tmp_path = tmp_file.name

    try:
        with st.spinner(f"🔄 Analyzing **{uploaded_file.name}**..."):
            audio = load_audio(tmp_path, target_sr=22050, mono=True)

        # ==============================================================
        # SECTION 1: File Info & Metadata
        # ==============================================================
        st.markdown(
            f'<span class="badge-success">✓ Loaded successfully</span>',
            unsafe_allow_html=True,
        )
        st.markdown(f"### {audio.file_name}")
        st.markdown("---")

        st.markdown("### 📋 File Metadata")

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.markdown(metric_card("Format", audio.file_format), unsafe_allow_html=True)
            st.markdown(metric_card("File Size", str(audio.file_size_mb), "MB"), unsafe_allow_html=True)

        with col2:
            st.markdown(metric_card("Duration", str(round(audio.duration, 2)), "sec"), unsafe_allow_html=True)
            st.markdown(metric_card("Channels", "Mono" if audio.num_channels == 1 else "Stereo"), unsafe_allow_html=True)

        with col3:
            st.markdown(metric_card("Sample Rate", f"{audio.sr:,}", "Hz"), unsafe_allow_html=True)
            st.markdown(metric_card("Total Samples", f"{audio.num_samples:,}"), unsafe_allow_html=True)

        with col4:
            bit_depth_val = str(audio.bit_depth) if audio.bit_depth else "N/A"
            st.markdown(metric_card("Bit Depth", bit_depth_val, "bit" if audio.bit_depth else ""), unsafe_allow_html=True)
            sig_range = f"{audio.y.min():.3f} to {audio.y.max():.3f}"
            st.markdown(metric_card("Signal Range", sig_range), unsafe_allow_html=True)

        with st.expander("🔬 Raw Signal Details"):
            st.markdown(f"**NumPy array shape:** `{audio.y.shape}` — {audio.num_samples:,} float32 values")
            st.markdown(f"**dtype:** `{audio.y.dtype}`")
            st.markdown(f"**First 10 samples:** `{audio.y[:10]}`")

        # Audio playback
        st.markdown("### ▶️ Audio Playback")
        st.audio(uploaded_file.getvalue(), format=f"audio/{suffix.lstrip('.')}")

        st.markdown("---")

        # ==============================================================
        # SECTION 2: Waveform
        # ==============================================================
        st.markdown("### 📈 Waveform — Time Domain")
        st.markdown(
            "Amplitude (air pressure) vs time. Tall peaks = loud moments. "
            "Flat sections = silence. Zoom and hover for detail."
        )

        waveform_fig = plot_waveform(audio)
        st.plotly_chart(waveform_fig, use_container_width=True)

        with st.expander("📊 Signal Statistics"):
            stats = get_waveform_stats(audio)
            s1, s2, s3, s4, s5 = st.columns(5)
            s1.metric("Peak Amplitude", stats["peak_amplitude"])
            s2.metric("RMS Amplitude", stats["rms_amplitude"])
            s3.metric("DC Offset", stats["dc_offset"])
            s4.metric("Dynamic Range", f"{stats['dynamic_range_db']} dB")
            clipped = "⚠️ YES" if stats["is_clipped"] else "✅ No"
            s5.metric("Clipping", clipped)
            if stats["is_clipped"]:
                st.warning("⚠️ Clipping detected — samples hit ±1.0, causing permanent distortion.")

        st.markdown("---")

        # ==============================================================
        # SECTION 3: Spectrogram
        # ==============================================================
        st.markdown("### 🎨 Spectrogram — Frequency vs Time")
        st.markdown(
            "The STFT converts the 1D waveform into a 2D heatmap. "
            "X = time, Y = frequency (Hz), color = energy (dB)."
        )

        spec_data = compute_spectrogram(audio, n_fft=2048, hop_length=512)
        spec_fig = plot_spectrogram(spec_data)
        st.plotly_chart(spec_fig, use_container_width=True)

        with st.expander("🔬 STFT Parameters"):
            p1, p2, p3, p4 = st.columns(4)
            p1.metric("FFT Size", f"{spec_data['n_fft']:,}")
            p2.metric("Hop Length", f"{spec_data['hop_length']:,}")
            p3.metric("Freq Resolution", f"{spec_data['freq_resolution']} Hz/bin")
            p4.metric("Time Resolution", f"{spec_data['time_resolution']} sec/frame")
            st.markdown(f"**Shape:** `{spec_data['S_db'].shape}` — {spec_data['S_db'].shape[0]} freq bins × {spec_data['S_db'].shape[1]} time frames")

        st.markdown("---")

        # ==============================================================
        # SECTION 4: Mel Spectrogram
        # ==============================================================
        st.markdown("### 🧠 Mel Spectrogram — Human Perception Scale")
        st.markdown(
            "The Mel Scale warps frequencies to match human hearing. "
            "Low frequencies are stretched, high frequencies compressed. "
            "**This is the #1 input for Whisper, CLAP, and AudioMAE.**"
        )

        mel_data = compute_mel_spectrogram(audio, n_fft=2048, hop_length=512, n_mels=128)
        mel_fig = plot_mel_spectrogram(mel_data)
        st.plotly_chart(mel_fig, use_container_width=True)

        with st.expander("🔬 Mel Spectrogram Details"):
            m1, m2, m3 = st.columns(3)
            m1.metric("Mel Bands", mel_data["n_mels"])
            m2.metric("FFT Size", f"{mel_data['n_fft']:,}")
            m3.metric("Hop Length", f"{mel_data['hop_length']:,}")
            st.markdown(
                f"**Shape:** `{mel_data['S_db'].shape}` — "
                f"compressed from {spec_data['S_db'].shape[0]} freq bins to {mel_data['S_db'].shape[0]} Mel bands (8× compression)"
            )
            st.markdown(
                f"**Mel frequency range:** {mel_data['mel_freqs'][0]:.0f} Hz → {mel_data['mel_freqs'][-1]:.0f} Hz"
            )

        st.markdown("---")

        # ==============================================================
        # SECTION 5: MFCCs
        # ==============================================================
        st.markdown("### 🎯 MFCCs — Mel-Frequency Cepstral Coefficients")
        st.markdown(
            "MFCCs compress the Mel Spectrogram into 13 coefficients per frame "
            "via the Discrete Cosine Transform (DCT). They capture the **spectral envelope** "
            "(vocal tract shape) — the classic feature for speech recognition and music classification."
        )

        mfcc_data = compute_mfcc(audio, n_mfcc=13, n_fft=2048, hop_length=512)

        # Plot MFCCs
        mfcc_fig = plot_mfcc(mfcc_data, title="MFCCs — Spectral Envelope")
        st.plotly_chart(mfcc_fig, use_container_width=True)

        # Plot Delta MFCCs
        delta_fig = plot_mfcc(
            {"mfccs": mfcc_data["delta_mfccs"], "times": mfcc_data["times"],
             "n_mfcc": mfcc_data["n_mfcc"], "n_fft": mfcc_data["n_fft"],
             "hop_length": mfcc_data["hop_length"]},
            title="Delta MFCCs — Rate of Spectral Change",
            color_scale="RdBu_r",
        )
        st.plotly_chart(delta_fig, use_container_width=True)

        with st.expander("🔬 MFCC Details"):
            c1, c2, c3 = st.columns(3)
            c1.metric("Coefficients", mfcc_data["n_mfcc"])
            c2.metric("FFT Size", f"{mfcc_data['n_fft']:,}")
            c3.metric("Hop Length", f"{mfcc_data['hop_length']:,}")
            st.markdown(
                f"**MFCC shape:** `{mfcc_data['mfccs'].shape}` — "
                f"{mfcc_data['n_mfcc']} coefficients × {mfcc_data['mfccs'].shape[1]} time frames"
            )
            st.markdown(
                f"**Delta MFCC shape:** `{mfcc_data['delta_mfccs'].shape}` — "
                f"first derivative of MFCCs (captures velocity of spectral change)"
            )
            st.markdown(
                "*MFCCs are computed by: Mel Spectrogram → log scale → "
                "DCT Type-II → keep first 13 coefficients. "
                "Coefficient 0 is overall energy, coefficients 1–12 capture the spectral envelope shape.*"
            )

        st.markdown("---")

        # ==============================================================
        # SECTION 6: Beat Tracking & Tempo
        # ==============================================================
        st.markdown("### 🥁 Tempo & Beat Tracking")
        st.markdown(
            "Automatic tempo estimation (BPM) and beat detection using "
            "librosa's onset strength envelope and dynamic programming."
        )

        with st.spinner("Detecting beats..."):
            rhythm_data = compute_rhythm(audio, hop_length=512)

        # Tempo display
        t1, t2, t3 = st.columns(3)
        t1.metric("🎵 Estimated Tempo", f"{rhythm_data['tempo']:.1f} BPM")
        t2.metric("🥁 Beats Detected", rhythm_data["beat_count"])
        if len(rhythm_data["beat_intervals"]) > 0:
            avg_interval = np.mean(rhythm_data["beat_intervals"])
            t3.metric("⏱️ Avg Beat Interval", f"{avg_interval:.3f} sec")

        # Beat overlay on waveform
        beat_waveform_fig = plot_beats_on_waveform(audio, rhythm_data)
        st.plotly_chart(beat_waveform_fig, use_container_width=True)

        # Beat interval consistency chart
        if len(rhythm_data["beat_intervals"]) > 1:
            interval_fig = plot_beat_intervals(rhythm_data)
            st.plotly_chart(interval_fig, use_container_width=True)

        with st.expander("🔬 Beat Tracking Details"):
            st.markdown(f"**Beat times (first 10):** `{rhythm_data['beat_times'][:10]}`")
            if len(rhythm_data["beat_intervals"]) > 0:
                st.markdown(f"**Interval std deviation:** `{np.std(rhythm_data['beat_intervals']):.4f}` sec")
                st.markdown(
                    "Lower standard deviation = more consistent tempo. "
                    "Electronic music: ~0.005s. Live performance: ~0.02-0.05s."
                )

        st.markdown("---")
        st.markdown(
            "🎓 **Audio Explorer** — Built as part of a 20-module AI + Music research curriculum. "
            "[View on GitHub](https://github.com/hey-shiv/Audio-Explorer)"
        )

    except Exception as e:
        st.error(f"❌ Failed to load audio file: {e}")
        import traceback
        with st.expander("Error details"):
            st.code(traceback.format_exc())

    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
