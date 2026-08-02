# 🎵 Audio Explorer

**A research-grade, interactive audio analysis toolkit built from first principles.**

Audio Explorer is a comprehensive audio analysis platform that transforms raw audio files into rich, interactive visualizations. Built as part of a 20-module AI + Music research curriculum, it serves as both a learning tool and a professional portfolio project.

[![Python 3.10](https://img.shields.io/badge/Python-3.10-blue.svg)](https://www.python.org/downloads/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.60-red.svg)](https://streamlit.io/)
[![Librosa](https://img.shields.io/badge/Librosa-0.11-green.svg)](https://librosa.org/)

---

## ✨ Features

| Feature | Description |
|---------|-------------|
| 📂 **Audio Loading** | Load WAV, MP3, FLAC with full metadata extraction (sample rate, duration, channels, bit depth) |
| 📈 **Waveform** | Interactive time-domain visualization with signal statistics (RMS, peak, DC offset, clipping detection) |
| 🎨 **Spectrogram** | STFT-based frequency vs time heatmap with configurable FFT parameters |
| 🧠 **Mel Spectrogram** | Perceptually-warped frequency representation — the standard input for Whisper, CLAP, and AudioMAE |
| 🎯 **MFCCs** | Mel-Frequency Cepstral Coefficients with Delta MFCCs for spectral envelope analysis |
| 🥁 **Beat Tracking** | Automatic tempo (BPM) estimation, beat detection overlay, and beat interval consistency analysis |

---

## 🚀 Quick Start

### 1. Clone the Repository

```bash
git clone https://github.com/hey-shiv/Audio-Explorer.git
cd Audio-Explorer
```

### 2. Create Environment

```bash
conda create -n audio_explorer python=3.10 -y
conda activate audio_explorer
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the App

```bash
streamlit run app.py
```

The app will open at `http://localhost:8501`. Upload any WAV, MP3, or FLAC file to begin analysis.

---

## 📁 Project Structure

```
Audio-Explorer/
├── app.py                              # Streamlit application (entry point)
├── src/
│   └── audio_explorer/
│       ├── __init__.py
│       ├── core/
│       │   ├── __init__.py
│       │   └── loader.py               # Audio file loading & metadata extraction
│       └── visualization/
│           ├── __init__.py
│           ├── waveform.py             # Time-domain waveform + signal stats
│           ├── spectrogram.py          # STFT-based spectrogram
│           ├── mel_spectrogram.py      # Mel-warped spectrogram
│           ├── mfcc.py                 # MFCC & Delta MFCC computation
│           └── rhythm.py              # Tempo estimation & beat tracking
├── data/                               # Sample audio files (not committed)
├── notebooks/                          # Jupyter notebooks for exploration
├── notes/                              # Course chapter notes
├── docs/                               # Project documentation
├── tests/                              # Unit tests
├── requirements.txt
├── pyproject.toml
└── LICENSE
```

---

## 🧠 The Audio AI Pipeline

Audio Explorer implements the exact preprocessing pipeline used by state-of-the-art audio AI systems:

```
                                        Audio Explorer Pipeline
                                        ═══════════════════════

Raw Audio File (MP3/WAV)
        │
        ▼
   librosa.load()  ──────────────────▶  Waveform (1D float array)
        │                                    │
        ▼                                    ▼
   STFT (FFT per window)  ──────────▶  Spectrogram (2D: freq × time)
        │                                    │
        ▼                                    ▼
   Mel Filterbank  ─────────────────▶  Mel Spectrogram (128 bands × time)
        │                                    │
        ▼                                    ▼
   DCT (Cosine Transform)  ─────────▶  MFCCs (13 coefficients × time)

   Beat Tracking  ──────────────────▶  Tempo (BPM) + Beat Times
```

**Real-world usage of these representations:**

| System | Organization | Input Format |
|--------|-------------|--------------|
| Whisper | OpenAI | 80-bin Mel Spectrogram |
| AudioMAE | Meta AI | 128-bin Mel Spectrogram |
| CLAP | LAION | 64-bin Mel Spectrogram |
| BEATs | Microsoft | Mel Spectrogram |
| MusicGen | Meta AI | Encoded Spectrogram |

---

## 🛠️ Tech Stack

- **Python 3.10** — Core language
- **Librosa 0.11** — Audio analysis (STFT, Mel, MFCC, beat tracking)
- **NumPy** — Numerical computing
- **SciPy** — Signal processing
- **Plotly** — Interactive visualizations
- **Streamlit** — Web application framework
- **SoundFile** — Audio file I/O

---

## 📚 Part of a Larger Curriculum

This project is the practical component of a 20-module, 75-chapter curriculum on **AI + Music**, covering:

1. **Foundations** (Modules 1–5): Physics of sound, human hearing, digital audio, DSP, mathematics
2. **Music & Features** (Modules 6–8): Music theory, MIR, feature engineering
3. **Engineering** (Modules 9–10): Software architecture, data pipelines
4. **Machine Learning** (Modules 11–14): CNNs/RNNs for audio, representation learning, CLAP
5. **Generative AI** (Modules 15–18): Audio Transformers, AudioLM, MusicGen, Diffusion
6. **Research** (Modules 19–20): Paper reproduction, research methodology

---

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.

---

## 👤 Author

**Shivashant** — B.Tech CS (AI & ML)  
Building towards AI + Music research at organizations like Google DeepMind and Suno AI.

- GitHub: [@hey-shiv](https://github.com/hey-shiv)
