# Audio Explorer 🎵🔍

**Audio Explorer** is a research-grade open-source project and learning curriculum dedicated to the fundamentals of **Computational Musicology, Digital Signal Processing (DSP), and Audio Artificial Intelligence**.

This repository is built incrementally from first principles—starting with the physics of sound and scaling up to the architectures powering state-of-the-art Generative AI systems (like MusicGen, Whisper, and CLAP).

## 🎯 Goals
- Deeply understand the mathematics and physics of audio signals.
- Extract semantic meaning (beats, tempo, pitch, timbre) using Music Information Retrieval (MIR).
- Master the preprocessing pipelines used by modern AI models (Spectrograms, MFCCs, Embeddings).
- Prepare for advanced research in Multimodal AI and Large Audio Models.

## 📁 Repository Architecture
```text
Audio-Explorer/
├── assets/          # Images, diagrams, and UI assets
├── data/            # Audio datasets (gitignored to save space)
├── docs/            # Formal API documentation
├── journal/         # Research reflections and weekly progress
├── notebooks/       # Jupyter notebooks for visual exploration
├── notes/           # Markdown summaries of theory and math
├── src/             # Core Python package (audio_explorer)
└── tests/           # Unit testing suite
```

## 🚀 Getting Started

### 1. Environment Setup
It is highly recommended to use `conda` to manage dependencies, as many DSP libraries rely on underlying C-libraries.

```bash
conda create -n audio_explorer python=3.10
conda activate audio_explorer
pip install -e .
```

### 2. Dependencies
The project relies on industry-standard tools:
- `numpy` & `scipy` (Core DSP and math)
- `librosa` (Audio analysis)
- `matplotlib` & `plotly` (Visualization)
- `streamlit` (Interactive UI)

## 📄 License
This project is licensed under the [MIT License](LICENSE).
