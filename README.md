<p align="center">
  <img src=".github/banner.jpg" alt="Audio Explorer" width="100%">
</p>

<h1 align="center">Audio Explorer</h1>

<p align="center">
  <em>The complete audio AI preprocessing pipeline — from raw samples to spectrograms, MFCCs, and beat detection.</em>
</p>

<p align="center">
  <code>Waveform</code> · <code>STFT</code> · <code>Mel Spectrogram</code> · <code>MFCC</code> · <code>Beat Tracking</code>
</p>

<br>

---

<br>

> Whisper, Spotify, Shazam, MusicGen — they all convert audio into the same mathematical representation before doing anything intelligent.  
> This project implements that representation from scratch.

<br>

## Quick start

```bash
git clone https://github.com/hey-shiv/Audio-Explorer.git && cd Audio-Explorer
pip install -r requirements.txt
streamlit run app.py
```

Opens at `localhost:8501`. Drop a WAV, MP3, or FLAC.

<br>

---

<br>

## The pipeline

Every major audio AI system preprocesses sound through the same chain of transformations. Audio Explorer implements each stage as an interactive visualization you can zoom, pan, and inspect sample-by-sample.

<br>

<table>
  <tr>
    <td width="160"><strong>Waveform</strong></td>
    <td>Decode audio into a 1D float array. 22,050 samples per second of audio. Compute signal health — RMS loudness, peak amplitude, DC offset, clipping detection, dynamic range in dB.</td>
  </tr>
  <tr>
    <td><strong>Spectrogram</strong></td>
    <td>Slide a 2048-sample Hann window with 512-sample hops. FFT each window. Stack magnitude spectra into a time-frequency matrix. 1025 frequency bins, one column per 23ms. Convert to dB.</td>
  </tr>
  <tr>
    <td><strong>Mel Spectrogram</strong></td>
    <td>Human hearing is logarithmic — 200→400 Hz is an octave, 8000→8200 Hz is nothing. Apply 128 triangular Mel filters to warp the spectrogram to match perception. 8x compression, zero perceptual loss. This is what Whisper sees.</td>
  </tr>
  <tr>
    <td><strong>MFCCs</strong></td>
    <td>DCT on the log-mel spectrum. 13 coefficients per frame. Coefficient 0 = energy. Coefficients 1–12 = spectral envelope (vocal tract shape). Delta MFCCs capture rate of change. The standard speech recognition feature since 1980.</td>
  </tr>
  <tr>
    <td><strong>Beat Tracking</strong></td>
    <td>Onset strength envelope → autocorrelation → dynamic programming. Outputs tempo in BPM + individual beat timestamps. Beat interval analysis reveals metronomic vs. human timing.</td>
  </tr>
</table>

<br>

---

<br>

## The math behind it

Four ideas, two centuries:

| Year | Idea | What it does |
|------|------|-------------|
| 1807 | Fourier Transform | Any signal = sum of sine waves. Find them. |
| 1937 | Mel Scale | `m = 2595 · log₁₀(1 + f/700)` — warp frequency to match human ears. |
| 1965 | FFT (Cooley-Tukey) | Compute DFT in O(N log N) instead of O(N²). 200,000x speedup. |
| 1980 | MFCCs | DCT on log-mel spectrum → 13 numbers that describe the spectral shape. |

<br>

---

<br>

## Who uses this pipeline

Every major audio AI system takes a mel spectrogram as input. The model architectures differ. The preprocessing is the same.

| System | Organization | Input |
|--------|-------------|-------|
| Whisper | OpenAI | 80-band mel spectrogram |
| AudioMAE | Meta | 128-band mel spectrogram |
| CLAP | LAION | 64-band mel spectrogram |
| BEATs | Microsoft | mel spectrogram |
| MusicGen | Meta | encoded spectrogram |

<br>

---

<br>

## Code

```
app.py                              entry point
src/audio_explorer/
    core/loader.py                  decode audio, extract metadata
    visualization/
        waveform.py                 amplitude vs time + signal stats
        spectrogram.py              STFT → frequency vs time
        mel_spectrogram.py          mel filterbank warping
        mfcc.py                     cepstral coefficients + deltas
        rhythm.py                   tempo estimation + beat detection
```

Each module: `compute_*()` returns arrays, `plot_*()` returns a Plotly figure. Computation and rendering fully separated.

**Stack:** Python 3.10 · Librosa · NumPy · SciPy · Plotly · Streamlit

<br>

---

<br>

<p align="center">
  MIT License · Built by <a href="https://github.com/hey-shiv">Shivashant</a>
</p>
