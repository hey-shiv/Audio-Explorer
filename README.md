# Audio Explorer

An interactive toolkit that implements the complete audio preprocessing pipeline used by Whisper, CLAP, AudioMAE, and every major audio AI system — from raw PCM samples to MFCCs and beat detection.

Drop any audio file. Get waveform analysis, spectrograms, mel spectrograms, cepstral coefficients, and tempo estimation. Everything runs locally in the browser.

```
git clone https://github.com/hey-shiv/Audio-Explorer.git
cd Audio-Explorer
pip install -r requirements.txt
streamlit run app.py
```

---

## What this does

Audio Explorer processes audio through six analysis stages, each building on the last:

```
                          ┌─────────────────────────────────────────────┐
                          │           THE AUDIO AI PIPELINE             │
                          └─────────────────────────────────────────────┘

  MP3 / WAV / FLAC
       │
       ├── librosa.load() ──────────── 1D float array (amplitude vs time)
       │                                     │
       ├── STFT ────────────────────── 2D matrix (1025 freq bins × time)
       │                                     │
       ├── Mel filterbank ──────────── 2D matrix (128 mel bands × time)
       │                                     │
       ├── DCT ─────────────────────── 2D matrix (13 coefficients × time)
       │
       └── onset detection ─────────── tempo (BPM) + beat positions
```

**Stage 1 — Waveform.** Decode compressed audio into a float32 array. Compute signal statistics: RMS, peak amplitude, DC offset, clipping detection, dynamic range.

**Stage 2 — Spectrogram.** Apply the Short-Time Fourier Transform. Slide a 2048-sample Hann window across the signal with a 512-sample hop, compute the FFT on each chunk, stack the magnitude spectra into a time-frequency matrix. Convert to decibels.

**Stage 3 — Mel spectrogram.** Warp the linear frequency axis to match human hearing using 128 triangular Mel filters. This is the exact input format used by Whisper (80 bands), AudioMAE (128), CLAP (64), and BEATs.

**Stage 4 — MFCCs.** Apply a Discrete Cosine Transform to the log-mel spectrogram and keep the first 13 coefficients. These capture the spectral envelope — the shape of the vocal tract — and have been the standard feature for speech recognition since 1980. Delta MFCCs capture the rate of spectral change.

**Stage 5 — Beat tracking.** Compute an onset strength envelope, estimate tempo via autocorrelation, and track individual beat positions using dynamic programming. Overlay beats on the waveform. Analyze beat interval consistency.

Every chart is interactive — zoom, pan, hover to inspect individual samples and frequency bins.

---

## How the code is organized

```
app.py                                     Streamlit entry point
src/audio_explorer/
    core/
        loader.py                          audio decoding + metadata
    visualization/
        waveform.py                        time-domain + signal stats
        spectrogram.py                     STFT computation + rendering
        mel_spectrogram.py                 mel filterbank + rendering
        mfcc.py                            cepstral coefficients + deltas
        rhythm.py                          tempo + beat detection
generate_animation.py                      15s pipeline animation script
```

Each module follows the same pattern: a `compute_*` function that returns a dict of arrays, and a `plot_*` function that returns a Plotly figure. Computation and rendering are fully separated.

---

## The math

The pipeline is built on four ideas spanning two centuries:

**Fourier Transform (1807).** Any signal decomposes into a sum of sine waves. The DFT finds the amplitude of each frequency component:

```
X[k] = Σ x[n] · e^(-j·2π·k·n/N)     for k = 0, 1, ..., N-1
```

**Fast Fourier Transform (1965).** Cooley-Tukey algorithm computes the DFT in O(N log N) instead of O(N²). For a 3-minute song at 22,050 Hz: ~200,000x speedup.

**Mel Scale (1937).** Human hearing is logarithmic. The jump from 200 Hz to 400 Hz sounds like an octave. The jump from 8,000 Hz to 8,200 Hz is barely noticeable. The Mel scale warps frequency to match:

```
m = 2595 · log₁₀(1 + f/700)
```

**MFCCs (1980).** Apply a DCT to the log-mel spectrum. The first 13 coefficients capture the smooth spectral envelope while discarding fine pitch detail. Coefficient 0 is overall energy, 1-12 encode vocal tract shape.

---

## Who uses this pipeline

| System     | Input representation       |
|------------|---------------------------|
| Whisper    | 80-band mel spectrogram   |
| AudioMAE   | 128-band mel spectrogram  |
| CLAP       | 64-band mel spectrogram   |
| BEATs      | mel spectrogram           |
| MusicGen   | encoded spectrogram       |

The preprocessing is the same across all of them. The model architectures differ. Audio Explorer implements the shared foundation.

---

## Requirements

Python 3.10+, librosa, numpy, scipy, plotly, streamlit, soundfile. Full list in `requirements.txt`.

```
conda create -n audio_explorer python=3.10 -y
conda activate audio_explorer
pip install -r requirements.txt
```

---

## Context

This is the first deliverable of a longer curriculum on audio and music AI. The preprocessing pipeline built here is the input stage for everything that comes next — classification, representation learning, and generative models.

---

MIT License. Built by [Shivashant](https://github.com/hey-shiv).
