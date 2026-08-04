"""
generate_animation.py

Creates a 15-second 4K animation showing the audio AI pipeline:
  Waveform → STFT sliding window → Spectrogram building up

Output: audio_explorer_animation.mp4
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.gridspec as gridspec

# ─────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────
FPS = 30
DURATION = 15  # seconds
TOTAL_FRAMES = FPS * DURATION  # 450 frames
WIDTH, HEIGHT = 1920, 1080  # Full HD (use 3840x2160 for true 4K)
DPI = 150

# ─────────────────────────────────────────────────────────
# Generate synthetic audio signal (multi-frequency)
# ─────────────────────────────────────────────────────────
sr = 4000  # low sample rate for visual clarity
duration_audio = 4.0  # seconds of audio
t = np.linspace(0, duration_audio, int(sr * duration_audio), endpoint=False)

# Create a rich signal: fundamental + harmonics + noise + envelope
np.random.seed(42)
signal = (
    0.6 * np.sin(2 * np.pi * 220 * t) *  # A3 fundamental
    (1 + 0.3 * np.sin(2 * np.pi * 0.5 * t))  # slow amplitude modulation
)
signal += 0.3 * np.sin(2 * np.pi * 440 * t)   # 2nd harmonic
signal += 0.15 * np.sin(2 * np.pi * 660 * t)  # 3rd harmonic
signal += 0.1 * np.sin(2 * np.pi * 880 * t)   # 4th harmonic

# Add a chirp (rising frequency) in the middle
chirp_mask = np.exp(-0.5 * ((t - 2.0) / 0.4) ** 2)
signal += 0.4 * np.sin(2 * np.pi * (300 + 200 * t) * t) * chirp_mask

# Add subtle noise
signal += 0.02 * np.random.randn(len(t))

# Normalize
signal = signal / np.max(np.abs(signal)) * 0.95

# ─────────────────────────────────────────────────────────
# Pre-compute the spectrogram
# ─────────────────────────────────────────────────────────
n_fft = 256
hop_length = 64
n_freq = n_fft // 2 + 1
n_frames = (len(signal) - n_fft) // hop_length + 1

# Hann window
window = 0.5 * (1 - np.cos(2 * np.pi * np.arange(n_fft) / n_fft))

# Compute STFT manually for visual clarity
spectrogram = np.zeros((n_freq, n_frames))
for i in range(n_frames):
    start = i * hop_length
    chunk = signal[start:start + n_fft] * window
    fft_result = np.fft.rfft(chunk)
    spectrogram[:, i] = np.abs(fft_result)

# Convert to dB
spectrogram_db = 20 * np.log10(spectrogram + 1e-10)
spectrogram_db = spectrogram_db - spectrogram_db.max()
spectrogram_db = np.clip(spectrogram_db, -60, 0)

# Frequency and time axes
freqs = np.linspace(0, sr / 2, n_freq)
times = np.arange(n_frames) * hop_length / sr

# ─────────────────────────────────────────────────────────
# Custom colormap (dark magma-like)
# ─────────────────────────────────────────────────────────
colors_custom = [
    (0.0, '#0a0a1a'),   # deep dark
    (0.15, '#1a0a3e'),  # deep purple
    (0.3, '#3b0764'),   # purple
    (0.45, '#7c2d12'),  # burnt orange
    (0.6, '#c2410c'),   # orange
    (0.8, '#fbbf24'),   # yellow
    (1.0, '#fef3c7'),   # cream white
]

cmap_vals = []
cmap_colors_r = []
cmap_colors_g = []
cmap_colors_b = []

for pos, hex_color in colors_custom:
    r = int(hex_color[1:3], 16) / 255
    g = int(hex_color[3:5], 16) / 255
    b = int(hex_color[5:7], 16) / 255
    cmap_vals.append(pos)
    cmap_colors_r.append((pos, r, r))
    cmap_colors_g.append((pos, g, g))
    cmap_colors_b.append((pos, b, b))

custom_cmap = LinearSegmentedColormap('AudioExplorer', {
    'red': cmap_colors_r,
    'green': cmap_colors_g,
    'blue': cmap_colors_b,
})

# ─────────────────────────────────────────────────────────
# Setup the figure
# ─────────────────────────────────────────────────────────
fig = plt.figure(figsize=(WIDTH / DPI, HEIGHT / DPI), dpi=DPI)
fig.patch.set_facecolor('#08080f')

gs = gridspec.GridSpec(
    2, 1, height_ratios=[1, 1.3],
    hspace=0.35, left=0.08, right=0.95, top=0.88, bottom=0.08
)

# Top: Waveform
ax_wave = fig.add_subplot(gs[0])
ax_wave.set_facecolor('#08080f')
ax_wave.set_xlim(0, duration_audio)
ax_wave.set_ylim(-1.15, 1.15)
ax_wave.set_ylabel('Amplitude', color='#94a3b8', fontsize=10, fontfamily='sans-serif')
ax_wave.set_xlabel('Time (s)', color='#64748b', fontsize=9, fontfamily='sans-serif')
ax_wave.tick_params(colors='#475569', labelsize=8)
ax_wave.spines['bottom'].set_color('#1e293b')
ax_wave.spines['left'].set_color('#1e293b')
ax_wave.spines['top'].set_visible(False)
ax_wave.spines['right'].set_visible(False)
ax_wave.grid(True, alpha=0.08, color='#334155')

# Bottom: Spectrogram
ax_spec = fig.add_subplot(gs[1])
ax_spec.set_facecolor('#08080f')
ax_spec.set_xlim(0, duration_audio)
ax_spec.set_ylim(0, sr / 2)
ax_spec.set_ylabel('Frequency (Hz)', color='#94a3b8', fontsize=10, fontfamily='sans-serif')
ax_spec.set_xlabel('Time (s)', color='#64748b', fontsize=9, fontfamily='sans-serif')
ax_spec.tick_params(colors='#475569', labelsize=8)
ax_spec.spines['bottom'].set_color('#1e293b')
ax_spec.spines['left'].set_color('#1e293b')
ax_spec.spines['top'].set_visible(False)
ax_spec.spines['right'].set_visible(False)

# Title
title_text = fig.text(
    0.5, 0.95, 'Audio Explorer',
    ha='center', va='center',
    fontsize=28, fontweight='bold',
    color='#a78bfa', fontfamily='sans-serif',
    alpha=0.0  # will fade in
)

subtitle_text = fig.text(
    0.5, 0.915, 'Building the Audio AI Pipeline from First Principles',
    ha='center', va='center',
    fontsize=12, color='#64748b', fontfamily='sans-serif',
    alpha=0.0
)

# ─────────────────────────────────────────────────────────
# Animation elements
# ─────────────────────────────────────────────────────────
# Waveform line (will draw progressively)
wave_line, = ax_wave.plot([], [], color='#818cf8', linewidth=0.8, alpha=0.9)
wave_glow, = ax_wave.plot([], [], color='#a78bfa', linewidth=2.5, alpha=0.15)

# STFT window indicator
window_rect = plt.Rectangle((0, -1.1), n_fft / sr, 2.2,
                              linewidth=1.5, edgecolor='#f87171',
                              facecolor='#f8717115', visible=False)
ax_wave.add_patch(window_rect)

# Spectrogram image placeholder (blank initially)
blank_spec = np.full_like(spectrogram_db, -60)
spec_img = ax_spec.imshow(
    blank_spec, aspect='auto', origin='lower',
    extent=[0, duration_audio, 0, sr / 2],
    cmap=custom_cmap, vmin=-60, vmax=0,
    interpolation='bilinear'
)

# Phase labels
phase_text = fig.text(
    0.5, 0.02, '',
    ha='center', fontsize=11, color='#94a3b8',
    fontfamily='sans-serif', alpha=0.0
)


def ease_in_out(x):
    """Smooth easing function."""
    return x * x * (3 - 2 * x)


def animate(frame):
    """Update function called for each frame."""
    progress = frame / TOTAL_FRAMES  # 0.0 to 1.0

    # ─── PHASE 1: Title fade in (0% - 8%) ───
    if progress < 0.08:
        p = ease_in_out(progress / 0.08)
        title_text.set_alpha(p)
        subtitle_text.set_alpha(p * 0.7)

    # ─── PHASE 2: Draw waveform progressively (5% - 40%) ───
    elif progress < 0.40:
        title_text.set_alpha(1.0)
        subtitle_text.set_alpha(0.7)

        p = ease_in_out((progress - 0.05) / 0.35)
        n_show = max(1, int(p * len(t)))
        wave_line.set_data(t[:n_show], signal[:n_show])
        wave_glow.set_data(t[:n_show], signal[:n_show])

        phase_text.set_text('① Waveform — raw amplitude over time')
        phase_text.set_alpha(min(1.0, (progress - 0.05) * 10))

    # ─── PHASE 3: STFT window slides across (40% - 80%) ───
    elif progress < 0.80:
        # Full waveform visible
        wave_line.set_data(t, signal)
        wave_glow.set_data(t, signal)

        p = ease_in_out((progress - 0.40) / 0.40)
        window_rect.set_visible(True)

        # Window position
        current_frame = int(p * (n_frames - 1))
        win_start = current_frame * hop_length / sr
        window_rect.set_x(win_start)
        window_rect.set_width(n_fft / sr)

        # Reveal spectrogram columns up to current frame
        revealed = blank_spec.copy()
        revealed[:, :current_frame + 1] = spectrogram_db[:, :current_frame + 1]
        spec_img.set_data(revealed)

        phase_text.set_text('② STFT — sliding window + FFT → spectrogram')
        phase_text.set_alpha(1.0)

    # ─── PHASE 4: Full spectrogram reveal + window fades (80% - 100%) ───
    else:
        wave_line.set_data(t, signal)
        wave_glow.set_data(t, signal)
        spec_img.set_data(spectrogram_db)

        p = ease_in_out((progress - 0.80) / 0.20)
        window_rect.set_alpha(1.0 - p)

        if progress > 0.85:
            phase_text.set_text('③ Complete — frequency content over time')
        phase_text.set_alpha(1.0)

    return [wave_line, wave_glow, window_rect, spec_img,
            title_text, subtitle_text, phase_text]


# ─────────────────────────────────────────────────────────
# Render
# ─────────────────────────────────────────────────────────
print(f"Rendering {TOTAL_FRAMES} frames at {FPS} fps ({DURATION}s)...")
print(f"Resolution: {WIDTH}x{HEIGHT}")

anim = animation.FuncAnimation(
    fig, animate, frames=TOTAL_FRAMES, interval=1000 / FPS, blit=False
)

output_path = "audio_explorer_animation.mp4"
writer = animation.FFMpegWriter(fps=FPS, bitrate=5000,
                                 extra_args=['-pix_fmt', 'yuv420p'])
anim.save(output_path, writer=writer, dpi=DPI)
print(f"\n✅ Saved: {output_path}")
print(f"   Size: {WIDTH}x{HEIGHT} @ {FPS}fps, {DURATION}s")

plt.close()
