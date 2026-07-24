import numpy as np
import matplotlib.pyplot as plt

def generate_sine_wave(frequency: float, duration: float, sample_rate: int = 44100, amplitude: float = 1.0, phase: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
    """
    Generates a pure sine wave.
    
    Parameters:
    - frequency: How many cycles per second (Hz).
    - duration: Length of the audio in seconds.
    - sample_rate: How many snapshots of the wave we take per second.
    - amplitude: The peak height of the wave.
    - phase: The starting shift in radians.
    
    Returns:
    - t: The time array.
    - y: The amplitude array.
    """
    # 1. Create an array of continuous time steps
    # np.linspace creates evenly spaced numbers over a specified interval.
    # Total samples = duration * sample_rate
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    
    # 2. Apply the physics equation: y(t) = A * sin(2 * pi * f * t + phi)
    y = amplitude * np.sin(2 * np.pi * frequency * t + phase)
    
    return t, y

if __name__ == "__main__":
    import os
    # Ensure assets directory exists
    os.makedirs("../../assets", exist_ok=True)
    
    # Generate a 440 Hz wave (A4 note) for 0.01 seconds
    t, y = generate_sine_wave(frequency=440.0, duration=0.01)
    
    # 3. Plot the result
    plt.figure(figsize=(10, 4))
    plt.plot(t, y, color='blue')
    plt.title("Pure Sine Wave (440 Hz)")
    plt.xlabel("Time (seconds)")
    plt.ylabel("Amplitude (Pressure)")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("../../assets/sine_wave_440hz.png")
    print("Waveform plotted and saved to assets/sine_wave_440hz.png")
