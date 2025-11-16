import numpy as np
import librosa
import soundfile as sf
import matplotlib.pyplot as plt
from scipy.signal import medfilt


# Parameters and file paths
INPUT = "input.wav"           # My vocal test recording file
OUTPUT = "output_clean.wav"

FRAME_LEN = 1024              # STFT window size (samples)
HOP = 256                     # Hop length between frames
ATTENUATION_DB = 12.0         # How much to reduce breath frames (in dB)
SMOOTH_KERNEL = 7             # Median filter size for smoothing mask

 
# Load audio
y, sr = librosa.load(INPUT, sr=None, mono=True) # Opens the audio file in mono
assert y.ndim == 1, "Expected mono audio after load" # Ensures the audio file is mono
print(f"Loaded {INPUT} at {sr} Hz, {len(y)/sr:.2f}s") # Prints a confirmation to the terminal

# STFT (Short-Time Fourier Transform)
S = librosa.stft(y, n_fft=FRAME_LEN, hop_length=HOP, window="hann") # Slices the file and performs a FFT on each
mag = np.abs(S) # Magnitude
phase = np.angle(S) # Phase
print(f"STFT computed: {mag.shape[0]} frequency bins × {mag.shape[1]} frames") # Prints a confirmation to the terminal


# Noise reduction mask
NOISE_PERCENTILE = 40        # 35–45 = stronger suppression
FLOOR_MARGIN_DB = 10.0       # 6–12 dB extra push below speech
MASK_POWER = 2.5             # 1.5–3.0; higher = harder attenuation

# Per-bin noise estimate and threshold
noise_floor = np.percentile(mag, NOISE_PERCENTILE, axis=1, keepdims=True)
noise_thresh = noise_floor * (10 ** (FLOOR_MARGIN_DB / 20.0))

# Ratio to threshold -> gain mask in [0..1], shaped by power
ratio = mag / (noise_thresh + 1e-12)
mask = np.clip(ratio, 0.0, 1.0) ** MASK_POWER

# Optional: light temporal smoothing to reduce “musical noise”
alpha = 0.7  # 0=none, 0.6–0.85 is nice
for t in range(1, mask.shape[1]):
    mask[:, t] = alpha * mask[:, t-1] + (1 - alpha) * mask[:, t]

mag_denoised = mag * mask


# Recombine with phase and reconstruct
S_denoised = mag_denoised * np.exp(1j * phase)
y_denoised = librosa.istft(S_denoised, hop_length=HOP, window="hann", length=len(y))

# Write the cleaned file
sf.write(OUTPUT, y_denoised, sr)
print(f"Saved denoised audio to {OUTPUT}")


# Visualize before/after
import librosa.display as ldisplay

fig, axes = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)

mag_db = librosa.amplitude_to_db(mag, ref=np.max)
mag_denoised_db = librosa.amplitude_to_db(np.abs(S_denoised), ref=np.max)

ldisplay.specshow(mag_db, sr=sr, hop_length=HOP, x_axis="time", y_axis="hz", ax=axes[0])
axes[0].set_title("Original")
axes[0].set_xlabel("Time (s)")
axes[0].set_ylabel("Hz")

ldisplay.specshow(mag_denoised_db, sr=sr, hop_length=HOP, x_axis="time", y_axis="hz", ax=axes[1])
axes[1].set_title("Denoised")
axes[1].set_xlabel("Time (s)")
axes[1].set_ylabel("Hz")

plt.show()
