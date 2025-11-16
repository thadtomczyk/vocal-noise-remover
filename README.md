# Vocal Noise Remover

A simple Python-based tool for reducing background noise in recorded vocals using spectral analysis and adaptive masking.

## Overview
This project applies short-time Fourier transform (STFT) techniques to identify and attenuate low-level background noise while preserving the main vocal signal.  
It’s designed as a learning and prototype project for understanding basic digital signal processing (DSP) concepts such as:
- STFT and inverse-STFT
- Noise floor estimation
- Spectral masking
- Temporal smoothing

The implementation uses `librosa`, `numpy`, and `soundfile` for the core signal processing pipeline.

---

## How It Works
1. Load a mono WAV file using `librosa`.
2. Compute the magnitude and phase via STFT.
3. Estimate a noise floor from quiet portions of the spectrum (using a percentile-based threshold).
4. Build a gain mask that attenuates bins close to that noise floor.
5. Apply optional smoothing across time to reduce musical noise.
6. Reconstruct the cleaned signal using the inverse STFT and write to disk.

---

## Technologies
- **Python 3.11**
- `librosa`
- `numpy`
- `scipy`
- `soundfile`
- `matplotlib` (for visualization)

---

## Instructions
1. Place your audio file in the project directory (e.g., `input.wav`).
2. Run:
   ```bash
   python main.py
