"""IMU recording, FFT and tremor features.

Gravity removal: subtract each axis' mean (hand held still). We FFT each axis
and combine the spectra (sqrt of summed power) rather than FFT-ing |a|, because
|a| of a de-meaned vector rectifies the signal (doubling the frequency) and |a|
including gravity hides tremor that is perpendicular to gravity.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np

from device import SAMPLE_HZ, Sample
from scoring import TREMOR_BAND_HZ, TremorFeatures, tremor_displacement_cm

MIN_SAMPLES = 100            # 1 s at 100 Hz
PEAK_TO_MEDIAN_RATIO = 6.0   # demo threshold for a "clear" peak
NOISE_FLOOR_ACCEL = 0.05     # m/s^2: below this a peak is just sensor noise
REFERENCE_BAND_HZ = (1.0, 20.0)
ZERO_PAD = 4


def resample(samples: Sequence[Sample], fs: float = SAMPLE_HZ) -> np.ndarray:
    """Uniform grid (N x 3) via linear interpolation on the Arduino timestamps."""
    arr = np.asarray(samples, dtype=float)
    t = (arr[:, 0] - arr[0, 0]) / 1000.0
    grid = np.arange(0.0, t[-1], 1.0 / fs)
    return np.column_stack([np.interp(grid, t, arr[:, i]) for i in (1, 2, 3)])


def spectrum(xyz: np.ndarray, fs: float = SAMPLE_HZ) -> tuple[np.ndarray, np.ndarray]:
    """Combined single-sided amplitude spectrum (m/s^2) of the de-meaned axes."""
    n = len(xyz)
    dyn = xyz - xyz.mean(axis=0)
    win = np.hanning(n)
    nfft = int(2 ** np.ceil(np.log2(n * ZERO_PAD)))
    amps = 2.0 * np.abs(np.fft.rfft(dyn * win[:, None], n=nfft, axis=0)) / win.sum()
    freqs = np.fft.rfftfreq(nfft, 1.0 / fs)
    return freqs, np.sqrt((amps ** 2).sum(axis=1))


def compute_features(samples: Sequence[Sample]) -> TremorFeatures:
    if len(samples) < MIN_SAMPLES:
        return TremorFeatures(0.0, 0.0, 0.0, False, len(samples) / SAMPLE_HZ)
    xyz = resample(samples)
    duration = len(xyz) / SAMPLE_HZ
    freqs, amp = spectrum(xyz)

    band = (freqs >= TREMOR_BAND_HZ[0]) & (freqs <= TREMOR_BAND_HZ[1])
    ref = (freqs >= REFERENCE_BAND_HZ[0]) & (freqs <= REFERENCE_BAND_HZ[1])
    i = np.flatnonzero(band)[np.argmax(amp[band])]
    peak_hz, peak_amp = float(freqs[i]), float(amp[i])
    median = float(np.median(amp[ref])) or 1e-12
    clear = peak_amp >= NOISE_FLOOR_ACCEL and peak_amp / median >= PEAK_TO_MEDIAN_RATIO
    return TremorFeatures(peak_hz=peak_hz, peak_accel=peak_amp,
                          displacement_cm=tremor_displacement_cm(peak_amp, peak_hz),
                          has_clear_peak=clear, duration_s=duration)


class TremorRecorder:
    """Accumulates samples for one test; keeps a short trace for the live plot."""

    def __init__(self) -> None:
        self.samples: list[Sample] = []

    def add(self, new: Sequence[Sample]) -> None:
        self.samples.extend(new)

    def recent_magnitude(self, seconds: float = 3.0) -> np.ndarray:
        """Dynamic accel magnitude (m/s^2) over the last few seconds, for display."""
        k = int(seconds * SAMPLE_HZ)
        if len(self.samples) < 2:
            return np.zeros(0)
        arr = np.asarray(self.samples[-k:], dtype=float)[:, 1:]
        return np.linalg.norm(arr - arr.mean(axis=0), axis=1)

    def features(self) -> TremorFeatures:
        return compute_features(self.samples)
