"""Tremor maths. PURE functions only: no hardware, no UI.

Pipeline (see CLAUDE.md): resample to uniform 100 Hz -> subtract each axis' mean
(removes gravity) -> linear detrend -> Hann-windowed FFT per axis -> sum power
spectra -> peak in 3-8 Hz -> sinusoid amplitude at that frequency on the
strongest axis -> displacement = a / (2*pi*f)^2.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
from scipy.signal import detrend

FS = 100.0
TREMOR_BAND_HZ = (3.0, 8.0)
PD_BAND_HZ = (4.0, 6.0)
CLEAR_PEAK_RATIO = 3.0        # peak power >= 3x median power in band (demo threshold)
MIN_TREMOR_CM = 0.1           # below this, no tremor (demo threshold)
MIN_SECONDS = 2.0             # less data than this -> cannot analyse
ZERO_PAD = 4
WINDOW_S = 1.0


@dataclass(frozen=True)
class TremorFeatures:
    valid: bool                   # enough data to analyse
    duration_s: float
    peak_hz: float
    peak_accel: float             # m/s^2 sinusoid amplitude, strongest axis
    displacement_cm: float
    clear_peak: bool
    peak_ratio: float             # peak power / median power in band
    window_pct: float             # % of 1-s windows that show tremor
    freqs: np.ndarray = field(repr=False, compare=False, default_factory=lambda: np.zeros(0))
    power: np.ndarray = field(repr=False, compare=False, default_factory=lambda: np.zeros(0))


def displacement_cm(accel: float, freq_hz: float) -> float:
    if freq_hz <= 0:
        return 0.0
    return accel / (2 * np.pi * freq_hz) ** 2 * 100.0


def resample_uniform(t_s: Sequence[float], xyz: np.ndarray, fs: float = FS) -> np.ndarray:
    """Linear interpolation of (N,3) samples at times t_s onto a uniform fs grid."""
    t = np.asarray(t_s, dtype=float)
    order = np.argsort(t, kind="stable")
    t, xyz = t[order], np.asarray(xyz, dtype=float)[order]
    t, idx = np.unique(t, return_index=True)          # drop duplicate timestamps
    xyz = xyz[idx]
    grid = np.arange(t[0], t[-1], 1.0 / fs)
    return np.column_stack([np.interp(grid, t, xyz[:, i]) for i in range(3)])


def spectrum(xyz: np.ndarray, fs: float = FS) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Returns (freqs, summed power, per-axis amplitude (F,3) in m/s^2)."""
    n = len(xyz)
    dyn = detrend(xyz - xyz.mean(axis=0), axis=0, type="linear")
    win = np.hanning(n)
    nfft = int(2 ** np.ceil(np.log2(n * ZERO_PAD)))
    spec = np.fft.rfft(dyn * win[:, None], n=nfft, axis=0)
    amp = 2.0 * np.abs(spec) / win.sum()
    freqs = np.fft.rfftfreq(nfft, 1.0 / fs)
    return freqs, (amp ** 2).sum(axis=1), amp


def band_peak(freqs: np.ndarray, power: np.ndarray, amp: np.ndarray
              ) -> tuple[float, float, float]:
    """(peak_hz, strongest-axis amplitude at peak, peak/median power ratio) in 3-8 Hz."""
    band = (freqs >= TREMOR_BAND_HZ[0]) & (freqs <= TREMOR_BAND_HZ[1])
    idx = np.flatnonzero(band)
    i = idx[np.argmax(power[idx])]
    median = float(np.median(power[idx]))
    ratio = float(power[i] / median) if median > 0 else (np.inf if power[i] > 0 else 0.0)
    return float(freqs[i]), float(amp[i].max()), ratio


def _window_has_tremor(xyz: np.ndarray, fs: float) -> bool:
    f, p, a = spectrum(xyz, fs)
    hz, accel, ratio = band_peak(f, p, a)
    return ratio >= CLEAR_PEAK_RATIO and displacement_cm(accel, hz) >= MIN_TREMOR_CM


def tremor_window_pct(xyz: np.ndarray, fs: float = FS) -> float:
    k = int(WINDOW_S * fs)
    windows = [xyz[i:i + k] for i in range(0, len(xyz) - k + 1, k)]
    if not windows:
        return 0.0
    return 100.0 * sum(_window_has_tremor(w, fs) for w in windows) / len(windows)


def analyze_tremor(t_s: Sequence[float], ax: Sequence[float], ay: Sequence[float],
                   az: Sequence[float], fs: float = FS) -> TremorFeatures:
    """t_s in seconds (may be uneven). Accelerations in m/s^2 including gravity."""
    n = len(t_s)
    duration = float(t_s[-1] - t_s[0]) if n > 1 else 0.0
    if n < 2 or duration < MIN_SECONDS:
        return TremorFeatures(False, duration, 0.0, 0.0, 0.0, False, 0.0, 0.0)
    xyz = resample_uniform(t_s, np.column_stack([ax, ay, az]), fs)
    freqs, power, amp = spectrum(xyz, fs)
    hz, accel, ratio = band_peak(freqs, power, amp)
    return TremorFeatures(
        valid=True, duration_s=duration, peak_hz=hz, peak_accel=accel,
        displacement_cm=displacement_cm(accel, hz),
        clear_peak=ratio >= CLEAR_PEAK_RATIO, peak_ratio=ratio,
        window_pct=tremor_window_pct(xyz, fs), freqs=freqs, power=power)


def analyze_samples(samples) -> TremorFeatures:
    """Convenience: list of (t_ms, ax, ay, az) tuples, e.g. device.Sample."""
    if len(samples) < 2:
        return analyze_tremor([0.0], [0.0], [0.0], [0.0])
    arr = np.asarray(samples, dtype=float)
    return analyze_tremor(arr[:, 0] / 1000.0, arr[:, 1], arr[:, 2], arr[:, 3])
