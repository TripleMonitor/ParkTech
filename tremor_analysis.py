"""Camera rest-tremor maths. PURE functions only: no camera, no UI.

Input: frame timestamps + MediaPipe hand landmarks in pixels (None when no hand).
Pipeline (see CLAUDE.md):
  pixels -> cm using the hand itself: wrist(0)-middle-MCP(9) = 9 cm (median over the recording)
  index fingertip (8) x/y in cm -> cubic-spline resample to uniform 30 Hz
  -> linear detrend + 1 Hz high-pass (removes slow drift of the whole arm)
  -> Hann-windowed FFT on x and y, power summed -> peak in 3-8 Hz
  -> displacement amplitude (cm) at the peak.

Amplitude is the movement's vector amplitude, measured as band-limited RMS within
+/-0.75 Hz of the peak over both axes (A = sqrt(2) * RMS), so tremor that drifts in
frequency, falls between FFT bins, or moves diagonally is not under-read.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.signal import butter, detrend, filtfilt

FS = 30.0                      # uniform resampling rate (Hz)
HAND_LENGTH_CM = 9.0           # wrist(0) -> middle-MCP(9), adult average
WRIST, INDEX_TIP, MIDDLE_MCP = 0, 8, 9
TREMOR_BAND_HZ = (3.0, 8.0)
PD_BAND_HZ = (4.0, 6.0)
CLEAR_PEAK_RATIO = 3.0         # peak power >= 3x median power in band (demo threshold)
MIN_TREMOR_CM = 0.1            # below this, no tremor (demo threshold)
MIN_SECONDS = 2.0              # less data than this -> cannot analyse
MIN_VISIBLE = 0.5              # hand must be found in >= 50% of frames
HIGHPASS_HZ = 1.0
ZERO_PAD = 4
WINDOW_S = 1.0
AMP_HALF_BAND_HZ = 0.75


@dataclass(frozen=True)
class TremorFeatures:
    valid: bool                   # enough data to analyse
    duration_s: float
    hand_visible: float           # fraction of frames with a hand (0..1)
    cm_per_px: float
    peak_hz: float
    displacement_cm: float        # fingertip displacement amplitude at the peak
    clear_peak: bool
    peak_ratio: float             # peak power / median power in band
    window_pct: float             # % of 1-s windows that show tremor
    freqs: np.ndarray = field(repr=False, compare=False, default_factory=lambda: np.zeros(0))
    power: np.ndarray = field(repr=False, compare=False, default_factory=lambda: np.zeros(0))


def effective_displacement_cm(f: TremorFeatures) -> float:
    """Displacement that counts: 0 when there is no clear tremor peak."""
    return f.displacement_cm if (f.valid and f.clear_peak) else 0.0


def cm_per_pixel(landmarks: Sequence[np.ndarray]) -> Optional[float]:
    lengths = [float(np.linalg.norm(lm[WRIST] - lm[MIDDLE_MCP])) for lm in landmarks]
    lengths = [x for x in lengths if np.isfinite(x) and x > 1e-6]
    return HAND_LENGTH_CM / float(np.median(lengths)) if lengths else None


def resample_uniform(t: np.ndarray, xy: np.ndarray, fs: float = FS) -> np.ndarray:
    """Cubic spline through (possibly uneven, gappy) samples onto a uniform fs grid.

    Cubic (not linear) interpolation: linear interpolation of a 30 fps camera
    attenuates a 5 Hz tremor by ~10%.
    """
    order = np.argsort(t, kind="stable")
    t, xy = t[order], xy[order]
    t, idx = np.unique(t, return_index=True)
    xy = xy[idx]
    grid = np.arange(t[0], t[-1], 1.0 / fs)
    return CubicSpline(t, xy, axis=0, bc_type="natural")(grid)


def highpass(xy: np.ndarray, fs: float = FS) -> np.ndarray:
    y = detrend(xy, axis=0, type="linear")
    b, a = butter(2, HIGHPASS_HZ / (fs / 2), btype="highpass")
    if len(y) <= 3 * max(len(a), len(b)):
        return y
    return filtfilt(b, a, y, axis=0)


def spectrum(xy: np.ndarray, fs: float = FS) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(freqs, power summed over x/y, complex windowed spectrum (F,2)).

    Power is scaled so a pure sinusoid of amplitude A cm on one axis gives A^2.
    """
    n = len(xy)
    win = np.hanning(n)
    nfft = int(2 ** np.ceil(np.log2(n * ZERO_PAD)))
    spec = np.fft.rfft(xy * win[:, None], n=nfft, axis=0)
    amp = 2.0 * np.abs(spec) / win.sum()
    return np.fft.rfftfreq(nfft, 1.0 / fs), (amp ** 2).sum(axis=1), spec


def band_peak(freqs: np.ndarray, power: np.ndarray) -> tuple[int, float, bool]:
    """(peak bin in 3-8 Hz, peak/median power ratio in band, is a true local maximum).

    A peak on the band edge with more power just outside is leakage from
    out-of-band movement, not tremor, so it is not a local maximum.
    """
    band = (freqs >= TREMOR_BAND_HZ[0]) & (freqs <= TREMOR_BAND_HZ[1])
    idx = np.flatnonzero(band)
    i = int(idx[np.argmax(power[idx])])
    median = float(np.median(power[idx]))
    ratio = float(power[i] / median) if median > 0 else (np.inf if power[i] > 0 else 0.0)
    lo, hi = max(i - 1, 0), min(i + 1, len(power) - 1)
    return i, ratio, bool(power[i] >= power[lo] and power[i] >= power[hi])


def band_amplitude(spec: np.ndarray, freqs: np.ndarray, peak_hz: float, n: int) -> float:
    """Vector amplitude (cm) from band-limited RMS within +/-0.75 Hz, both axes.

    Parseval on the Hann-windowed, zero-padded spectrum; dividing by sum(w^2)
    undoes the window's energy loss. For linear sinusoidal motion of amplitude A
    in any direction this returns A.
    """
    nfft = 2 * (len(freqs) - 1)
    win = np.hanning(n)
    mask = np.abs(freqs - peak_hz) <= AMP_HALF_BAND_HZ
    energy = 2.0 * (np.abs(spec[mask]) ** 2).sum() / nfft     # sum over time and axes of y^2
    rms_vec = np.sqrt(energy / (win ** 2).sum())
    return float(np.sqrt(2.0) * rms_vec)


def _peak(xy: np.ndarray, fs: float):
    freqs, power, spec = spectrum(xy, fs)
    i, ratio, local_max = band_peak(freqs, power)
    hz = float(freqs[i])
    return freqs, power, hz, band_amplitude(spec, freqs, hz, len(xy)), ratio, \
        ratio >= CLEAR_PEAK_RATIO and local_max


def tremor_window_pct(xy: np.ndarray, fs: float = FS) -> float:
    k = int(WINDOW_S * fs)
    windows = [xy[i:i + k] for i in range(0, len(xy) - k + 1, k)]
    if not windows:
        return 0.0
    hits = 0
    for w in windows:
        _, _, _, amp, _, clear = _peak(w - w.mean(axis=0), fs)
        hits += clear and amp >= MIN_TREMOR_CM
    return 100.0 * hits / len(windows)


def _invalid(duration: float, visible: float, cm_px: float = 0.0) -> TremorFeatures:
    return TremorFeatures(False, duration, visible, cm_px, 0.0, 0.0, False, 0.0, 0.0)


def analyze_tremor(t_s: Sequence[float], landmarks: Sequence[Optional[np.ndarray]],
                   fs: float = FS) -> TremorFeatures:
    """t_s: frame times (s, may be uneven). landmarks: (21, 2) pixel arrays or None."""
    n = len(t_s)
    duration = float(t_s[-1] - t_s[0]) if n > 1 else 0.0
    valid = [(t, np.asarray(lm, float)) for t, lm in zip(t_s, landmarks) if lm is not None]
    visible = len(valid) / n if n else 0.0
    if len(valid) < 10 or visible < MIN_VISIBLE:
        return _invalid(duration, visible)
    tv = np.array([v[0] for v in valid])
    cm_px = cm_per_pixel([v[1] for v in valid])
    if cm_px is None or tv[-1] - tv[0] < MIN_SECONDS:
        return _invalid(duration, visible, cm_px or 0.0)
    tip_cm = np.array([v[1][INDEX_TIP] for v in valid]) * cm_px
    xy = highpass(resample_uniform(tv, tip_cm, fs), fs)
    freqs, power, hz, amp, ratio, clear = _peak(xy, fs)
    return TremorFeatures(valid=True, duration_s=duration, hand_visible=visible, cm_per_px=cm_px,
                          peak_hz=hz, displacement_cm=amp, clear_peak=clear, peak_ratio=ratio,
                          window_pct=tremor_window_pct(xy, fs), freqs=freqs, power=power)
