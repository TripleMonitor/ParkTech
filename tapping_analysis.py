"""Finger-tapping maths. PURE functions only: no camera, no MediaPipe, no UI.

Input: timestamps (s) + normalised thumb-index distance per frame (None when no
hand was found). Each opening (local maximum) of the distance is one tap; its
amplitude is the peak's prominence (how far the fingers opened from closed).
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Optional, Sequence

import numpy as np
from scipy.signal import find_peaks

THUMB_TIP, INDEX_TIP, WRIST, MIDDLE_MCP = 4, 8, 0, 9
GRID_HZ = 60.0
MIN_TAP_PROMINENCE = 0.15   # normalised units: smaller wiggles aren't taps
MIN_TAP_GAP_S = 0.08        # faster than ~12 taps/s is jitter
EDGE_WINDOW_S = 3.0         # decrement = first 3 s vs last 3 s
HESITATION_FACTOR = 2.0     # interval > 2x median
MAX_TRACKING_GAP_S = 0.15   # no hand for longer than this = tracking dropout, not the patient


@dataclass(frozen=True)
class TappingFeatures:
    n_taps: int
    taps_per_sec: float
    mean_amplitude: float       # normalised by wrist-to-middle-MCP length
    decrement: float            # fraction lost, first 3 s vs last 3 s (0..1)
    interval_cv: float
    hesitations: int
    hand_visible: float         # fraction of frames with a hand (0..1)
    duration_s: float
    tracking_lost_s: float = 0.0  # total time in tracking gaps > 0.15 s
    tap_times: tuple = field(default=(), repr=False)


def normalised_distance(landmarks) -> Optional[float]:
    """landmarks: (21, 2+) array-like in any consistent units."""
    pts = np.asarray(landmarks, dtype=float)[:, :2]
    scale = np.linalg.norm(pts[WRIST] - pts[MIDDLE_MCP])
    if not np.isfinite(scale) or scale < 1e-6:
        return None
    return float(np.linalg.norm(pts[THUMB_TIP] - pts[INDEX_TIP]) / scale)


def _empty(visible: float, duration: float) -> TappingFeatures:
    return TappingFeatures(0, 0.0, 0.0, 0.0, 0.0, 0, visible, duration)


def detect_taps(t_s: np.ndarray, d: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Returns (tap times relative to t_s[0], tap amplitudes)."""
    grid = np.arange(t_s[0], t_s[-1], 1.0 / GRID_HZ)
    sig = np.interp(grid, t_s, d)
    peaks, props = find_peaks(sig, prominence=MIN_TAP_PROMINENCE,
                              distance=max(1, int(MIN_TAP_GAP_S * GRID_HZ)))
    return grid[peaks] - t_s[0], props["prominences"]


def decrement(tap_t: np.ndarray, amps: np.ndarray, duration: float) -> float:
    first = amps[tap_t < EDGE_WINDOW_S]
    last = amps[tap_t > duration - EDGE_WINDOW_S]
    if len(first) == 0 or len(last) == 0 or first.mean() <= 0:
        return 0.0
    return float(max(0.0, 1.0 - last.mean() / first.mean()))


def tracking_gaps(t_valid: np.ndarray) -> list[tuple[float, float]]:
    """(start, end) of stretches longer than MAX_TRACKING_GAP_S with no hand."""
    d = np.diff(t_valid)
    return [(float(t_valid[i]), float(t_valid[i + 1]))
            for i in np.flatnonzero(d > MAX_TRACKING_GAP_S)]


def rhythm(tap_t: np.ndarray, gaps: Sequence[tuple[float, float]] = ()) -> tuple[float, int]:
    """(interval coefficient of variation, number of hesitations).

    Intervals that overlap a tracking gap are dropped: the camera lost the
    hand, so we don't know what the patient did there.
    """
    intervals = np.array([b - a for a, b in zip(tap_t[:-1], tap_t[1:])
                          if not any(a < g1 and g0 < b for g0, g1 in gaps)])
    if len(intervals) < 2:
        return 0.0, 0
    cv = float(intervals.std() / intervals.mean())
    hes = int((intervals > HESITATION_FACTOR * np.median(intervals)).sum())
    return cv, hes


def analyze_tapping(t_s: Sequence[float], dists: Sequence[Optional[float]]) -> TappingFeatures:
    n_frames = len(t_s)
    duration = float(t_s[-1] - t_s[0]) if n_frames > 1 else 0.0
    valid = [(t, d) for t, d in zip(t_s, dists) if d is not None and np.isfinite(d)]
    visible = len(valid) / n_frames if n_frames else 0.0
    if len(valid) < 10 or duration <= 0:
        return _empty(visible, duration)

    t0 = float(t_s[0])
    tv = np.array([v[0] for v in valid], dtype=float)
    dv = np.array([v[1] for v in valid], dtype=float)
    gaps = [(a - t0, b - t0) for a, b in tracking_gaps(tv)]
    lost = float(sum(b - a for a, b in gaps))
    tap_t, amps = detect_taps(tv, dv)
    tap_t = tap_t + (tv[0] - t0)
    if len(tap_t) == 0:
        return replace(_empty(visible, duration), tracking_lost_s=lost)
    cv, hes = rhythm(tap_t, gaps)
    return TappingFeatures(
        n_taps=len(tap_t), taps_per_sec=len(tap_t) / duration,
        mean_amplitude=float(amps.mean()), decrement=decrement(tap_t, amps, duration),
        interval_cv=cv, hesitations=hes, hand_visible=visible, duration_s=duration,
        tracking_lost_s=lost, tap_times=tuple(float(x) for x in tap_t))
