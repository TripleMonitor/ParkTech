"""Camera + tilt-switch fusion for hand flipping. PURE functions only.

Camera: palm normal = (p5 - p0) x (p17 - p0) from MediaPipe 3D world landmarks
(wrist, index MCP, pinky MCP). Rotation angle = angle between the current normal and
the palm-down reference normal (first frames of the test), 0 deg = palm down,
~180 deg = palm up. Independent of how the forearm points relative to the camera.

Swings between turning points of that angle are camera half-flips; their size is
the flip amplitude. The switch gives precise timing; the camera gives amplitude and
an independent count. Counts agreeing within 15% = FUSION LOCKED.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np
from scipy.signal import find_peaks

WRIST, INDEX_MCP, PINKY_MCP = 0, 5, 17
MIN_SWING_DEG = 45.0          # smaller rotations are wobble, not a half-flip
AGREE_TOL = 0.15              # counts within 15% -> locked
MIN_VISIBLE = 0.5             # need the hand in >= 50% of frames to use the camera
EDGE_WINDOW_S = 4.0
SMALL_AMPLITUDE_DEG = 120.0   # demo threshold
AMP_DECREMENT = 0.25          # demo threshold


@dataclass(frozen=True)
class FusionResult:
    camera_ok: bool               # enough hand frames to trust the camera
    visible: float
    camera_half_flips: int
    switch_half_flips: int
    locked: bool
    median_amplitude_deg: float
    amplitude_decrement: float    # 1 - median(last 4 s) / median(first 4 s), >= 0
    swing_times: tuple = ()
    swing_amplitudes: tuple = ()

    @property
    def status(self) -> str:
        if not self.camera_ok:
            return "CAMERA: hand not tracked - switch only"
        return "FUSION LOCKED" if self.locked else "FUSION MISMATCH - low confidence"


def palm_normal(world: np.ndarray) -> Optional[np.ndarray]:
    w = np.asarray(world, dtype=float)
    n = np.cross(w[INDEX_MCP] - w[WRIST], w[PINKY_MCP] - w[WRIST])
    norm = np.linalg.norm(n)
    return None if norm < 1e-9 else n / norm


def rotation_series(t_s: Sequence[float], worlds: Sequence[Optional[np.ndarray]],
                    ref_normal: Optional[np.ndarray] = None
                    ) -> tuple[np.ndarray, np.ndarray, float]:
    """(times, angle_deg from the palm-down reference, fraction of frames with a hand).

    ref_normal: palm normal while the hand is held palm-down (the app captures it just
    before recording). Without it the first tracked frame is the reference.
    """
    pairs = [(t, palm_normal(w)) for t, w in zip(t_s, worlds) if w is not None]
    pairs = [(t, n) for t, n in pairs if n is not None]
    visible = len(pairs) / len(t_s) if len(t_s) else 0.0
    if len(pairs) < 3:
        return np.zeros(0), np.zeros(0), visible
    ref = pairs[0][1] if ref_normal is None else np.asarray(ref_normal, float)
    ref = ref / np.linalg.norm(ref)
    t = np.array([p[0] for p in pairs])
    ang = np.degrees(np.arccos(np.clip([float(np.dot(n, ref)) for _, n in pairs], -1, 1)))
    return t, ang, visible


def swings(t: np.ndarray, ang: np.ndarray) -> tuple[list[float], list[float]]:
    """(end time, size in deg) of each rotation swing >= MIN_SWING_DEG."""
    if len(ang) < 3:
        return [], []
    hi, _ = find_peaks(ang, prominence=MIN_SWING_DEG)
    lo, _ = find_peaks(-ang, prominence=MIN_SWING_DEG)
    turns = sorted([0, *hi.tolist(), *lo.tolist(), len(ang) - 1])
    times, sizes = [], []
    for a, b in zip(turns[:-1], turns[1:]):
        size = abs(float(ang[b] - ang[a]))
        if size >= MIN_SWING_DEG:
            times.append(float(t[b]))
            sizes.append(size)
    return times, sizes


def fuse(t_s: Sequence[float], worlds: Sequence[Optional[np.ndarray]], switch_half_flips: int,
         t_start: float, duration_s: float,
         ref_normal: Optional[np.ndarray] = None) -> FusionResult:
    t, ang, visible = rotation_series(t_s, worlds, ref_normal)
    ok = visible >= MIN_VISIBLE and len(t) > 0
    st, sizes = swings(t, ang) if ok else ([], [])
    rel = [x - t_start for x in st]
    n_cam = len(sizes)
    big = max(n_cam, switch_half_flips)
    locked = ok and big > 0 and abs(n_cam - switch_half_flips) / big <= AGREE_TOL
    med = float(np.median(sizes)) if sizes else 0.0
    first = [s for s, r in zip(sizes, rel) if r < EDGE_WINDOW_S]
    last = [s for s, r in zip(sizes, rel) if r > duration_s - EDGE_WINDOW_S]
    dec = 0.0
    if first and last and np.median(first) > 0:
        dec = max(0.0, 1.0 - float(np.median(last)) / float(np.median(first)))
    return FusionResult(ok, visible, n_cam, switch_half_flips, locked, med, dec,
                        tuple(rel), tuple(sizes))


def current_angle(worlds_recent: Sequence[Optional[np.ndarray]], ref_normal: Optional[np.ndarray]
                  ) -> Optional[float]:
    """Live gauge: angle of the latest normal vs the reference."""
    if ref_normal is None:
        return None
    for w in reversed(list(worlds_recent)):
        if w is not None:
            n = palm_normal(w)
            if n is not None:
                return float(np.degrees(np.arccos(np.clip(np.dot(n, ref_normal), -1, 1))))
    return None
