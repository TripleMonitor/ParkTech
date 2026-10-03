"""Finger tapping: MediaPipe hand tracking, features, overlay.

Signal: distance thumb-tip(4) <-> index-tip(8), normalised by the hand size
wrist(0) <-> middle-MCP(9) so it doesn't depend on distance to the camera.
Each opening (local maximum) is one tap.
"""
from __future__ import annotations

from typing import Optional, Sequence

import cv2
import numpy as np
from scipy.signal import find_peaks

from scoring import TappingFeatures

THUMB_TIP, INDEX_TIP, WRIST, MIDDLE_MCP = 4, 8, 0, 9
GRID_HZ = 60.0
MIN_TAP_PROMINENCE = 0.15   # normalised units: smaller wiggles aren't taps
MIN_TAP_GAP_S = 0.08
WINDOW_S = 3.0              # decrement compares first vs last 3 s
HESITATION_FACTOR = 2.0

GREEN, YELLOW, RED, WHITE = (80, 200, 80), (0, 210, 255), (60, 60, 230), (255, 255, 255)


def normalised_distance(landmarks: np.ndarray) -> float:
    """landmarks: (21, 2) array in pixels."""
    scale = np.linalg.norm(landmarks[WRIST] - landmarks[MIDDLE_MCP])
    if scale < 1e-6:
        return 0.0
    return float(np.linalg.norm(landmarks[THUMB_TIP] - landmarks[INDEX_TIP]) / scale)


def compute_features(times: Sequence[float], dists: Sequence[Optional[float]]) -> TappingFeatures:
    """times in seconds; dists None where no hand was found."""
    n_frames = len(times)
    duration = (times[-1] - times[0]) if n_frames > 1 else 0.0
    valid = [(t, d) for t, d in zip(times, dists) if d is not None]
    visible = len(valid) / n_frames if n_frames else 0.0
    if len(valid) < 10 or duration <= 0:
        return TappingFeatures(0, 0.0, 0.0, 0.0, 0.0, 0, visible, duration)

    t_valid = np.array([v[0] for v in valid]) - times[0]
    d_valid = np.array([v[1] for v in valid])
    grid = np.arange(0.0, t_valid[-1], 1.0 / GRID_HZ)
    sig = np.interp(grid, t_valid, d_valid)
    peaks, props = find_peaks(sig, prominence=MIN_TAP_PROMINENCE,
                              distance=max(1, int(MIN_TAP_GAP_S * GRID_HZ)))
    n = len(peaks)
    if n == 0:
        return TappingFeatures(0, 0.0, 0.0, 0.0, 0.0, 0, visible, duration)

    peak_t = grid[peaks]
    amps = props["prominences"]
    first = amps[peak_t < WINDOW_S]
    last = amps[peak_t > duration - WINDOW_S]
    decrement = 0.0
    if len(first) and len(last) and first.mean() > 0:
        decrement = max(0.0, 1.0 - last.mean() / first.mean())

    intervals = np.diff(peak_t)
    cv, hesitations = 0.0, 0
    if len(intervals) >= 2:
        cv = float(intervals.std() / intervals.mean())
        hesitations = int((intervals > HESITATION_FACTOR * np.median(intervals)).sum())

    return TappingFeatures(n_taps=n, taps_per_sec=n / duration,
                           mean_amplitude=float(amps.mean()), decrement=decrement,
                           interval_cv=cv, hesitations=hesitations,
                           hand_visible=visible, duration_s=duration)


class TappingTracker:
    """Runs MediaPipe Hands on mirrored webcam frames and records the tap signal."""

    def __init__(self) -> None:
        import mediapipe as mp
        self._hands = mp.solutions.hands.Hands(
            static_image_mode=False, max_num_hands=1, model_complexity=0,
            min_detection_confidence=0.6, min_tracking_confidence=0.5)
        self.reset()

    def reset(self) -> None:
        self.times: list[float] = []
        self.dists: list[Optional[float]] = []

    def detect(self, frame_bgr: np.ndarray) -> tuple[Optional[np.ndarray], Optional[str]]:
        """Returns ((21,2) pixel landmarks, 'Left'/'Right') or (None, None)."""
        h, w = frame_bgr.shape[:2]
        res = self._hands.process(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
        if not res.multi_hand_landmarks:
            return None, None
        lm = res.multi_hand_landmarks[0].landmark
        pts = np.array([[p.x * w, p.y * h] for p in lm])
        label = res.multi_handedness[0].classification[0].label
        return pts, label

    def process(self, frame_bgr: np.ndarray, t: float, expected_hand: str,
                record: bool) -> Optional[float]:
        """Detect, draw overlay onto frame in place, optionally record. Returns distance."""
        pts, label = self.detect(frame_bgr)
        d = normalised_distance(pts) if pts is not None else None
        if record:
            self.times.append(t)
            self.dists.append(d)
        draw_overlay(frame_bgr, pts, label, d, expected_hand)
        return d

    def features(self) -> TappingFeatures:
        return compute_features(self.times, self.dists)

    def close(self) -> None:
        self._hands.close()


def draw_overlay(frame: np.ndarray, pts: Optional[np.ndarray], label: Optional[str],
                 d: Optional[float], expected_hand: str) -> None:
    if pts is None:
        cv2.putText(frame, "Show your hand to the camera", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, RED, 2)
        return
    for i, p in enumerate(pts.astype(int)):
        r = 7 if i in (THUMB_TIP, INDEX_TIP) else 3
        cv2.circle(frame, tuple(p), r, YELLOW if r > 3 else WHITE, -1)
    a, b = tuple(pts[THUMB_TIP].astype(int)), tuple(pts[INDEX_TIP].astype(int))
    cv2.line(frame, a, b, GREEN, 3)
    cv2.line(frame, tuple(pts[WRIST].astype(int)), tuple(pts[MIDDLE_MCP].astype(int)), WHITE, 1)
    cv2.putText(frame, f"d = {d:.2f}", (b[0] + 10, b[1] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, GREEN, 2)
    # Frame is mirrored before detection, so MediaPipe's label matches the user's hand.
    if label and label.lower() != expected_hand.lower():
        cv2.putText(frame, f"That looks like your {label.upper()} hand", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, YELLOW, 2)
