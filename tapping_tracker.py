"""Hand sources for the camera tests (tremor + finger tapping) + drawing helpers.

CameraHand: webcam + MediaPipe Hands.  FakeHand: synthetic landmarks for --sim.
Both implement:  read(now, hand, detect, test) -> HandFrame,  reset(now),  close().
The frame is mirrored (selfie view) so MediaPipe's Left/Right label matches the user.
"""
from __future__ import annotations

import logging
import math
import random
from dataclasses import dataclass
from typing import NamedTuple, Optional, Sequence

import cv2
import numpy as np

from tapping_analysis import INDEX_TIP, MIDDLE_MCP, THUMB_TIP, WRIST, normalised_distance

log = logging.getLogger(__name__)

GREEN, YELLOW, RED, WHITE, GREY = (90, 200, 90), (0, 210, 255), (60, 60, 230), (240, 240, 240), (150, 150, 150)
ACCENT = (255, 190, 60)
FONT = cv2.FONT_HERSHEY_SIMPLEX
HAND_CONNECTIONS = ((0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8),
                    (5, 9), (9, 10), (10, 11), (11, 12), (9, 13), (13, 14), (14, 15),
                    (15, 16), (13, 17), (17, 18), (18, 19), (19, 20), (0, 17))


class HandFrame(NamedTuple):
    frame: Optional[np.ndarray]        # BGR image to show (None = no camera)
    landmarks: Optional[np.ndarray]    # (21, 2) pixels, None = no hand found
    label: Optional[str]               # "Left"/"Right" as seen by MediaPipe

    @property
    def distance(self) -> Optional[float]:
        return None if self.landmarks is None else normalised_distance(self.landmarks)


# --- real camera ---------------------------------------------------------------------
class CameraHand:
    def __init__(self, index: int = 0):
        self._cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
        if not self._cap.isOpened():
            self._cap = cv2.VideoCapture(index)
        self.ok = self._cap.isOpened()
        self.status = "Camera OK" if self.ok else f"Camera {index} unavailable"
        self._hands = None
        if self.ok:
            import mediapipe as mp
            self._hands = mp.solutions.hands.Hands(
                static_image_mode=False, max_num_hands=1, model_complexity=0,
                min_detection_confidence=0.6, min_tracking_confidence=0.5)

    def reset(self, now: float) -> None:
        pass

    def read(self, now: float, hand: str, detect: bool, test: str = "tapping") -> HandFrame:
        if not self.ok:
            return HandFrame(None, None, None)
        ok, raw = self._cap.read()
        if not ok or raw is None:
            self.status = "Camera read failed"
            return HandFrame(None, None, None)
        self.status = "Camera OK"
        frame = cv2.flip(raw, 1)
        if not detect:
            return HandFrame(frame, None, None)
        res = self._hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        if not res.multi_hand_landmarks:
            return HandFrame(frame, None, None)
        h, w = frame.shape[:2]
        pts = np.array([[p.x * w, p.y * h] for p in res.multi_hand_landmarks[0].landmark])
        label = res.multi_handedness[0].classification[0].label
        return HandFrame(frame, pts, label)

    def close(self) -> None:
        if self._hands is not None:
            self._hands.close()
        self._cap.release()


# --- synthetic hand --------------------------------------------------------------------
@dataclass(frozen=True)
class TapProfile:
    rate: float = 3.5          # taps per second at the start
    amp: float = 1.0           # opening, normalised units
    shrink: float = 0.0        # fraction of amplitude lost by the end of 10 s
    slowing: float = 0.0       # fraction of rate lost by the end of 10 s
    closed: float = 0.15
    noise: float = 0.01


NORMAL = TapProfile()
IMPAIRED = TapProfile(rate=1.6, amp=0.8, shrink=0.5, slowing=0.2)

# 640x480 template, wrist->middle-MCP = 150 px. Points 3, 4, 7, 8 are set per frame.
_TEMPLATE = np.array([
    (380, 430), (330, 400), (295, 365), (0, 0), (0, 0),          # wrist, thumb
    (330, 300), (300, 255), (0, 0), (0, 0),                      # index
    (380, 280), (380, 215), (380, 170), (380, 130),              # middle
    (425, 290), (430, 230), (432, 190), (434, 155),              # ring
    (465, 310), (475, 265), (480, 235), (485, 205)], float)      # pinky
_PINCH = np.array([250.0, 285.0])
_SCALE = 150.0


# Open palm facing the camera (tremor test): fingers spread, wrist->middle-MCP = 150 px.
_PALM = np.array([
    (320, 430), (270, 400), (235, 360), (210, 325), (190, 295),   # wrist, thumb
    (280, 290), (270, 230), (265, 195), (262, 165),               # index
    (320, 280), (320, 215), (320, 175), (320, 140),               # middle
    (360, 290), (370, 232), (375, 197), (378, 168),               # ring
    (395, 310), (410, 265), (418, 238), (424, 212)], float)
_TREMOR_DIR = np.array([np.cos(np.radians(20)), np.sin(np.radians(20))])
_PX_PER_CM = _SCALE / 9.0      # tremor_analysis assumes wrist->middle-MCP = 9 cm


class FakeHand:
    """Synthetic hand for --sim.

    tapping: per-hand TapProfile, e.g. {"Right": IMPAIRED, "Left": NORMAL}.
    tremor:  palm held still with a `tremor_hz` oscillation of tremor_cm[hand] cm
             (toggle all tremor with `tremor_enabled`), plus +/-1.5 px landmark jitter.
    """

    status = "Simulated hand"
    ok = True

    def __init__(self, profiles: Optional[dict] = None, seed: Optional[int] = None,
                 test_seconds: float = 10.0, tremor_cm: Optional[dict] = None,
                 tremor_hz: float = 5.0):
        self.profiles = dict(profiles or {"Right": IMPAIRED, "Left": NORMAL})
        self.tremor_cm = dict(tremor_cm if tremor_cm is not None else {"Right": 1.5, "Left": 0.0})
        self.tremor_hz = tremor_hz
        self.tremor_enabled = True
        self._rng = random.Random(seed)
        self._seconds = test_seconds
        self._t0 = 0.0
        self._phase = 0.0
        self._last_t = 0.0

    def reset(self, now: float) -> None:
        self._t0, self._last_t, self._phase = now, now, 0.0

    def distance_at(self, now: float, hand: str) -> float:
        p = self.profiles.get(hand, NORMAL)
        t = max(0.0, now - self._t0)
        frac = min(1.0, t / self._seconds)
        self._phase += p.rate * (1 - p.slowing * frac) * max(0.0, now - self._last_t)
        self._last_t = now
        a = p.amp * (1 - p.shrink * frac)
        return p.closed + a * (0.5 - 0.5 * math.cos(2 * math.pi * self._phase)) \
            + self._rng.gauss(0, p.noise)

    def landmarks_for(self, d: float) -> np.ndarray:
        pts = _TEMPLATE.copy()
        half = max(d, 0.0) * _SCALE / 2
        pts[INDEX_TIP] = _PINCH + (0, -half)
        pts[THUMB_TIP] = _PINCH + (0, half)
        pts[7] = (pts[6] + pts[8]) / 2
        pts[3] = (pts[2] + pts[4]) / 2
        return pts

    def palm_at(self, now: float, hand: str) -> np.ndarray:
        cm = self.tremor_cm.get(hand, 0.0) if self.tremor_enabled else 0.0
        shift = _TREMOR_DIR * cm * _PX_PER_CM * math.sin(2 * math.pi * self.tremor_hz * now)
        jitter = np.array([[self._rng.uniform(-1.5, 1.5) for _ in range(2)] for _ in range(21)])
        return _PALM + shift + jitter

    def read(self, now: float, hand: str, detect: bool, test: str = "tapping") -> HandFrame:
        frame = np.full((480, 640, 3), (60, 50, 45), np.uint8)
        cv2.putText(frame, "SIMULATED HAND (--sim)", (20, 460), FONT, 0.6, GREY, 1, cv2.LINE_AA)
        if not detect:
            return HandFrame(frame, None, None)
        if test == "tremor":
            return HandFrame(frame, self.palm_at(now, hand), hand)
        return HandFrame(frame, self.landmarks_for(self.distance_at(now, hand)), hand)

    def close(self) -> None:
        pass


# --- drawing -------------------------------------------------------------------------------
def draw_hand(frame: np.ndarray, hf: HandFrame, expected_hand: str, test: str = "tapping") -> None:
    """Skeleton plus the test's key landmarks, drawn onto `frame` in place.

    tapping: thumb-index line and normalised distance.  tremor: index fingertip ring.
    """
    if hf.landmarks is None:
        cv2.putText(frame, "Show your hand to the camera", (20, 40), FONT, 0.9, RED, 2, cv2.LINE_AA)
        return
    if test == "tremor":
        pts = hf.landmarks.astype(int)
        for a, b in HAND_CONNECTIONS:
            cv2.line(frame, tuple(pts[a]), tuple(pts[b]), WHITE, 2, cv2.LINE_AA)
        for p in pts:
            cv2.circle(frame, tuple(p), 4, ACCENT, -1, cv2.LINE_AA)
        cv2.circle(frame, tuple(pts[INDEX_TIP]), 12, YELLOW, 2, cv2.LINE_AA)
        cv2.line(frame, tuple(pts[WRIST]), tuple(pts[MIDDLE_MCP]), GREEN, 2, cv2.LINE_AA)
        cv2.putText(frame, "9 cm", tuple(((pts[WRIST] + pts[MIDDLE_MCP]) // 2) + (8, 0)), FONT,
                    0.55, GREEN, 1, cv2.LINE_AA)
        if hf.label and hf.label.lower() != expected_hand.lower():
            cv2.putText(frame, f"That looks like your {hf.label.upper()} hand", (20, 40), FONT,
                        0.8, YELLOW, 2, cv2.LINE_AA)
        return
    pts = hf.landmarks.astype(int)
    for a, b in HAND_CONNECTIONS:
        cv2.line(frame, tuple(pts[a]), tuple(pts[b]), WHITE, 2, cv2.LINE_AA)
    for i, p in enumerate(pts):
        cv2.circle(frame, tuple(p), 7 if i in (THUMB_TIP, INDEX_TIP) else 4,
                   YELLOW if i in (THUMB_TIP, INDEX_TIP) else ACCENT, -1, cv2.LINE_AA)
    cv2.line(frame, tuple(pts[THUMB_TIP]), tuple(pts[INDEX_TIP]), GREEN, 4, cv2.LINE_AA)
    cv2.line(frame, tuple(pts[WRIST]), tuple(pts[MIDDLE_MCP]), GREY, 1, cv2.LINE_AA)
    d = hf.distance
    if d is not None:
        tip = pts[INDEX_TIP]
        cv2.putText(frame, f"d = {d:.2f}", (int(tip[0]) + 12, int(tip[1])), FONT, 0.7, GREEN, 2,
                    cv2.LINE_AA)
    if hf.label and hf.label.lower() != expected_hand.lower():
        cv2.putText(frame, f"That looks like your {hf.label.upper()} hand", (20, 40), FONT, 0.8,
                    YELLOW, 2, cv2.LINE_AA)


def draw_distance_graph(canvas: np.ndarray, times: Sequence[float],
                        dists: Sequence[Optional[float]], now: float,
                        x: int, y: int, w: int, h: int, window_s: float = 5.0,
                        vmax: float = 1.6) -> None:
    """Scrolling plot of the last `window_s` seconds of thumb-index distance."""
    cv2.rectangle(canvas, (x, y), (x + w, y + h), (52, 46, 42), -1)
    cv2.putText(canvas, "thumb-index distance", (x + 8, y + 18), FONT, 0.5, GREY, 1, cv2.LINE_AA)
    pts = []
    for t, d in zip(times, dists):
        if d is None or now - t > window_s:
            continue
        px = x + int(w * (1 - (now - t) / window_s))
        py = y + h - int(h * min(d, vmax) / vmax)
        pts.append((px, py))
    if len(pts) >= 2:
        cv2.polylines(canvas, [np.array(pts, np.int32)], False, GREEN, 2, cv2.LINE_AA)
