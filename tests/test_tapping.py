import math

import numpy as np

from scoring import score_tapping
from tapping import compute_features, normalised_distance


def synth(freq=4.0, amp=0.9, seconds=10, fps=30, decay=0.0, drop_every=0, pause_at=None):
    times, dists = [], []
    for i in range(int(seconds * fps)):
        t = i / fps
        if pause_at and pause_at[0] <= t < pause_at[1]:
            d = 0.2
        else:
            a = amp * (1 - decay * t / seconds)
            d = 0.2 + a * (0.5 - 0.5 * math.cos(2 * math.pi * freq * t))
        times.append(t)
        dists.append(None if drop_every and i % drop_every == 0 else d)
    return times, dists


def test_normalised_distance():
    pts = np.zeros((21, 2))
    pts[9] = [0, 100]      # hand length 100 px
    pts[4] = [0, 0]
    pts[8] = [50, 0]
    assert normalised_distance(pts) == 0.5


def test_normal_tapping():
    f = compute_features(*synth())
    assert abs(f.taps_per_sec - 4.0) < 0.3
    assert abs(f.mean_amplitude - 0.9) < 0.05
    assert f.interval_cv < 0.1 and f.hesitations == 0
    assert score_tapping(f).score == 0


def test_decrement_detected():
    f = compute_features(*synth(decay=0.6))
    assert f.decrement > 0.4


def test_hesitation_detected():
    f = compute_features(*synth(pause_at=(4.0, 5.0)))
    assert f.hesitations >= 1


def test_tolerates_dropped_frames():
    f = compute_features(*synth(drop_every=5))
    assert 0.75 < f.hand_visible < 0.85
    assert abs(f.taps_per_sec - 4.0) < 0.3


def test_no_hand():
    times = [i / 30 for i in range(300)]
    f = compute_features(times, [None] * 300)
    assert f.hand_visible == 0 and score_tapping(f).score is None
