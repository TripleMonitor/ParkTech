import numpy as np
import pytest

from tapping_analysis import analyze_tapping, normalised_distance

FPS, SECONDS = 30, 10.0


def taps(rate=3.0, amp=1.0, shrink=0.0, pauses=(), closed=0.2, noise=0.0, drop_every=0, seed=0):
    """Synthetic distance signal. Pauses hold the fingers closed for 1 s each.

    Phase only advances outside pauses, so taps resume cleanly afterwards.
    """
    rng = np.random.default_rng(seed)
    times, dists, phase = [], [], 0.0
    for i in range(int(SECONDS * FPS)):
        t = i / FPS
        paused = any(p <= t < p + 1.0 for p in pauses)
        if not paused:
            phase += rate / FPS
        a = amp * (1 - shrink * t / SECONDS)
        d = closed if paused else closed + a * (0.5 - 0.5 * np.cos(2 * np.pi * phase))
        d += rng.normal(0, noise) if noise else 0.0
        times.append(t)
        dists.append(None if drop_every and i % drop_every == 0 else d)
    return times, dists


def test_normalised_distance():
    pts = np.zeros((21, 2))
    pts[9] = [0, 100]
    pts[8] = [50, 0]
    assert normalised_distance(pts) == 0.5
    assert normalised_distance(np.zeros((21, 2))) is None


def test_steady_3hz():
    f = analyze_tapping(*taps())
    assert 28 <= f.n_taps <= 31
    assert f.mean_amplitude == pytest.approx(1.0, abs=0.05)
    assert f.decrement < 0.05 and f.interval_cv < 0.1 and f.hesitations == 0


def test_shrinking_amplitude_decrement():
    f = analyze_tapping(*taps(shrink=0.5))
    assert f.decrement > 0.3


def test_two_pauses_two_hesitations():
    f = analyze_tapping(*taps(pauses=(3.0, 6.5)))
    assert f.hesitations == 2


def test_flat_signal_no_taps():
    f = analyze_tapping([i / FPS for i in range(300)], [0.2] * 300)
    assert f.n_taps == 0


def test_noise_and_dropped_frames_are_tolerated():
    f = analyze_tapping(*taps(noise=0.02, drop_every=4))
    assert 0.7 < f.hand_visible < 0.8
    assert 28 <= f.n_taps <= 31


def test_no_hand_at_all():
    f = analyze_tapping([i / FPS for i in range(300)], [None] * 300)
    assert f.hand_visible == 0 and f.n_taps == 0
