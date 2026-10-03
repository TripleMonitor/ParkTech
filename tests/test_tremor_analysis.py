"""Camera tremor analysis on synthetic landmark tracks with known answers."""
import numpy as np
import pytest

from tremor_analysis import analyze_tremor, effective_displacement_cm

HAND_PX = 150.0                  # wrist -> middle-MCP in pixels  (= 9 cm)
PX_PER_CM = HAND_PX / 9.0


def track(freq=5.0, amp_cm=0.5, seconds=10.0, fps=30.0, jitter_px=0.0, drop_every=0,
          timing_jitter_ms=0.0, angle_deg=30.0, drift_px=0.0, seed=0):
    """Whole hand oscillating along `angle_deg`; per-landmark pixel jitter."""
    rng = np.random.default_rng(seed)
    t = np.arange(0, seconds, 1 / fps)
    if timing_jitter_ms:
        t = np.sort(t + rng.uniform(-timing_jitter_ms, timing_jitter_ms, len(t)) / 1000)
    base = np.zeros((21, 2))
    base[0] = (320, 420)
    base[9] = (320, 420 - HAND_PX)
    base[8] = (280, 160)
    d = np.array([np.cos(np.radians(angle_deg)), np.sin(np.radians(angle_deg))])
    lms = []
    for i, ti in enumerate(t):
        if drop_every and i % drop_every == 0:
            lms.append(None)
            continue
        shift = d * amp_cm * PX_PER_CM * np.sin(2 * np.pi * freq * ti) + (drift_px * ti / seconds, 0)
        lms.append(base + shift + rng.uniform(-jitter_px, jitter_px, (21, 2)))
    return list(t), lms


@pytest.mark.parametrize("cm", [0.5, 2.0, 5.0])
def test_5hz_frequency_and_displacement(cm):
    f = analyze_tremor(*track(amp_cm=cm, jitter_px=2.0))
    assert f.valid and f.clear_peak
    assert f.peak_hz == pytest.approx(5.0, abs=0.3)
    assert f.displacement_cm == pytest.approx(cm, rel=0.2)
    assert f.cm_per_px == pytest.approx(9.0 / HAND_PX, rel=0.02)


def test_jitter_only_is_tiny():
    f = analyze_tremor(*track(amp_cm=0.0, jitter_px=2.0))
    assert effective_displacement_cm(f) < 0.1


@pytest.mark.parametrize("kw", [{"drop_every": 4}, {"drop_every": 3},
                                {"timing_jitter_ms": 8.0}, {"fps": 22.0},
                                {"fps": 24.0, "drop_every": 5, "timing_jitter_ms": 5.0}])
def test_dropped_frames_and_uneven_timing(kw):
    f = analyze_tremor(*track(amp_cm=2.0, jitter_px=2.0, **kw))
    assert f.peak_hz == pytest.approx(5.0, abs=0.3)
    assert f.displacement_cm == pytest.approx(2.0, rel=0.2)


def test_slow_drift_is_removed():
    f = analyze_tremor(*track(amp_cm=0.0, jitter_px=1.0, drift_px=80.0))
    assert effective_displacement_cm(f) < 0.1


def test_slow_1hz_sway_outside_band():
    f = analyze_tremor(*track(freq=1.0, amp_cm=3.0, jitter_px=1.0))
    assert effective_displacement_cm(f) < 0.1


def test_diagonal_motion_not_under_read():
    for angle in (0, 45, 90):
        f = analyze_tremor(*track(amp_cm=2.0, angle_deg=angle))
        assert f.displacement_cm == pytest.approx(2.0, rel=0.05)


def test_hand_mostly_missing_is_invalid():
    t, lms = track(amp_cm=2.0)
    lms = [lm if i % 3 == 0 else None for i, lm in enumerate(lms)]
    assert not analyze_tremor(t, lms).valid


def test_too_short_is_invalid():
    assert not analyze_tremor(*track(seconds=1.0)).valid
    assert not analyze_tremor([], []).valid
