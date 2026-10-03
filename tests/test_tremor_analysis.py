import numpy as np
import pytest

from tremor_analysis import analyze_samples, analyze_tremor, displacement_cm

G = 9.81


def signal(freq=5.0, accel=4.93, seconds=10.0, fs=100.0, noise=0.02, jitter_ms=0.0, seed=0):
    rng = np.random.default_rng(seed)
    t = np.arange(0, seconds, 1 / fs)
    if jitter_ms:
        t = t + rng.uniform(-jitter_ms, jitter_ms, len(t)) / 1000.0
    ax = accel * np.sin(2 * np.pi * freq * t) + rng.normal(0, noise, len(t))
    ay = rng.normal(0, noise, len(t))
    az = G + rng.normal(0, noise, len(t))
    return t, ax, ay, az


def test_displacement_formula():
    assert displacement_cm(4.93, 5.0) == pytest.approx(0.4995, abs=1e-3)


@pytest.mark.parametrize("accel,cm", [(4.93, 0.5), (19.7, 2.0), (49.0, 5.0)])
def test_5hz_sine_frequency_and_displacement(accel, cm):
    f = analyze_tremor(*signal(accel=accel))
    assert f.valid and f.clear_peak
    assert f.peak_hz == pytest.approx(5.0, abs=0.3)
    assert f.displacement_cm == pytest.approx(cm, rel=0.05)
    assert f.window_pct >= 80


def test_noise_only_has_tiny_displacement():
    f = analyze_tremor(*signal(accel=0.0))
    assert f.displacement_cm < 0.1
    assert f.window_pct == 0


def test_slow_1hz_movement_is_outside_band():
    f = analyze_tremor(*signal(freq=1.0, accel=3.0))
    assert f.displacement_cm < 0.1


def test_uneven_timestamps_still_find_frequency():
    f = analyze_tremor(*signal(freq=5.5, accel=10.0, jitter_ms=3.0))
    assert f.peak_hz == pytest.approx(5.5, abs=0.3)
    assert f.displacement_cm == pytest.approx(displacement_cm(10.0, 5.5), rel=0.1)


def test_tremor_on_other_axis_through_gravity():
    t, ax, ay, az = signal(accel=0.0)
    az = az + 8.0 * np.sin(2 * np.pi * 4.5 * t)
    f = analyze_tremor(t, ax, ay, az)
    assert f.peak_hz == pytest.approx(4.5, abs=0.3)
    assert f.displacement_cm == pytest.approx(displacement_cm(8.0, 4.5), rel=0.05)


def test_too_short_is_invalid():
    f = analyze_tremor(*signal(seconds=1.0))
    assert not f.valid


def test_analyze_samples_from_device_tuples():
    t, ax, ay, az = signal()
    samples = list(zip(t * 1000, ax, ay, az))
    assert analyze_samples(samples).peak_hz == pytest.approx(5.0, abs=0.3)
    assert not analyze_samples([]).valid
