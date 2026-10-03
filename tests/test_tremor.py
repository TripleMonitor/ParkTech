import math
import random

from device import MockDevice, Sample
from scoring import score_tremor
from tremor import compute_features


def synth(freq=5.0, accel=5.0, seconds=10, noise=0.05, seed=0):
    rng = random.Random(seed)
    out = []
    for i in range(int(seconds * 100)):
        t = i / 100
        a = accel * math.sin(2 * math.pi * freq * t)
        # tremor perpendicular to gravity: |a| would barely change
        out.append(Sample(i * 10.0, a + rng.gauss(0, noise), rng.gauss(0, noise),
                          9.81 + rng.gauss(0, noise)))
    return out


def test_detects_5hz_tremor_and_amplitude():
    f = compute_features(synth(freq=5.0, accel=9.8696))
    assert f.has_clear_peak
    assert abs(f.peak_hz - 5.0) < 0.15
    assert abs(f.displacement_cm - 1.0) < 0.1


def test_noise_only_has_no_clear_peak():
    f = compute_features(synth(accel=0.0))
    assert not f.has_clear_peak
    assert score_tremor(f).score == 0


def test_too_few_samples():
    assert not compute_features(synth(seconds=0.5)).has_clear_peak


def test_mock_device_tremor_scores_two():
    dev = MockDevice(seed=3)
    dev.tremor_on = True
    samples = [dev._sample(i * 10.0) for i in range(1000)]
    result = score_tremor(compute_features(samples))
    assert result.score == 2, result.reasons
