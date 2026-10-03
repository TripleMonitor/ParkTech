"""Fusion on synthetic hand rotations with known answers."""
import numpy as np
import pytest

from fusion import fuse, palm_normal, rotation_series

# canonical palm (metres): fingers along -z (towards the camera), palm-down normal along y
PALM = np.zeros((21, 3))
PALM[5] = (-0.03, 0.0, -0.085)
PALM[17] = (0.035, 0.0, -0.07)
PALM[9] = (0.0, 0.0, -0.09)


def rot(axis, deg):
    a = np.radians(deg)
    c, s = np.cos(a), np.sin(a)
    if axis == "z":
        return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])          # about x


def rotating(rate=2.0, amp_deg=160.0, seconds=10.0, fps=30.0, axis="z", decay=0.0,
             drop_every=0, noise_deg=0.0, seed=0):
    """Hand pronating/supinating: angle swings 0 <-> amp at `rate` full flips/s."""
    rng = np.random.default_rng(seed)
    t = np.arange(0, seconds, 1 / fps)
    worlds = []
    for i, ti in enumerate(t):
        a = amp_deg * (1 - decay * ti / seconds)
        ang = a * (0.5 - 0.5 * np.cos(2 * np.pi * rate * ti)) + rng.normal(0, noise_deg)
        w = PALM @ rot(axis, ang).T
        worlds.append(None if drop_every and i % drop_every == 0 else w)
    return list(t), worlds


def test_palm_normal_is_unit():
    n = palm_normal(PALM)
    assert np.linalg.norm(n) == pytest.approx(1.0)


@pytest.mark.parametrize("axis", ["z", "x"])
def test_rotation_angle_independent_of_forearm_axis(axis):
    t, worlds = rotating(axis=axis)
    _, ang, vis = rotation_series(t, worlds)
    assert vis == 1.0 and ang.max() == pytest.approx(160.0, abs=3) and ang.min() < 3


def test_counts_agree_and_amplitude():
    t, worlds = rotating(rate=2.0, amp_deg=160.0, noise_deg=3.0)
    r = fuse(t, worlds, switch_half_flips=40, t_start=0.0, duration_s=10.0)
    assert r.camera_half_flips == pytest.approx(40, abs=2)
    assert r.locked and r.median_amplitude_deg == pytest.approx(160, abs=8)
    assert r.amplitude_decrement < 0.1


def test_mismatch_when_switch_count_differs():
    t, worlds = rotating(rate=2.0)
    r = fuse(t, worlds, switch_half_flips=25, t_start=0.0, duration_s=10.0)
    assert not r.locked and "MISMATCH" in r.status


def test_small_and_shrinking_amplitude():
    t, worlds = rotating(rate=1.5, amp_deg=100.0, decay=0.5)
    r = fuse(t, worlds, switch_half_flips=30, t_start=0.0, duration_s=10.0)
    assert r.median_amplitude_deg < 120 and r.amplitude_decrement > 0.25


def test_dropped_frames_still_count():
    t, worlds = rotating(rate=2.0, drop_every=4)
    r = fuse(t, worlds, switch_half_flips=40, t_start=0.0, duration_s=10.0)
    assert r.locked


def test_no_hand_is_switch_only():
    t, _ = rotating()
    r = fuse(t, [None] * len(t), switch_half_flips=40, t_start=0.0, duration_s=10.0)
    assert not r.camera_ok and not r.locked and "switch only" in r.status


def test_wobble_is_not_a_flip():
    t, worlds = rotating(rate=2.0, amp_deg=20.0)
    r = fuse(t, worlds, switch_half_flips=0, t_start=0.0, duration_s=10.0)
    assert r.camera_half_flips == 0
