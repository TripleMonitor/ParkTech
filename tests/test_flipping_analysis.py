import numpy as np
import pytest

from flipping_analysis import analyze_flipping, debounce

DUR = 10.0


def flips(rate=2.0, slowing=0.0, pauses=(), bounce_every=0, seconds=DUR, start=100.0,
          linear=False):
    """Synthetic switch reports: half-flip every 1/(2*rate). Pauses are (start, 1 s).

    slowing: the last 4 s run `slowing` slower than the first 4 s (linear ramp
    from 4 s to 6 s). linear=True instead ramps over the whole test.
    bounce_every=n adds a 3-change bounce burst (within 20 ms) to every n-th change.
    """
    def r(t):
        if linear:
            return rate * (1 - slowing * t / seconds)
        k = min(1.0, max(0.0, (t - 4.0) / 2.0))
        return rate * (1 - slowing * k)

    times, states, t, s, i = [start], [0], 0.0, 0, 0
    while True:
        t += 1.0 / (2 * r(t))
        for p in pauses:
            if p <= t < p + 1.0:
                t = p + 1.0
        if t > seconds:
            break
        s = 1 - s
        i += 1
        if bounce_every and i % bounce_every == 0:
            times += [start + t, start + t + 0.008, start + t + 0.018]
            states += [s, 1 - s, s]
        else:
            times.append(start + t)
            states.append(s)
    return times, states


def test_debounce_ignores_bounce_and_state_repeats():
    t = [0.0, 1.0, 1.008, 1.018, 1.5, 1.5, 2.0, 2.005]
    s = [0, 1, 0, 1, 1, 1, 0, 1]          # burst at 1.0 = one change; repeats ignored;
    # 2.0 -> 2.005 returns to 1 within 5 ms -> net nothing
    assert debounce(t, s) == [(1.0, 1)]


def test_steady_2_flips_per_second():
    f = analyze_flipping(*flips(rate=2.0), t_start=100.0, duration_s=DUR)
    assert f.flips_per_sec == pytest.approx(2.0, abs=0.1)
    assert f.full_flips in (19, 20)
    assert f.interval_cv < 0.05 and f.hesitations == 0 and f.decrement < 0.05


def test_slowing_rate_gives_decrement():
    f = analyze_flipping(*flips(rate=2.0, slowing=0.4), t_start=100.0, duration_s=DUR)
    assert f.decrement == pytest.approx(0.4, abs=0.05)


def test_linear_40pct_over_whole_test_is_borderline():
    """Documented boundary: a 40% drop spread linearly over 10 s compares the
    first 4 s with the last 4 s, which only differ by ~25%."""
    f = analyze_flipping(*flips(rate=2.0, slowing=0.4, linear=True), t_start=100.0,
                         duration_s=DUR)
    assert 0.2 < f.decrement < 0.3


def test_two_pauses_two_hesitations():
    f = analyze_flipping(*flips(rate=2.0, pauses=(3.0, 6.5)), t_start=100.0, duration_s=DUR)
    assert f.hesitations == 2


def test_bounce_bursts_counted_once():
    clean = analyze_flipping(*flips(rate=2.0), t_start=100.0, duration_s=DUR)
    bouncy = analyze_flipping(*flips(rate=2.0, bounce_every=2), t_start=100.0, duration_s=DUR)
    assert bouncy.half_flips == clean.half_flips


def test_no_flips():
    f = analyze_flipping([100.0], [0], t_start=100.0, duration_s=DUR)
    assert f.half_flips == 0 and f.first_change_s is None and f.stuck_at_start


def test_late_start_is_flagged_stuck():
    times, states = flips(rate=2.0)
    keep = [(t, s) for t, s in zip(times, states) if t == 100.0 or t > 104.0]
    t2, s2 = zip(*keep)
    f = analyze_flipping(list(t2), list(s2), t_start=100.0, duration_s=DUR)
    assert f.stuck_at_start and f.first_change_s > 3.0


def test_changes_outside_window_ignored():
    times, states = flips(rate=2.0, seconds=20.0)
    f = analyze_flipping(times, states, t_start=100.0, duration_s=DUR)
    assert f.flips_per_sec == pytest.approx(2.0, abs=0.1)


def test_unsorted_input_is_sorted():
    times, states = flips(rate=2.0)
    idx = np.random.default_rng(0).permutation(len(times))
    f = analyze_flipping([times[i] for i in idx], [states[i] for i in idx],
                         t_start=100.0, duration_s=DUR, initial_state=0)
    assert f.flips_per_sec == pytest.approx(2.0, abs=0.1)


def test_live_rate_and_colour():
    from flipping_analysis import live_colour, live_rate
    fast = [i * 0.25 for i in range(1, 40)]          # 2 full flips/s
    assert live_rate(fast, 1.0) is None
    assert live_rate(fast, 6.0) == pytest.approx(2.0)
    assert live_colour(live_rate(fast, 6.0)) == "G"
    slow = [i * 0.4 for i in range(1, 30)]           # 1.25 full flips/s
    assert live_colour(live_rate(slow, 6.0)) == "Y"
    assert live_colour(live_rate([], 6.0)) == "R"    # stopped


def test_closures_only_mode():
    from flipping_analysis import analyze_closures
    steady = [100.0 + 0.5 * k for k in range(1, 21)]            # 2 full flips/s
    f = analyze_closures(steady, 100.0, 10.0)
    assert f.full_flips == 20 and f.half_flips == 40
    assert f.flips_per_sec == pytest.approx(2.0) and f.interval_cv < 0.01 and f.hesitations == 0
    bouncy = sorted(steady + [t + 0.02 for t in steady[::3]])    # bounce closures 20 ms later
    assert analyze_closures(bouncy, 100.0, 10.0).full_flips == 20
    paused = [t for t in steady if not (104.0 < t < 105.6)]
    assert analyze_closures(paused, 100.0, 10.0).hesitations == 1
    assert analyze_closures([], 100.0, 10.0).first_change_s is None
