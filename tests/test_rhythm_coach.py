import numpy as np
import pytest

from rhythm_coach import (HIT_WINDOW_S, MAX_RATE, MIN_RATE, Coach, SimulatedPatient, credit,
                          interval_ms, match_flip, simulate, summarise, warm_start)


def test_credit_is_graded():
    assert credit(0.0) == 1.0 and credit(-0.07) == 1.0
    assert credit(0.1125) == pytest.approx(0.5)
    assert credit(0.2) == 0.0 and credit(None) == 0.0


def test_match_flip_nearest_within_window_and_used_once():
    used = set()
    assert match_flip(1.0, 0.5, [0.7, 0.98, 1.1], used) == pytest.approx(-0.02)
    assert match_flip(1.0, 0.5, [0.7, 0.98, 1.1], used) == pytest.approx(0.1)   # 0.98 used
    assert match_flip(5.0, 0.5, [0.7], set()) is None


def test_interval_clamped():
    assert interval_ms(100.0) == 200 and interval_ms(0.1) == 1500
    assert MIN_RATE == pytest.approx(1000 / 1500) and MAX_RATE == pytest.approx(5.0)


def test_tempo_rises_while_on_time_and_respects_step_limit():
    c = Coach(start_rate=2.0)
    rates = [c.beat_result(i * 0.5, 0.0) for i in range(30)]
    assert rates[-1] > 2.0
    assert max(np.diff([2.0] + rates)) <= 0.15 + 1e-9


def test_tempo_falls_when_late():
    c = Coach(start_rate=3.0)
    for i in range(10):
        c.beat_result(i * 0.33, 0.0)
    r0 = c.rate
    for i in range(10, 25):
        c.beat_result(i * 0.33, 0.2)            # consistently 200 ms late
    assert c.rate < r0


def test_random_miss_after_on_time_beat_is_excluded():
    c = Coach(start_rate=2.0)
    for i in range(8):
        c.beat_result(i * 0.5, 0.0)
    c.beat_result(4.0, None)
    assert c.records[-1].on_time is None and c.on_time_rate() == 1.0


def test_miss_after_late_beat_counts_as_late():
    c = Coach(start_rate=2.0)
    c.beat_result(0.0, 0.2)
    c.beat_result(0.5, None)
    assert c.records[-1].on_time is False


def test_no_windup_during_stop():
    c = Coach(start_rate=2.5)
    for i in range(20):
        c.beat_result(i * 0.4, 0.0)
    before = c.rate
    for i in range(20, 35):                    # 6 s of nothing
        c.beat_result(i * 0.4, None)
    assert c.rate == before                    # no responses -> no change


@pytest.mark.parametrize("kw", [{}, {"miss_p": 0.2}, {"uncued_ratio": 0.6}])
def test_simulation_settles_near_true_max(kw):
    coach, flips = simulate(SimulatedPatient(max_rate=3.0, seed=1, **kw))
    late = [r.rate for r in coach.records if r.t >= coach.records[-1].t - 10]
    assert np.median(late) == pytest.approx(3.0, rel=0.10)


def test_summary():
    p = SimulatedPatient(max_rate=3.0, seed=2)
    coach, flips = simulate(p)
    s = summarise(coach, flips, coach.records[-1].t, 24, 10.0)
    assert s.max_sustainable_rate == pytest.approx(3.0, rel=0.1)
    assert s.uncued_rate == pytest.approx(2.4) and s.beats > 50
    assert s.mean_asynchrony_ms is not None and abs(s.mean_asynchrony_ms) < HIT_WINDOW_S * 1000


def test_warm_start():
    assert warm_start(2.0) == pytest.approx(2.3) and warm_start(0.0) == 1.5
