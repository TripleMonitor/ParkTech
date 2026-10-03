"""Scoring tests. Signals are synthetic with known answers (see test_*_analysis too)."""
from dataclasses import replace

import numpy as np
import pytest

from scoring import (ScoreResult, asymmetry, score_tapping, score_tremor, tapping_asymmetry,
                     tremor_asymmetry)
from tapping_analysis import TappingFeatures, analyze_tapping
from tests.test_tapping_analysis import taps
from tests.test_tremor_analysis import signal
from tremor_analysis import analyze_tremor


# --- tremor: synthetic signal -> analysis -> score -----------------------------------
@pytest.mark.parametrize("accel,score", [(4.93, 1), (19.7, 2), (49.0, 3)])
def test_tremor_scores_from_sine(accel, score):
    r = score_tremor(analyze_tremor(*signal(accel=accel)))
    assert r.score == score
    assert any("5.0 Hz in Parkinson's range" in s for s in r.reasons)


def test_tremor_score_4_from_large_sine():
    assert score_tremor(analyze_tremor(*signal(accel=110.0))).score == 4


def test_tremor_noise_only_scores_zero_with_reason():
    r = score_tremor(analyze_tremor(*signal(accel=0.0)))
    assert r.score == 0
    assert "no clear tremor" in r.reasons[0].lower()


def test_tremor_slow_1hz_scores_zero():
    assert score_tremor(analyze_tremor(*signal(freq=1.0, accel=3.0))).score == 0


def test_tremor_outside_pd_range_says_so():
    r = score_tremor(analyze_tremor(*signal(freq=7.0, accel=30.0)))
    assert r.score >= 1 and "outside Parkinson's range" in r.reasons[0]


def test_tremor_invalid_data_is_unscored():
    r = score_tremor(analyze_tremor(*signal(seconds=1.0)))
    assert r.score is None and r.reasons


# --- tapping ----------------------------------------------------------------------------
def test_tapping_steady_3hz_scores_zero():
    r = score_tapping(analyze_tapping(*taps()))
    assert r.score == 0


def test_tapping_shrinking_flags_decrement():
    r = score_tapping(analyze_tapping(*taps(shrink=0.5)))
    assert r.score >= 1 and any("shrink" in s for s in r.reasons)


def test_tapping_two_pauses_scores_at_least_2():
    r = score_tapping(analyze_tapping(*taps(pauses=(3.0, 6.5))))
    assert r.score >= 2 and any("hesitation" in s for s in r.reasons)


def test_tapping_flat_scores_4():
    f = analyze_tapping([i / 30 for i in range(300)], [0.2] * 300)
    assert score_tapping(f).score == 4


def test_tapping_problem_count_caps_at_3():
    f = TappingFeatures(20, 1.5, 0.3, 0.5, 0.5, 2, 1.0, 10.0)
    r = score_tapping(f)
    assert r.score == 3 and len(r.reasons) == 5


def test_tapping_no_hand_is_unscored():
    f = analyze_tapping([i / 30 for i in range(300)], [None] * 300)
    assert score_tapping(f).score is None


# --- asymmetry --------------------------------------------------------------------------
def test_asymmetry_right3_left0_flagged():
    assert asymmetry("Tapping", ScoreResult(3, ["x"]), ScoreResult(0, ["y"]))


def test_asymmetry_right1_left1_not_flagged():
    assert asymmetry("Tapping", ScoreResult(1, ["x"]), ScoreResult(1, ["y"]),
                     "taps/s", 3.0, 2.9) == []


def test_asymmetry_feature_25_percent():
    flags = asymmetry("Tapping", ScoreResult(0, ["x"]), ScoreResult(0, ["y"]), "taps/s", 4.0, 2.9)
    assert len(flags) == 1 and "taps/s" in flags[0]


def test_tremor_asymmetry_ignores_noise_levels():
    a, b = analyze_tremor(*signal(accel=0.0, seed=1)), analyze_tremor(*signal(accel=0.0, seed=2))
    assert tremor_asymmetry(a, b, score_tremor(a), score_tremor(b)) == []


def test_tremor_asymmetry_real_difference():
    a, b = analyze_tremor(*signal(accel=19.7)), analyze_tremor(*signal(accel=0.0))
    flags = tremor_asymmetry(a, b, score_tremor(a), score_tremor(b))
    assert any("Right worse" in f for f in flags) and any("displacement" in f for f in flags)


def test_tapping_asymmetry_from_signals():
    r, l = analyze_tapping(*taps(rate=1.5)), analyze_tapping(*taps(rate=3.0))
    flags = tapping_asymmetry(r, l, score_tapping(r), score_tapping(l))
    assert flags


# --- every score has a reason -------------------------------------------------------------
def test_every_score_has_a_reason():
    tremors = [analyze_tremor(*signal(accel=a)) for a in (0, 4.93, 19.7, 49, 110)]
    tremors.append(analyze_tremor(*signal(seconds=1.0)))
    tapping = [analyze_tapping(*taps(**kw)) for kw in
               ({}, {"shrink": 0.5}, {"pauses": (3.0, 6.5)}, {"rate": 1.0, "amp": 0.3})]
    tapping.append(analyze_tapping([i / 30 for i in range(300)], [0.2] * 300))
    results = [score_tremor(f) for f in tremors] + [score_tapping(f) for f in tapping]
    assert {r.score for r in results} >= {0, 1, 2, 3, 4}
    for r in results:
        assert r.reasons and all(isinstance(s, str) and s for s in r.reasons)
