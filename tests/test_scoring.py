"""Scoring tests from synthetic signals with known answers."""
import pytest

from flipping_analysis import FlippingFeatures, analyze_flipping
from scoring import (SENSOR_CHECK, ScoreResult, asymmetry, flipping_asymmetry, score_flipping,
                     score_tapping, score_tremor, tapping_asymmetry, tremor_asymmetry)
from tapping_analysis import TappingFeatures, analyze_tapping
from tests.test_flipping_analysis import flips
from tests.test_tapping_analysis import taps
from tests.test_tremor_analysis import track
from tremor_analysis import TremorFeatures, analyze_tremor


def tremor(**kw):
    return analyze_tremor(*track(jitter_px=2.0, **kw))


def flip(**kw):
    return analyze_flipping(*flips(**kw), t_start=100.0, duration_s=10.0)


# --- camera tremor ---------------------------------------------------------------------
@pytest.mark.parametrize("cm,score", [(0.5, 1), (2.0, 2), (5.0, 3)])
def test_tremor_scores(cm, score):
    r = score_tremor(tremor(amp_cm=cm))
    assert r.score == score
    assert any("5.0 Hz in Parkinson's range" in s for s in r.reasons)


def test_tremor_score_4():
    assert score_tremor(tremor(amp_cm=12.0)).score == 4


def test_tremor_jitter_only_scores_zero_with_reason():
    r = score_tremor(tremor(amp_cm=0.0))
    assert r.score == 0 and "no clear tremor" in r.reasons[0].lower() and "Hz" in r.reasons[0]


def test_tremor_outside_pd_range_says_so():
    r = score_tremor(tremor(freq=7.0, amp_cm=2.0))
    assert r.score == 2 and "outside Parkinson's range" in r.reasons[0]


def test_tremor_no_hand_is_unscored():
    t, lms = track(amp_cm=2.0)
    r = score_tremor(analyze_tremor(t, [None] * len(lms)))
    assert r.score is None and "Hand seen" in r.reasons[0]


def test_tremor_partial_recording_is_unscored():
    r = score_tremor(tremor(amp_cm=2.0, seconds=5.0), expected_s=10.0)
    assert r.score is None and "of 10 s" in r.reasons[0]


# --- finger tapping -------------------------------------------------------------------------
def test_tapping_steady_3hz_scores_zero():
    assert score_tapping(analyze_tapping(*taps())).score == 0


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
    r = score_tapping(TappingFeatures(20, 1.0, 0.3, 0.5, 0.7, 2, 1.0, 10.0))   # 5 problems
    assert r.score == 3 and len(r.reasons) == 5


def test_tapping_no_hand_is_unscored():
    f = analyze_tapping([i / 30 for i in range(300)], [None] * 300)
    assert score_tapping(f).score is None


def test_tracking_dropout_scores_zero_with_note():
    times, dists = taps(rate=3.5)
    dists = [None if 4.0 <= t < 4.8 else d for t, d in zip(times, dists)]
    r = score_tapping(analyze_tapping(times, dists))
    assert r.score == 0 and any("tracking lost" in s for s in r.reasons)


# --- hand flipping --------------------------------------------------------------------------
def test_flipping_steady_2_per_s_scores_zero():
    assert score_flipping(flip(rate=2.0)).score == 0


def test_flipping_slowing_flags_decrement():
    r = score_flipping(flip(rate=2.0, slowing=0.4))
    assert r.score >= 1 and any("Slows down" in s for s in r.reasons)


def test_flipping_two_pauses():
    f = flip(rate=2.0, pauses=(3.0, 6.5))
    r = score_flipping(f)
    assert f.hesitations == 2 and any("2 hesitation" in s for s in r.reasons)


def test_flipping_no_flips_scores_4_with_sensor_check():
    r = score_flipping(analyze_flipping([100.0], [0], t_start=100.0, duration_s=10.0))
    assert r.score == 4 and SENSOR_CHECK in r.reasons


def test_flipping_few_flips_scores_4():
    r = score_flipping(flip(rate=0.3))
    assert r.score == 4 and "Barely able" in r.reasons[0]


def test_flipping_problem_count():
    r = score_flipping(FlippingFeatures(20, 10, 1.0, 0.5, 0.5, 2, 10.0, 0.5))
    assert r.score == 3 and len(r.reasons) == 4


# --- asymmetry ------------------------------------------------------------------------------
def test_asymmetry_right3_left0_flagged():
    assert asymmetry("Tapping", ScoreResult(3, ["x"]), ScoreResult(0, ["y"]))


def test_asymmetry_right1_left1_not_flagged():
    assert asymmetry("Tapping", ScoreResult(1, ["x"]), ScoreResult(1, ["y"]),
                     "taps/s", 3.0, 2.9) == []


def test_asymmetry_feature_25_percent():
    flags = asymmetry("Tapping", ScoreResult(0, ["x"]), ScoreResult(0, ["y"]), "taps/s", 4.0, 2.9)
    assert len(flags) == 1 and "taps/s" in flags[0]


def test_tremor_asymmetry_ignores_noise_levels():
    a, b = tremor(amp_cm=0.0, seed=1), tremor(amp_cm=0.0, seed=2)
    assert tremor_asymmetry(a, b, score_tremor(a), score_tremor(b)) == []


def test_tremor_asymmetry_ignores_unclear_peaks():
    r = TremorFeatures(True, 10, 1.0, 0.06, 5.0, 0.15, False, 2.0, 0.0)
    l = TremorFeatures(True, 10, 1.0, 0.06, 5.0, 0.02, False, 2.0, 0.0)
    assert tremor_asymmetry(r, l, score_tremor(r), score_tremor(l)) == []


def test_tremor_asymmetry_real_difference():
    a, b = tremor(amp_cm=2.0), tremor(amp_cm=0.0)
    flags = tremor_asymmetry(a, b, score_tremor(a), score_tremor(b))
    assert any("Right worse" in f for f in flags) and any("displacement" in f for f in flags)


def test_tapping_asymmetry_from_signals():
    r, l = analyze_tapping(*taps(rate=1.5)), analyze_tapping(*taps(rate=3.0))
    assert tapping_asymmetry(r, l, score_tapping(r), score_tapping(l))


def test_flipping_asymmetry_rate_only():
    r, l = flip(rate=1.6), flip(rate=2.4)          # both score 0, rate differs 33%
    rs, ls = score_flipping(r), score_flipping(l)
    assert rs.score == ls.score == 0
    flags = flipping_asymmetry(r, l, rs, ls)
    assert len(flags) == 1 and "flips/s" in flags[0]


def test_flipping_asymmetry_similar_not_flagged():
    r, l = flip(rate=2.0), flip(rate=2.1)
    assert flipping_asymmetry(r, l, score_flipping(r), score_flipping(l)) == []


# --- every score has a reason ------------------------------------------------------------------
def test_every_score_has_a_reason():
    results = [score_tremor(tremor(amp_cm=c)) for c in (0, 0.5, 2, 5, 12)]
    results.append(score_tremor(analyze_tremor([], [])))
    results += [score_tapping(analyze_tapping(*taps(**kw))) for kw in
                ({}, {"shrink": 0.5}, {"pauses": (3.0, 6.5)}, {"rate": 1.0, "amp": 0.3})]
    results.append(score_tapping(analyze_tapping([i / 30 for i in range(300)], [0.2] * 300)))
    results += [score_flipping(flip(**kw)) for kw in
                ({"rate": 2.0}, {"rate": 2.0, "slowing": 0.4}, {"rate": 1.0, "pauses": (3.0,)},
                 {"rate": 0.3})]
    results.append(score_flipping(analyze_flipping([100.0], [0], 100.0, 10.0)))
    assert {r.score for r in results} >= {None, 0, 1, 2, 3, 4}
    for r in results:
        assert r.reasons and all(isinstance(s, str) and s for s in r.reasons)
