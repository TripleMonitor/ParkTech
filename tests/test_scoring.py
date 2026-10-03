from dataclasses import replace

from scoring import (ScoreResult, TappingFeatures, TremorFeatures, score_tapping,
                     score_tremor, tapping_asymmetry, tremor_asymmetry,
                     tremor_displacement_cm)


def tremor(peak_hz=5.0, disp=0.5, clear=True):
    return TremorFeatures(peak_hz=peak_hz, peak_accel=0.0, displacement_cm=disp,
                          has_clear_peak=clear, duration_s=10)


NORMAL_TAP = TappingFeatures(n_taps=40, taps_per_sec=4.0, mean_amplitude=0.9,
                             decrement=0.05, interval_cv=0.1, hesitations=0,
                             hand_visible=1.0, duration_s=10)


def test_displacement_formula():
    # 1 cm at 5 Hz needs a = 0.01 * (2*pi*5)^2 ~= 9.87 m/s^2
    assert abs(tremor_displacement_cm(9.8696, 5.0) - 1.0) < 1e-3


def test_tremor_no_peak_scores_zero():
    assert score_tremor(tremor(clear=False)).score == 0


def test_tremor_amplitude_bands():
    assert [score_tremor(tremor(disp=d)).score for d in (0.05, 0.5, 2, 5, 12)] == [0, 1, 2, 3, 4]


def test_tremor_reason_mentions_pd_range():
    reasons = score_tremor(tremor(peak_hz=5.2)).reasons
    assert any("Parkinson's range" in r and "5.2 Hz" in r for r in reasons)


def test_tapping_normal_is_zero():
    assert score_tapping(NORMAL_TAP).score == 0


def test_tapping_problem_counts():
    one = replace(NORMAL_TAP, interval_cv=0.4)
    two = replace(one, decrement=0.4)
    many = replace(two, mean_amplitude=0.3, taps_per_sec=1.5)
    assert score_tapping(one).score == 1
    assert score_tapping(two).score == 2
    assert score_tapping(many).score == 3
    assert len(score_tapping(many).reasons) == 4


def test_tapping_barely_able():
    assert score_tapping(replace(NORMAL_TAP, n_taps=2)).score == 4


def test_tapping_no_hand_is_unscored():
    assert score_tapping(replace(NORMAL_TAP, hand_visible=0.2)).score is None


def test_asymmetry_flags_score_and_feature():
    r, l = NORMAL_TAP, replace(NORMAL_TAP, taps_per_sec=2.5)
    flags = tapping_asymmetry(r, l, ScoreResult(0, ()), ScoreResult(1, ()))
    assert any("left side worse" in f for f in flags)
    assert any("taps/s" in f for f in flags)


def test_tremor_asymmetry_ignores_noise_level():
    flags = tremor_asymmetry(tremor(disp=0.01), tremor(disp=0.05),
                             ScoreResult(0, ()), ScoreResult(0, ()))
    assert flags == ()
