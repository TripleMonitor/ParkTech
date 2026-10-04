"""Metric helpers for the dataset evaluation."""
import pytest

from tapping_analysis import TappingFeatures
from tools.hubu_eval import Clip, cross_validate, current_params, qwk, score_with
from scoring import score_tapping


def test_qwk_perfect_and_opposite():
    assert qwk([0, 1, 2, 3], [0, 1, 2, 3]) == pytest.approx(1.0)
    assert qwk([0, 0, 3, 3], [3, 3, 0, 0]) < 0


def test_score_with_matches_app_scorer():
    import itertools
    for n, rate, amp, dec, cv, hes in itertools.product((3, 30), (1.5, 3.0), (0.3, 0.9),
                                                        (0.1, 0.4), (0.1, 0.4), (0, 2)):
        f = TappingFeatures(n, rate, amp, dec, cv, hes, 1.0, 10.0)
        assert score_with(f, current_params()) == score_tapping(f).score


def test_cross_validation_runs_by_participant():
    clips = []
    for i in range(20):
        r = i % 4
        f = TappingFeatures(30, 3.5 - 0.6 * r, 1.0 - 0.15 * r, 0.1 + 0.1 * r, 0.1, 0, 1.0, 10.0)
        clips.append(Clip(f"v{i}", f"p{i // 2}", r, f))
    m, t, p, chosen = cross_validate(clips, folds=5)
    assert m["n"] == 20 and len(chosen) == 5
