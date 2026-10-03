"""The WHY THIS SCORE panel must always agree with the scorer."""
import itertools

import pytest

from flipping_analysis import FlippingFeatures
from scoring import (explain_flipping, explain_tapping, explain_tremor, score_flipping,
                     score_from_rules, score_tapping, score_tremor)
from tapping_analysis import TappingFeatures
from tremor_analysis import TremorFeatures


@pytest.mark.parametrize("clear,cm,hz", list(itertools.product(
    (True, False), (0.0, 0.05, 0.1, 0.5, 0.99, 1.0, 2.9, 3.0, 9.9, 10.0, 15.0), (3.5, 5.0, 7.5))))
def test_tremor_panel_matches_scorer(clear, cm, hz):
    f = TremorFeatures(True, 10.0, 1.0, 0.06, hz, cm, clear, 5.0, 50.0)
    e = explain_tremor(f)
    assert e.score == score_tremor(f).score == score_from_rules("tremor", e)


@pytest.mark.parametrize("n,rate,amp,dec,cv,hes", list(itertools.product(
    (3, 30), (1.5, 3.0), (0.3, 0.9), (0.1, 0.4), (0.1, 0.4), (0, 2))))
def test_tapping_panel_matches_scorer(n, rate, amp, dec, cv, hes):
    f = TappingFeatures(n, rate, amp, dec, cv, hes, 1.0, 10.0)
    e = explain_tapping(f)
    assert e.score == score_tapping(f).score == score_from_rules("tapping", e)


@pytest.mark.parametrize("full,rate,cv,dec,hes", list(itertools.product(
    (2, 20), (1.0, 2.2), (0.1, 0.5), (0.1, 0.4), (0, 1))))
def test_flipping_panel_matches_scorer(full, rate, cv, dec, hes):
    f = FlippingFeatures(full * 2, full, rate, cv, dec, hes, 10.0, 0.4)
    e = explain_flipping(f)
    assert e.score == score_flipping(f).score == score_from_rules("flipping", e)


def test_unscored_has_no_rules():
    f = TremorFeatures(False, 1.0, 0.1, 0.0, 0.0, 0.0, False, 0.0, 0.0)
    e = explain_tremor(f)
    assert e.score is None and e.rules == []


def test_rules_show_measured_and_threshold():
    f = TappingFeatures(30, 1.4, 0.9, 0.34, 0.1, 0, 1.0, 10.0)
    rows = {r.label: r for r in explain_tapping(f).rules}
    assert rows["Decrement"].measured == "34%" and rows["Decrement"].threshold == "> 30%"
    assert rows["Decrement"].fired and not rows["Hesitations"].fired
