"""Camera-only hand-flipping trial: pure maths (no app, no hardware)."""
import numpy as np

from dashboard import merge_flipping, neuroscore
from flipping_analysis import analyze_half_flips
from fusion import FusionResult
from scoring import CAMERA_NOT_TRACKED, score_flipping_camera


def fusion(ok=True, n=20, amp=160.0, dec=0.0, visible=1.0):
    return FusionResult(ok, visible, n, 0, False, amp, dec)


def test_half_flips_from_relative_times():
    t = np.arange(0.25, 10.0, 0.25)                 # 2 full flips/s
    f = analyze_half_flips(t.tolist() + [-1.0, 11.0], 10.0)  # outside the window ignored
    assert f.half_flips == len(t) and f.full_flips == len(t) // 2
    assert abs(f.flips_per_sec - 1.95) < 0.06 and f.interval_cv < 0.01 and f.hesitations == 0


def test_camera_trial_scores_good_flips_0():
    f = analyze_half_flips(np.arange(0.25, 10.0, 0.25).tolist(), 10.0)
    r = score_flipping_camera(f, fusion())
    assert r.score == 0 and r.reasons


def test_camera_trial_flags_small_and_slow_flips():
    f = analyze_half_flips(np.arange(0.5, 10.0, 0.5).tolist(), 10.0)   # 1 full flip/s
    r = score_flipping_camera(f, fusion(amp=80.0))
    assert r.score == 2 and any("Slow" in x for x in r.reasons) \
        and any("Small flips" in x for x in r.reasons)


def test_camera_trial_unscored_when_hand_not_tracked():
    f = analyze_half_flips([], 10.0)
    r = score_flipping_camera(f, fusion(ok=False, visible=0.2))
    assert r.score is None and r.reasons[0].startswith(CAMERA_NOT_TRACKED) and "20%" in r.reasons[0]
    assert score_flipping_camera(f, None).score is None


def test_camera_trial_no_flips_is_4_without_sensor_wording():
    r = score_flipping_camera(analyze_half_flips([], 10.0), fusion(n=0))
    assert r.score == 4 and not any("taped" in x for x in r.reasons)


def test_merge_prefers_sensor_trial_and_falls_back_to_camera():
    scores = {("tremor", "Right"): 0, ("flipcam", "Right"): 3, ("flipping", "Right"): 1,
              ("flipcam", "Left"): 2, ("flipping", "Left"): None}
    m = merge_flipping(scores)
    assert m == {("tremor", "Right"): 0, ("flipping", "Right"): 1, ("flipping", "Left"): 2}
    ns, formula = neuroscore(m)                     # 3 scored tests, flipping not doubled
    assert "4 x 3" in formula and ns is not None
