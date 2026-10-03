import numpy as np

from scoring import score_tapping
from tapping_analysis import analyze_tapping
from tapping_tracker import IMPAIRED, NORMAL, FakeHand, HandFrame, draw_distance_graph, draw_hand


def record(hand, profile):
    fh = FakeHand({hand: profile}, seed=0)
    fh.reset(0.0)
    times, dists = [], []
    for i in range(300):
        t = i / 30
        hf = fh.read(t, hand, detect=True)
        times.append(t)
        dists.append(hf.distance)
    return times, dists, hf


def test_fake_hand_landmarks_give_requested_distance():
    fh = FakeHand()
    for d in (0.15, 0.5, 1.2):
        pts = fh.landmarks_for(d)
        assert pts.shape == (21, 2)
        assert abs(HandFrame(None, pts, "Right").distance - d) < 1e-9


def test_fake_normal_hand_scores_zero():
    times, dists, _ = record("Left", NORMAL)
    f = analyze_tapping(times, dists)
    assert 3.0 <= f.taps_per_sec <= 3.8
    assert score_tapping(f).score == 0


def test_fake_impaired_hand_scores_worse():
    times, dists, _ = record("Right", IMPAIRED)
    r = score_tapping(analyze_tapping(times, dists))
    assert r.score >= 2, r.reasons


def test_drawing_does_not_crash():
    times, dists, hf = record("Right", NORMAL)
    frame = hf.frame.copy()
    draw_hand(frame, hf, "Left")              # wrong-hand warning path
    draw_hand(frame, HandFrame(frame, None, None), "Right")
    canvas = np.zeros((300, 600, 3), np.uint8)
    draw_distance_graph(canvas, times, dists, times[-1], 10, 10, 500, 200)
    assert canvas.any()


def test_fake_hand_tremor_mode_measures_set_size():
    from scoring import score_tremor
    from tremor_analysis import analyze_tremor
    fh = FakeHand(tremor_cm={"Right": 2.0, "Left": 0.0}, seed=1)
    for hand, expected_score in (("Right", 2), ("Left", 0)):
        times, lms = [], []
        for i in range(300):
            t = i / 30
            hf = fh.read(t, hand, detect=True, test="tremor")
            times.append(t)
            lms.append(hf.landmarks)
        f = analyze_tremor(times, lms)
        assert score_tremor(f).score == expected_score
        if hand == "Right":
            assert abs(f.displacement_cm - 2.0) < 0.2 and abs(f.peak_hz - 5.0) < 0.3


def test_fake_hand_tremor_toggle():
    fh = FakeHand(seed=1)
    fh.tremor_enabled = False
    a = fh.palm_at(0.05, "Right")
    b = fh.palm_at(0.10, "Right")
    assert abs(a - b).max() < 4      # jitter only


def test_tremor_overlay_draws():
    fh = FakeHand(seed=1)
    hf = fh.read(0.1, "Right", detect=True, test="tremor")
    frame = hf.frame.copy()
    draw_hand(frame, hf, "Right", test="tremor")
    assert (frame != hf.frame).any()
