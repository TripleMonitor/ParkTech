import pytest

from quality import signal_quality


def frames(n=300, fps=30.0):
    return [i / fps for i in range(n)]


def test_perfect_recording():
    q = signal_quality(frames(), [True] * 300, [0.95] * 300)
    assert q.detected == 1.0 and q.confidence == pytest.approx(0.95)
    assert q.dropped_frames == 0 and not q.low


def test_timing_gaps_count_as_dropped_frames():
    t = frames()
    t = t[:100] + t[110:]                       # 10 frames missing
    q = signal_quality(t, [True] * len(t), [0.9] * len(t))
    assert q.dropped_frames == 10
    assert q.dropped == pytest.approx(10 / 300, abs=0.005)


def test_failed_reads_counted():
    q = signal_quality(frames(), [True] * 300, [0.9] * 300, failed_reads=40)
    assert q.dropped_frames == 40 and q.low


def test_low_detection_and_confidence_flagged():
    det = [i % 2 == 0 for i in range(300)]
    q = signal_quality(frames(), det, [0.5] * 300)
    assert q.detected == pytest.approx(0.5) and q.confidence == pytest.approx(0.5)
    assert q.low and len(q.problems) == 2


def test_empty():
    q = signal_quality([], [], [])
    assert q.low and q.confidence is None
