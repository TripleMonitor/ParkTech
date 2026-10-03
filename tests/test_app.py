"""Headless smoke test: drive the whole session flow with simulated time."""
from app import App, KEY_SPACE, TESTS
from device import MockDevice, Sample
from history import load_sessions


class FakeClockDevice(MockDevice):
    """MockDevice whose samples follow the test's fake clock."""

    def __init__(self):
        super().__init__(seed=0)
        self.now_ms = 0.0
        self.beeps = 0

    def _now_ms(self):
        return self.now_ms

    def beep(self, n=1):
        self.beeps += n


def run_session(tmp_path, tremor_on):
    dev = FakeClockDevice()
    dev.tremor_on = tremor_on
    path = str(tmp_path / "s.csv")
    app = App(dev, tracker=None, seconds=2.0, history_path=path, sim=True)
    now = 0.0

    def tick(key=-1):
        dev.now_ms = now * 1000
        return app.tick(key, now, None)

    tick(KEY_SPACE)                      # welcome -> ready
    for _ in TESTS:
        assert app.state == "ready"
        tick(KEY_SPACE)                  # -> countdown
        while app.state != "ready" and app.state != "results":
            now += 1 / 30
            img = tick()
            assert img.shape == (720, 1280, 3)
    return app, dev, path


def test_full_session_with_tremor(tmp_path):
    app, dev, path = run_session(tmp_path, tremor_on=True)
    assert app.state == "results"
    assert app.results[("tremor", "Right")].result.score == 2
    assert app.results[("tapping", "Right")].result.score is None   # no camera
    assert dev.beeps == 4 * (3 + 1)                                 # 3-2-1 + done
    assert dev.last_led == "R"
    rows = load_sessions(path)
    assert len(rows) == 1 and rows[0]["tremor_R"] == "2"


def test_full_session_without_tremor(tmp_path):
    app, dev, _ = run_session(tmp_path, tremor_on=False)
    assert app.results[("tremor", "Left")].result.score == 0
    assert dev.last_led == "G"


def test_trend_screen_renders(tmp_path):
    app, _, _ = run_session(tmp_path, tremor_on=False)
    img = app.tick(ord("h"), 999.0, None)
    assert app.state == "trend" and img.shape == (720, 1280, 3)
    app.tick(KEY_SPACE, 999.1, None)
    assert app.state == "results"
