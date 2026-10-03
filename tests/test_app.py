"""App state machine tests, headless, on a fake clock."""
import pytest

from app import KEY_SPACE, App, led_for
from device import MockDevice
from selftest import FakeClock, checks, run_session
from tapping_tracker import FakeHand, HandFrame


def test_full_selftest_session_passes(tmp_path):
    path = str(tmp_path / "s.csv")
    app, dev, _, trend = run_session(path, seconds=10.0)
    failed = [(name, detail) for name, ok, detail in checks(app, dev, path, trend) if not ok]
    assert failed == []


def drive(app, clock, key=-1, frames=1):
    img = None
    for _ in range(frames):
        img = app.tick(key, clock())
        key = -1
        clock.t += 1 / 30
    return img


def make_app(tmp_path, device=None, hands=None, seconds=2.0):
    clock = FakeClock()
    device = device or MockDevice(seed=0, clock=clock)
    hands = hands or FakeHand(seed=0, test_seconds=seconds)
    app = App(device, hands, seconds=seconds, history_path=str(tmp_path / "s.csv"),
              clock=clock)
    app.restart(clock())
    return app, clock


def run_to_results(app, clock):
    drive(app, clock, KEY_SPACE)                 # welcome -> ready
    for i in range(4):
        assert app.state == "ready"
        drive(app, clock, KEY_SPACE)             # ready -> countdown
        n = 0
        while app.state != "done":
            drive(app, clock)
            n += 1
            assert n < 2000, f"stuck in {app.state}"
        drive(app, clock, KEY_SPACE)             # done -> ready / results
    assert app.state == "results"


@pytest.mark.parametrize("target", ["ready", "countdown", "recording", "done", "results", "trend"])
def test_r_restarts_from_any_screen(tmp_path, target):
    app, clock = make_app(tmp_path)
    drive(app, clock, KEY_SPACE)
    for _ in range(5000):
        if app.state == target:
            break
        key = KEY_SPACE if app.state in ("ready", "done", "results") else -1
        drive(app, clock, key)
    assert app.state == target
    drive(app, clock, ord("r"))
    assert app.state == "welcome" and app.results == {}


def test_q_and_esc_quit(tmp_path):
    for key in (ord("q"), 27):
        app, clock = make_app(tmp_path)
        drive(app, clock, key)
        assert app.running is False


def test_partial_tremor_data_is_unscored(tmp_path):
    """Arduino delivers only the first 2.5 s of a 10 s test (e.g. USB glitch)."""
    clock = FakeClock()

    class Glitchy(MockDevice):
        def drain(self):
            out = super().drain()
            return [s for s in out if s.t_ms - self._start_ms < 2500]

        def start(self):
            super().start()
            self._start_ms = self._now_ms()

    dev = Glitchy(clock=clock)
    dev.tremor_on = True
    app, clock = make_app(tmp_path, device=dev, seconds=10.0)
    dev._clock = clock
    run_to_results(app, clock)
    r = app.results[("tremor", "Right")].result
    assert r.score is None and "of 10 s" in r.reasons[0]


def test_every_test_has_a_ready_screen(tmp_path):
    app, clock = make_app(tmp_path)
    seen = []
    drive(app, clock, KEY_SPACE)
    for _ in range(4):
        seen.append((app.state, app.test))
        drive(app, clock, KEY_SPACE)
        while app.state != "done":
            drive(app, clock)
        drive(app, clock, KEY_SPACE)
    assert [s for s, _ in seen] == ["ready"] * 4


class DeadDevice(MockDevice):
    """Arduino unplugged: no data, connected=False."""
    connected = False
    status = "Arduino disconnected (COM9) - plug it back in"

    def drain(self):
        return []


def test_disconnected_arduino_does_not_crash(tmp_path):
    clock = FakeClock()
    app, clock = make_app(tmp_path, device=DeadDevice(clock=clock))
    img = drive(app, clock)
    assert img.shape == (720, 1280, 3)
    run_to_results(app, clock)
    r = app.results[("tremor", "Right")].result
    assert r.score is None and any("disconnected" in s for s in r.reasons)


class NoHand(FakeHand):
    ok = False
    status = "Camera 0 unavailable"

    def read(self, now, hand, detect):
        return HandFrame(None, None, None)


def test_missing_hand_or_camera_does_not_crash(tmp_path):
    app, clock = make_app(tmp_path, hands=NoHand())
    run_to_results(app, clock)
    r = app.results[("tapping", "Left")].result
    assert r.score is None and r.reasons


def test_led_mapping():
    assert led_for([0, 1, None]) == "G"
    assert led_for([2, 0]) == "Y"
    assert led_for([3, 0]) == "R" and led_for([4]) == "R"
    assert led_for([None, None]) == "OFF"


def test_t_toggles_fake_tremor(tmp_path):
    app, clock = make_app(tmp_path)
    drive(app, clock, ord("t"))
    assert app.device.tremor_on is True
    drive(app, clock, ord("t"))
    assert app.device.tremor_on is False
