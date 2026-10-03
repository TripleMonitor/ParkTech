"""App state machine tests, headless, on a fake clock."""
import pytest

from app import KEY_SPACE, TESTS, App, led_for
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
    for i in range(len(TESTS)):
        assert app.state == "ready"
        drive(app, clock, KEY_SPACE)             # ready -> countdown
        n = 0
        while app.state != "done":
            drive(app, clock)
            n += 1
            assert n < 2000, f"stuck in {app.state}"
        drive(app, clock, KEY_SPACE)             # done -> ready / results
    assert app.state == "results"


@pytest.mark.parametrize("target", ["ready", "countdown", "recording", "done", "results",
                                    "dashboard", "trend"])
def test_r_restarts_from_any_screen(tmp_path, target):
    app, clock = make_app(tmp_path)
    drive(app, clock, KEY_SPACE)
    for _ in range(5000):
        if app.state == target:
            break
        key = KEY_SPACE if app.state in ("ready", "done", "results", "dashboard") else -1
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
    """Hand only visible for the first part of the tremor test."""
    class Vanishing(FakeHand):
        def read(self, now, hand, detect, test="tapping"):
            hf = super().read(now, hand, detect, test)
            if test == "tremor" and now - self._t0 > 3.0:
                return HandFrame(hf.frame, None, None)
            return hf

    app, clock = make_app(tmp_path, hands=Vanishing(seed=0), seconds=10.0)
    run_to_results(app, clock)
    r = app.results[("tremor", "Right")].result
    assert r.score is None and r.reasons


def test_stuck_flip_sensor_scores_4_with_sensor_check(tmp_path):
    app, clock = make_app(tmp_path)
    app.device.flipping = False
    run_to_results(app, clock)
    r = app.results[("flipping", "Left")].result
    assert r.score == 4 and any("taped on" in s for s in r.reasons)


def test_every_test_has_a_ready_screen(tmp_path):
    app, clock = make_app(tmp_path)
    seen = []
    drive(app, clock, KEY_SPACE)
    for _ in range(len(TESTS)):
        seen.append((app.state, app.test))
        drive(app, clock, KEY_SPACE)
        while app.state != "done":
            drive(app, clock)
        drive(app, clock, KEY_SPACE)
    assert [s for s, _ in seen] == ["ready"] * len(TESTS)


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
    r = app.results[("flipping", "Right")].result
    assert r.score is None and any("disconnected" in s for s in r.reasons)
    assert app.results[("tremor", "Right")].result.score is not None   # camera still works


class NoHand(FakeHand):
    ok = False
    status = "Camera 0 unavailable"

    def read(self, now, hand, detect, test="tapping"):
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


def test_t_and_f_sim_toggles(tmp_path):
    app, clock = make_app(tmp_path)
    drive(app, clock, ord("t"))
    assert app.hands.tremor_enabled is False
    drive(app, clock, ord("f"))
    assert app.device.flipping is False


def test_ready_screen_has_no_stale_data_from_previous_test(tmp_path):
    app, clock = make_app(tmp_path)
    drive(app, clock, KEY_SPACE)
    drive(app, clock, KEY_SPACE)
    while app.state != "done":
        drive(app, clock)
    assert app.trem_lm                       # right-hand data present on the done screen
    drive(app, clock, KEY_SPACE)             # -> ready for the left hand
    assert app.state == "ready" and app.trem_lm == []


def test_flipping_calibration_uses_ready_state(tmp_path):
    app, clock = make_app(tmp_path)
    app.device.state = 1
    drive(app, clock, KEY_SPACE)
    while app.test[0] != "flipping":
        drive(app, clock, KEY_SPACE if app.state in ("ready", "done") else -1)
    assert app.state == "ready" and app._palm_down() is True
    drive(app, clock, KEY_SPACE)
    assert app.calib == app.device.state


def test_r_aborts_coach_and_turns_metronome_off(tmp_path):
    app, clock = make_app(tmp_path)
    drive(app, clock, ord("c"))
    assert app.state == "coach" and app.coach.phase == "ready"
    drive(app, clock, KEY_SPACE)
    for _ in range(int(30 * 15)):              # countdown + uncued 10 s -> cued
        drive(app, clock)
    assert app.coach.phase == "cued" and app.device.cue
    drive(app, clock, ord("r"))
    assert app.state == "welcome" and app.device.cue is False


def test_dose_input(tmp_path):
    app, clock = make_app(tmp_path)
    for k in (ord("1"), ord("."), ord("5"), 8, ord("7")):
        drive(app, clock, k)
    assert app.dose_hours == 1.7
    drive(app, clock, ord("n"))
    assert app.dose_hours is None and app.dose_unknown
