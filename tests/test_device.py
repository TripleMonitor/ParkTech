import pytest

from device import (IMPAIRED_FLIPS, NORMAL_FLIPS, ClockMapper, MockDevice, SwitchEvent,
                    lcd_command, parse_line)
from flipping_analysis import analyze_flipping
from scoring import score_flipping


@pytest.mark.parametrize("line", [
    "", "\r\n", "READY", "S,", "S,12", "S,12,2", "S,12,1,0", "S,x,1", "S,-5,1",
    "S,nan,1", "S,inf,0", "T,1,2,3,4", "garbage!!", "\x00\xff", "S,12,", "s,12,1",
])
def test_parser_rejects_bad_lines_without_crashing(line):
    assert parse_line(line) is None


def test_parser_accepts_good_lines():
    assert parse_line("S,1234,1\r\n") == (1234.0, 1)
    assert parse_line(b"S,5,0\n") == (5.0, 0)
    assert parse_line(b"\xff\xfeS,7,1") == (7.0, 1)       # garbage bytes before a line


def test_parse_beat():
    from device import parse_beat
    assert parse_beat("C,1234\r\n") == 1234.0
    for bad in ("", "C,", "C,x", "C,-1", "C,1,2", "S,1,1", "c,5"):
        assert parse_beat(bad) is None


def test_mock_metronome_and_simulated_patient():
    clock = FakeClock()
    dev = MockDevice(clock=clock)
    dev.sim_hand = "Left"
    dev.start()
    dev.cue_on(400)
    beats, flips = [], []
    while clock.t < 10.0:
        clock.t += 1 / 30
        beats += dev.drain_beats()
        flips += dev.drain()
    assert len(beats) == pytest.approx(25, abs=1)          # 10 s at 400 ms
    assert len(flips) >= 20                                # patient answers most beats
    dev.tempo(5000)                                        # clamped to 1500 ms
    n = len(beats)
    while clock.t < 16.0:
        clock.t += 1 / 30
        beats += dev.drain_beats()
    assert len(beats) - n <= 5
    dev.cue_off()


def test_lcd_command_truncates_and_sanitises():
    assert lcd_command("A" * 20, "x|y,z") == "LCD," + "A" * 16 + "|x/y z"


def test_clock_mapper_uses_lowest_latency():
    m = ClockMapper()
    assert m.to_pc(1000, 50.020) == pytest.approx(50.020)    # first: 20 ms latency assumed 0
    assert m.to_pc(2000, 51.005) == pytest.approx(51.005)    # lower latency -> new offset
    assert m.to_pc(3000, 52.200) == pytest.approx(52.005)    # late arrival doesn't shift time


class FakeClock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t


def run_mock(hand, seconds=10.0):
    clock = FakeClock()
    dev = MockDevice(seed=3, clock=clock, test_seconds=seconds)
    dev.sim_hand = hand
    clock.t = 5.0
    dev.start()
    t0 = clock()
    events = []
    while clock.t < t0 + seconds:
        clock.t += 1 / 30
        events += dev.drain()
    dev.stop()
    return analyze_flipping([e.t for e in events], [e.state for e in events], t0, seconds)


def test_mock_normal_left_hand_scores_zero():
    f = run_mock("Left")
    assert f.flips_per_sec == pytest.approx(NORMAL_FLIPS.rate, rel=0.15)
    assert score_flipping(f).score == 0


def test_mock_impaired_right_hand_scores_worse():
    r = score_flipping(run_mock("Right"))
    assert r.score >= 2, r.reasons


def test_mock_streams_only_when_started_and_answers_state():
    clock = FakeClock()
    dev = MockDevice(clock=clock)
    clock.t = 3.0
    assert dev.drain() == []
    dev.request_state()
    assert dev.drain() == [SwitchEvent(3.0, 0)]
    dev.start()
    clock.t = 6.0
    assert len(dev.drain()) > 5
    dev.stop()
    clock.t = 9.0
    assert dev.drain() == []


def test_mock_not_flipping():
    clock = FakeClock()
    dev = MockDevice(clock=clock)
    dev.flipping = False
    dev.start()
    clock.t = 10.0
    assert len(dev.drain()) == 1           # only the STATE reply


def test_mock_records_outputs():
    dev = MockDevice(clock=FakeClock())
    dev.beep(3)
    dev.led("R")
    dev.lcd("hello", "world")
    assert dev.beeps == 3 and dev.last_led == "R" and dev.last_lcd == ("hello", "world")
    with pytest.raises(ValueError):
        dev.led("PURPLE")


def test_profiles_are_distinct():
    assert IMPAIRED_FLIPS.rate < NORMAL_FLIPS.rate and IMPAIRED_FLIPS.pauses
