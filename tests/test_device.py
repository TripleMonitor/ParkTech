import pytest

from device import ArduinoDevice, MockDevice, lcd_command, parse_line


@pytest.mark.parametrize("line", [
    "", "\r\n", "READY", "T,", "T,12,0.1", "T,1,2,3", "T,1,2,x,4", "T,1,2,3,4,5",
    "garbage!!", "\x00\xff", "T,nan,1,2,3", "T,1,inf,2,3",
])
def test_parser_rejects_bad_lines_without_crashing(line):
    assert parse_line(line) is None


def test_parser_accepts_good_lines():
    s = parse_line("T,1234,0.100,-0.200,9.810\r\n")
    assert s == (1234.0, 0.1, -0.2, 9.81)
    assert parse_line(b"T,5,1,2,3\n") == (5.0, 1.0, 2.0, 3.0)
    assert parse_line(b"\xff\xfeT,1,2,3,4") == (1.0, 2.0, 3.0, 4.0)  # garbage bytes dropped


def test_lcd_command_truncates_and_sanitises():
    assert lcd_command("A" * 20, "x|y,z") == "LCD," + "A" * 16 + "|x/y z"


class FakeClock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t


def test_mock_streams_100hz_only_when_started():
    clock = FakeClock()
    dev = MockDevice(seed=1, clock=clock)
    clock.t = 1.0
    assert dev.drain() == []
    dev.start()
    clock.t = 2.0
    assert len(dev.drain()) in (100, 101)
    dev.stop()
    clock.t = 3.0
    assert dev.drain() == []


def test_mock_records_outputs():
    dev = MockDevice(clock=FakeClock())
    dev.beep(3)
    dev.led("R")
    dev.lcd("hello", "world")
    assert dev.beeps == 3 and dev.last_led == "R" and dev.last_lcd == ("hello", "world")
    with pytest.raises(ValueError):
        dev.led("PURPLE")


def test_arduino_device_without_board_does_not_crash():
    dev = ArduinoDevice(port="COM_DOES_NOT_EXIST")
    try:
        dev.beep(1)          # silently dropped
        dev.led("G")
        dev.start()
        assert dev.drain() == []
        assert dev.connected is False
    finally:
        dev.close()
