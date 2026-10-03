import time

from device import MockDevice, lcd_command, parse_line


def test_parse_line_ok_and_bad():
    s = parse_line("T,1234,0.1,-0.2,9.8\r\n")
    assert s is not None and s.t_ms == 1234 and s.az == 9.8
    assert parse_line("READY") is None
    assert parse_line("T,1,2,x,4") is None


def test_lcd_command_truncates_and_sanitises():
    assert lcd_command("A" * 20, "x|y") == "LCD," + "A" * 16 + "|x/y"


def test_mock_streams_only_when_started():
    dev = MockDevice(seed=1)
    assert dev.drain() == []
    dev.start()
    time.sleep(0.2)
    samples = dev.drain()
    assert 10 <= len(samples) <= 40
    dev.stop()
    assert dev.drain() == []


def test_mock_tremor_toggle():
    dev = MockDevice()
    assert dev.toggle_tremor() is True
    assert dev.toggle_tremor() is False
