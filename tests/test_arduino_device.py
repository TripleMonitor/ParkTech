"""ArduinoDevice against a scripted fake serial port (no hardware needed)."""
import time

import pytest
import serial

import device
from device import ArduinoDevice


class FakeSerial:
    """Each open() pops the next script: a list of bytes lines, or an Exception to raise."""
    scripts: list = []
    opened: list = []

    def __init__(self, port, baud, timeout=None, write_timeout=None):
        self.lines = list(FakeSerial.scripts.pop(0)) if FakeSerial.scripts else []
        self.written: list[str] = []
        self.closed = False
        self.write_delay = 0.0
        FakeSerial.opened.append(self)

    def readline(self):
        if self.closed:
            raise serial.SerialException("port closed")
        if not self.lines:
            time.sleep(0.02)
            return b""
        item = self.lines.pop(0)
        if isinstance(item, Exception):
            self.closed = True
            raise item
        return item

    def write(self, data):
        time.sleep(self.write_delay)
        self.written.append(data.decode().strip())

    def close(self):
        self.closed = True


@pytest.fixture
def fake_serial(monkeypatch):
    FakeSerial.scripts, FakeSerial.opened = [], []
    monkeypatch.setattr(serial, "Serial", FakeSerial)
    monkeypatch.setattr(device, "READY_TIMEOUT_S", 0.5)
    monkeypatch.setattr(device, "RETRY_S", 0.05)
    return FakeSerial


def wait_for(cond, timeout=3.0):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if cond():
            return True
        time.sleep(0.01)
    return False


def test_ready_handshake_and_switch_events(fake_serial):
    fake_serial.scripts = [[b"boot noise\r\n", b"READY\r\n", b"S,1000,1\r\n",
                            b"S,1250,", b"S,1500,0\r\n"]]              # 2nd is partial
    dev = ArduinoDevice(port="FAKE")
    try:
        assert wait_for(lambda: dev.connected)
        assert dev.warning == ""
        assert wait_for(lambda: dev._events.qsize() >= 2)
        events = dev.drain()
        assert [e.state for e in events] == [1, 0]                    # partial line ignored
        assert events[1].t - events[0].t == pytest.approx(0.5, abs=0.05)   # Arduino timing
        assert dev.state == 0
    finally:
        dev.close()


def test_missing_ready_sets_warning(fake_serial):
    fake_serial.scripts = [[b"hello\r\n"]]
    dev = ArduinoDevice(port="FAKE")
    try:
        assert wait_for(lambda: dev.connected)
        assert "No READY" in dev.warning
    finally:
        dev.close()


def test_firmware_error_is_shown(fake_serial):
    fake_serial.scripts = [[b"ERROR something broke\r\n", b"READY\r\n"]]
    dev = ArduinoDevice(port="FAKE")
    try:
        assert wait_for(lambda: "something broke" in dev.warning)
    finally:
        dev.close()


def test_state_request_is_sent_on_connect(fake_serial):
    fake_serial.scripts = [[b"READY\r\n"]]
    dev = ArduinoDevice(port="FAKE")
    try:
        assert wait_for(lambda: "STATE" in FakeSerial.opened[0].written)
        FakeSerial.opened[0].lines.append(b"S,42,1\r\n")
        assert wait_for(lambda: dev.state == 1)
    finally:
        dev.close()


def test_commands_do_not_block_ui(fake_serial):
    fake_serial.scripts = [[b"READY\r\n"]]
    dev = ArduinoDevice(port="FAKE")
    try:
        assert wait_for(lambda: dev.connected)
        assert wait_for(lambda: "STATE" in FakeSerial.opened[0].written)
        FakeSerial.opened[0].write_delay = 0.4      # a stuck/slow port
        t0 = time.perf_counter()
        for _ in range(5):
            dev.beep(1)
        assert time.perf_counter() - t0 < 0.05
        assert wait_for(lambda: FakeSerial.opened[0].written.count("BEEP,1") == 5, timeout=4)
    finally:
        dev.close()


def test_reconnect_resends_led_lcd_start_and_state(fake_serial):
    fake_serial.scripts = [[b"READY\r\n"], [b"READY\r\n"]]
    dev = ArduinoDevice(port="FAKE")
    try:
        assert wait_for(lambda: dev.connected)
        dev.led("Y")
        dev.lcd("Right flipping", "Recording...")
        dev.start()
        first = FakeSerial.opened[0]
        assert wait_for(lambda: first.written.count("STATE") >= 2 and "START" in first.written)
        first.lines.append(serial.SerialException("unplugged"))
        assert wait_for(lambda: len(FakeSerial.opened) == 2 and dev.connected)
        second = FakeSerial.opened[1]
        assert wait_for(lambda: len(second.written) >= 4)
        assert second.written[:4] == ["LED,Y", "LCD,Right flipping|Recording...", "START", "STATE"]
    finally:
        dev.close()


def test_commands_while_disconnected_are_dropped(fake_serial):
    dev = ArduinoDevice(port="FAKE")
    try:
        dev.beep(2)          # not connected yet: dropped, no exception
        assert dev._writes.empty()
    finally:
        dev.close()
