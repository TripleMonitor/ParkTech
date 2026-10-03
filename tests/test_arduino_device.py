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


def test_ready_handshake_and_samples(fake_serial):
    fake_serial.scripts = [[b"boot noise\r\n", b"READY\r\n", b"T,10,0.1,0.2,9.8\r\n",
                            b"T,20,0.1,0.2,9", b"T,30,0.3,0.2,9.8\r\n"]]   # 2nd is partial
    dev = ArduinoDevice(port="FAKE")
    try:
        assert wait_for(lambda: dev.connected)
        assert dev.warning == ""
        assert wait_for(lambda: dev._samples.qsize() >= 2)
        samples = dev.drain()
        assert [s.t_ms for s in samples] == [10.0, 30.0]       # partial line ignored
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


def test_firmware_error_shown_then_cleared_by_data(fake_serial):
    fake_serial.scripts = [[b"ERROR MPU6050 not responding\r\n", b"READY\r\n"]]
    dev = ArduinoDevice(port="FAKE")
    try:
        assert wait_for(lambda: "MPU6050" in dev.warning)
        FakeSerial.opened[0].lines.append(b"T,1,0,0,9.8\r\n")
        assert wait_for(lambda: dev.warning == "")
    finally:
        dev.close()


def test_commands_do_not_block_ui(fake_serial):
    fake_serial.scripts = [[b"READY\r\n"]]
    dev = ArduinoDevice(port="FAKE")
    try:
        assert wait_for(lambda: dev.connected)
        FakeSerial.opened[0].write_delay = 0.4      # a stuck/slow port
        t0 = time.perf_counter()
        for _ in range(5):
            dev.beep(1)
        assert time.perf_counter() - t0 < 0.05
        assert wait_for(lambda: len(FakeSerial.opened[0].written) == 5, timeout=4)
    finally:
        dev.close()


def test_reconnect_resends_led_lcd_and_start(fake_serial):
    fake_serial.scripts = [[b"READY\r\n"], [b"READY\r\n"]]
    dev = ArduinoDevice(port="FAKE")
    try:
        assert wait_for(lambda: dev.connected)
        dev.led("Y")
        dev.lcd("Right tremor", "Recording...")
        dev.start()
        first = FakeSerial.opened[0]
        assert wait_for(lambda: "START" in first.written)
        first.lines.append(serial.SerialException("unplugged"))
        assert wait_for(lambda: len(FakeSerial.opened) == 2 and dev.connected)
        second = FakeSerial.opened[1]
        assert wait_for(lambda: len(second.written) >= 3)
        assert second.written[:3] == ["LED,Y", "LCD,Right tremor|Recording...", "START"]
    finally:
        dev.close()


def test_commands_while_disconnected_are_dropped(fake_serial):
    fake_serial.scripts = []
    dev = ArduinoDevice(port="FAKE")
    try:
        dev.beep(2)          # not connected yet: dropped, no exception
        assert dev._writes.empty()
    finally:
        dev.close()
