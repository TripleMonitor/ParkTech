"""Arduino serial link + MockDevice for --sim. Both share the same interface.

Protocol (115200 baud, newline-terminated):
  Arduino -> PC:  T,millis,ax,ay,az   (m/s^2, 100 Hz, only while recording)
  PC -> Arduino:  START / STOP / BEEP,n / LED,G|Y|R|OFF / LCD,line1|line2
"""
from __future__ import annotations

import logging
import math
import queue
import random
import threading
import time
from typing import NamedTuple, Optional

log = logging.getLogger(__name__)

BAUD = 115200
SAMPLE_HZ = 100
LCD_WIDTH = 16
LED_COLOURS = ("G", "Y", "R", "OFF")


class Sample(NamedTuple):
    t_ms: float
    ax: float
    ay: float
    az: float


def parse_line(line: str) -> Optional[Sample]:
    """Parse 'T,millis,ax,ay,az'. Returns None for anything else / malformed."""
    parts = line.strip().split(",")
    if len(parts) != 5 or parts[0] != "T":
        return None
    try:
        return Sample(*(float(p) for p in parts[1:]))
    except ValueError:
        return None


def lcd_command(line1: str, line2: str = "") -> str:
    clean = [s.replace("|", "/").replace("\n", " ")[:LCD_WIDTH] for s in (line1, line2)]
    return f"LCD,{clean[0]}|{clean[1]}"


class SerialDevice:
    def __init__(self, port: Optional[str] = None):
        import serial  # imported lazily so --sim works without a port
        port = port or find_arduino_port()
        if port is None:
            raise RuntimeError("No Arduino serial port found (use --port COMx or --sim)")
        self._ser = serial.Serial(port, BAUD, timeout=0.1)
        self.port = port
        self._samples: "queue.Queue[Sample]" = queue.Queue()
        self._running = True
        self._lock = threading.Lock()
        time.sleep(2.0)  # Uno resets when the port opens
        self._ser.reset_input_buffer()
        self._thread = threading.Thread(target=self._reader, daemon=True)
        self._thread.start()

    def _reader(self) -> None:
        while self._running:
            try:
                raw = self._ser.readline()
            except Exception as exc:  # serial unplugged etc.
                log.error("Serial read failed: %s", exc)
                self._running = False
                return
            if not raw:
                continue
            line = raw.decode("ascii", errors="ignore")
            sample = parse_line(line)
            if sample is not None:
                self._samples.put(sample)
            elif line.strip():
                log.info("arduino: %s", line.strip())

    def _send(self, cmd: str) -> None:
        with self._lock:
            try:
                self._ser.write((cmd + "\n").encode("ascii", errors="replace"))
            except Exception as exc:
                log.error("Serial write failed (%s): %s", cmd, exc)

    # --- shared interface ---------------------------------------------------
    def start(self) -> None:
        self.drain()
        self._send("START")

    def stop(self) -> None:
        self._send("STOP")

    def drain(self) -> list[Sample]:
        out = []
        while True:
            try:
                out.append(self._samples.get_nowait())
            except queue.Empty:
                return out

    def beep(self, n: int = 1) -> None:
        self._send(f"BEEP,{int(n)}")

    def led(self, colour: str) -> None:
        if colour not in LED_COLOURS:
            raise ValueError(f"LED colour must be one of {LED_COLOURS}")
        self._send(f"LED,{colour}")

    def lcd(self, line1: str, line2: str = "") -> None:
        self._send(lcd_command(line1, line2))

    def close(self) -> None:
        self._running = False
        self.stop()
        self._ser.close()


def find_arduino_port() -> Optional[str]:
    from serial.tools import list_ports
    hints = ("arduino", "ch340", "usb serial", "usb-serial", "cp210")
    for p in list_ports.comports():
        if any(h in (p.description or "").lower() for h in hints):
            return p.device
    return None


class MockDevice:
    """Fake IMU at 100 Hz. Noise + slow hand sway; toggle a 5 Hz tremor with 'T'."""

    TREMOR_HZ = 5.0
    TREMOR_ACCEL = 12.0   # m/s^2 -> ~1.2 cm at 5 Hz -> demo score 2
    NOISE_STD = 0.05
    port = "SIM"

    def __init__(self, seed: Optional[int] = None):
        self._rng = random.Random(seed)
        self.tremor_on = False
        self._streaming = False
        self._t0 = time.monotonic()
        self._next_ms = 0.0
        self.last_lcd = ("", "")
        self.last_led = "OFF"

    def toggle_tremor(self) -> bool:
        self.tremor_on = not self.tremor_on
        return self.tremor_on

    def _now_ms(self) -> float:
        return (time.monotonic() - self._t0) * 1000.0

    def _sample(self, t_ms: float) -> Sample:
        t = t_ms / 1000.0
        n = self._rng.gauss
        sway = 0.3 * math.sin(2 * math.pi * 0.4 * t)
        trem = self.TREMOR_ACCEL * math.sin(2 * math.pi * self.TREMOR_HZ * t) if self.tremor_on else 0.0
        return Sample(t_ms,
                      sway + 0.8 * trem + n(0, self.NOISE_STD),
                      0.6 * trem + n(0, self.NOISE_STD),
                      9.81 + n(0, self.NOISE_STD))

    def start(self) -> None:
        self._streaming = True
        self._next_ms = self._now_ms()

    def stop(self) -> None:
        self._streaming = False

    def drain(self) -> list[Sample]:
        if not self._streaming:
            return []
        now = self._now_ms()
        step = 1000.0 / SAMPLE_HZ
        out = []
        while self._next_ms <= now:
            out.append(self._sample(self._next_ms))
            self._next_ms += step
        return out

    def beep(self, n: int = 1) -> None:
        def _play() -> None:
            try:
                import winsound
                for _ in range(n):
                    winsound.Beep(1800, 80)
                    time.sleep(0.06)
            except (ImportError, RuntimeError):
                print("\a" * n, end="", flush=True)
        threading.Thread(target=_play, daemon=True).start()

    def led(self, colour: str) -> None:
        if colour not in LED_COLOURS:
            raise ValueError(f"LED colour must be one of {LED_COLOURS}")
        self.last_led = colour

    def lcd(self, line1: str, line2: str = "") -> None:
        self.last_lcd = (line1[:LCD_WIDTH], line2[:LCD_WIDTH])

    def close(self) -> None:
        self._streaming = False
