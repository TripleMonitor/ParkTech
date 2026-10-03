"""Boot self-check: real checks, run in a background thread so the UI can show
each line as it completes. Hardware lines say SIM when running simulated.

Status values: PENDING, OK, WARN, FAIL, SENT (command sent; confirm by eye/ear), SIM.
"""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Callable

ARDUINO_WAIT_S = 6.0
MIN_GOOD_FPS = 15.0


@dataclass
class Check:
    name: str
    status: str = "PENDING"
    detail: str = ""


class BootChecks:
    def __init__(self, device, hands, pace_s: float = 0.25):
        self.device, self.hands, self.pace_s = device, hands, pace_s
        self.lines: list[Check] = []
        self.done = False
        self._thread: threading.Thread | None = None

    # each step returns (status, detail)
    def _camera_open(self):
        if self.hands.sim:
            return "SIM", "synthetic hand (--sim)"
        if not getattr(self.hands, "_cap", None) or not self.hands._cap.isOpened():
            return "FAIL", self.hands.status
        return "OK", f"opened in {self.hands.open_ms:.0f} ms"

    def _camera_fps(self):
        if self.hands.sim:
            return "SIM", "no camera in --sim"
        fps = self.hands.measure_fps(20)
        if fps is None:
            return "FAIL", "no frames"
        return ("OK" if fps >= MIN_GOOD_FPS else "WARN"), f"{fps:.1f} fps (20 frames timed)"

    def _model(self):
        if self.hands.sim:
            return "SIM", "not loaded in --sim"
        if self.hands.model_ms is None:
            return "FAIL", self.hands.model_error or "not loaded"
        return "OK", f"MediaPipe Hands loaded in {self.hands.model_ms:.0f} ms"

    def _arduino(self):
        if self.device.sim:
            return "SIM", "MockDevice"
        deadline = time.monotonic() + ARDUINO_WAIT_S
        while not self.device.connected and time.monotonic() < deadline:
            time.sleep(0.05)
        if not self.device.connected:
            return "FAIL", self.device.status
        return "OK", f"found on {self.device.port}"

    def _ready(self):
        if self.device.sim:
            return "SIM", ""
        if not self.device.connected:
            return "FAIL", "not connected"
        if self.device.ready_received:
            return "OK", "READY received"
        return "WARN", "no READY - is the NeuroCheck firmware uploaded?"

    def _latency(self):
        if self.device.sim:
            return "SIM", ""
        ms = self.device.measure_latency(1.0)
        if ms is None:
            return "FAIL", "no reply to STATE within 1 s"
        return "OK", f"STATE round trip {ms:.1f} ms"

    def _tilt(self):
        if self.device.sim:
            return "SIM", f"simulated state {self.device.state}"
        if self.device.state is None:
            return "FAIL", "no switch state received"
        return "OK", f"switch reads {self.device.state}"

    def _buzzer(self):
        if self.device.sim:
            return "SIM", ""
        if not self.device.connected:
            return "FAIL", "not connected"
        self.device.beep(1)
        return "SENT", "BEEP,1 sent - listen for one beep"

    def _led(self):
        if self.device.sim:
            return "SIM", ""
        if not self.device.connected:
            return "FAIL", "not connected"
        for colour in ("R", "G", "B", "OFF"):
            self.device.led(colour)
            time.sleep(0.25)
        return "SENT", "red, green, blue sent - watch the LED"

    STEPS: list[tuple[str, str]] = [
        ("Camera opens", "_camera_open"), ("Camera frame rate", "_camera_fps"),
        ("Hand model", "_model"), ("Arduino on serial port", "_arduino"),
        ("READY handshake", "_ready"), ("Round-trip latency", "_latency"),
        ("Tilt switch reading", "_tilt"), ("Buzzer test", "_buzzer"),
        ("LED colour cycle", "_led"),
    ]

    def run(self) -> None:
        for name, method in self.STEPS:
            line = Check(name)
            self.lines.append(line)
            try:
                line.status, line.detail = getattr(self, method)()
            except Exception as exc:      # a broken check must not stop the boot screen
                line.status, line.detail = "FAIL", f"{type(exc).__name__}: {exc}"
            if self.pace_s:
                time.sleep(self.pace_s)   # presentation pacing only; values above are real
        self.done = True

    def start(self) -> "BootChecks":
        self._thread = threading.Thread(target=self.run, daemon=True)
        self._thread.start()
        return self

    def wait(self, timeout: float = 30.0) -> None:
        if self._thread:
            self._thread.join(timeout)

    @property
    def failures(self) -> int:
        return sum(c.status == "FAIL" for c in self.lines)
