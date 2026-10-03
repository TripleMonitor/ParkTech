"""Arduino serial link + MockDevice for --sim. Both share the same interface:

    start() stop() drain()->list[Sample] beep(n) led(c) lcd(l1, l2) close()
    connected: bool      status: str (human-readable, for the UI)

Protocol (115200 baud, newline-terminated):
  Arduino -> PC:  READY on boot; T,millis,ax,ay,az  (m/s^2, 100 Hz, only while streaming)
  PC -> Arduino:  START / STOP / BEEP,n / LED,G|Y|R|OFF / LCD,line1|line2
"""
from __future__ import annotations

import logging
import math
import queue
import random
import threading
import time
from typing import Callable, NamedTuple, Optional

log = logging.getLogger(__name__)

BAUD = 115200
SAMPLE_HZ = 100
LCD_WIDTH = 16
LED_COLOURS = ("G", "Y", "R", "B", "OFF")
READY_TIMEOUT_S = 4.0
RETRY_S = 1.5
MAX_BUFFER = 60 * SAMPLE_HZ   # drop oldest beyond 60 s so a stuck UI can't eat RAM

# USB vendor IDs: Arduino, Arduino (new), WCH CH340, FTDI, SiLabs CP210x
ARDUINO_VIDS = {0x2341, 0x2A03, 0x1A86, 0x0403, 0x10C4}


class Sample(NamedTuple):
    t_ms: float
    ax: float
    ay: float
    az: float


def parse_line(line) -> Optional[Sample]:
    """Parse 'T,millis,ax,ay,az'. None for anything else (READY, garbage, partial, empty)."""
    if isinstance(line, bytes):
        line = line.decode("ascii", errors="ignore")
    parts = line.strip().split(",")
    if len(parts) != 5 or parts[0] != "T":
        return None
    try:
        vals = [float(p) for p in parts[1:]]
    except ValueError:
        return None
    if not all(math.isfinite(v) for v in vals):
        return None
    return Sample(*vals)


def lcd_command(line1: str, line2: str = "") -> str:
    clean = [s.replace("|", "/").replace("\n", " ").replace(",", " ")[:LCD_WIDTH]
             for s in (line1, line2)]
    return f"LCD,{clean[0]}|{clean[1]}"


def _check_led(colour: str) -> None:
    if colour not in LED_COLOURS:
        raise ValueError(f"LED colour must be one of {LED_COLOURS}")


def find_arduino_port() -> Optional[str]:
    from serial.tools import list_ports
    ports = list(list_ports.comports())
    for p in ports:
        if p.vid in ARDUINO_VIDS:
            return p.device
    hints = ("arduino", "ch340", "usb serial", "usb-serial", "cp210")
    for p in ports:
        if any(h in (p.description or "").lower() for h in hints):
            return p.device
    return None


class ArduinoDevice:
    """Serial link that keeps (re)connecting in a background thread.

    Never raises into the UI: if the board is missing or unplugged, `connected`
    is False, `status` says why, commands are dropped, and it retries.
    """

    def __init__(self, port: Optional[str] = None):
        self._want_port = port
        self.port = port or "?"
        self._samples: "queue.Queue[Sample]" = queue.Queue()
        self._ser = None
        self._ser_lock = threading.Lock()
        self._stop = threading.Event()
        self.connected = False
        self.status = "Looking for Arduino..."
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    # --- background connection manager ----------------------------------------
    def _run(self) -> None:
        while not self._stop.is_set():
            if not self._open():
                self._stop.wait(RETRY_S)
                continue
            self._read_until_error()
            self._close_port()
            if not self._stop.is_set():
                self.status = f"Arduino disconnected ({self.port}) - plug it back in"
                log.error(self.status)
                self._stop.wait(RETRY_S)

    def _open(self) -> bool:
        import serial
        port = self._want_port or find_arduino_port()
        if port is None:
            self.status = "Arduino not found - plug in USB (or run with --sim)"
            return False
        try:
            ser = serial.Serial(port, BAUD, timeout=0.2, write_timeout=0.5)
        except (serial.SerialException, OSError) as exc:
            self.status = f"Cannot open {port}: {exc}"
            return False
        self.port = port
        if not self._wait_ready(ser):
            log.warning("No READY from %s within %.0fs - continuing anyway", port, READY_TIMEOUT_S)
        with self._ser_lock:
            self._ser = ser
        self.connected = True
        self.status = f"Arduino {port}"
        log.info("Connected to %s", port)
        return True

    def _wait_ready(self, ser) -> bool:
        deadline = time.monotonic() + READY_TIMEOUT_S   # Uno resets when the port opens
        while time.monotonic() < deadline and not self._stop.is_set():
            try:
                line = ser.readline().decode("ascii", errors="ignore").strip()
            except Exception:
                return False
            if line.startswith("READY"):
                return True
            if line.startswith("ERROR"):
                log.error("arduino: %s", line)
        return False

    def _read_until_error(self) -> None:
        while not self._stop.is_set():
            try:
                raw = self._ser.readline()
            except Exception as exc:
                log.error("Serial read failed: %s", exc)
                return
            if not raw:
                continue
            sample = parse_line(raw)
            if sample is not None:
                if self._samples.qsize() < MAX_BUFFER:
                    self._samples.put(sample)
            else:
                text = raw.decode("ascii", errors="ignore").strip()
                if text:
                    log.info("arduino: %s", text)

    def _close_port(self) -> None:
        self.connected = False
        with self._ser_lock:
            ser, self._ser = self._ser, None
        if ser is not None:
            try:
                ser.close()
            except Exception:
                pass

    def _send(self, cmd: str) -> None:
        with self._ser_lock:
            if self._ser is None:
                log.debug("dropped (not connected): %s", cmd)
                return
            try:
                self._ser.write((cmd + "\n").encode("ascii", errors="replace"))
            except Exception as exc:
                log.error("Serial write failed (%s): %s", cmd, exc)
                try:
                    self._ser.close()   # makes the reader thread notice and reconnect
                except Exception:
                    pass

    # --- shared interface -------------------------------------------------------
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
        _check_led(colour)
        self._send(f"LED,{colour}")

    def lcd(self, line1: str, line2: str = "") -> None:
        self._send(lcd_command(line1, line2))

    def close(self) -> None:
        self.stop()
        self._stop.set()
        self._close_port()


class MockDevice:
    """Fake IMU at 100 Hz: noise + slow sway, or a 5 Hz tremor while `tremor_on` is set."""

    TREMOR_HZ = 5.0
    TREMOR_ACCEL = 12.0   # m/s^2 on x -> ~1.2 cm at 5 Hz -> demo score 2
    NOISE_STD = 0.05
    connected = True
    port = "SIM"
    status = "SIM device"

    def __init__(self, seed: Optional[int] = None,
                 clock: Callable[[], float] = time.monotonic):
        self._rng = random.Random(seed)
        self._clock = clock
        self._t0 = clock()
        self.tremor_on = False
        self._streaming = False
        self._next_ms = 0.0
        self.beeps = 0
        self.last_lcd = ("", "")
        self.last_led = "OFF"

    def _now_ms(self) -> float:
        return (self._clock() - self._t0) * 1000.0

    def sample_at(self, t_ms: float) -> Sample:
        t = t_ms / 1000.0
        n = self._rng.gauss
        sway = 0.3 * math.sin(2 * math.pi * 0.4 * t)
        trem = self.TREMOR_ACCEL * math.sin(2 * math.pi * self.TREMOR_HZ * t) if self.tremor_on else 0.0
        return Sample(t_ms, sway + trem + n(0, self.NOISE_STD),
                      0.3 * trem + n(0, self.NOISE_STD), 9.81 + n(0, self.NOISE_STD))

    def start(self) -> None:
        self._streaming = True
        self._next_ms = self._now_ms()

    def stop(self) -> None:
        self._streaming = False

    def drain(self) -> list[Sample]:
        if not self._streaming:
            return []
        now = self._now_ms()
        out = []
        while self._next_ms <= now:
            out.append(self.sample_at(self._next_ms))
            self._next_ms += 1000.0 / SAMPLE_HZ
        return out

    def beep(self, n: int = 1) -> None:
        self.beeps += n
        if self._clock is not time.monotonic:
            return   # headless/selftest: count only, no sound

        def _play() -> None:
            try:
                import winsound
                for _ in range(n):
                    winsound.Beep(1800, 80)
                    time.sleep(0.06)
            except (ImportError, RuntimeError):
                pass
        threading.Thread(target=_play, daemon=True).start()

    def led(self, colour: str) -> None:
        _check_led(colour)
        self.last_led = colour

    def lcd(self, line1: str, line2: str = "") -> None:
        self.last_lcd = (line1[:LCD_WIDTH], line2[:LCD_WIDTH])

    def close(self) -> None:
        self._streaming = False
