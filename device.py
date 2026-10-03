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
MAX_BUFFER = 60 * SAMPLE_HZ   # newest samples are dropped beyond 60 s so a stuck UI can't eat RAM
WRITE_QUEUE_MAX = 50

# Port preference: official Arduino, then CH340 clones, then generic USB-serial chips
PREFERRED_VIDS = ((0x2341, 0x2A03), (0x1A86,), (0x0403, 0x10C4))


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
    for vids in PREFERRED_VIDS:
        for p in ports:
            if p.vid in vids:
                return p.device
    hints = ("arduino", "ch340", "usb serial", "usb-serial", "cp210")
    for p in ports:
        if any(h in (p.description or "").lower() for h in hints):
            return p.device
    return None


class ArduinoDevice:
    """Serial link that keeps (re)connecting in a background thread.

    Never raises into or blocks the UI: commands go through a writer thread;
    if the board is missing or unplugged, `connected` is False, `status` says
    why, commands are dropped, and it retries. After a reconnect (the Uno
    resets) the last LED/LCD state and START are re-sent. `warning` holds a
    firmware ERROR line or a missing READY, for the UI to show.
    """

    def __init__(self, port: Optional[str] = None):
        self._want_port = port
        self.port = port or "?"
        self._samples: "queue.Queue[Sample]" = queue.Queue()
        self._writes: "queue.Queue[str]" = queue.Queue(maxsize=WRITE_QUEUE_MAX)
        self._ser = None
        self._ser_lock = threading.Lock()
        self._stop = threading.Event()
        self.connected = False
        self.status = "Looking for Arduino..."
        self.warning = ""
        self._streaming_wanted = False
        self._last_led: Optional[str] = None
        self._last_lcd: Optional[str] = None
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        self._writer = threading.Thread(target=self._write_loop, daemon=True)
        self._writer.start()

    # --- background connection manager ----------------------------------------
    def _run(self) -> None:
        while not self._stop.is_set():
            if not self._open():
                self._stop.wait(RETRY_S)
                continue
            self._resync()
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
        if self._wait_ready(ser):
            if self.warning.startswith("No READY"):
                self.warning = ""
        else:
            self.warning = f"No READY from {port} - is the NeuroCheck firmware uploaded?"
            log.warning(self.warning)
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
                self._firmware_error(line)
        return False

    def _firmware_error(self, line: str) -> None:
        self.warning = line[len("ERROR"):].strip() or line
        log.error("arduino: %s", line)

    def _resync(self) -> None:
        """Board just (re)booted: restore LED/LCD and streaming state."""
        for cmd in (self._last_led, self._last_lcd):
            if cmd:
                self._enqueue(cmd)
        if self._streaming_wanted:
            self._enqueue("START")

    def _read_until_error(self) -> None:
        while not self._stop.is_set():
            ser = self._ser
            if ser is None:
                return
            try:
                raw = ser.readline()
            except Exception as exc:
                log.error("Serial read failed: %s", exc)
                return
            if not raw.endswith(b"\n"):
                continue        # empty (timeout) or partial line: never half-parse a sample
            sample = parse_line(raw)
            if sample is not None:
                if "MPU" in self.warning:
                    self.warning = ""      # sensor is delivering data again
                if self._samples.qsize() < MAX_BUFFER:
                    self._samples.put(sample)
                continue
            text = raw.decode("ascii", errors="ignore").strip()
            if text.startswith("ERROR"):
                self._firmware_error(text)
            elif text:
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

    def _write_loop(self) -> None:
        while not self._stop.is_set():
            try:
                cmd = self._writes.get(timeout=0.2)
            except queue.Empty:
                continue
            with self._ser_lock:
                ser = self._ser
                if ser is None:
                    log.debug("dropped (not connected): %s", cmd)
                    continue
                try:
                    ser.write((cmd + "\n").encode("ascii", errors="replace"))
                except Exception as exc:
                    log.error("Serial write failed (%s): %s", cmd, exc)
                    try:
                        ser.close()   # makes the reader thread notice and reconnect
                    except Exception:
                        pass

    def _enqueue(self, cmd: str) -> None:
        try:
            self._writes.put_nowait(cmd)
        except queue.Full:
            log.warning("serial write queue full, dropped: %s", cmd)

    def _send(self, cmd: str) -> None:
        """Non-blocking: the writer thread does the actual (possibly slow) write."""
        if self._ser is None:
            log.debug("dropped (not connected): %s", cmd)
            return
        self._enqueue(cmd)

    # --- shared interface -------------------------------------------------------
    def start(self) -> None:
        self.drain()
        self._streaming_wanted = True
        self._send("START")

    def stop(self) -> None:
        self._streaming_wanted = False
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
        self._last_led = f"LED,{colour}"
        self._send(self._last_led)

    def lcd(self, line1: str, line2: str = "") -> None:
        self._last_lcd = lcd_command(line1, line2)
        self._send(self._last_lcd)

    def close(self) -> None:
        self.stop()
        deadline = time.monotonic() + 0.5     # give STOP / LED,OFF a moment to go out
        while not self._writes.empty() and self._ser is not None and time.monotonic() < deadline:
            time.sleep(0.02)
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
    warning = ""

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
