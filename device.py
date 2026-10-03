"""Arduino serial link (SW-520D tilt switch, buzzer, LED, LCD) + MockDevice for --sim.

Both share the same interface:
    start() stop() request_state() drain()->list[SwitchEvent] beep(n) led(c) lcd(l1, l2) close()
    state: Optional[int]   connected: bool   status: str   warning: str

Protocol (115200 baud, newline-terminated):
  Arduino -> PC:  READY on boot;  S,millis,0|1  on every debounced change (between START/STOP)
                  and as the reply to STATE
  PC -> Arduino:  START / STOP / STATE / BEEP,n / LED,G|Y|R|OFF / LCD,line1|line2
"""
from __future__ import annotations

import logging
import math
import queue
import random
import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import Callable, NamedTuple, Optional

log = logging.getLogger(__name__)

BAUD = 115200
LCD_WIDTH = 16
LED_COLOURS = ("G", "Y", "R", "B", "OFF")
READY_TIMEOUT_S = 4.0
RETRY_S = 1.5
MAX_BUFFER = 5000            # newest events are dropped beyond this so a stuck UI can't eat RAM
WRITE_QUEUE_MAX = 50

# Port preference: official Arduino, then CH340 clones, then generic USB-serial chips
PREFERRED_VIDS = ((0x2341, 0x2A03), (0x1A86,), (0x0403, 0x10C4))


class SwitchReport(NamedTuple):
    """One parsed 'S,millis,state' line (Arduino clock)."""
    t_ms: float
    state: int


class SwitchEvent(NamedTuple):
    """A switch report placed on the PC clock (seconds, same clock as the app)."""
    t: float
    state: int


def parse_line(line) -> Optional[SwitchReport]:
    """Parse 'S,millis,0|1'. None for anything else (READY, garbage, partial, empty)."""
    if isinstance(line, bytes):
        line = line.decode("ascii", errors="ignore")
    parts = line.strip().split(",")
    if len(parts) != 3 or parts[0] != "S" or parts[2] not in ("0", "1"):
        return None
    try:
        t_ms = float(parts[1])
    except ValueError:
        return None
    if not math.isfinite(t_ms) or t_ms < 0:
        return None
    return SwitchReport(t_ms, int(parts[2]))


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


class ClockMapper:
    """Maps Arduino millis onto the PC clock.

    offset = min over reports of (pc_receive_time - arduino_time): the report with
    the least USB latency gives the best estimate, so intervals keep the Arduino's
    millisecond precision while lining up with the PC's recording window.
    """

    def __init__(self) -> None:
        self.offset: Optional[float] = None

    def reset(self) -> None:
        self.offset = None

    def observe(self, t_ms: float, received: float) -> None:
        cand = received - t_ms / 1000.0
        if self.offset is None or cand < self.offset:
            self.offset = cand

    def to_pc(self, t_ms: float, received: Optional[float] = None) -> float:
        if received is not None:
            self.observe(t_ms, received)
        return t_ms / 1000.0 + (self.offset or 0.0)


class ArduinoDevice:
    """Serial link that keeps (re)connecting in a background thread.

    Never raises into or blocks the UI: commands go through a writer thread;
    if the board is missing or unplugged, `connected` is False, `status` says
    why, commands are dropped, and it retries. After a reconnect (the Uno
    resets) the last LED/LCD state and START are re-sent. `warning` holds a
    firmware ERROR line or a missing READY, for the UI to show.
    """

    def __init__(self, port: Optional[str] = None, clock: Callable[[], float] = time.perf_counter):
        self._want_port = port
        self._clock = clock
        self.port = port or "?"
        self._events: "queue.Queue[SwitchReport]" = queue.Queue()   # raw Arduino times
        self._writes: "queue.Queue[str]" = queue.Queue(maxsize=WRITE_QUEUE_MAX)
        self._ser = None
        self._ser_lock = threading.Lock()
        self._stop = threading.Event()
        self._mapper = ClockMapper()
        self.connected = False
        self.status = "Looking for Arduino..."
        self.warning = ""
        self.state: Optional[int] = None
        self.sim = False
        self.ready_received = False
        self._rx_times: deque = deque(maxlen=2000)
        self._ping = threading.Event()
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
        self._mapper.reset()          # the Uno resets on open: millis restart from 0
        self.ready_received = self._wait_ready(ser)
        if self.ready_received:
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
        deadline = time.monotonic() + READY_TIMEOUT_S
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
        self._enqueue("STATE")

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
            received = self._clock()
            if not raw.endswith(b"\n"):
                continue        # empty (timeout) or partial line: never half-parse
            self._rx_times.append(received)
            rep = parse_line(raw)
            if rep is not None:
                self.state = rep.state
                self._ping.set()
                self._mapper.observe(rep.t_ms, received)
                if self._events.qsize() < MAX_BUFFER:
                    self._events.put(rep)
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

    # --- telemetry ------------------------------------------------------------------
    @property
    def packets_per_sec(self) -> float:
        """Serial lines received in the last second (real)."""
        now = self._clock()
        return float(sum(1 for t in list(self._rx_times) if now - t <= 1.0))

    def measure_latency(self, timeout: float = 1.0) -> Optional[float]:
        """Round trip in ms: send STATE, wait for the S reply. Blocking - call off the UI thread."""
        if self._ser is None:
            return None
        self._ping.clear()
        t0 = time.perf_counter()
        self._send("STATE")
        if not self._ping.wait(timeout):
            return None
        return (time.perf_counter() - t0) * 1000.0

    # --- shared interface -------------------------------------------------------
    def start(self) -> None:
        self.drain()
        self._streaming_wanted = True
        self._send("START")
        self._send("STATE")          # reference state at t=0 of the recording

    def stop(self) -> None:
        self._streaming_wanted = False
        self._send("STOP")

    def request_state(self) -> None:
        self._send("STATE")

    def drain(self) -> list[SwitchEvent]:
        """Events on the PC clock. The whole batch uses the current best offset, so
        intervals between events are exact Arduino-millisecond differences."""
        reps = []
        while True:
            try:
                reps.append(self._events.get_nowait())
            except queue.Empty:
                break
        return [SwitchEvent(self._mapper.to_pc(r.t_ms), r.state) for r in reps]

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


# --------------------------------------------------------------------------- simulation
@dataclass(frozen=True)
class FlipProfile:
    rate: float = 2.4              # full flips per second at the start
    slowing: float = 0.0           # fraction of rate lost by the end of the test
    pauses: tuple = ()             # (start_s, duration_s) with no flipping
    bounce_p: float = 0.15         # chance a change comes with a contact-bounce burst


NORMAL_FLIPS = FlipProfile()
IMPAIRED_FLIPS = FlipProfile(rate=1.5, slowing=0.45, pauses=((5.0, 1.2),))


class MockDevice:
    """Fake SW-520D: emits flip events while streaming, following the profile for `sim_hand`."""

    connected = True
    port = "SIM"
    status = "SIM device"
    warning = ""
    sim = True
    ready_received = True

    def __init__(self, seed: Optional[int] = None, clock: Callable[[], float] = time.perf_counter,
                 profiles: Optional[dict] = None, test_seconds: float = 10.0):
        self._rng = random.Random(seed)
        self._clock = clock
        self.profiles = dict(profiles or {"Right": IMPAIRED_FLIPS, "Left": NORMAL_FLIPS})
        self.sim_hand = "Left"
        self.flipping = True            # False = sensor not moving (e.g. fell off)
        self._seconds = test_seconds
        self._streaming = False
        self._pending: list[SwitchEvent] = []
        self.state: Optional[int] = 0
        self._t0 = 0.0
        self._next_t = math.inf
        self.beeps = 0
        self.last_lcd = ("", "")
        self.last_led = "OFF"
        self._rx_times: deque = deque(maxlen=2000)

    @property
    def packets_per_sec(self) -> float:
        now = self._clock()
        return float(sum(1 for t in list(self._rx_times) if now - t <= 1.0))

    def measure_latency(self, timeout: float = 1.0) -> Optional[float]:
        return None          # nothing to measure in simulation

    def _rate(self, t: float) -> float:
        p = self.profiles.get(self.sim_hand, NORMAL_FLIPS)
        return p.rate * (1 - p.slowing * min(1.0, t / self._seconds))

    def _paused_until(self, t: float) -> Optional[float]:
        for start, dur in self.profiles.get(self.sim_hand, NORMAL_FLIPS).pauses:
            if start <= t < start + dur:
                return start + dur
        return None

    def _schedule(self, after: float) -> None:
        t = after - self._t0
        nxt = t + self._rng.uniform(0.9, 1.1) / (2 * self._rate(t))
        resume = self._paused_until(nxt)
        if resume is not None:
            nxt = resume + 1.0 / (2 * self._rate(resume))
        self._next_t = self._t0 + nxt

    def _generate(self, now: float) -> None:
        p = self.profiles.get(self.sim_hand, NORMAL_FLIPS)
        while self._streaming and self.flipping and self._next_t <= now:
            t = self._next_t
            self.state = 1 - self.state
            if self._rng.random() < p.bounce_p:          # ball bounce: x -> y -> x -> y in ~10 ms
                self._pending += [SwitchEvent(t, self.state), SwitchEvent(t + 0.004, 1 - self.state),
                                  SwitchEvent(t + 0.009, self.state)]
            else:
                self._pending.append(SwitchEvent(t, self.state))
            self._schedule(t)

    def start(self) -> None:
        self._pending = []
        self._streaming = True
        self._t0 = self._clock()
        self._pending.append(SwitchEvent(self._t0, self.state))      # STATE reply
        self._schedule(self._t0)

    def stop(self) -> None:
        self._streaming = False

    def request_state(self) -> None:
        self._pending.append(SwitchEvent(self._clock(), self.state))

    def drain(self) -> list[SwitchEvent]:
        self._generate(self._clock())
        out, self._pending = self._pending, []
        self._rx_times.extend(e.t for e in out)
        return out

    def beep(self, n: int = 1) -> None:
        self.beeps += n
        if self._clock is not time.perf_counter:
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
