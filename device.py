"""Arduino serial link (SW-520D tilt switch, buzzer, LED, LCD) + MockDevice for --sim.

Both share the same interface:
    start() stop() request_state() drain()->list[SwitchEvent] beep(n) led(c) lcd(l1, l2) close()
    state: Optional[int]   connected: bool   status: str   warning: str

Protocol (115200 baud, newline-terminated):
  Arduino -> PC:  READY on boot;  S,millis,0|1  on every debounced change (between START/STOP)
                  and as the reply to STATE;  C,millis  one per metronome beat (cue on)
  PC -> Arduino:  START / STOP / STATE / BEEP,n / LED,G|Y|R|OFF / LCD,line1|line2
                  CUE,ON,<ms> / CUE,OFF / TEMPO,<ms>   (beat interval, clamped 200-1500)
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


def parse_beat(line) -> Optional[float]:
    """Parse 'C,millis' (metronome beat). Returns Arduino millis or None."""
    if isinstance(line, bytes):
        line = line.decode("ascii", errors="ignore")
    parts = line.strip().split(",")
    if len(parts) != 2 or parts[0] != "C":
        return None
    try:
        t = float(parts[1])
    except ValueError:
        return None
    return t if math.isfinite(t) and t >= 0 else None


CUE_MIN_MS, CUE_MAX_MS = 200, 1500


def clamp_interval(ms: float) -> int:
    return int(min(CUE_MAX_MS, max(CUE_MIN_MS, round(ms))))


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

    offset = min over RECENT reports (last WINDOW_S seconds) of (pc_receive_time -
    arduino_time): the report with the least USB latency gives the best estimate, so
    intervals keep the Arduino's millisecond precision while lining up with the PC's
    recording window. The sliding window (plus a reset at every START) tracks a slow
    or fast Arduino resonator instead of drifting away from it.
    """

    WINDOW_S = 20.0

    def __init__(self) -> None:
        self.offset: Optional[float] = None
        self._obs: deque = deque()

    def reset(self) -> None:
        self.offset = None
        self._obs.clear()

    def observe(self, t_ms: float, received: float) -> None:
        self._obs.append((received, received - t_ms / 1000.0))
        while self._obs and received - self._obs[0][0] > self.WINDOW_S:
            self._obs.popleft()
        self.offset = min(c for _, c in self._obs)

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
        self._beats: "queue.Queue[float]" = queue.Queue()            # raw Arduino millis
        self._writes: "queue.Queue[str]" = queue.Queue(maxsize=WRITE_QUEUE_MAX)
        self._ser = None
        self._ser_lock = threading.Lock()
        self._stop = threading.Event()
        self._mapper = ClockMapper()
        self.connected = False
        self.status = "Looking for Arduino..."
        self.warning = ""
        self.state: Optional[int] = None
        self.state_at = -1e9                 # PC time of the last switch report
        self._cue_ms: Optional[int] = None   # metronome interval while the cue is on
        self._rx_buf = b""
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
            self.warning = f"No READY from {port} - is the ParkTech firmware (firmware/neurocheck) uploaded?"
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
        if self._cue_ms is not None:
            self._enqueue(f"CUE,ON,{self._cue_ms}")
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
            if not raw:
                continue
            if not raw.endswith(b"\n"):
                self._rx_buf = (self._rx_buf + raw)[-256:]   # timed out mid-line: keep it
                continue
            raw, self._rx_buf = self._rx_buf + raw, b""
            self._rx_times.append(received)
            rep = parse_line(raw)
            if rep is not None:
                self.state = rep.state
                self.state_at = received
                self._ping.set()
                self._mapper.observe(rep.t_ms, received)
                if self._events.qsize() < MAX_BUFFER:
                    self._events.put(rep)
                continue
            beat = parse_beat(raw)
            if beat is not None:
                self._mapper.observe(beat, received)
                if self._beats.qsize() < MAX_BUFFER:
                    self._beats.put(beat)
                continue
            text = raw.decode("ascii", errors="ignore").strip()
            if text.startswith("ERROR"):
                self._firmware_error(text)
            elif text:
                log.info("arduino: %s", text)

    def _close_port(self) -> None:
        self.connected = False
        self.state = None                    # no stale tilt value while disconnected
        self._rx_buf = b""
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
        self._mapper.reset()         # fresh clock offset for every recording
        self._streaming_wanted = True
        self._send("START")
        self._send("STATE")          # reference state at t=0 of the recording

    @property
    def state_fresh(self) -> bool:
        """Tilt state is trustworthy: streaming (changes reported) or a recent report."""
        return self.state is not None and (self._streaming_wanted or
                                           self._clock() - self.state_at < 0.6)

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

    def drain_beats(self) -> list[float]:
        """Metronome beat times on the PC clock."""
        out = []
        while True:
            try:
                out.append(self._mapper.to_pc(self._beats.get_nowait()))
            except queue.Empty:
                return out

    def cue_on(self, interval_ms: float) -> None:
        self._cue_ms = clamp_interval(interval_ms)
        self._send(f"CUE,ON,{self._cue_ms}")

    def cue_off(self) -> None:
        self._cue_ms = None
        self._send("CUE,OFF")

    def tempo(self, interval_ms: float) -> None:
        if self._cue_ms is not None:
            self._cue_ms = clamp_interval(interval_ms)
        self._send(f"TEMPO,{clamp_interval(interval_ms)}")

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
        self.cue_off()                         # never leave the metronome clicking
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
    state_fresh = True

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
        # metronome (rhythm coach): beats + a simulated patient answering them
        self.cue = False
        self._cue_interval = 0.6
        self._cue_pending = 0.6
        self._next_beat = math.inf
        self._beats_out: list[float] = []
        self._future_flips: list[float] = []
        self.coach_patients: dict = {}

    @property
    def packets_per_sec(self) -> float:
        now = self._clock()
        return float(sum(1 for t in list(self._rx_times) if now - t <= 1.0))

    def measure_latency(self, timeout: float = 1.0) -> Optional[float]:
        return None          # nothing to measure in simulation

    def _patient(self):
        from rhythm_coach import SimulatedPatient
        if self.sim_hand not in self.coach_patients:
            mx = 2.6 if self.sim_hand == "Right" else 4.0
            self.coach_patients[self.sim_hand] = SimulatedPatient(max_rate=mx, seed=11,
                                                                  fatigue=0.1)
        return self.coach_patients[self.sim_hand]

    def cue_on(self, interval_ms: float) -> None:
        self._cue_interval = self._cue_pending = clamp_interval(interval_ms) / 1000.0
        self.cue = True
        self._next_beat = self._clock()

    def cue_off(self) -> None:
        self.cue = False
        self._next_beat = math.inf

    def tempo(self, interval_ms: float) -> None:
        self._cue_pending = clamp_interval(interval_ms) / 1000.0

    def drain_beats(self) -> list[float]:
        self._generate(self._clock())
        out, self._beats_out = self._beats_out, []
        return out

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
        if self.cue:
            self._generate_cued(now)
            return
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

    def _generate_cued(self, now: float) -> None:
        patient = self._patient()
        while self._next_beat <= now:
            beat = self._next_beat
            self._beats_out.append(beat)
            if self._streaming and self.flipping:
                flip = patient.respond(beat)
                if flip is not None:
                    self._future_flips.append(flip)
            self._cue_interval = self._cue_pending          # TEMPO applies from the next beat
            self._next_beat = beat + self._cue_interval
        due = sorted(f for f in self._future_flips if f <= now)
        self._future_flips = [f for f in self._future_flips if f > now]
        for f in due:
            self.state = 1 - self.state
            self._pending.append(SwitchEvent(f, self.state))

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


# --------------------------------------------------------------------------- original sketch
OG_BAUD = 9600
OG_BANNER = "Parkinson's Tremor Monitor Active."


def parse_og_line(line) -> tuple[str, Optional[float]]:
    """Original sketch output -> ("interval", ms) | ("status", None) | ("banner"|"other", None)."""
    if isinstance(line, bytes):
        line = line.decode("ascii", errors="ignore")
    s = line.strip()
    if s.startswith("Interval:"):
        try:
            ms = float(s[len("Interval:"):].replace("ms", "").strip())
        except ValueError:
            return "other", None
        return ("interval", ms) if math.isfinite(ms) and ms >= 0 else ("other", None)
    if s.startswith("["):
        return "status", None
    if s == OG_BANNER:
        return "banner", None
    return "other", None


class OgArduinoDevice:
    """Read-only link to the team's ORIGINAL tremor-monitor sketch (9600 baud).

    The sketch prints 'Interval: N ms' each time the tilt switch closes (plus a status line
    such as '[!] Tremor Detected (Fast)') and drives its own LEDs. We turn each closure into
    an event timed by the Arduino's own intervals. Commands (beep, LED, LCD, metronome) are
    not supported by that sketch and are ignored. closures_only=True tells the app to count
    one event = one full flip.
    """

    closures_only = True
    sim = False

    def __init__(self, port: Optional[str] = None, clock: Callable[[], float] = time.perf_counter):
        self._want_port = port
        self._clock = clock
        self.port = port or "?"
        self.connected = False
        self.ready_received = False
        self.status = "Looking for Arduino (original sketch)..."
        self.warning = ""
        self.state: Optional[int] = None
        self.state_fresh = False
        self.last_status = ""                 # the sketch's own text, e.g. "[!] Tremor Detected"
        self.last_interval_ms: Optional[float] = None
        self._events: "queue.Queue[float]" = queue.Queue()    # Arduino ms of each closure
        self._rx_times: deque = deque(maxlen=2000)
        self._mapper = ClockMapper()
        self._ard_ms = 0.0
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self) -> None:
        import serial
        while not self._stop.is_set():
            port = self._want_port or find_arduino_port()
            if port is None:
                self.status = "Arduino not found - plug in USB"
                self._stop.wait(RETRY_S)
                continue
            try:
                ser = serial.Serial(port, OG_BAUD, timeout=0.2)
            except (serial.SerialException, OSError) as exc:
                self.status = f"Cannot open {port}: {exc}"
                self._stop.wait(RETRY_S)
                continue
            self.port, self.connected = port, True
            self.status = f"Arduino {port} (original sketch)"
            self._mapper.reset()
            self._ard_ms = 0.0
            buf = b""
            try:
                while not self._stop.is_set():
                    raw = ser.readline()
                    if not raw:
                        continue
                    if not raw.endswith(b"\n"):
                        buf = (buf + raw)[-256:]
                        continue
                    raw, buf = buf + raw, b""
                    self._handle(raw, self._clock())
            except Exception as exc:
                log.error("Serial read failed: %s", exc)
            finally:
                try:
                    ser.close()
                except Exception:
                    pass
            self.connected, self.state = False, None
            if not self._stop.is_set():
                self.status = f"Arduino disconnected ({self.port}) - plug it back in"
                self._stop.wait(RETRY_S)

    def _handle(self, raw: bytes, received: float) -> None:
        self._rx_times.append(received)
        kind, ms = parse_og_line(raw)
        if kind == "banner":
            self.ready_received = True
            self._mapper.reset()
            self._ard_ms = 0.0
        elif kind == "status":
            self.last_status = raw.decode("ascii", errors="ignore").strip()
        elif kind == "interval":
            self._ard_ms += ms                 # Arduino-measured time between closures
            self._mapper.observe(self._ard_ms, received)
            self.last_interval_ms = ms
            self.state, self.state_fresh = 1, True
            self._events.put(self._ard_ms)     # mapped to PC time at drain (exact intervals)

    # --- telemetry -------------------------------------------------------------------
    @property
    def packets_per_sec(self) -> float:
        now = self._clock()
        return float(sum(1 for t in list(self._rx_times) if now - t <= 1.0))

    def measure_latency(self, timeout: float = 1.0) -> Optional[float]:
        return None                            # the original sketch has no request/reply

    # --- shared interface (commands are not supported by the original sketch) ------------
    def start(self) -> None:
        self.drain()

    def stop(self) -> None:
        pass

    def request_state(self) -> None:
        pass

    def drain(self) -> list[SwitchEvent]:
        out = []
        while True:
            try:
                out.append(SwitchEvent(self._mapper.to_pc(self._events.get_nowait()), 1))
            except queue.Empty:
                return out

    def drain_beats(self) -> list[float]:
        return []

    def cue_on(self, interval_ms: float) -> None:
        pass

    def cue_off(self) -> None:
        pass

    def tempo(self, interval_ms: float) -> None:
        pass

    def beep(self, n: int = 1) -> None:
        pass

    def led(self, colour: str) -> None:
        _check_led(colour)

    def lcd(self, line1: str, line2: str = "") -> None:
        pass

    def close(self) -> None:
        self._stop.set()
