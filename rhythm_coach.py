"""PID rhythm coach: finds the fastest hand-flipping rhythm a patient can sustain.

The Arduino plays a metronome (one beat = one half-flip). After every beat we
check whether the patient's half-flip landed on time (|asynchrony| <= 150 ms).
The controlled variable is the rolling ON-TIME RATE over the last 8 answered beats,
with graded credit so the controller sees lag building up before beats are lost:

    credit = 1 if |asynchrony| <= 75 ms, falling linearly to 0 at 150 ms (and beyond)
    on-time rate = mean credit over the last 8 answered beats

Beats with no flip near them are MISSED. A miss right after an on-time answer is a
random skip: it is reported but excluded from the rate (otherwise a patient who
randomly skips 20% of beats could never reach the 0.85 set point). A miss right
after a LATE answer means the patient is falling behind and re-syncing, so it counts
as not on time. A too-fast tempo shows up as flips lagging further behind the beats.

The PID (reverse acting, set point 0.85) outputs a tempo CHANGE per beat, limited to
+/-0.15 beats/s.

Saturation probing: when every one of the last 8 answered beats was on time AND the
last 4 flips were tight (mean |asynchrony| < 60 ms), the measurement is saturated at
1.0 and says nothing about how far below the limit the tempo is, so the tempo rises
by at least PROBE_STEP per beat (like TCP slow start). Growing lag or any late beat
hands control back to the PID alone. PURE logic: the same code runs live (beats from the Arduino) and in
simulation (beats + SimulatedPatient).
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np

from pid import PID

HIT_WINDOW_S = 0.150
FULL_CREDIT_S = 0.075
ROLLING_BEATS = 8
SETPOINT = 0.85
MAX_STEP = 0.15                 # beats/s per beat
MIN_INTERVAL_MS, MAX_INTERVAL_MS = 200, 1500
MIN_RATE, MAX_RATE = 1000 / MAX_INTERVAL_MS, 1000 / MIN_INTERVAL_MS
MIN_RESPONSES = 4               # don't adjust before this many answered beats
UNCUED_S, CUED_S = 10.0, 35.0   # 45 s session
SETTLE_WINDOW_S = 10.0          # "max sustainable rate" = median tempo over the last 10 s
RESYNC_S = 0.25                 # simulated patient skips a beat when this far behind
DEFAULT_GAINS = (0.01, 0.0, 0.0)     # tools/tune_pid.py: only P-only settings pass (see docs)
D_TAU = 0.5                          # derivative low-pass (beats)
WARM_START = 1.15                    # cued tempo starts at 1.15 x the measured uncued rate
PROBE_STEP = 0.02                    # beats/s per beat while the window is saturated (8/8)
PROBE_MAX_ASYNC_S = 0.060            # ...and the last 4 flips are this tight


def clamp_rate(r: float) -> float:
    return min(MAX_RATE, max(MIN_RATE, r))


def warm_start(uncued_rate: float) -> float:
    """Feedforward: begin a little above the patient's own self-paced rate."""
    return clamp_rate(WARM_START * uncued_rate if uncued_rate > 0 else 1.5)


def interval_ms(rate: float) -> int:
    return int(round(1000.0 / clamp_rate(rate)))


@dataclass
class BeatRecord:
    t: float
    rate: float                     # tempo in force for this beat (beats/s)
    asynchrony: Optional[float]     # flip time - beat time (s), None = missed
    on_time: Optional[bool]         # None = missed
    on_time_rate: Optional[float]   # rolling, after this beat


def credit(asynchrony: Optional[float]) -> float:
    """Graded on-time credit: 1 within +/-75 ms, linear to 0 at +/-150 ms."""
    if asynchrony is None:
        return 0.0
    a = abs(asynchrony)
    if a <= FULL_CREDIT_S:
        return 1.0
    return max(0.0, (HIT_WINDOW_S - a) / (HIT_WINDOW_S - FULL_CREDIT_S))


def match_flip(beat_t: float, ibi: float, flips: Sequence[float], used: set) -> Optional[float]:
    """Nearest unused flip within +/- half an inter-beat interval of the beat."""
    win = max(HIT_WINDOW_S, 0.5 * ibi)
    best = None
    for i, f in enumerate(flips):
        if i in used or abs(f - beat_t) > win:
            continue
        if best is None or abs(f - beat_t) < abs(flips[best] - beat_t):
            best = i
    if best is None:
        return None
    used.add(best)
    return flips[best] - beat_t


@dataclass
class Coach:
    start_rate: float = 1.5
    gains: tuple = DEFAULT_GAINS
    rate: float = field(init=False)
    records: list = field(default_factory=list, init=False)

    def __post_init__(self) -> None:
        kp, ki, kd = self.gains
        self.pid = PID(kp, ki, kd, setpoint=SETPOINT, out_min=-MAX_STEP, out_max=MAX_STEP,
                       d_tau=D_TAU, reverse=True)
        self.rate = clamp_rate(self.start_rate)
        self.probing = False

    def on_time_rate(self) -> Optional[float]:
        answered = [credit(r.asynchrony) for r in self.records if r.on_time is not None]
        answered = answered[-ROLLING_BEATS:]
        return float(np.mean(answered)) if answered else None

    def beat_result(self, beat_t: float, asynchrony: Optional[float]) -> float:
        """Record one beat's outcome; returns the tempo for the next beat."""
        on_time = None if asynchrony is None else abs(asynchrony) <= HIT_WINDOW_S
        if asynchrony is None:
            prev = [r.on_time for r in self.records if r.on_time is not None]
            if prev and prev[-1] is False:
                on_time = False                  # falling behind, not a random skip
        self.records.append(BeatRecord(beat_t, self.rate, asynchrony, on_time, None))
        rate_now = self.on_time_rate()
        self.records[-1].on_time_rate = rate_now
        answered = sum(r.on_time is not None for r in self.records)
        if on_time is not None and answered >= MIN_RESPONSES and rate_now is not None:
            step = self.pid.update(rate_now)
            window = [r.on_time for r in self.records if r.on_time is not None][-ROLLING_BEATS:]
            recent = [abs(r.asynchrony) for r in self.records if r.asynchrony is not None][-4:]
            tight = len(recent) == 4 and float(np.mean(recent)) < PROBE_MAX_ASYNC_S
            self.probing = len(window) == ROLLING_BEATS and all(window) and tight
            if self.probing:
                step = max(step, PROBE_STEP)
            self.rate = clamp_rate(self.rate + step)
        return self.rate


MIN_VALID_BEATS = 8                 # fewer answered beats -> "not measured"


@dataclass(frozen=True)
class CoachSummary:
    max_sustainable_rate: float     # beats (half-flips) per second
    mean_asynchrony_ms: Optional[float]
    on_time_rate: Optional[float]
    missed: int
    beats: int
    cued_rate: float                # patient half-flips/s in the last 10 s of cueing
    uncued_rate: float              # patient half-flips/s in the uncued phase
    invalid_reason: str = ""        # non-empty -> result is not a measurement

    @property
    def valid(self) -> bool:
        return not self.invalid_reason


def summarise(coach: Coach, flips_cued: Sequence[float], cue_end: float,
              uncued_flips: int, uncued_s: float) -> CoachSummary:
    recs = coach.records
    if not recs:
        return CoachSummary(0.0, None, None, 0, 0, 0.0, uncued_flips / uncued_s if uncued_s else 0.0,
                            "no metronome beats received (check the Arduino)")
    answered = sum(r.asynchrony is not None for r in recs)
    on_time = sum(bool(r.on_time) for r in recs)
    reason = ""
    if len(recs) < MIN_VALID_BEATS:
        reason = f"only {len(recs)} beats received (need {MIN_VALID_BEATS})"
    elif answered < MIN_VALID_BEATS:
        reason = (f"only {answered} of {len(recs)} beats answered by a flip - "
                  "check the sensor is taped on and flipping")
    elif on_time == 0:
        reason = "no flip landed on a beat"
    elif sum(r.asynchrony is not None for r in recs[-ROLLING_BEATS:]) == 0:
        reason = "the patient stopped flipping before the end - tempo not established"
    t_end = recs[-1].t
    late = [r for r in recs if r.t >= t_end - SETTLE_WINDOW_S] or recs
    asyncs = [r.asynchrony * 1000 for r in recs if r.on_time]
    last = [f for f in flips_cued if cue_end - SETTLE_WINDOW_S <= f <= cue_end]
    return CoachSummary(
        max_sustainable_rate=float(np.median([r.rate for r in late])),
        mean_asynchrony_ms=float(np.mean(asyncs)) if asyncs else None,
        on_time_rate=coach.on_time_rate(),
        missed=sum(r.on_time is None for r in recs), beats=len(recs),
        cued_rate=len(last) / SETTLE_WINDOW_S,
        uncued_rate=uncued_flips / uncued_s if uncued_s else 0.0, invalid_reason=reason)


# --------------------------------------------------------------------------- simulation
@dataclass
class SimulatedPatient:
    """Half-flip generator responding to beats.

    max_rate: fastest sustainable half-flips/s. Each flip aims at its beat with
    N(-0.02, noise) s asynchrony but can't come sooner than 1/max_rate after the
    previous flip, so a too-fast tempo makes flips lag behind; once a flip would be
    more than RESYNC_S late the patient skips that beat and re-syncs (as people do).
    fatigue: fraction of max_rate lost over 45 s. miss_p: chance a flip is not detected
    (the patient moved but the beat registers as missed).
    stop: (start_s, duration_s) of a complete stop.
    """
    max_rate: float = 3.0
    noise_s: float = 0.03
    fatigue: float = 0.0
    miss_p: float = 0.0
    stop: Optional[tuple] = None
    uncued_ratio: float = 0.8       # self-paced rate as a fraction of max_rate
    seed: int = 0
    last_flip: float = -math.inf
    rng: random.Random = field(init=False)

    def __post_init__(self) -> None:
        self.rng = random.Random(self.seed)

    def current_max(self, t: float) -> float:
        return self.max_rate * (1 - self.fatigue * min(1.0, t / (UNCUED_S + CUED_S)))

    def respond(self, beat_t: float) -> Optional[float]:
        if self.stop and self.stop[0] <= beat_t < self.stop[0] + self.stop[1]:
            return None
        aim = beat_t + self.rng.gauss(-0.02, self.noise_s)
        flip = max(aim, self.last_flip + 1.0 / self.current_max(beat_t))
        if flip - beat_t > RESYNC_S:
            self.last_flip = beat_t          # skip this beat, catch up for the next one
            return None
        self.last_flip = flip
        if self.rng.random() < self.miss_p:
            return None                      # flipped, but not detected
        return flip

    def uncued_flips(self, t0: float, seconds: float) -> list[float]:
        rate = self.uncued_ratio * self.current_max(t0)   # self-paced rate
        n = int(seconds * rate)
        return [t0 + (i + 1) / rate + self.rng.gauss(0, self.noise_s) for i in range(n)]


def simulate(patient: SimulatedPatient, gains=DEFAULT_GAINS, start_rate: Optional[float] = None,
             cued_s: float = CUED_S, t0: float = UNCUED_S) -> tuple[Coach, list[float]]:
    """Run the cued phase offline. Returns (coach with per-beat records, flip times).

    Like the live coach, the tempo starts at WARM_START x the measured uncued rate."""
    if start_rate is None:
        start_rate = warm_start(len(patient.uncued_flips(0.0, UNCUED_S)) / UNCUED_S)
    coach = Coach(start_rate, gains)
    t, flips = t0, []
    while t < t0 + cued_s:
        f = patient.respond(t)
        if f is not None:
            flips.append(f)
        coach.beat_result(t, None if f is None else f - t)
        t += 1.0 / coach.rate
    return coach, flips
