"""Live rhythm-coach session on top of the device API (Arduino or MockDevice).

Phases: ready -> countdown (3 s) -> uncued (10 s, self-paced flipping) ->
cued (35 s, Arduino metronome + PID tempo) -> done.
Each beat is judged once the flip that answers it has had time to arrive.
"""
from __future__ import annotations

from typing import Optional

from flipping_analysis import debounce
from rhythm_coach import (CUED_S, DEFAULT_GAINS, PROBE_STEP, SETPOINT, UNCUED_S, Coach,
                          CoachSummary, interval_ms, match_flip, summarise, warm_start)

COUNTDOWN_S = 3.0
JUDGE_DELAY_S = 0.35            # wait this long after a beat before judging it
GRAPH_EVERY_S = 0.25


class CoachSession:
    def __init__(self, device, hand: str, clock, uncued_s: float = UNCUED_S,
                 cued_s: float = CUED_S):
        self.device, self.hand, self.clock = device, hand, clock
        self.uncued_s, self.cued_s = uncued_s, cued_s
        self.phase = "ready"
        self.t_phase = clock()
        self.events: list = []
        self.calib: Optional[int] = None
        self.coach: Optional[Coach] = None
        self.beats: list[float] = []
        self.judged = 0
        self.used: set = set()
        self.flips: list[float] = []
        self.uncued_flips = 0
        self.cue_start = 0.0
        self.last_beat: Optional[float] = None
        self.series: list[tuple] = []       # (t, tempo, patient_rate, on_time_rate)
        self._graph_at = 0.0
        self.summary: Optional[CoachSummary] = None
        self.last_beep = 0
        device.request_state()
        if hasattr(device, "sim_hand"):
            device.sim_hand = hand

    # ---------------------------------------------------------------- control
    def start(self) -> None:
        self.calib = self.device.state
        self.phase, self.t_phase = "countdown", self.clock()
        self.last_beep = 0

    def abort(self) -> None:
        self.device.cue_off()
        self.device.stop()

    def _flip_times(self, since: float) -> list[float]:
        ch = debounce([e.t for e in self.events], [e.state for e in self.events], self.calib)
        return [t for t, _ in ch if t >= since]

    def tick(self) -> None:
        now = self.clock()
        el = now - self.t_phase
        if self.phase == "countdown":
            n = int(el) + 1
            if n <= COUNTDOWN_S and n > self.last_beep:
                self.last_beep = n
                self.device.beep(1)
            if el >= COUNTDOWN_S:
                self.phase, self.t_phase = "uncued", now
                self.device.start()
            return
        if self.phase in ("uncued", "cued"):
            self.events.extend(self.device.drain())
        if self.phase == "uncued":
            if el >= self.uncued_s:
                self.uncued_flips = len(self._flip_times(self.t_phase))
                rate0 = warm_start(self.uncued_flips / self.uncued_s)
                self.coach = Coach(rate0, DEFAULT_GAINS)
                self.phase, self.t_phase = "cued", now
                self.cue_start = now
                self.device.cue_on(interval_ms(rate0))
            return
        if self.phase == "cued":
            new = self.device.drain_beats()
            if new:
                self.last_beat = new[-1]
            self.beats.extend(new)
            self.flips = self._flip_times(self.cue_start - 0.5)
            self._judge(now)
            if now - self._graph_at >= GRAPH_EVERY_S:
                self._graph_at = now
                recent = [f for f in self.flips if now - f <= 3.0]
                self.series.append((now - self.cue_start, self.coach.rate, len(recent) / 3.0,
                                    self.coach.on_time_rate()))
            if el >= self.cued_s:
                self.device.cue_off()
                self.device.stop()
                self.events.extend(self.device.drain())
                self.flips = self._flip_times(self.cue_start - 0.5)
                self._judge(now + 10)                 # judge the remaining beats
                self.summary = summarise(self.coach, self.flips, now, self.uncued_flips,
                                         self.uncued_s)
                self.phase, self.t_phase = "done", now

    def _judge(self, now: float) -> None:
        old_rate = self.coach.rate
        while self.judged < len(self.beats) and self.beats[self.judged] <= now - JUDGE_DELAY_S:
            b = self.beats[self.judged]
            a = match_flip(b, 1.0 / self.coach.rate, self.flips, self.used)
            self.coach.beat_result(b, a)
            self.judged += 1
        if self.coach.rate != old_rate and self.phase == "cued":
            self.device.tempo(interval_ms(self.coach.rate))

    # ---------------------------------------------------------------- display helpers
    @property
    def palm_down(self) -> Optional[bool]:
        st = self.device.state
        if st is None:
            return None
        return True if self.calib is None else st == self.calib

    def beat_pulse(self) -> float:
        """1.0 right on a beat, decaying to 0 over 250 ms (drives the pulsing circle)."""
        if self.last_beat is None or self.phase != "cued":
            return 0.0
        return max(0.0, 1.0 - (self.clock() - self.last_beat) / 0.25)

    @staticmethod
    def gains_text() -> str:
        kp, ki, kd = DEFAULT_GAINS
        return (f"PID kp {kp:g}  ki {ki:g}  kd {kd:g} | set point {SETPOINT:g} | "
                f"probe +{PROBE_STEP:g}/beat")
