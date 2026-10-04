"""Rapid hand flipping (MDS-UPDRS 3.6 style) from SW-520D tilt-switch events.
PURE functions only: no serial, no UI.

Each debounced switch change is one half-flip (palm-down <-> palm-up); two make
one full flip. The firmware already debounces (30 ms) but the ball can still
chatter, so here any change that is followed by another change within 60 ms is
ignored: a state only counts once it has been held for >= 60 ms. A bounce burst
x->y->x->y therefore counts once, and a burst that returns to x counts as nothing.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np

MIN_CHANGE_GAP_S = 0.060      # ball bounce: ignore changes faster than this
EDGE_WINDOW_S = 4.0           # decrement = rate in first 4 s vs last 4 s
HESITATION_FACTOR = 2.0       # half-flip interval > 2x median
NO_CHANGE_WARN_S = 3.0        # no change in the first 3 s -> sensor check


@dataclass(frozen=True)
class FlippingFeatures:
    half_flips: int
    full_flips: int
    flips_per_sec: float          # full flips per second
    interval_cv: float            # of half-flip intervals
    decrement: float              # 1 - rate(last 4 s) / rate(first 4 s), >= 0
    hesitations: int
    duration_s: float
    first_change_s: Optional[float]   # None if the switch never changed
    flip_times: tuple = ()        # accepted half-flip times, relative to start

    @property
    def stuck_at_start(self) -> bool:
        return self.first_change_s is None or self.first_change_s > NO_CHANGE_WARN_S


def debounce(times: Sequence[float], states: Sequence[int],
             initial_state: Optional[int] = None,
             min_gap: float = MIN_CHANGE_GAP_S) -> list[tuple[float, int]]:
    """Accepted (time, new_state) changes from raw state reports (any order of repeats).

    Reports that repeat the current state (e.g. STATE replies) are not changes.
    A raw change only counts if the new state is held for >= min_gap; its time is
    the start of the burst that led to it.
    """
    pairs = sorted(zip(times, states), key=lambda p: p[0])
    if not pairs:
        return []
    prev = pairs[0][1] if initial_state is None else initial_state
    raw = []
    for t, s in pairs:
        if s != prev:
            raw.append((float(t), int(s)))
            prev = s
    accepted, acc = [], (pairs[0][1] if initial_state is None else initial_state)
    burst_start = None
    for i, (t, s) in enumerate(raw):
        if burst_start is None:
            burst_start = t
        held = i + 1 == len(raw) or raw[i + 1][0] - t >= min_gap
        if held:
            if s != acc:
                accepted.append((burst_start, s))
                acc = s
            burst_start = None
    return accepted


def _rate(intervals: np.ndarray) -> float:
    return 0.0 if len(intervals) == 0 else float(1.0 / intervals.mean())


def analyze_flipping(times: Sequence[float], states: Sequence[int], t_start: float,
                     duration_s: float, initial_state: Optional[int] = None) -> FlippingFeatures:
    """times on any clock (s); only changes inside [t_start, t_start + duration] count."""
    changes = [(t - t_start, s) for t, s in debounce(times, states, initial_state)
               if 0.0 <= t - t_start <= duration_s]
    ft = np.array([t for t, _ in changes], dtype=float)
    half = len(ft)
    first = float(ft[0]) if half else None
    intervals = np.diff(ft)
    cv, hes, dec = 0.0, 0, 0.0
    if len(intervals) >= 2:
        cv = float(intervals.std() / intervals.mean())
        hes = int((intervals > HESITATION_FACTOR * np.median(intervals)).sum())
        mids = (ft[:-1] + ft[1:]) / 2
        r_first = _rate(intervals[mids < EDGE_WINDOW_S])
        r_last = _rate(intervals[mids > duration_s - EDGE_WINDOW_S])
        if r_first > 0 and r_last > 0:
            dec = max(0.0, 1.0 - r_last / r_first)
    return FlippingFeatures(
        half_flips=half, full_flips=half // 2,
        flips_per_sec=(half / 2) / duration_s if duration_s > 0 else 0.0,
        interval_cv=cv, decrement=dec, hesitations=hes, duration_s=duration_s,
        first_change_s=first, flip_times=tuple(float(x) for x in ft))


LIVE_WINDOW_S = 2.0
LIVE_GREEN, LIVE_YELLOW = 1.5, 1.0     # full flips/s (demo thresholds, same as scoring "slow")


def live_rate(flip_times: Sequence[float], elapsed: float, window: float = LIVE_WINDOW_S
              ) -> Optional[float]:
    """Full flips/s over the last `window` seconds (None until a full window exists)."""
    if elapsed < window:
        return None
    n = sum(1 for t in flip_times if elapsed - window < t <= elapsed)
    return (n / 2) / window


def live_colour(rate: Optional[float]) -> Optional[str]:
    """LED colour for the live flip speed: G >= 1.5, Y >= 1.0, else R (None = no window yet)."""
    if rate is None:
        return None
    return "G" if rate >= LIVE_GREEN else "Y" if rate >= LIVE_YELLOW else "R"


def analyze_closures(times: Sequence[float], t_start: float, duration_s: float,
                     min_gap: float = MIN_CHANGE_GAP_S) -> FlippingFeatures:
    """Hand flipping from switch CLOSURES only (the team's original sketch reports one
    'Interval' line each time the tilt switch closes; it never reports the opening).

    One closure = one full flip (palm down -> up -> down). Intervals are closure-to-closure,
    i.e. full-flip intervals; CV, decrement and hesitations are computed on those.
    half_flips is reported as 2 x full flips (each closure implies a complete open/close
    cycle) so fusion with the camera's half-flip count stays comparable.
    """
    ts = sorted(t - t_start for t in times if 0.0 <= t - t_start <= duration_s)
    kept: list[float] = []
    for t in ts:                                   # ignore bounce closures < 60 ms apart
        if not kept or t - kept[-1] >= min_gap:
            kept.append(t)
    ft = np.array(kept, dtype=float)
    n = len(ft)
    intervals = np.diff(ft)
    cv, hes, dec = 0.0, 0, 0.0
    if len(intervals) >= 2:
        cv = float(intervals.std() / intervals.mean())
        hes = int((intervals > HESITATION_FACTOR * np.median(intervals)).sum())
        mids = (ft[:-1] + ft[1:]) / 2
        r_first = _rate(intervals[mids < EDGE_WINDOW_S])
        r_last = _rate(intervals[mids > duration_s - EDGE_WINDOW_S])
        if r_first > 0 and r_last > 0:
            dec = max(0.0, 1.0 - r_last / r_first)
    return FlippingFeatures(
        half_flips=2 * n, full_flips=n,
        flips_per_sec=n / duration_s if duration_s > 0 else 0.0,
        interval_cv=cv, decrement=dec, hesitations=hes, duration_s=duration_s,
        first_change_s=float(ft[0]) if n else None, flip_times=tuple(float(x) for x in ft))
