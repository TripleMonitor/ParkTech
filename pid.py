"""Small, explicit PID controller.

- output limits
- anti-windup: the integrator only accumulates when that doesn't push a saturated
  output further into saturation, and it is itself clamped to the output range
- derivative on measurement (no "derivative kick" when the set point changes),
  low-pass filtered with time constant d_tau
- reverse=True: error = measurement - set point (output rises when measurement is high)
"""
from __future__ import annotations

from typing import Optional


class PID:
    def __init__(self, kp: float, ki: float, kd: float, setpoint: float = 0.0,
                 out_min: float = -1.0, out_max: float = 1.0, d_tau: float = 0.0,
                 reverse: bool = False):
        if out_min >= out_max:
            raise ValueError("out_min must be < out_max")
        self.kp, self.ki, self.kd = kp, ki, kd
        self.setpoint = setpoint
        self.out_min, self.out_max = out_min, out_max
        self.d_tau = d_tau
        self.reverse = reverse
        self.reset()

    def reset(self) -> None:
        self.integral = 0.0
        self._d_filt = 0.0
        self._last_meas: Optional[float] = None
        self.last = (0.0, 0.0, 0.0)          # (p, i, d) terms of the last update

    def _clamp(self, v: float) -> float:
        return min(self.out_max, max(self.out_min, v))

    def update(self, measurement: float, dt: float = 1.0) -> float:
        if dt <= 0:
            raise ValueError("dt must be > 0")
        e = (measurement - self.setpoint) if self.reverse else (self.setpoint - measurement)
        p = self.kp * e
        # derivative on measurement, sign follows the error convention
        raw_d = 0.0
        if self._last_meas is not None:
            slope = (measurement - self._last_meas) / dt
            raw_d = slope if self.reverse else -slope
        self._last_meas = measurement
        alpha = dt / (self.d_tau + dt) if self.d_tau > 0 else 1.0
        self._d_filt += alpha * (raw_d - self._d_filt)
        d = self.kd * self._d_filt
        # integrator with conditional integration (anti-windup)
        cand = self.integral + self.ki * e * dt
        unsat = p + cand + d
        pushing_high = unsat > self.out_max and e * (1 if self.ki >= 0 else -1) > 0
        pushing_low = unsat < self.out_min and e * (1 if self.ki >= 0 else -1) < 0
        if not (pushing_high or pushing_low):
            self.integral = self._clamp(cand)
        out = self._clamp(p + self.integral + d)
        self.last = (p, self.integral, d)
        return out
