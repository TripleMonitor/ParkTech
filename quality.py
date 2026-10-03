"""Signal quality for the camera tests. PURE functions only.

All three numbers are measured from the recording itself:
  detected   fraction of frames where MediaPipe found a hand
  confidence mean MediaPipe HANDEDNESS score over detected frames (the only per-frame
             score the Hands solution exposes; shown for information, it is NOT a
             tracking confidence and does not trigger LOW CONFIDENCE)
  dropped    frames lost: failed camera reads + timing gaps > 2x the median frame interval
A result is "low confidence" if detection or dropped frames are out of range (demo thresholds).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np

MIN_DETECTED = 0.80
MIN_CONFIDENCE = 0.70
MAX_DROPPED = 0.10
GAP_FACTOR = 2.0


@dataclass(frozen=True)
class SignalQuality:
    frames: int
    detected: float               # 0..1
    confidence: Optional[float]   # 0..1, None if no hand ever detected
    dropped: float                # 0..1 of expected frames
    dropped_frames: int

    @property
    def problems(self) -> list[str]:
        out = []
        if self.detected < MIN_DETECTED:
            out.append(f"hand found in {self.detected:.0%} of frames (< {MIN_DETECTED:.0%})")
        if self.dropped > MAX_DROPPED:
            out.append(f"{self.dropped:.0%} frames dropped (> {MAX_DROPPED:.0%})")
        return out

    @property
    def low(self) -> bool:
        return bool(self.problems)


def signal_quality(times: Sequence[float], detected: Sequence[bool],
                   scores: Sequence[Optional[float]], failed_reads: int = 0) -> SignalQuality:
    n = len(times)
    if n == 0:
        return SignalQuality(0, 0.0, None, 1.0, failed_reads)
    det = float(np.mean([bool(d) for d in detected])) if detected else 0.0
    sc = [s for s, d in zip(scores, detected) if d and s is not None]
    conf = float(np.mean(sc)) if sc else None
    gaps_lost = 0
    if n >= 3:
        dt = np.diff(np.asarray(times, dtype=float))
        med = float(np.median(dt))
        if med > 0:
            big = dt[dt > GAP_FACTOR * med]
            gaps_lost = int(np.sum(np.round(big / med) - 1))
    lost = gaps_lost + failed_reads
    return SignalQuality(n, det, conf, lost / (n + lost), lost)
