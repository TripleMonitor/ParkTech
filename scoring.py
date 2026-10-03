"""Pure scoring functions: features -> score 0-4 + plain-English reasons.

ALL THRESHOLDS ARE DEMO THRESHOLDS. Not validated clinically. This is a
tracking / decision-support aid, NOT a diagnosis.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

# --- Tremor (MDS-UPDRS 3.17 style), demo thresholds --------------------------
TREMOR_BAND_HZ = (3.0, 8.0)
PD_BAND_HZ = (4.0, 6.0)
TREMOR_NONE_CM = 0.1
TREMOR_CUTS_CM = (1.0, 3.0, 10.0)  # <1 ->1, 1-3 ->2, 3-10 ->3, >=10 ->4

# --- Tapping (MDS-UPDRS 3.4 style), demo thresholds --------------------------
TAP_SLOW_HZ = 2.0            # fewer taps/sec than this = slow
TAP_VERY_SLOW_HZ = 1.0
TAP_SMALL_AMP = 0.5          # normalised amplitude below this = small
TAP_DECREMENT = 0.25         # >25% amplitude loss first 3s vs last 3s
TAP_IRREGULAR_CV = 0.25      # interval coefficient of variation
TAP_MANY_HESITATIONS = 3
TAP_BARELY_ABLE_TAPS = 3     # this few taps in the whole test = barely able
TAP_MIN_HAND_VISIBLE = 0.5   # fraction of frames with a hand

# --- Asymmetry ----------------------------------------------------------------
ASYM_SCORE_DIFF = 1
ASYM_FEATURE_RATIO = 0.25


@dataclass(frozen=True)
class ScoreResult:
    score: Optional[int]          # None = could not score (e.g. no hand seen)
    reasons: tuple[str, ...]


@dataclass(frozen=True)
class TremorFeatures:
    peak_hz: float
    peak_accel: float             # m/s^2 amplitude at the peak
    displacement_cm: float
    has_clear_peak: bool
    duration_s: float


@dataclass(frozen=True)
class TappingFeatures:
    n_taps: int
    taps_per_sec: float
    mean_amplitude: float         # normalised by wrist-to-middle-MCP length
    decrement: float              # fraction lost, first 3s vs last 3s
    interval_cv: float
    hesitations: int
    hand_visible: float           # fraction of frames with a hand
    duration_s: float


def tremor_displacement_cm(accel: float, freq_hz: float) -> float:
    """Sinusoid: x = a / (2*pi*f)^2. Returns centimetres."""
    if freq_hz <= 0:
        return 0.0
    import math
    return accel / (2 * math.pi * freq_hz) ** 2 * 100.0


def score_tremor(f: TremorFeatures) -> ScoreResult:
    lo, hi = TREMOR_BAND_HZ
    if not f.has_clear_peak:
        return ScoreResult(0, (f"No clear peak in the {lo:g}-{hi:g} Hz tremor band",))

    reasons = []
    if PD_BAND_HZ[0] <= f.peak_hz <= PD_BAND_HZ[1]:
        reasons.append(f"Peak {f.peak_hz:.1f} Hz in Parkinson's range "
                       f"({PD_BAND_HZ[0]:g}-{PD_BAND_HZ[1]:g} Hz)")
    else:
        reasons.append(f"Peak {f.peak_hz:.1f} Hz in tremor band "
                       f"({lo:g}-{hi:g} Hz), outside typical Parkinson's 4-6 Hz")

    d = f.displacement_cm
    c1, c2, c3 = TREMOR_CUTS_CM
    if d < TREMOR_NONE_CM:
        score = 0
        reasons.append(f"Amplitude {d:.2f} cm < {TREMOR_NONE_CM} cm (negligible)")
    elif d < c1:
        score = 1
        reasons.append(f"Amplitude {d:.2f} cm < {c1:g} cm (slight)")
    elif d < c2:
        score = 2
        reasons.append(f"Amplitude {d:.2f} cm in {c1:g}-{c2:g} cm (mild)")
    elif d < c3:
        score = 3
        reasons.append(f"Amplitude {d:.2f} cm in {c2:g}-{c3:g} cm (moderate)")
    else:
        score = 4
        reasons.append(f"Amplitude {d:.2f} cm >= {c3:g} cm (severe)")
    return ScoreResult(score, tuple(reasons))


def _tapping_problems(f: TappingFeatures) -> tuple[list[str], int]:
    """Returns (reasons, weighted problem count). Severe problems count 2."""
    reasons: list[str] = []
    count = 0
    if f.taps_per_sec < TAP_VERY_SLOW_HZ:
        reasons.append(f"Very slow: {f.taps_per_sec:.1f} taps/s (< {TAP_VERY_SLOW_HZ:g})")
        count += 2
    elif f.taps_per_sec < TAP_SLOW_HZ:
        reasons.append(f"Slow: {f.taps_per_sec:.1f} taps/s (< {TAP_SLOW_HZ:g})")
        count += 1
    if f.mean_amplitude < TAP_SMALL_AMP:
        reasons.append(f"Small taps: amplitude {f.mean_amplitude:.2f} (< {TAP_SMALL_AMP:g})")
        count += 1
    if f.decrement > TAP_DECREMENT:
        reasons.append(f"Taps shrink: {f.decrement:.0%} smaller at end vs start "
                       f"(> {TAP_DECREMENT:.0%})")
        count += 1
    if f.interval_cv > TAP_IRREGULAR_CV:
        reasons.append(f"Irregular rhythm: interval CV {f.interval_cv:.2f} "
                       f"(> {TAP_IRREGULAR_CV:g})")
        count += 1
    if f.hesitations >= TAP_MANY_HESITATIONS:
        reasons.append(f"{f.hesitations} hesitations (gap > 2x median)")
        count += 2
    elif f.hesitations >= 1:
        reasons.append(f"{f.hesitations} hesitation(s) (gap > 2x median)")
        count += 1
    return reasons, count


def score_tapping(f: TappingFeatures) -> ScoreResult:
    if f.hand_visible < TAP_MIN_HAND_VISIBLE:
        return ScoreResult(None, (f"Hand visible only {f.hand_visible:.0%} of the time - "
                                  "cannot score, please repeat",))
    if f.n_taps <= TAP_BARELY_ABLE_TAPS:
        return ScoreResult(4, (f"Barely able: only {f.n_taps} taps in {f.duration_s:.0f} s",))

    reasons, count = _tapping_problems(f)
    if count == 0:
        return ScoreResult(0, (f"Normal: {f.taps_per_sec:.1f} taps/s, steady size and rhythm",))
    score = 1 if count == 1 else 2 if count == 2 else 3
    return ScoreResult(score, tuple(reasons))


def _ratio_diff(a: float, b: float) -> float:
    big = max(abs(a), abs(b))
    return 0.0 if big == 0 else abs(a - b) / big


def asymmetry(name: str, right: ScoreResult, left: ScoreResult,
              key_features: dict[str, tuple[float, float]]) -> tuple[str, ...]:
    """key_features: {label: (right_value, left_value)}. Returns flag reasons."""
    flags: list[str] = []
    if right.score is not None and left.score is not None \
            and abs(right.score - left.score) >= ASYM_SCORE_DIFF:
        worse = "right" if right.score > left.score else "left"
        flags.append(f"{name}: {worse} side worse (R {right.score} vs L {left.score})")
    for label, (r, l) in key_features.items():
        if _ratio_diff(r, l) >= ASYM_FEATURE_RATIO:
            flags.append(f"{name}: {label} differs {_ratio_diff(r, l):.0%} "
                         f"(R {r:.2f} vs L {l:.2f})")
    return tuple(flags)


def tremor_asymmetry(rf: TremorFeatures, lf: TremorFeatures,
                     rs: ScoreResult, ls: ScoreResult) -> tuple[str, ...]:
    # Only compare amplitude when at least one side has a measurable tremor.
    feats = {}
    if max(rf.displacement_cm, lf.displacement_cm) >= TREMOR_NONE_CM:
        feats["amplitude cm"] = (rf.displacement_cm, lf.displacement_cm)
    return asymmetry("Tremor", rs, ls, feats)


def tapping_asymmetry(rf: TappingFeatures, lf: TappingFeatures,
                      rs: ScoreResult, ls: ScoreResult) -> tuple[str, ...]:
    if rs.score is None or ls.score is None:
        return ()
    return asymmetry("Tapping", rs, ls, {
        "taps/s": (rf.taps_per_sec, lf.taps_per_sec),
        "amplitude": (rf.mean_amplitude, lf.mean_amplitude),
    })
