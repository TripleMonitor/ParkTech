"""Scoring. PURE functions: features -> (score 0-4, [plain-English reasons]).

ALL THRESHOLDS ARE DEMO THRESHOLDS, not clinically validated. This is a
tracking / decision-support aid, NOT a diagnosis.

score is None only when the test could not be measured (no data / no hand);
the reasons then say why and what to do.
"""
from __future__ import annotations

from typing import NamedTuple, Optional

from tapping_analysis import TappingFeatures
from tremor_analysis import (CLEAR_PEAK_RATIO, MIN_TREMOR_CM, PD_BAND_HZ, TREMOR_BAND_HZ,
                             TremorFeatures, effective_displacement_cm)

# Tremor amplitude bands (cm): <1 -> 1, 1-3 -> 2, 3-10 -> 3, >=10 -> 4
TREMOR_CUTS_CM = (1.0, 3.0, 10.0)

# Tapping, demo thresholds
TAP_SLOW = 2.0               # taps/s
TAP_SMALL = 0.5              # normalised amplitude
TAP_DECREMENT = 0.30         # fraction
TAP_IRREGULAR_CV = 0.30
TAP_MIN_TAPS = 5             # fewer -> score 4 "barely able"
TAP_MIN_VISIBLE = 0.5        # fraction of frames with a hand

MIN_DATA_FRACTION = 0.8     # tremor recording must cover 80% of the planned test

# Asymmetry: % difference is relative to the LARGER value (4.0 vs 3.0 -> 25%)
ASYM_POINTS = 1
ASYM_RATIO = 0.25


class ScoreResult(NamedTuple):
    score: Optional[int]
    reasons: list[str]


def _band(lo_hi: tuple[float, float]) -> str:
    return f"{lo_hi[0]:g}-{lo_hi[1]:g} Hz"


def score_tremor(f: TremorFeatures, expected_s: Optional[float] = None) -> ScoreResult:
    if not f.valid or (expected_s and f.duration_s < MIN_DATA_FRACTION * expected_s):
        planned = f" of {expected_s:g} s" if expected_s else ""
        return ScoreResult(None, [f"Only {f.duration_s:.1f} s{planned} of sensor data - "
                                  "check the Arduino and repeat"])
    if not f.clear_peak:
        return ScoreResult(0, [f"No clear tremor: no distinct peak in {_band(TREMOR_BAND_HZ)} "
                               f"(largest at {f.peak_hz:.1f} Hz, {f.peak_ratio:.1f}x median, "
                               f"needs a {CLEAR_PEAK_RATIO:g}x local peak)"])
    d = f.displacement_cm
    if d < MIN_TREMOR_CM:
        return ScoreResult(0, [f"No clear tremor: {f.peak_hz:.1f} Hz peak only {d:.2f} cm "
                               f"(< {MIN_TREMOR_CM:g} cm)"])

    in_pd = PD_BAND_HZ[0] <= f.peak_hz <= PD_BAND_HZ[1]
    reasons = [f"Peak {f.peak_hz:.1f} Hz in Parkinson's range ({_band(PD_BAND_HZ)})" if in_pd
               else f"Peak {f.peak_hz:.1f} Hz: tremor band but outside Parkinson's range "
                    f"({_band(PD_BAND_HZ)})"]
    c1, c2, c3 = TREMOR_CUTS_CM
    if d < c1:
        score, label = 1, f"< {c1:g} cm (slight)"
    elif d < c2:
        score, label = 2, f"{c1:g}-{c2:g} cm (mild)"
    elif d < c3:
        score, label = 3, f"{c2:g}-{c3:g} cm (moderate)"
    else:
        score, label = 4, f">= {c3:g} cm (severe)"
    reasons.append(f"Amplitude {d:.2f} cm: {label}")
    reasons.append(f"Tremor present in {f.window_pct:.0f}% of 1-second windows")
    return ScoreResult(score, reasons)


def tapping_problems(f: TappingFeatures) -> list[str]:
    problems = []
    if f.taps_per_sec < TAP_SLOW:
        problems.append(f"Slow: {f.taps_per_sec:.1f} taps/s (< {TAP_SLOW:g})")
    if f.mean_amplitude < TAP_SMALL:
        problems.append(f"Small taps: amplitude {f.mean_amplitude:.2f} (< {TAP_SMALL:g})")
    if f.decrement > TAP_DECREMENT:
        problems.append(f"Taps shrink: {f.decrement:.0%} smaller in last 3 s vs first 3 s "
                        f"(> {TAP_DECREMENT:.0%})")
    if f.interval_cv > TAP_IRREGULAR_CV:
        problems.append(f"Irregular rhythm: interval CV {f.interval_cv:.2f} (> {TAP_IRREGULAR_CV:g})")
    if f.hesitations >= 1:
        problems.append(f"{f.hesitations} hesitation(s): pause > 2x the usual gap")
    return problems


def _tracking_note(f: TappingFeatures) -> list[str]:
    if f.tracking_lost_s <= 0:
        return []
    return [f"Note: hand tracking lost for {f.tracking_lost_s:.1f} s "
            "(those gaps are ignored for rhythm)"]


def score_tapping(f: TappingFeatures) -> ScoreResult:
    if f.hand_visible < TAP_MIN_VISIBLE:
        return ScoreResult(None, [f"Hand seen in only {f.hand_visible:.0%} of frames - "
                                  "cannot score, please repeat"])
    if f.n_taps < TAP_MIN_TAPS:
        return ScoreResult(4, [f"Barely able: only {f.n_taps} taps detected in "
                               f"{f.duration_s:.0f} s (< {TAP_MIN_TAPS}); openings smaller "
                               "than 0.15 aren't counted"] + _tracking_note(f))
    problems = tapping_problems(f)
    if not problems:
        return ScoreResult(0, [f"No problems: {f.taps_per_sec:.1f} taps/s, amplitude "
                               f"{f.mean_amplitude:.2f}, steady rhythm"] + _tracking_note(f))
    return ScoreResult(min(3, len(problems)), problems + _tracking_note(f))


def _differs(a: float, b: float) -> float:
    big = max(abs(a), abs(b))
    return 0.0 if big == 0 else abs(a - b) / big


def asymmetry(test: str, right: ScoreResult, left: ScoreResult,
              feature: str = "", right_value: float = 0.0, left_value: float = 0.0,
              compare_feature: bool = True) -> list[str]:
    """Flags if scores differ by >= 1 point or the key feature differs by >= 25%."""
    flags = []
    if right.score is not None and left.score is not None \
            and abs(right.score - left.score) >= ASYM_POINTS:
        worse = "Right" if right.score > left.score else "Left"
        flags.append(f"{test}: {worse} worse (R {right.score} vs L {left.score})")
    if feature and compare_feature and _differs(right_value, left_value) >= ASYM_RATIO:
        flags.append(f"{test}: {feature} differs {_differs(right_value, left_value):.0%} "
                     f"(R {right_value:.2f} vs L {left_value:.2f})")
    return flags


def tremor_asymmetry(rf: TremorFeatures, lf: TremorFeatures,
                     rs: ScoreResult, ls: ScoreResult) -> list[str]:
    # Only displacement from a clear tremor peak counts (otherwise 0), and the %
    # rule only applies once either side reaches the tremor threshold, so two
    # noise-level readings (e.g. 0.01 vs 0.02 cm) are never flagged.
    rd, ld = effective_displacement_cm(rf), effective_displacement_cm(lf)
    measurable = rs.score is not None and ls.score is not None and max(rd, ld) >= MIN_TREMOR_CM
    return asymmetry("Tremor", rs, ls, "displacement cm", rd, ld, compare_feature=measurable)


def tapping_asymmetry(rf: TappingFeatures, lf: TappingFeatures,
                      rs: ScoreResult, ls: ScoreResult) -> list[str]:
    # taps/s is only compared when both sides could be scored (hand visible)
    both = rs.score is not None and ls.score is not None
    return asymmetry("Tapping", rs, ls, "taps/s", rf.taps_per_sec, lf.taps_per_sec,
                     compare_feature=both)
