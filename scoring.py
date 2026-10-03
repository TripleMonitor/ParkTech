"""Scoring. PURE functions: features -> (score 0-4, [plain-English reasons]).

ALL THRESHOLDS ARE DEMO THRESHOLDS, not clinically validated. This is a
tracking / decision-support aid, NOT a diagnosis.

score is None only when the test could not be measured (no hand / no data);
the reasons then say why and what to do.
"""
from __future__ import annotations

from typing import NamedTuple, Optional

from flipping_analysis import NO_CHANGE_WARN_S, FlippingFeatures
from tapping_analysis import TappingFeatures
from tremor_analysis import (CLEAR_PEAK_RATIO, MIN_TREMOR_CM, PD_BAND_HZ, TREMOR_BAND_HZ,
                             TremorFeatures, effective_displacement_cm)

# Tremor amplitude bands (cm): <1 -> 1, 1-3 -> 2, 3-10 -> 3, >=10 -> 4
TREMOR_CUTS_CM = (1.0, 3.0, 10.0)
MIN_DATA_FRACTION = 0.8      # a recording must cover 80% of the planned test

# Finger tapping
TAP_SLOW = 2.0               # taps/s
TAP_SMALL = 0.5              # normalised amplitude
TAP_DECREMENT = 0.30
TAP_IRREGULAR_CV = 0.30
TAP_MIN_TAPS = 5
TAP_MIN_VISIBLE = 0.5

# Hand flipping
FLIP_SLOW = 1.5              # full flips/s
FLIP_IRREGULAR_CV = 0.35
FLIP_DECREMENT = 0.25
FLIP_MIN_FULL = 5

# Asymmetry: % difference is relative to the LARGER value (4.0 vs 3.0 -> 25%)
ASYM_POINTS = 1
ASYM_RATIO = 0.25

SENSOR_CHECK = "Sensor not flipping - check it's taped on and upright"


class ScoreResult(NamedTuple):
    score: Optional[int]
    reasons: list[str]


def _band(lo_hi: tuple[float, float]) -> str:
    return f"{lo_hi[0]:g}-{lo_hi[1]:g} Hz"


# --------------------------------------------------------------------------- tremor
def score_tremor(f: TremorFeatures, expected_s: Optional[float] = None) -> ScoreResult:
    if not f.valid:
        if f.hand_visible < 0.5:
            return ScoreResult(None, [f"Hand seen in only {f.hand_visible:.0%} of frames - "
                                      "keep the palm facing the camera and repeat"])
        return ScoreResult(None, [f"Only {f.duration_s:.1f} s of usable video - repeat"])
    if expected_s and f.duration_s < MIN_DATA_FRACTION * expected_s:
        return ScoreResult(None, [f"Only {f.duration_s:.1f} s of {expected_s:g} s recorded - repeat"])
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
    reasons.append(f"Fingertip moves {d:.2f} cm: {label}")
    reasons.append(f"Tremor present in {f.window_pct:.0f}% of 1-second windows")
    return ScoreResult(score, reasons)


# --------------------------------------------------------------------------- tapping
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


# --------------------------------------------------------------------------- flipping
def flipping_problems(f: FlippingFeatures) -> list[str]:
    problems = []
    if f.flips_per_sec < FLIP_SLOW:
        problems.append(f"Slow: {f.flips_per_sec:.1f} full flips/s (< {FLIP_SLOW:g})")
    if f.interval_cv > FLIP_IRREGULAR_CV:
        problems.append(f"Irregular rhythm: interval CV {f.interval_cv:.2f} (> {FLIP_IRREGULAR_CV:g})")
    if f.decrement > FLIP_DECREMENT:
        problems.append(f"Slows down: {f.decrement:.0%} slower in last 4 s vs first 4 s "
                        f"(> {FLIP_DECREMENT:.0%})")
    if f.hesitations >= 1:
        problems.append(f"{f.hesitations} hesitation(s): pause > 2x the usual gap")
    return problems


def score_flipping(f: FlippingFeatures) -> ScoreResult:
    if f.half_flips == 0:
        return ScoreResult(4, [f"No flips detected in {f.duration_s:.0f} s", SENSOR_CHECK])
    note = [f"Note: no flip in the first {NO_CHANGE_WARN_S:g} s - {SENSOR_CHECK.lower()}"] \
        if f.stuck_at_start else []
    if f.full_flips < FLIP_MIN_FULL:
        return ScoreResult(4, [f"Barely able: only {f.full_flips} full flips in "
                               f"{f.duration_s:.0f} s (< {FLIP_MIN_FULL})"] + note)
    problems = flipping_problems(f)
    if not problems:
        return ScoreResult(0, [f"No problems: {f.flips_per_sec:.1f} full flips/s "
                               f"({f.full_flips} flips), steady rhythm"] + note)
    return ScoreResult(min(3, len(problems)), problems + note)


# --------------------------------------------------------------------------- asymmetry
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


def _both_scored(rs: ScoreResult, ls: ScoreResult) -> bool:
    return rs.score is not None and ls.score is not None


def tremor_asymmetry(rf: TremorFeatures, lf: TremorFeatures,
                     rs: ScoreResult, ls: ScoreResult) -> list[str]:
    # Only displacement from a clear peak counts (otherwise 0), and the % rule only
    # applies once either side reaches the tremor threshold, so two noise-level
    # readings (e.g. 0.01 vs 0.02 cm) are never flagged.
    rd, ld = effective_displacement_cm(rf), effective_displacement_cm(lf)
    measurable = _both_scored(rs, ls) and max(rd, ld) >= MIN_TREMOR_CM
    return asymmetry("Tremor", rs, ls, "displacement cm", rd, ld, compare_feature=measurable)


def tapping_asymmetry(rf: TappingFeatures, lf: TappingFeatures,
                      rs: ScoreResult, ls: ScoreResult) -> list[str]:
    return asymmetry("Tapping", rs, ls, "taps/s", rf.taps_per_sec, lf.taps_per_sec,
                     compare_feature=_both_scored(rs, ls))


def flipping_asymmetry(rf: FlippingFeatures, lf: FlippingFeatures,
                       rs: ScoreResult, ls: ScoreResult) -> list[str]:
    return asymmetry("Flipping", rs, ls, "flips/s", rf.flips_per_sec, lf.flips_per_sec,
                     compare_feature=_both_scored(rs, ls))


# --------------------------------------------------------------------------- explainability
class Rule(NamedTuple):
    """One scoring rule as shown in the WHY THIS SCORE panel."""
    label: str
    measured: str
    threshold: str
    fired: bool          # True = this rule found a problem / raised the score


class Explanation(NamedTuple):
    score: Optional[int]
    rules: list[Rule]
    formula: str


def explain_tremor(f: TremorFeatures, expected_s: Optional[float] = None) -> Explanation:
    s = score_tremor(f, expected_s).score
    if s is None:
        return Explanation(None, [], "not scored: not enough usable data")
    d = f.displacement_cm if f.clear_peak else 0.0
    c1, c2, c3 = TREMOR_CUTS_CM
    in_pd = f.clear_peak and PD_BAND_HZ[0] <= f.peak_hz <= PD_BAND_HZ[1]
    rules = [
        Rule("Distinct peak 3-8 Hz", ">99x" if f.peak_ratio > 99 else f"{f.peak_ratio:.1f}x", f">= {CLEAR_PEAK_RATIO:g}x + local max",
             f.clear_peak),
        Rule("Peak in PD range", f"{f.peak_hz:.1f} Hz", "4-6 Hz", in_pd),
        Rule("Amplitude", f"{d:.2f} cm", f">= {MIN_TREMOR_CM:g} cm", d >= MIN_TREMOR_CM),
        Rule("Amplitude", f"{d:.2f} cm", f">= {c1:g} cm", d >= c1),
        Rule("Amplitude", f"{d:.2f} cm", f">= {c2:g} cm", d >= c2),
        Rule("Amplitude", f"{d:.2f} cm", f">= {c3:g} cm", d >= c3),
    ]
    return Explanation(s, rules, "score = number of amplitude rules fired "
                                 "(0 if no distinct peak); PD-range row is info only")


def explain_tapping(f: TappingFeatures) -> Explanation:
    s = score_tapping(f).score
    if s is None:
        return Explanation(None, [], "not scored: hand not visible enough")
    rules = [
        Rule("Taps detected", f"{f.n_taps}", f"< {TAP_MIN_TAPS} -> score 4", f.n_taps < TAP_MIN_TAPS),
        Rule("Speed", f"{f.taps_per_sec:.1f}/s", f"< {TAP_SLOW:g}/s", f.taps_per_sec < TAP_SLOW),
        Rule("Amplitude", f"{f.mean_amplitude:.2f}", f"< {TAP_SMALL:g}", f.mean_amplitude < TAP_SMALL),
        Rule("Decrement", f"{f.decrement:.0%}", f"> {TAP_DECREMENT:.0%}", f.decrement > TAP_DECREMENT),
        Rule("Rhythm CV", f"{f.interval_cv:.2f}", f"> {TAP_IRREGULAR_CV:g}",
             f.interval_cv > TAP_IRREGULAR_CV),
        Rule("Hesitations", f"{f.hesitations}", ">= 1", f.hesitations >= 1),
    ]
    return Explanation(s, rules, "score = min(3, problems fired); 4 if < 5 taps")


def explain_flipping(f: FlippingFeatures) -> Explanation:
    s = score_flipping(f).score
    if s is None:
        return Explanation(None, [], "not scored")
    rules = [
        Rule("Full flips", f"{f.full_flips}", f"< {FLIP_MIN_FULL} -> score 4",
             f.full_flips < FLIP_MIN_FULL),
        Rule("Speed", f"{f.flips_per_sec:.2f}/s", f"< {FLIP_SLOW:g}/s", f.flips_per_sec < FLIP_SLOW),
        Rule("Rhythm CV", f"{f.interval_cv:.2f}", f"> {FLIP_IRREGULAR_CV:g}",
             f.interval_cv > FLIP_IRREGULAR_CV),
        Rule("Slowdown", f"{f.decrement:.0%}", f"> {FLIP_DECREMENT:.0%}", f.decrement > FLIP_DECREMENT),
        Rule("Hesitations", f"{f.hesitations}", ">= 1", f.hesitations >= 1),
    ]
    return Explanation(s, rules, "score = min(3, problems fired); 4 if < 5 full flips")


def score_from_rules(kind: str, e: Explanation) -> Optional[int]:
    """Recompute the score from the rule list alone (used to prove panel == scorer)."""
    if e.score is None:
        return None
    if kind == "tremor":
        return 0 if not e.rules[0].fired else sum(r.fired for r in e.rules[2:])
    if e.rules[0].fired:
        return 4
    return min(3, sum(r.fired for r in e.rules[1:]))
