"""Calibrate the demo thresholds from real recordings (app.py --record --label NAME).

Label convention: a label containing "normal", "healthy" or "baseline" is a NORMAL run
(healthy volunteer doing the test properly); anything else (e.g. "alex_acted_tremor",
"sam_acted_slow") is an ACTED impairment run.

For every recording the features and score are recomputed with the CURRENT maths
(recorder.reanalyse), then per test we report:
  - how many normal runs score > 0      (false alarms - want 0)
  - how many acted runs score 0         (misses - want 0)
  - key features: median and 10th/90th percentile, normal vs acted
  - SUGGESTED thresholds: placed just outside the normal range (only suggestions;
    edit the constants at the top of scoring.py / tremor_analysis.py to apply them)

Usage: python tools/calibrate.py [recordings_dir]      (writes <dir>/calibration_report.txt)
This calibrates the tool to YOUR camera, lighting and sensor using healthy volunteers.
It is not clinical validation.
"""
from __future__ import annotations

import glob
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import scoring  # noqa: E402
import tremor_analysis  # noqa: E402
from fusion import SMALL_AMPLITUDE_DEG  # noqa: E402
from recorder import load_record, reanalyse  # noqa: E402

NORMAL_WORDS = ("normal", "healthy", "baseline")


def is_normal(label: str) -> bool:
    return any(w in label.lower() for w in NORMAL_WORDS)


def features_of(kind: str, f, fu) -> dict:
    if kind == "tremor":
        return {"displacement_cm": tremor_analysis.effective_displacement_cm(f),
                "peak_hz": f.peak_hz if f.clear_peak else np.nan}
    if kind == "tapping":
        return {"taps_per_sec": f.taps_per_sec, "amplitude": f.mean_amplitude,
                "decrement": f.decrement, "interval_cv": f.interval_cv,
                "hesitations": f.hesitations}
    out = {"flips_per_sec": f.flips_per_sec, "interval_cv": f.interval_cv,
           "decrement": f.decrement, "hesitations": f.hesitations}
    if fu is not None and fu.camera_ok and fu.camera_half_flips:
        out["rotation_deg"] = fu.median_amplitude_deg
        out["fusion_locked"] = float(fu.locked)
    return out


def load_all(directory: str) -> list[dict]:
    rows = []
    for path in sorted(glob.glob(os.path.join(directory, "*.json"))):
        try:
            rec = load_record(path)
            f, r, fu = reanalyse(rec)
        except Exception as exc:
            print(f"skip {os.path.basename(path)}: {exc}")
            continue
        m = rec["meta"]
        rows.append({"file": os.path.basename(path), "test": m["test"], "hand": m["hand"],
                     "label": m.get("label", ""), "normal": is_normal(m.get("label", "")),
                     "mode": m.get("mode", "?"), "score": r.score,
                     "app_score": rec["app_result"]["score"], "features": features_of(m["test"], f, fu)})
    return rows


def pct(vals: list, q: float) -> float:
    v = [x for x in vals if x is not None and not (isinstance(x, float) and np.isnan(x))]
    return float(np.percentile(v, q)) if v else float("nan")


def suggestions(test: str, normal: list[dict]) -> list[str]:
    """Thresholds placed just outside the normal range (needs >= 3 normal runs)."""
    if len(normal) < 3:
        return [f"need >= 3 normal runs for suggestions (have {len(normal)})"]
    feats = lambda k: [r["features"].get(k) for r in normal]  # noqa: E731
    out = []
    if test == "tremor":
        p95 = pct(feats("displacement_cm"), 95)
        sug = max(tremor_analysis.MIN_TREMOR_CM, round(1.5 * p95, 2))
        out.append(f"MIN_TREMOR_CM (tremor_analysis.py): now {tremor_analysis.MIN_TREMOR_CM}, "
                   f"normal p95 {p95:.3f} cm -> suggest {sug}")
    elif test == "tapping":
        out.append(f"TAP_SLOW: now {scoring.TAP_SLOW}, normal p10 {pct(feats('taps_per_sec'), 10):.2f}"
                   f" -> suggest {min(scoring.TAP_SLOW, round(0.8 * pct(feats('taps_per_sec'), 10), 2))}")
        out.append(f"TAP_SMALL: now {scoring.TAP_SMALL}, normal p10 {pct(feats('amplitude'), 10):.2f}"
                   f" -> suggest {min(scoring.TAP_SMALL, round(0.7 * pct(feats('amplitude'), 10), 2))}")
        out.append(f"TAP_IRREGULAR_CV: now {scoring.TAP_IRREGULAR_CV}, normal p90 "
                   f"{pct(feats('interval_cv'), 90):.2f} -> suggest "
                   f"{max(scoring.TAP_IRREGULAR_CV, round(1.3 * pct(feats('interval_cv'), 90), 2))}")
        out.append(f"TAP_DECREMENT: now {scoring.TAP_DECREMENT}, normal p90 "
                   f"{pct(feats('decrement'), 90):.2f} -> suggest "
                   f"{max(scoring.TAP_DECREMENT, round(pct(feats('decrement'), 90) + 0.1, 2))}")
    else:
        out.append(f"FLIP_SLOW: now {scoring.FLIP_SLOW}, normal p10 {pct(feats('flips_per_sec'), 10):.2f}"
                   f" -> suggest {min(scoring.FLIP_SLOW, round(0.8 * pct(feats('flips_per_sec'), 10), 2))}")
        out.append(f"FLIP_IRREGULAR_CV: now {scoring.FLIP_IRREGULAR_CV}, normal p90 "
                   f"{pct(feats('interval_cv'), 90):.2f} -> suggest "
                   f"{max(scoring.FLIP_IRREGULAR_CV, round(1.3 * pct(feats('interval_cv'), 90), 2))}")
        rot = pct(feats("rotation_deg"), 10)
        if not np.isnan(rot):
            out.append(f"SMALL_AMPLITUDE_DEG (fusion.py): now {SMALL_AMPLITUDE_DEG}, normal p10 "
                       f"{rot:.0f} -> suggest {min(SMALL_AMPLITUDE_DEG, round(0.8 * rot))}")
    return out


def report(rows: list[dict]) -> str:
    lines = ["NeuroCheck calibration report (healthy volunteers + acted runs; NOT clinical data)", ""]
    if not rows:
        return "\n".join(lines + ["no recordings found - run app.py --record --label NAME"])
    sims = sum(r["mode"] != "LIVE" for r in rows)
    if sims:
        lines.append(f"WARNING: {sims} recording(s) are SIM - calibrate on LIVE recordings only")
    changed = [r for r in rows if r["score"] != r["app_score"]]
    lines.append(f"{len(rows)} recordings; {len(changed)} would score differently with the current "
                 "code than when recorded")
    for test in ("tremor", "tapping", "flipping"):
        rs = [r for r in rows if r["test"] == test]
        if not rs:
            continue
        normal = [r for r in rs if r["normal"]]
        acted = [r for r in rs if not r["normal"]]
        fa = [r for r in normal if (r["score"] or 0) > 0]
        miss = [r for r in acted if r["score"] == 0]
        lines += ["", f"== {test.upper()}  ({len(normal)} normal, {len(acted)} acted)",
                  f"  false alarms (normal scoring > 0): {len(fa)}"
                  + (f"  -> {', '.join(r['file'] for r in fa)}" if fa else ""),
                  f"  misses (acted scoring 0): {len(miss)}"
                  + (f"  -> {', '.join(r['file'] for r in miss)}" if miss else "")]
        keys = sorted({k for r in rs for k in r["features"]})
        lines.append(f"  {'feature':16s} {'normal p10/med/p90':>26s}   {'acted p10/med/p90':>26s}")
        for k in keys:
            def trio(group):
                v = [r["features"].get(k) for r in group]
                return f"{pct(v, 10):7.2f} {pct(v, 50):7.2f} {pct(v, 90):7.2f}" if group else "      -"
            lines.append(f"  {k:16s} {trio(normal):>26s}   {trio(acted):>26s}")
        lines.append("  suggestions:")
        lines += [f"    {s}" for s in suggestions(test, normal)]
    return "\n".join(lines)


def main() -> int:
    directory = sys.argv[1] if len(sys.argv) > 1 else "recordings"
    text = report(load_all(directory))
    print(text)
    if os.path.isdir(directory):
        with open(os.path.join(directory, "calibration_report.txt"), "w", encoding="utf-8") as f:
            f.write(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
