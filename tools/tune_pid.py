"""Tune the rhythm-coach PID gains against SimulatedPatient and plot the result.

All numbers here come from a SIMULATED patient - this is controller engineering, not
patient data. Criteria, judged against the patient's TRUE max at each moment
(it declines over time in the fatigue scenario):
  converge   tempo within 10% of true max within 20 s of the cued phase, and stays there
  overshoot  peak tempo < 115% of true max
  no oscillation: tempo std over the last 10 s < 5% of true max
  5 s stop:  converged before the stop, no windup after it (post-stop peak < 115%),
             back within 10% by the end
Start: WARM_START x measured uncued rate (feedforward); slow/fast self-pacer scenarios
make that start wrong on purpose. A cold start (1.5 beats/s) is reported as info.
Usage: python tools/tune_pid.py   (writes docs/pid_tuning.png, prints the table)
"""
from __future__ import annotations

import itertools
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import rhythm_coach  # noqa: E402
from rhythm_coach import UNCUED_S, SimulatedPatient, simulate  # noqa: E402

TRUE_MAX = 3.0
STOP = (25.0, 5.0)              # in cued-phase seconds
SCENARIOS = {
    "nominal": dict(),
    "20% missed beats": dict(miss_p=0.2),
    "5 s stop at 25 s": dict(stop=(UNCUED_S + STOP[0], STOP[1])),
    "noisy timing (60 ms)": dict(noise_s=0.06),
    "slow self-pacer (uncued 60%)": dict(uncued_ratio=0.6),
    "fast self-pacer (uncued 95%)": dict(uncued_ratio=0.95),
    "fatigue 15% + 20% missed": dict(fatigue=0.15, miss_p=0.2),
}
SEEDS = (0, 1, 2)


def run(gains, seed: int, kw: dict, start_rate=None):
    p = SimulatedPatient(max_rate=TRUE_MAX, seed=seed, **kw)
    coach, _ = simulate(p, gains, start_rate=start_rate)
    return coach, p


def _converged_at(tc, within, end: int):
    return next((float(tc[i]) for i in range(end) if within[i:end].all()), None)


def metrics(coach, patient) -> dict:
    t = np.array([r.t for r in coach.records])
    rate = np.array([r.rate for r in coach.records])
    true = np.array([patient.current_max(x) for x in t])
    rel = rate / true
    tc = t - UNCUED_S
    within = np.abs(rel - 1) <= 0.10
    late = tc >= tc[-1] - 10
    m = {"converge_s": _converged_at(tc, within, len(rel)), "overshoot": float(rel.max() - 1),
         "late_std": float(rel[late].std()), "final": float(np.median(rate[late])),
         "final_true": float(true[-1])}
    if patient.stop:
        s0 = patient.stop[0] - UNCUED_S
        n_before = int((tc < s0).sum())
        after = tc >= s0
        m.update(converge_s=_converged_at(tc, within, n_before),
                 overshoot=float(rel[after].max() - 1) if after.any() else 0.0,
                 recovered=bool(within[-3:].all()),
                 late_std=float(rel[tc < s0][-20:].std()))     # steadiness before the stop
    return m


def passes(m: dict) -> bool:
    ok = (m["converge_s"] is not None and m["converge_s"] <= 20.0 and m["overshoot"] < 0.15
          and m["late_std"] < 0.05)
    return ok and m.get("recovered", True)


def evaluate(gains) -> tuple[bool, float, dict]:
    worst, allok, results = 0.0, True, {}
    for name, kw in SCENARIOS.items():
        ms = [metrics(*run(gains, s, kw)) for s in SEEDS]
        ok = all(passes(m) for m in ms)
        allok &= ok
        conv = [m["converge_s"] if m["converge_s"] is not None else 99 for m in ms]
        worst += max(conv) + 200 * max(m["late_std"] for m in ms) \
            + 100 * max(0.0, max(m["overshoot"] for m in ms))
        results[name] = (ok, ms)
    return allok, worst, results


def main() -> int:
    grid = list(itertools.product((0.01, 0.02, 0.03, 0.05), (0.0, 0.005, 0.01, 0.02),
                                  (0.0, 0.5, 1.0, 2.0), (0.02, 0.03, 0.04)))
    scored = []
    for kp, ki, kd, probe in grid:
        rhythm_coach.PROBE_STEP = probe
        scored.append((evaluate((kp, ki, kd)), (kp, ki, kd), probe))
    passing = [(w, g, pr, r) for (ok, w, r), g, pr in scored if ok]
    pool = passing or [(w, g, pr, r) for (ok, w, r), g, pr in scored]
    w, best, probe, results = min(pool, key=lambda x: x[0])
    rhythm_coach.PROBE_STEP = probe
    print(f"searched {len(grid)} settings; {len(passing)} pass all scenarios")
    print(f"best: kp={best[0]} ki={best[1]} kd={best[2]} probe_step={probe}  (cost {w:.1f})")
    for name, (ok, ms) in results.items():
        conv = ", ".join("-" if m["converge_s"] is None else f"{m['converge_s']:.1f}s" for m in ms)
        print(f"  {name:30s} {'PASS' if ok else 'FAIL'}  converge {conv}  "
              f"overshoot {max(m['overshoot'] for m in ms):+.0%}  "
              f"late std {max(m['late_std'] for m in ms):.1%}  "
              f"final {np.mean([m['final'] for m in ms]):.2f}/s "
              f"(true {np.mean([m['final_true'] for m in ms]):.2f})")
    cold = [metrics(*run(best, s, {}, start_rate=1.5)) for s in SEEDS]
    print(f"  {'cold start 1.5/s (info only)':30s}      converge "
          + ", ".join("-" if m["converge_s"] is None else f"{m['converge_s']:.1f}s" for m in cold)
          + f"  overshoot {max(m['overshoot'] for m in cold):+.0%}")
    plot(best)
    return 0 if passing else 1


def plot(gains) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    os.makedirs("docs", exist_ok=True)
    fig, axes = plt.subplots(len(SCENARIOS), 1, figsize=(10, 2.1 * len(SCENARIOS)), sharex=True)
    for ax, (name, kw) in zip(axes, SCENARIOS.items()):
        coach, p = run(gains, 0, kw)
        t = np.array([r.t for r in coach.records])
        ax.plot(t - UNCUED_S, [r.rate for r in coach.records], lw=2, label="tempo (beats/s)")
        true = np.array([p.current_max(x) for x in t])
        ax.plot(t - UNCUED_S, true, "k--", lw=1, label="patient true max")
        ax.fill_between(t - UNCUED_S, 0.9 * true, 1.1 * true, color="g", alpha=0.1, label="+/-10%")
        ax2 = ax.twinx()
        ax2.plot(t - UNCUED_S, [r.on_time_rate if r.on_time_rate is not None else np.nan
                                for r in coach.records], color="tab:orange", lw=1, alpha=0.8)
        ax2.axhline(0.85, color="tab:orange", ls=":", lw=1)
        ax2.set_ylim(0, 1.05)
        ax2.set_ylabel("on-time", color="tab:orange", fontsize=8)
        ax.set_title(f"{name}   (kp={gains[0]}, ki={gains[1]}, kd={gains[2]}, "
                     f"probe={rhythm_coach.PROBE_STEP})", fontsize=9)
        ax.set_ylabel("beats/s")
        ax.set_ylim(1.0, 4.0)
        if ax is axes[0]:
            ax.legend(loc="lower right", fontsize=8)
    axes[-1].set_xlabel("time in cued phase (s) - SIMULATED PATIENT, not patient data")
    fig.suptitle("Rhythm coach PID tuning (simulation)")
    fig.tight_layout()
    fig.savefig("docs/pid_tuning.png", dpi=110)
    print("wrote docs/pid_tuning.png")


if __name__ == "__main__":
    sys.exit(main())
