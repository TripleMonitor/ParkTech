"""Headless full-session self-test: MockDevice + FakeHand on a fake clock, no window.

Right hand: 1.5 cm 5 Hz tremor, slow shrinking taps, slow decrementing flips with a pause.
Left hand: no tremor, normal taps, normal flips.
Usage:  python selftest.py [--screens DIR]      exit code 0 = all PASS
"""
from __future__ import annotations

import argparse
import os
import sys
import tempfile

import cv2

from app import KEY_SPACE, TESTS, App, led_for
from boot import BootChecks
from device import IMPAIRED_FLIPS, NORMAL_FLIPS, MockDevice
from history import load_sessions, trend_image
from tapping_tracker import IMPAIRED, NORMAL, FakeHand

FPS = 30.0


class FakeClock:
    def __init__(self) -> None:
        self.t = 1000.0

    def __call__(self) -> float:
        return self.t


def run_session(history_path: str, screens_dir: str | None = None, seconds: float = 10.0):
    clock = FakeClock()
    dev = MockDevice(seed=7, clock=clock, test_seconds=seconds,
                     profiles={"Right": IMPAIRED_FLIPS, "Left": NORMAL_FLIPS})
    hands = FakeHand({"Right": IMPAIRED, "Left": NORMAL}, seed=7, test_seconds=seconds,
                     tremor_cm={"Right": 1.5, "Left": 0.0})
    boot = BootChecks(dev, hands, pace_s=0)
    boot.run()                                        # synchronous in the self-test
    app = App(dev, hands, seconds=seconds, history_path=history_path, clock=clock, boot=boot)
    shots: dict[str, object] = {}

    def tick(key: int = -1):
        img = app.tick(key, clock())
        clock.t += 1 / FPS
        return img

    def snap(name: str, img) -> None:
        if name not in shots:
            shots[name] = img

    snap("00_boot", tick())
    tick(KEY_SPACE)                                   # boot -> welcome
    snap("01_welcome", tick())
    for k in (ord("2"), ord("."), ord("5")):          # hours since levodopa = 2.5
        tick(k)
    tick(KEY_SPACE)                                   # welcome -> ready (test 1)
    for i, (kind, hand) in enumerate(TESTS):
        if app.state != "ready":
            raise RuntimeError(f"expected ready before test {i + 1}, got {app.state}")
        snap(f"02_ready_{kind}_{hand}", tick())
        tick(KEY_SPACE)                               # ready -> countdown
        while app.state != "done":
            img = tick()
            if app.state == "countdown":
                snap(f"03_countdown_{kind}", img)
            if app.state == "recording" and clock.t - app.t_state > seconds * 0.6:
                snap(f"04_recording_{kind}_{hand}", img)
            if clock.t > 10_000:
                raise RuntimeError(f"stuck in state {app.state}")
        snap(f"05_done_{kind}_{hand}", tick())
        if i + 1 < len(TESTS):
            tick(KEY_SPACE)                           # done -> ready (next test)
    results_img = tick(KEY_SPACE)                     # last done -> results
    snap("06_results", results_img)
    # rhythm coach (C) from the results screen, right hand, full 45 s
    tick(ord("c"))
    snap("08_coach_ready", tick())
    tick(KEY_SPACE)
    while app.coach.phase != "done":
        img = tick()
        if app.coach.phase == "cued" and clock.t - app.coach.t_phase > 25:
            snap("09_coach_cued", img)
        if clock.t > 20_000:
            raise RuntimeError(f"coach stuck in {app.coach.phase}")
    snap("10_coach_done", tick())
    tick(KEY_SPACE)                                   # back to results
    results_img = tick()
    tick(ord("p"))                                    # doctor PDF
    tick(ord("f"))                                    # FHIR JSON
    snap("11_dashboard", tick(KEY_SPACE))             # results -> dashboard
    tick()
    trend_img = tick(KEY_SPACE)                       # results -> trend
    snap("07_trend", trend_img)
    if screens_dir:
        os.makedirs(screens_dir, exist_ok=True)
        for name, img in shots.items():
            cv2.imwrite(os.path.join(screens_dir, f"selftest_{name}.png"), img)
    return app, dev, results_img, trend_img


def _exports_ok(app: App, history_path: str) -> bool:
    import json

    from fhir import validate_bundle
    d = os.path.join(os.path.dirname(history_path), "exports")
    pdf = os.path.join(d, f"{app.session_id}.pdf")
    fj = os.path.join(d, f"{app.session_id}_fhir.json")
    if not (os.path.exists(pdf) and os.path.exists(fj)):
        return False
    with open(pdf, "rb") as f:
        if not f.read(4) == b"%PDF":
            return False
    with open(fj, encoding="utf-8") as f:
        return validate_bundle(json.load(f)) == []


def checks(app: App, dev: MockDevice, history_path: str, trend) -> list[tuple[str, bool, str]]:
    res = app.results

    def score(kind: str, hand: str):
        return res[(kind, hand)].result.score

    rt, lt = score("tremor", "Right"), score("tremor", "Left")
    rp, lp = score("tapping", "Right"), score("tapping", "Left")
    rf, lf = score("flipping", "Right"), score("flipping", "Left")
    rows = load_sessions(history_path)
    boot = app.boot
    out = [
        ("Boot self-check: 9 lines, sim lines say SIM",
         boot is not None and boot.done and len(boot.lines) == 9 and boot.failures == 0
         and all(l.status == "SIM" for l in boot.lines),
         "" if boot is None else ", ".join(l.status for l in boot.lines)),
        ("All 6 tests completed", len(res) == 6, f"{len(res)} results"),
        ("Right tremor > left tremor", rt is not None and lt is not None and rt > lt,
         f"R {rt} vs L {lt}"),
        ("Right tapping > left tapping", rp is not None and lp is not None and rp > lp,
         f"R {rp} vs L {lp}"),
        ("Right flipping > left flipping", rf is not None and lf is not None and rf > lf,
         f"R {rf} vs L {lf}"),
        ("Right tremor peak 4-6 Hz, ~1.5 cm",
         4 <= res[("tremor", "Right")].features.peak_hz <= 6
         and abs(res[("tremor", "Right")].features.displacement_cm - 1.5) < 0.3,
         f"{res[('tremor', 'Right')].features.peak_hz:.2f} Hz, "
         f"{res[('tremor', 'Right')].features.displacement_cm:.2f} cm"),
        ("Every score has a reason", all(r.result.reasons for r in res.values()), ""),
        ("Asymmetry flagged (all 3 tests)",
         all(any(f.startswith(t) for f in app.flags) for t in ("Tremor", "Tapping", "Flipping")),
         f"{len(app.flags)} flag(s)"),
        ("sessions.csv written", len(rows) >= 1 and rows[0]["tremor_R"] == str(rt)
         and rows[0]["flip_R"] == str(rf) and rows[0]["session_id"] == app.session_id,
         f"{len(rows)} row(s)"),
        ("Fusion locked on both hands (camera agrees with switch)",
         all(res[("flipping", h)].fusion is not None and res[("flipping", h)].fusion.locked
             for h in ("Right", "Left")),
         " / ".join(f"{h[0]} sw {res[('flipping', h)].fusion.switch_half_flips} cam "
                    f"{res[('flipping', h)].fusion.camera_half_flips}" for h in ("Right", "Left"))),
        ("Right flips smaller (camera rotation)",
         res[("flipping", "Right")].fusion.median_amplitude_deg
         < res[("flipping", "Left")].fusion.median_amplitude_deg,
         f"R {res[('flipping', 'Right')].fusion.median_amplitude_deg:.0f} deg vs "
         f"L {res[('flipping', 'Left')].fusion.median_amplitude_deg:.0f} deg"),
        ("Rhythm coach finished, rate near SIM patient max (2.6/s)",
         "Right" in app.coach_results
         and abs(app.coach_results["Right"].max_sustainable_rate - 2.6) / 2.6 < 0.2,
         "" if "Right" not in app.coach_results else
         f"{app.coach_results['Right'].max_sustainable_rate:.2f} beats/s, "
         f"{app.coach_results['Right'].beats} beats"),
        ("NeuroScore + dose saved", rows[0]["neuroscore"] != "" and float(rows[0]["dose_hours"] or 0) == 2.5,
         f"NeuroScore {rows[0]['neuroscore']}, dose {rows[0]['dose_hours']} h"),
        ("Exports: PDF + valid FHIR bundle", _exports_ok(app, history_path),
         ", ".join(sorted(os.listdir(os.path.join(os.path.dirname(history_path), "exports"))))
         if os.path.isdir(os.path.join(os.path.dirname(history_path), "exports")) else "none"),
        ("Signal quality measured on camera tests",
         all(res[(k, h)].quality is not None and res[(k, h)].quality.frames > 250
             for k in ("tremor", "tapping") for h in ("Right", "Left")),
         f"{res[('tremor', 'Right')].quality.frames} frames"),
        ("Beeps: 3 countdown + 2 done per test, +3 coach countdown",
         dev.beeps == 6 * (3 + 2) + 3, f"{dev.beeps} beeps"),
        ("LED colour from results", dev.last_led == led_for([rt, lt, rp, lp, rf, lf]),
         f"LED {dev.last_led}"),
        ("LCD shows scores", dev.last_lcd[0].startswith("R tr") and dev.last_lcd[1].startswith("L tr")
         and all(len(x) <= 16 for x in dev.last_lcd), " / ".join(dev.last_lcd)),
        ("Trend chart rendered", trend is not None and trend.shape == (720, 1280, 3)
         and trend.std() > 10, f"{None if trend is None else trend.shape}"),
        ("Ended on trend screen", app.state == "trend", app.state),
    ]
    return out


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--screens", help="save screenshots of each screen into this folder")
    args = p.parse_args()
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "sessions.csv")
        try:
            app, dev, _, trend = run_session(path, args.screens)
            rows = checks(app, dev, path, trend)
        except Exception as exc:   # report, don't hide
            import traceback
            traceback.print_exc()
            rows = [("Session ran without exception", False, repr(exc))]
    width = max(len(r[0]) for r in rows)
    print(f"\n{'CHECK'.ljust(width)}  RESULT  DETAIL")
    for name, ok, detail in rows:
        print(f"{name.ljust(width)}  {'PASS' if ok else 'FAIL'}    {detail}")
    passed = sum(ok for _, ok, _ in rows)
    print(f"\n{passed}/{len(rows)} passed -> {'PASS' if passed == len(rows) else 'FAIL'}")
    return 0 if passed == len(rows) else 1


if __name__ == "__main__":
    sys.exit(main())
