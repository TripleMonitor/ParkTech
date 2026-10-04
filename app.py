"""NeuroCheck - Parkinson's motor check station (hackathon demo).

TRACKING / DECISION-SUPPORT TOOL, NOT A DIAGNOSIS. All thresholds are demo thresholds.
Tests per hand: rest tremor (webcam), finger tapping (webcam), hand flipping (SW-520D tilt switch).

  python app.py                 real Arduino (auto-detect port) + webcam
  python app.py --sim           MockDevice + FakeHand: no hardware at all
  python app.py --sim-device    MockDevice + real webcam
  python app.py --seed-history  add 14 days x 2 sessions of DEMO DATA first (trend demo)

Keys: SPACE next   R restart   Q/ESC quit   H trend
      sim only: T toggle fake tremor   F toggle fake flip sensor (stuck)
"""
from __future__ import annotations

import argparse
import logging
import os
import time
from datetime import datetime
import traceback
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np

import history
import hud
import ui
from boot import BootChecks
from coach_session import CoachSession
from dashboard import neuroscore, radar_image
from quality import SignalQuality, signal_quality
from flipping_analysis import (FlippingFeatures, analyze_closures, analyze_flipping, debounce,
                               live_colour, live_rate)
from fusion import FusionResult, current_angle, fuse, palm_normal
from scoring import (ScoreResult, explain_flipping, explain_tapping, explain_tremor,
                     flipping_asymmetry, score_flipping, score_tapping, score_tremor,
                     tapping_asymmetry, tremor_asymmetry)
from tapping_analysis import TappingFeatures, analyze_tapping, detect_taps
from tapping_tracker import HandFrame, draw_distance_graph, draw_hand
from tremor_analysis import (HAND_LENGTH_CM, INDEX_TIP, MIDDLE_MCP, WRIST, TremorFeatures,
                             analyze_tremor, effective_displacement_cm)

log = logging.getLogger("neurocheck")

COUNTDOWN_S = 3
WINDOW = "NeuroCheck"
TESTS = (("tremor", "Right"), ("tremor", "Left"), ("tapping", "Right"), ("tapping", "Left"),
         ("flipping", "Right"), ("flipping", "Left"))
INSTRUCTIONS = {
    "tremor": ["Hold your {hand} hand up, palm facing", "the camera, fingers spread.",
               "Relax and keep it as still as you can."],
    "tapping": ["Hold your {hand} hand up to the camera.", "Tap index finger on thumb",
                "as FAST and as BIG as you can."],
    "flipping": ["Tilt sensor taped to the back of your", "{hand} hand. Hold it PALM-DOWN now.",
                 "Then flip palm-up / palm-down", "as FAST and as FULLY as you can."],
}
TITLES = {"tremor": "Rest tremor (3.17)", "tapping": "Finger tapping (3.4)",
          "flipping": "Hand flipping (3.6)"}
LCD_SHORT = {"tremor": "tr", "tapping": "tap", "flipping": "fl"}
CAMERA_NOTE = "Camera tremor: may miss very small tremors (< ~0.5 cm)"
SENSOR_STUCK = "Sensor not flipping - check it's taped on and upright"
DISCLAIMER = "Demo thresholds - tracking aid, NOT a diagnosis"
KEY_SPACE, KEY_ENTER, KEY_ESC = 32, 13, 27
LIVE_EVERY_S = 0.25
STUCK_AFTER_S = 3.0


@dataclass(frozen=True)
class TestResult:
    kind: str
    hand: str
    features: object                    # TremorFeatures | TappingFeatures | FlippingFeatures
    result: ScoreResult
    image: Optional[np.ndarray] = None  # spectrum for tremor
    quality: Optional[SignalQuality] = None   # camera tests
    fusion: Optional[FusionResult] = None     # hand flipping

    @property
    def low_confidence(self) -> bool:
        if self.result.score is None:
            return False
        bad_q = self.quality is not None and self.quality.low
        bad_f = self.fusion is not None and self.fusion.camera_ok and not self.fusion.locked
        return bad_q or bad_f


def led_for(scores: list[Optional[int]]) -> str:
    """Results LED: worst score 0-1 green, 2 yellow, 3-4 red; nothing scored -> off."""
    real = [s for s in scores if s is not None]
    if not real:
        return "OFF"
    worst = max(real)
    return "G" if worst <= 1 else "Y" if worst == 2 else "R"


def fingertip_cm(lm: Optional[np.ndarray]) -> Optional[np.ndarray]:
    """Live-view only: fingertip position in cm using this frame's hand length."""
    if lm is None:
        return None
    hand_px = float(np.linalg.norm(lm[WRIST] - lm[MIDDLE_MCP]))
    return None if hand_px < 1e-6 else lm[INDEX_TIP] * HAND_LENGTH_CM / hand_px


class App:
    def __init__(self, device, hands, seconds: float = 10.0,
                 history_path: str = history.DEFAULT_PATH, clock=time.perf_counter,
                 boot: Optional[BootChecks] = None, raw_dir: Optional[str] = None,
                 label: str = ""):
        self.device, self.hands, self.seconds = device, hands, seconds
        self.history_path = history_path
        self.raw_dir, self.label = raw_dir, label     # --record: save raw data per test
        self.raw_msg = ""
        self.clock = clock          # perf_counter: monotonic() is 15.6 ms on Windows
        self.running = True
        self.boot = boot
        if hasattr(hands, "flip_state"):          # sim: fake palm follows the fake switch
            hands.flip_state = lambda: (self.device.state,
                                        self.calib if self.calib is not None else
                                        (self.device.state if self.state == "ready" else None))
        self.restart(0.0)
        if boot is not None:
            self.state = "boot"

    @property
    def closures_only(self) -> bool:
        """--og: the original sketch reports only switch closures (1 closure = 1 full flip)."""
        return bool(getattr(self.device, "closures_only", False))

    @property
    def switch_half_flips(self) -> int:
        return len(self.flip_times) * (2 if self.closures_only else 1)

    @property
    def sim_hand(self) -> bool:
        return bool(getattr(self.hands, "sim", False))

    @property
    def sim_dev(self) -> bool:
        return bool(getattr(self.device, "sim", False))

    # ---------------------------------------------------------------- state changes
    def restart(self, now: float) -> None:
        self.device.stop()
        self.state, self.idx, self.t_state = "welcome", 0, now
        self.results: dict[tuple[str, str], TestResult] = {}
        self.flags: list[str] = []
        self._clear_live()
        self.trend_img: Optional[np.ndarray] = None
        self.save_msg = ""
        self.session_id = hud.session_id()
        if getattr(self, "coach", None) is not None:
            self.coach.abort()
        self.coach: Optional[CoachSession] = None
        self.coach_results: dict = {}
        self.coach_return = "welcome"
        self.dose_text = ""
        self.dose_unknown = False
        self.radar_img: Optional[np.ndarray] = None
        self.export_msg = ""
        self.started_at = datetime.now().astimezone().isoformat(timespec="seconds")
        self._poll_at = 0.0
        self.last_beep = 0
        self.device.led("OFF")
        self.device.lcd("NeuroCheck", "Press SPACE")

    def _clear_live(self) -> None:
        self.hf = HandFrame(None, None, None)
        self.trem_t: list[float] = []
        self.trem_lm: list[Optional[np.ndarray]] = []
        self.trem_xy: list[Optional[np.ndarray]] = []
        self.tap_t: list[float] = []
        self.tap_d: list[Optional[float]] = []
        self.live_taps, self._live_at = 0, 0.0
        self.events: list = []
        self.calib: Optional[int] = None
        self.flip_times: list[float] = []
        self.rec_start = 0.0
        self.device_lost = False
        self.q_t: list[float] = []
        self.q_det: list[bool] = []
        self.q_score: list[Optional[float]] = []
        self.fails0 = 0
        self.quality: Optional[SignalQuality] = None
        self.fl_t: list[float] = []
        self.fl_world: list[Optional[np.ndarray]] = []
        self.flip_ref: Optional[np.ndarray] = None
        self._ref_candidate: Optional[np.ndarray] = None
        self.fusion: Optional[FusionResult] = None
        self.flip_angle: Optional[float] = None
        self._fuse_at = 0.0
        self._live_led: Optional[str] = None

    @property
    def test(self) -> tuple[str, str]:
        return TESTS[self.idx]

    def _lcd_test(self, line2: str) -> None:
        kind, hand = self.test
        self.device.lcd(f"{hand} {kind}", line2)

    def _goto(self, state: str, now: float) -> None:
        self.state, self.t_state = state, now

    def _enter_ready(self, now: float) -> None:
        """Ready screen for self.test: fresh live views (no data from the previous test)."""
        self._goto("ready", now)
        self._clear_live()
        self.device.led("OFF")
        self._lcd_test("SPACE to start")
        if self.test[0] == "flipping":
            self._lcd_test("Palm DOWN, SPACE")
            self.device.request_state()       # calibration: current state = palm-down

    def _begin_countdown(self, now: float) -> None:
        kind, hand = self.test
        self._goto("countdown", now)
        self.last_beep = 0
        if kind == "flipping":
            self.calib = None                  # set from the STATE reply sent with START
            if hasattr(self.device, "sim_hand"):
                self.device.sim_hand = hand
        self.hands.reset(now + COUNTDOWN_S)

    def _begin_recording(self, now: float) -> None:
        self._goto("recording", now)
        self.rec_start = self.clock()
        self._lcd_test("Recording...")
        tel = getattr(self.hands, "telemetry", None)
        self.fails0 = tel.failed_reads if tel else 0
        if self.test[0] == "tremor":
            self.trem_t, self.trem_xy, self.trem_lm = [], [], []    # drop preview frames
        if self.test[0] == "flipping":
            self.flip_ref = self._ref_candidate      # palm-down normal from the countdown
            self.device.start()

    def _finish_test(self, now: float) -> None:
        kind, hand = self.test
        self.device.beep(2)            # done beep first: rendering below can take ~0.2 s
        if kind == "tremor":
            self.results[self.test] = self._tremor_result(hand)
        elif kind == "tapping":
            self.results[self.test] = self._tapping_result(hand)
        else:
            self.device.stop()
            self.events.extend(self.device.drain())
            self._update_live_flips(now)
            self._update_fusion(force=True)
            self.results[self.test] = self._flipping_result(hand)
        score = self.results[self.test].result.score
        self._lcd_test(f"Score {'-' if score is None else score}")
        if self.raw_dir:
            self._save_raw(kind, hand)
        self._goto("done", now)

    def _save_raw(self, kind: str, hand: str) -> None:
        """--record: write this test's raw signals so tools/calibrate.py can re-score them."""
        from recorder import build_record, save_record
        try:
            data = {"q_t": self.q_t, "q_det": self.q_det, "q_score": self.q_score}
            if kind == "tremor":
                data.update(t=self.trem_t, landmarks=self.trem_lm)
            elif kind == "tapping":
                data.update(t=self.tap_t, d=self.tap_d)
            else:
                data.update(events=[(e.t, e.state) for e in self.events], rec_start=self.rec_start,
                            calib=self.calib, cam_t=self.fl_t, world=self.fl_world,
                            ref_normal=self.flip_ref)
            meta = {"session_id": self.session_id, "label": self.label, "hand": hand,
                    "seconds": self.seconds, "timestamp": self.started_at,
                    "mode": "LIVE" if not (self.sim_hand or self.sim_dev) else "SIM",
                    "dose_hours": self.dose_hours, "closures_only": self.closures_only}
            r = self.results[(kind, hand)].result
            path = save_record(self.raw_dir, build_record(meta, kind, data, r.score, r.reasons))
            self.raw_msg = f"raw data saved: {os.path.basename(path)}"
        except Exception as exc:                  # recording must never break a test
            log.exception("raw save failed")
            self.raw_msg = f"raw data NOT saved: {exc}"

    def _final_quality(self) -> SignalQuality:
        tel = getattr(self.hands, "telemetry", None)
        fails = (tel.failed_reads if tel else 0) - self.fails0
        return signal_quality(self.q_t, self.q_det, self.q_score, failed_reads=fails)

    @staticmethod
    def _mark_quality(r: ScoreResult, q: SignalQuality) -> ScoreResult:
        if r.score is None or not q.low:
            return r
        return ScoreResult(r.score, r.reasons + ["LOW CONFIDENCE: " + "; ".join(q.problems)])

    def _tremor_result(self, hand: str) -> TestResult:
        f = analyze_tremor(self.trem_t, self.trem_lm)
        q = self._final_quality()
        r = self._mark_quality(score_tremor(f, expected_s=self.seconds), q)
        if r.score is None and not self.hands.ok:
            r = ScoreResult(None, r.reasons + [self.hands.status])
        img = None
        try:
            img = ui.spectrum_image(f.freqs, f.power, f.peak_hz, f.displacement_cm,
                                    r.score, 720, 540, clear_peak=f.clear_peak)
        except Exception:   # a chart must never take the demo down
            log.exception("spectrum render failed")
        return TestResult("tremor", hand, f, r, img, q)

    def _tapping_result(self, hand: str) -> TestResult:
        f = analyze_tapping(self.tap_t, self.tap_d) if self.tap_t else \
            TappingFeatures(0, 0.0, 0.0, 0.0, 0.0, 0, 0.0, 0.0)
        q = self._final_quality()
        r = self._mark_quality(score_tapping(f), q)
        if r.score is None and not self.hands.ok:
            r = ScoreResult(None, r.reasons + [self.hands.status])
        return TestResult("tapping", hand, f, r, None, q)

    def _flipping_features(self) -> FlippingFeatures:
        if self.closures_only:
            return analyze_closures([e.t for e in self.events], self.rec_start, self.seconds)
        return analyze_flipping([e.t for e in self.events], [e.state for e in self.events],
                                self.rec_start, self.seconds, initial_state=self.calib)

    def _flipping_result(self, hand: str) -> TestResult:
        f = self._flipping_features()
        if self.device_lost or not self.device.connected:
            r = ScoreResult(None, ["Arduino disconnected during the test - repeat",
                                   self.device.status])
        else:
            fu = self.fusion
            r = score_flipping(f, fu)
            if f.half_flips == 0 and getattr(self.device, "warning", ""):
                r = ScoreResult(r.score, r.reasons + [self.device.warning])
            if fu is not None and fu.camera_ok and not fu.locked and r.score is not None:
                r = ScoreResult(r.score, r.reasons + [
                    f"LOW CONFIDENCE: fusion mismatch (switch {fu.switch_half_flips} vs camera "
                    f"{fu.camera_half_flips} half-flips)"])
            elif fu is not None and fu.locked:
                r = ScoreResult(r.score, r.reasons + [
                    f"Fusion locked: switch {fu.switch_half_flips} / camera "
                    f"{fu.camera_half_flips} half-flips agree"])
        return TestResult("flipping", hand, f, r, fusion=self.fusion)

    def _enter_results(self, now: float) -> None:
        self._goto("results", now)
        self.flags = self._asymmetry()
        self.device.led(led_for([r.result.score for r in self.results.values()]))
        self.device.lcd(self._lcd_scores("Right"), self._lcd_scores("Left"))
        try:
            history.save_session(self._session_row(), self.history_path)
            self.save_msg = f"Saved to {self.history_path}"
        except OSError as exc:
            log.error("Could not save session: %s", exc)
            self.save_msg = f"NOT saved: {exc.strerror} (is {self.history_path} open in Excel?)"
        except Exception as exc:              # corrupt CSV etc.
            log.exception("Could not save session")
            self.save_msg = f"NOT saved: {type(exc).__name__} - check {self.history_path}"

    @property
    def dose_hours(self) -> Optional[float]:
        if self.dose_unknown or not self.dose_text:
            return None
        try:
            return float(self.dose_text)
        except ValueError:
            return None

    def _dose_key(self, ch: str, key: int) -> None:
        if key == 8:
            self.dose_text = self.dose_text[:-1]
        elif ch == "n":
            self.dose_unknown, self.dose_text = True, ""
            return
        elif len(self.dose_text) < 4 and not (ch == "." and "." in self.dose_text):
            self.dose_text += ch
        self.dose_unknown = False

    def _scores(self) -> dict:
        return {(k, h): (self.results[(k, h)].result.score if (k, h) in self.results else None)
                for k, h in TESTS}

    def _neuroscore(self):
        return neuroscore(self._scores())

    def session_snapshot(self) -> dict:
        """Plain, JSON-able summary of this session (used by the PDF and FHIR exports)."""
        tests = []
        for (kind, hand), tr in self.results.items():
            f = tr.features
            feats = {}
            if isinstance(f, TremorFeatures) and f.valid and f.clear_peak \
                    and tr.result.score is not None:
                feats = {"peak_hz": f.peak_hz, "displacement_cm": f.displacement_cm}
            elif isinstance(f, TappingFeatures) and tr.result.score is not None:
                feats = {"taps_per_sec": f.taps_per_sec, "mean_amplitude": f.mean_amplitude,
                         "decrement": f.decrement}
            elif isinstance(f, FlippingFeatures) and tr.result.score is not None:
                feats = {"flips_per_sec": f.flips_per_sec, "interval_cv": f.interval_cv}
                if tr.fusion is not None and tr.fusion.camera_ok and tr.fusion.camera_half_flips:
                    feats["median_amplitude_deg"] = tr.fusion.median_amplitude_deg
            tests.append({"kind": kind, "hand": hand, "score": tr.result.score,
                          "reasons": list(tr.result.reasons), "features": feats})
        ns, formula = self._neuroscore()
        return {"session_id": self.session_id, "timestamp": self.started_at,
                "mode": "LIVE" if not (self.sim_hand or self.sim_dev) else "SIMULATION",
                "dose_hours": self.dose_hours, "neuroscore": ns, "neuroscore_formula": formula,
                "asymmetry": list(self.flags), "tests": tests,
                "coach_rates": {h: s.max_sustainable_rate for h, s in self.coach_results.items()}}

    def _export_dir(self) -> str:
        d = os.path.join(os.path.dirname(os.path.abspath(self.history_path)), "exports")
        os.makedirs(d, exist_ok=True)
        return d

    def _export(self, what: str) -> None:
        try:
            snap = self.session_snapshot()
            base = os.path.join(self._export_dir(), self.session_id)
            if what == "pdf":
                from report import save_pdf
                path = save_pdf(snap, history.load_sessions(self.history_path), base + ".pdf")
            else:
                from fhir import save_bundle
                path = save_bundle(snap, base + "_fhir.json")
            self.export_msg = f"Saved exports/{os.path.basename(path)}"
            log.info(self.export_msg)
        except Exception as exc:          # an export must never take the demo down
            log.exception("export failed")
            self.export_msg = f"Export failed: {exc}"

    def _enter_dashboard(self, now: float) -> None:
        self._goto("dashboard", now)
        try:
            self.radar_img = radar_image(self._scores(), {h: s.max_sustainable_rate for h, s
                                                          in self.coach_results.items()}, 600, 520)
        except Exception:
            log.exception("radar render failed")
            self.radar_img = None

    def _save_coach_row(self, hand: str, summary) -> None:
        try:
            history.save_session({"session_id": self.session_id,
                                  "mode": "LIVE" if not (self.sim_hand or self.sim_dev) else "SIM",
                                  f"coach_{hand[0]}_rate": summary.max_sustainable_rate},
                                 self.history_path)
        except Exception as exc:
            log.error("Could not save coach result: %s", exc)

    def _enter_trend(self, now: float) -> None:
        self.trend_return = self.state if self.state in ("welcome", "results", "dashboard") \
            else "welcome"
        self._goto("trend", now)
        try:
            self.trend_img = history.trend_image(history.load_sessions(self.history_path),
                                                 ui.W, hud.HINT_Y - 22)
        except Exception:
            log.exception("trend render failed")
            self.trend_img = None

    def _lcd_scores(self, hand: str) -> str:
        parts = []
        for kind in ("tremor", "tapping", "flipping"):
            r = self.results.get((kind, hand))
            s = "-" if r is None or r.result.score is None else str(r.result.score)
            parts.append(f"{LCD_SHORT[kind]}{s}")
        return f"{hand[0]} " + " ".join(parts)          # e.g. "R tr2 tap1 fl3"

    def _asymmetry(self) -> list[str]:
        flags: list[str] = []
        for kind, fn in (("tremor", tremor_asymmetry), ("tapping", tapping_asymmetry),
                         ("flipping", flipping_asymmetry)):
            r, l = self.results.get((kind, "Right")), self.results.get((kind, "Left"))
            if r and l:
                flags.extend(fn(r.features, l.features, r.result, l.result))
        return flags

    def _session_row(self) -> dict:
        row: dict = {"asymmetry": " ; ".join(self.flags), "session_id": self.session_id,
                     "mode": "LIVE" if not (self.sim_hand or self.sim_dev) else "SIM",
                     "dose_hours": self.dose_hours, "neuroscore": self._neuroscore()[0]}
        for hand, s in self.coach_results.items():
            row[f"coach_{hand[0]}_rate"] = s.max_sustainable_rate
        for (kind, hand), tr in self.results.items():
            side, f = hand[0], tr.features
            scored = tr.result.score is not None
            if isinstance(f, TremorFeatures):
                peak = scored and f.valid and f.clear_peak
                row.update({f"tremor_{side}": tr.result.score,
                            f"tremor_{side}_cm": f.displacement_cm if peak else None,
                            f"tremor_{side}_hz": f.peak_hz if peak else None})
            elif isinstance(f, TappingFeatures):
                row.update({f"tap_{side}": tr.result.score,
                            f"tap_{side}_rate": f.taps_per_sec if scored else None,
                            f"tap_{side}_amp": f.mean_amplitude if scored else None})
            elif isinstance(f, FlippingFeatures):
                ok = tr.result.score is not None
                fu = tr.fusion
                row.update({f"flip_{side}": tr.result.score,
                            f"flip_{side}_rate": f.flips_per_sec if ok else None,
                            f"flip_{side}_amp_deg": fu.median_amplitude_deg
                            if ok and fu is not None and fu.camera_ok and fu.camera_half_flips
                            else None})
        return row

    # ---------------------------------------------------------------- input
    def handle_key(self, key: int, now: float) -> None:
        if key is None or key < 0:
            return
        key &= 0xFF
        ch = chr(key).lower()
        go = key in (KEY_SPACE, KEY_ENTER)
        if key == KEY_ESC or ch == "q":
            self.running = False
        elif ch == "r":
            self.restart(now)
        elif ch == "t" and hasattr(self.hands, "tremor_enabled"):
            self.hands.tremor_enabled = not self.hands.tremor_enabled
        elif ch == "f" and hasattr(self.device, "flipping") and self.state not in ("results", "dashboard"):
            self.device.flipping = not self.device.flipping
        elif ch == "h" and self.state in ("welcome", "results"):
            self._enter_trend(now)
        elif self.state == "welcome" and (ch.isdigit() or ch == "." or key == 8 or ch == "n"):
            self._dose_key(ch, key)
        elif ch == "p" and self.state in ("results", "dashboard"):
            self._export("pdf")
        elif ch == "f" and self.state in ("results", "dashboard"):
            self._export("fhir")
        elif ch == "c" and self.state in ("welcome", "results"):
            self.coach_return = self.state
            self.coach = CoachSession(self.device, "Right", self.clock)
            self._goto("coach", now)
        elif ch == "l" and self.state == "coach" and self.coach.phase == "ready":
            other = "Left" if self.coach.hand == "Right" else "Right"
            self.coach = CoachSession(self.device, other, self.clock)
        elif self.state == "coach" and go:
            if self.coach.phase == "ready":
                self.coach.start()
            elif self.coach.phase == "done":
                if self.coach.summary is not None and self.coach.summary.valid:
                    self.coach_results[self.coach.hand] = self.coach.summary
                    if self.coach_return == "results":     # session row already saved
                        self._save_coach_row(self.coach.hand, self.coach.summary)
                self._goto(self.coach_return, now)
        elif not go:
            return
        elif self.state == "boot":
            if self.boot is None or self.boot.done:
                self._goto("welcome", now)
        elif self.state == "welcome":
            self.idx = 0
            self._enter_ready(now)
        elif self.state == "ready":
            self._begin_countdown(now)
        elif self.state == "done":
            if self.idx + 1 < len(TESTS):
                self.idx += 1
                self._enter_ready(now)            # time to position the hand / sensor
            else:
                self._enter_results(now)
        elif self.state == "results":
            self._enter_dashboard(now)
        elif self.state == "dashboard":
            self._enter_trend(now)
        elif self.state == "trend":
            self._goto(getattr(self, "trend_return", "welcome"), now)

    # ---------------------------------------------------------------- update
    def update(self, now: float) -> None:
        if self.state == "coach":
            self.coach.tick()
            return
        kind, hand = self.test
        elapsed = now - self.t_state
        if self.state == "countdown":
            n = int(elapsed) + 1
            if n <= COUNTDOWN_S and n > self.last_beep:
                self.last_beep = n
                self.device.beep(1)
                self._lcd_test(f"Starting in {COUNTDOWN_S - n + 1}")
            if elapsed >= COUNTDOWN_S:
                self._begin_recording(now)
                elapsed = 0.0

        active = self.state in ("ready", "countdown", "recording")
        recording = self.state == "recording"
        if active and kind in ("tremor", "tapping"):
            self.hf = self.hands.read(now, hand, detect=True, test=kind)
            t_frame = self.hf.t if self.hf.t is not None else self.clock()   # capture time
            if recording:
                self.q_t.append(t_frame)
                self.q_det.append(self.hf.landmarks is not None)
                self.q_score.append(self.hf.score)
                if len(self.q_t) % 5 == 0:
                    self.quality = signal_quality(self.q_t, self.q_det, self.q_score)
            if kind == "tremor":
                self.trem_xy.append(fingertip_cm(self.hf.landmarks))
                self.trem_t.append(t_frame)
                if recording:
                    self.trem_lm.append(self.hf.landmarks)
                else:                                # preview only: keep the last 3 s
                    self.trem_t, self.trem_xy = self.trem_t[-120:], self.trem_xy[-120:]
            elif recording:
                self.tap_t.append(t_frame)
                self.tap_d.append(self.hf.distance)
                self._update_live_taps(now)
        if kind == "flipping" and active:
            self.hf = self.hands.read(now, hand, detect=True, test="flipping")
            t_frame = self.hf.t if self.hf.t is not None else self.clock()
            if self.hf.world is not None and not recording:
                n = palm_normal(self.hf.world)
                if n is not None:
                    self._ref_candidate = n          # latest palm-down pose before START
            if recording:
                self.fl_t.append(t_frame)
                self.fl_world.append(self.hf.world)
                self.events.extend(self.device.drain())
                if self.calib is None and self.events:
                    self.calib = self.events[0].state  # first report after START = palm-down
                self.device_lost |= not self.device.connected
                self._update_live_flips(now)
                self._update_fusion()
                rate = live_rate(self.flip_times, self.clock() - self.rec_start)
                if rate is not None and self.closures_only:
                    rate *= 2                          # live_rate assumes half-flips
                colour = live_colour(rate)
                if colour is not None and colour != self._live_led:
                    self._live_led = colour          # LED follows the measured flip speed
                    self.device.led(colour)
            else:
                self.device.drain()                  # discard stray reports before START
                if self.clock() - self._poll_at >= 0.2:
                    self._poll_at = self.clock()
                    self.device.request_state()      # keep the live PALM indicator fresh

        if recording and elapsed >= self.seconds:
            self._finish_test(now)

    def _update_live_taps(self, now: float) -> None:
        if now - self._live_at < LIVE_EVERY_S:
            return
        self._live_at = now
        valid = [(t, d) for t, d in zip(self.tap_t, self.tap_d) if d is not None]
        if len(valid) >= 10:
            t, d = np.array(valid, dtype=float).T
            self.live_taps = len(detect_taps(t, d)[0])

    def _update_live_flips(self, now: float) -> None:
        if self.closures_only:                 # each closure = one full flip
            self.flip_times = list(analyze_closures([e.t for e in self.events], self.rec_start,
                                                    1e9).flip_times)
            return
        changes = debounce([e.t for e in self.events], [e.state for e in self.events],
                           self.calib)
        self.flip_times = [t - self.rec_start for t, _ in changes if t >= self.rec_start]

    def _update_fusion(self, force: bool = False) -> None:
        now = self.clock()
        if not force and now - self._fuse_at < LIVE_EVERY_S:
            return
        self._fuse_at = now
        self.fusion = fuse(self.fl_t, self.fl_world, self.switch_half_flips, self.rec_start,
                           self.seconds, self.flip_ref)
        self.flip_angle = current_angle(self.fl_world[-3:], self.flip_ref)

    def _palm_down(self) -> Optional[bool]:
        """Ready/countdown: the patient is told to hold palm-down, so a FRESH reading is
        palm-down. Recording: compare with the state reported at START. Stale -> None."""
        state = self.device.state
        if state is None or not getattr(self.device, "state_fresh", True):
            return None
        if self.state in ("ready", "countdown") or self.calib is None:
            return True if self.state in ("ready", "countdown") else None
        return state == self.calib

    def tick(self, key: int, now: float) -> np.ndarray:
        self.handle_key(key, now)
        if self.running:
            self.update(now)
        return self.draw(now)

    # ---------------------------------------------------------------- drawing
    def draw(self, now: float) -> np.ndarray:
        if self.state == "trend":
            return self._draw_trend()
        c = ui.blank()
        self._draw_header(c)
        getattr(self, f"_draw_{self.state}")(c, now)
        self._draw_footer(c)
        return c

    def _hints(self) -> str:
        if self.state == "boot":
            return "SPACE continue (when done)   Q quit"
        if self.state in ("results", "dashboard"):
            return "SPACE next  P pdf  F fhir  C coach  R restart  H trend  Q quit"
        hints = "SPACE next  C coach  R restart  H trend  Q quit"
        sim = [k for k, ok in (("T tremor", hasattr(self.hands, "tremor_enabled")),
                               ("F stuck sensor", hasattr(self.device, "flipping"))) if ok]
        return ("sim: " + " ".join(sim) + " | " + hints) if sim else hints

    def _draw_footer(self, c, hints: Optional[str] = None) -> None:
        hud.footer_line(c, hints or self._hints())
        hud.telemetry(c, self.hands, self.device, self.sim_hand, self.sim_dev)

    def _draw_header(self, c) -> None:
        dev = self.device
        in_test = self.state in ("ready", "countdown", "recording", "done")
        status = dev.status
        if hasattr(self.hands, "tremor_enabled"):
            status += f" | fake tremor {'ON' if self.hands.tremor_enabled else 'off'}"
        if getattr(self.device, "flipping", True) is False:
            status += " | STUCK SENSOR (SIM)"
        hud.header(c, self.session_id, self.test[1].upper() if in_test else None,
                   live=not (self.sim_hand or self.sim_dev), status=status,
                   status_ok=dev.connected)
        if not dev.connected:
            hud.banner(c, f"{dev.status}  -  reconnecting automatically", (40, 40, 150))
        elif getattr(dev, "warning", ""):
            hud.banner(c, f"Arduino: {dev.warning}", (0, 110, 160))

    def _draw_coach(self, c, now) -> None:
        co = self.coach
        ui.text(c, "RHYTHM COACH", (20, 100), 0.9, ui.ACCENT, 2)
        ui.text(c, f"{co.hand.upper()} hand  |  hand flipping to a metronome", (290, 98), 0.5, ui.GREY)
        ui.text(c, co.gains_text(), (20, 124), 0.42, ui.DIM)
        if co.phase == "ready":
            ui.panel(c, 20, 140, 1240, 500)
            lines = ["Tilt sensor on the back of the hand, hold it PALM-DOWN.",
                     "1) 10 s: flip at your own pace (uncued baseline).",
                     "2) 35 s: flip once per beat. The tempo adapts with a PID controller",
                     "   to find the fastest rhythm you can keep (target: 85% on time).",
                     "", "SPACE start    L switch hand    R cancel"]
            for i, line in enumerate(lines):
                ui.text(c, line, (60, 200 + i * 40), 0.6, ui.WHITE if i < 5 else ui.ACCENT)
            ui.flip_indicator(c, co.palm_down, 760, 470, 460, 120)
            return
        if co.phase == "countdown":
            n = max(1, 3 - int(co.clock() - co.t_phase))
            ui.centred(c, str(n), 420, 4.0, ui.ACCENT, 4)
            return
        if co.phase == "done":
            self._draw_coach_done(c, co)
            return
        el = co.clock() - co.t_phase
        total = co.uncued_s if co.phase == "uncued" else co.cued_s
        label = "UNCUED - flip at your own pace" if co.phase == "uncued" else \
            f"CUED - tempo {co.coach.rate:.2f} beats/s ({1000 / co.coach.rate:.0f} ms)"
        ui.text(c, label, (20, 160), 0.6, ui.WHITE, 2)
        ui.text(c, f"{max(0.0, total - el):4.1f}s", (1100, 160), 0.8, ui.REC, 2)
        ui.beat_circle(c, 180, 330, co.beat_pulse(),
                       "beat (buzzer + LED from the Arduino)" if co.phase == "cued" else "no cue yet")
        ui.flip_indicator(c, co.palm_down, 20, 500, 340, 80)
        ui.coach_graph(c, co.series, 380, 180, 880, 400, co.cued_s)
        if co.coach is not None:
            otr = co.coach.on_time_rate()
            recs = co.coach.records
            stats = (f"beats {len(recs)}   on-time rate {'--' if otr is None else f'{otr:.2f}'}   "
                     f"missed {sum(r.on_time is None for r in recs)}   "
                     f"{'PROBING' if co.coach.probing else 'PID'}")
            ui.text(c, stats, (380, 610), 0.5, ui.GREY)
        ui.text(c, f"half-flips {len(co.flips)}", (20, 610), 0.5, ui.GREY)

    def _draw_coach_done(self, c, co) -> None:
        s = co.summary
        if s is None or not s.valid:
            ui.panel(c, 20, 140, 1240, 460)
            ui.text(c, "COACH RESULT: NOT MEASURED", (40, 190), 0.8, ui.REC, 2)
            why = co.error or (s.invalid_reason if s is not None else "no data")
            for i, line in enumerate(ui.wrap(why, 1160, 0.6)[:4]):
                ui.text(c, line, (40, 240 + i * 34), 0.6, ui.WHITE)
            ui.text(c, "Nothing is saved or exported for this run.", (40, 400), 0.55, ui.GREY)
            ui.text(c, "SPACE to continue", (40, 560), 0.7, ui.ACCENT, 2)
            return
        ui.panel(c, 20, 140, 600, 460)
        ui.text(c, "COACH RESULT", (40, 180), 0.7, ui.ACCENT, 2)
        rows = [("Max sustainable rhythm", f"{s.max_sustainable_rate:.2f} beats/s"),
                ("  = full flips/s", f"{s.max_sustainable_rate / 2:.2f}"),
                ("Mean asynchrony", "--" if s.mean_asynchrony_ms is None else f"{s.mean_asynchrony_ms:+.0f} ms"),
                ("On-time rate (last 8)", "--" if s.on_time_rate is None else f"{s.on_time_rate:.2f}"),
                ("Beats / missed", f"{s.beats} / {s.missed}"),
                ("Uncued rate", f"{s.uncued_rate:.2f} half-flips/s"),
                ("Cued rate (last 10 s)", f"{s.cued_rate:.2f} half-flips/s"),
                ("Cueing effect", f"{(s.cued_rate / s.uncued_rate - 1):+.0%}" if s.uncued_rate else "--")]
        for i, (k, v) in enumerate(rows):
            ui.text(c, k, (40, 225 + i * 42), 0.5, ui.GREY)
            ui.text(c, v, (330, 225 + i * 42), 0.55, ui.WHITE)
        ui.text(c, "SIM patient - not a measurement" if self.sim_dev else "demo tracking metric",
                (40, 580), 0.45, ui.WARN)
        ui.coach_graph(c, co.series, 640, 140, 620, 460, co.cued_s)
        ui.text(c, "SPACE to continue", (640, 640), 0.7, ui.ACCENT, 2)

    def _draw_boot(self, c, now) -> None:
        hud.boot_screen(c, self.boot, now)

    def _draw_welcome(self, c, now) -> None:
        ui.panel(c, 300, 110, 680, 500)
        ui.centred(c, "MOTOR CHECK PROTOCOL", 165, 1.1, ui.ACCENT, 2)
        ui.centred(c, "3 tests x 2 hands, scored 0-4 with every rule shown", 198, 0.45, ui.GREY)
        for i, (kind, hand) in enumerate(TESTS):
            y = 245 + i * 38
            ui.text(c, f"{i + 1:02d}", (360, y), 0.65, ui.ACCENT, 2)
            ui.text(c, f"{hand.upper():<6} {TITLES[kind]}", (420, y), 0.65, ui.WHITE)
            ui.text(c, f"{self.seconds:4.0f} s", (850, y), 0.65, ui.GREY)
        dose = "unknown" if self.dose_unknown else (self.dose_text or "_")
        ui.centred(c, f"Hours since last levodopa dose?  [ {dose} ]", 510, 0.6, ui.WHITE)
        ui.centred(c, "type a number (e.g. 2.5), N = unknown, Backspace = edit", 532, 0.4, ui.DIM)
        ui.centred(c, "PRESS SPACE TO BEGIN", 568, 0.85, ui.ACCENT, 2)
        ui.centred(c, "H: history trend    C: rhythm coach", 596, 0.45, ui.GREY)

    def _test_title(self, c) -> None:
        kind, hand = self.test
        ui.text(c, f"Test {self.idx + 1}/{len(TESTS)}", (20, 112), 0.8, ui.GREY)
        ui.text(c, f"{hand.upper()} hand {kind}", (150, 112), 1.1, ui.WHITE, 2)

    def _draw_camera(self, c) -> None:
        kind, hand = self.test
        frame = self.hf.frame
        if frame is None:
            ui.panel(c, 20, 135, 720, 520)
            ui.centred(c, self.hands.status, 380, 0.9, ui.REC, 1, 20, 740)
            return
        frame = frame.copy()
        draw_hand(frame, self.hf, hand, test=kind)
        ui.paste(c, frame, 20, 135, 693, 520)
        ui.brackets(c, 20, 135, 693, 520)
        if self.state == "recording":
            hud.scan_line(c, 20, 135, 693, 520, self.clock())
            ui.badge(c, "REC", 32, 147, ui.REC, 0.45)

    def _draw_left_live(self, c, now) -> None:
        kind, _ = self.test
        if kind == "flipping":
            self._draw_flipping_left(c)
            return
        self._draw_camera(c)
        if kind == "tremor":
            ui.text(c, CAMERA_NOTE, (30, 645), 0.5, ui.WARN)

    def _draw_flipping_left(self, c) -> None:
        kind, hand = self.test
        frame = self.hf.frame
        if frame is None:
            ui.panel(c, 20, 135, 360, 270)
            ui.centred(c, "camera unavailable", 260, 0.55, ui.WARN, 1, 20, 380)
            ui.centred(c, "switch only", 290, 0.5, ui.GREY, 1, 20, 380)
        else:
            frame = frame.copy()
            draw_hand(frame, self.hf, hand, test="tremor")
            ui.paste(c, frame, 20, 135, 360, 270)
            ui.brackets(c, 20, 135, 360, 270)
            if self.state == "recording":
                hud.scan_line(c, 20, 135, 360, 270, self.clock())
        angle = self.flip_angle if self.state == "recording" else None
        ui.rotation_gauge(c, 392, 135, 348, 270, angle, self.fusion if self.state == "recording" else None,
                          self.switch_half_flips)
        if self.closures_only:
            self._draw_og_status(c)
        else:
            ui.flip_indicator(c, self._palm_down(), 20, 415, 720, 70)
            ui.tilt_plot(c, self.events, self.calib, self.clock(), 20, 495, 720, 75)
        if self.state == "ready":
            ui.panel(c, 20, 580, 720, 75)
            ui.centred(c, "Calibrating: hold your hand PALM-DOWN", 612, 0.6, ui.WARN, 2, 20, 740)
            fresh = self.device.state is not None and getattr(self.device, "state_fresh", True)
            st = self.device.state if fresh else "?"
            ui.centred(c, f"switch reads {st}  (this state = palm-down)", 640, 0.45, ui.GREY, 1, 20, 740)
            return
        elapsed = (self.clock() - self.rec_start) if self.state == "recording" else None
        ui.flip_timeline(c, self.flip_times, self.seconds, elapsed, 20, 580, 720, 75)
        if self.state == "recording" and elapsed is not None and \
                elapsed > STUCK_AFTER_S and not self.flip_times:
            ui.panel(c, 20, 540, 720, 34, (40, 40, 150))
            ui.centred(c, SENSOR_STUCK, 563, 0.55, ui.WHITE, 2, 20, 740)

    def _draw_og_status(self, c) -> None:
        """--og: show what the original sketch itself reports (it drives its own LEDs)."""
        dev = self.device
        st = getattr(dev, "last_status", "") or "waiting for the first tilt..."
        col = ui.REC if "[!]" in st else ui.WARN if "[.]" in st else ui.OK if "[ ]" in st else ui.DIM
        ui.panel(c, 20, 415, 720, 155)
        ui.text(c, "ARDUINO (your original sketch) SAYS", (36, 442), 0.45, ui.ACCENT, 2)
        ui.text(c, st[:40], (36, 492), 0.9, col, 2)
        iv = getattr(dev, "last_interval_ms", None)
        ui.text(c, "last interval: " + ("--" if iv is None else f"{iv:.0f} ms between tilts"),
                (36, 530), 0.5, ui.GREY)
        ui.text(c, "its own LEDs: fast / medium / calm", (36, 556), 0.42, ui.DIM)

    def _draw_right_column(self, c, big: str, sub: str, colour) -> None:
        kind, hand = self.test
        x = 770
        for i, line in enumerate(INSTRUCTIONS[kind]):
            ui.text(c, line.format(hand=hand.upper()), (x, 160 + i * 32), 0.66)
        ui.text(c, big, (x, 345), 3.0 if len(big) <= 2 else 2.0, colour, 4)
        ui.text(c, sub, (x, 395), 0.8, ui.GREY)

    def _draw_ready(self, c, now) -> None:
        self._test_title(c)
        self._draw_left_live(c, now)
        self._draw_right_column(c, "Ready?", "SPACE to start", ui.ACCENT)

    def _draw_countdown(self, c, now) -> None:
        self._test_title(c)
        self._draw_left_live(c, now)
        n = max(1, COUNTDOWN_S - int(now - self.t_state))
        self._draw_right_column(c, str(n), "Get ready...", ui.ACCENT)

    def _draw_recording(self, c, now) -> None:
        kind, _ = self.test
        self._test_title(c)
        self._draw_left_live(c, now)
        left = max(0.0, self.seconds - (now - self.t_state))
        self._draw_right_column(c, f"{left:.1f}s", "Recording...", ui.REC)
        frac = min(1.0, 1 - left / self.seconds)
        ui.panel(c, 770, 420, 490, 22)
        cv2.rectangle(c, (770, 420), (770 + int(490 * frac), 442), ui.REC, -1)
        if kind == "tapping":
            draw_distance_graph(c, self.tap_t, self.tap_d, self.clock(), 770, 452, 490, 98)
            ui.text(c, f"TAPS {self.live_taps:3d}", (1150, 470), 0.5, ui.OK, 2)
        elif kind == "tremor":
            ui.fingertip_chart(c, self.trem_t, self.trem_xy, self.clock(), 770, 452, 490, 98)
        if kind in ("tremor", "tapping"):
            hud.quality_meter(c, self.quality, 770, 558, 490, self.sim_hand)
        else:
            n = len(self.flip_times) if self.closures_only else len(self.flip_times) // 2
            ui.text(c, f"FLIPS {n:3d}", (770, 520), 1.4, ui.OK, 3)
            ui.text(c, "closures from your sketch (1 = 1 full flip)" if self.closures_only
                    else f"{len(self.flip_times)} half-flips (switch)", (770, 556), 0.5, ui.GREY)

    def _draw_done_left(self, c, tr: TestResult) -> None:
        if tr.kind == "tremor":
            if tr.image is not None:
                ui.paste(c, tr.image, 20, 135, 720, 520)
            return
        if tr.kind == "tapping":
            t0 = self.tap_t[0] if self.tap_t else 0.0
            span = (self.tap_t[-1] - t0) if self.tap_t else self.seconds
            draw_distance_graph(c, self.tap_t, self.tap_d, t0 + span, 20, 135, 720, 520,
                                window_s=max(span, 1.0), vmax=1.6)
            f = tr.features
            for tt in getattr(f, "tap_times", ()):
                px = 20 + int(720 * tt / max(span, 1.0))
                cv2.line(c, (px, 640), (px, 655), ui.ACCENT, 2)
            ui.text(c, f"{f.n_taps} taps, {f.taps_per_sec:.1f}/s  (ticks = detected taps)",
                    (40, 630), 0.6, ui.WHITE)
            return
        f = tr.features
        ui.flip_timeline(c, list(f.flip_times), self.seconds, None, 20, 135, 720, 130)
        fu = tr.fusion
        lines = [("SWITCH", f"{f.full_flips} full flips ({f.half_flips} half-flips)"),
                 ("RATE", f"{f.flips_per_sec:.2f} full flips per second"),
                 ("RHYTHM", f"CV {f.interval_cv:.2f}   slowdown {f.decrement:.0%}   "
                            f"{f.hesitations} hesitation(s)")]
        if fu is not None:
            cam = f"{fu.camera_half_flips} half-flips" if fu.camera_ok else "hand not tracked"
            lines += [("CAMERA", cam),
                      ("ROTATION", f"median {fu.median_amplitude_deg:.0f} deg, shrinks "
                                   f"{fu.amplitude_decrement:.0%}" if fu.camera_ok else "n/a"),
                      ("FUSION", fu.status)]
        ui.panel(c, 20, 280, 720, 50 + 44 * len(lines))
        for i, (label, value) in enumerate(lines):
            ui.text(c, label, (40, 318 + i * 44), 0.45, ui.ACCENT, 2)
            col = ui.WHITE
            if label == "FUSION":
                col = ui.OK if fu.locked else ui.WARN
            ui.text(c, value, (180, 318 + i * 44), 0.55, col)

    def _draw_done(self, c, now) -> None:
        self._test_title(c)
        tr = self.results[self.test]
        self._draw_done_left(c, tr)
        if self.raw_dir:
            ui.text(c, ("REC " + self.label + " | " if self.label else "REC | ") + self.raw_msg,
                    (20, 670), 0.42, ui.REC if "NOT" in self.raw_msg else ui.OK)
        ui.rule_panel(c, 760, 125, 500, 420, f"{tr.hand} - {TITLES[tr.kind]}", self._explain(tr),
                      flag="LOW CONFIDENCE" if tr.low_confidence else "")
        if self.idx + 1 < len(TESTS):
            kind, hand = TESTS[self.idx + 1]
            ui.text(c, f"NEXT: {hand.upper()} {TITLES[kind]}", (770, 585), 0.55, ui.WHITE)
            ui.text(c, "SPACE to continue", (770, 630), 0.75, ui.ACCENT, 2)
        else:
            ui.text(c, "SPACE for results", (770, 610), 0.85, ui.ACCENT, 2)

    def _explain(self, tr: TestResult):
        if tr.kind == "tremor":
            e = explain_tremor(tr.features, self.seconds)
        elif tr.kind == "tapping":
            e = explain_tapping(tr.features)
        else:
            e = explain_flipping(tr.features, tr.fusion)
        if tr.result.score is None:          # e.g. Arduino lost: show why, no rules
            return e._replace(score=None, rules=[], formula="not scored: " + tr.result.reasons[0])
        return e

    def _draw_results(self, c, now) -> None:
        ui.text(c, "Results", (20, 100), 1.0, ui.ACCENT, 2)
        if self.save_msg:
            col = ui.REC if self.save_msg.startswith("NOT") else ui.DIM
            ui.text(c, self.save_msg, (180, 98), 0.5, col)
        for kind, hand in TESTS:
            tr = self.results.get((kind, hand))
            res = tr.result if tr else ScoreResult(None, ["Not run"])
            row = ("tremor", "tapping", "flipping").index(kind)
            col = 0 if hand == "Right" else 1
            ui.score_card(c, 20 + col * 625, 112 + row * 166, 615, 158,
                          f"{hand} - {TITLES[kind]}", res.score, res.reasons, compact=True,
                          flag="LOW CONFIDENCE" if tr and tr.low_confidence else "")
        if self.export_msg:
            ui.text(c, self.export_msg[-90:], (620, 98), 0.42,
                    ui.REC if self.export_msg.startswith("Export failed") else ui.OK)
        y = 628
        if self.flags:
            ui.text(c, "ASYMMETRY", (20, y + 14), 0.65, ui.WARN, 2)
            lines = ui.wrap(" | ".join(self.flags), ui.W - 200, 0.42)
            if len(lines) > 3:
                lines = lines[:3]
                lines[2] = lines[2][:max(0, len(lines[2]) - 14)] + " (+ more)"
            for j, part in enumerate(lines):
                ui.text(c, part, (165, y + 8 + j * 16), 0.42, ui.WARN)
        else:
            ui.text(c, "No left/right asymmetry flagged", (20, y + 14), 0.6, ui.GREY)

    def _draw_dashboard(self, c, now) -> None:
        ui.text(c, "RESULTS DASHBOARD", (20, 100), 0.9, ui.ACCENT, 2)
        if self.radar_img is not None:
            ui.paste(c, self.radar_img, 20, 115, 600, 520)
        ns, formula = self._neuroscore()
        ui.panel(c, 640, 115, 620, 250)
        ui.text(c, "NEUROSCORE", (665, 150), 0.6, ui.ACCENT, 2)
        col = ui.DIM if ns is None else (ui.OK if ns >= 75 else ui.WARN if ns >= 50 else ui.REC)
        ui.text(c, "--" if ns is None else f"{ns:.0f}", (665, 255), 3.2, col, 4)
        ui.text(c, "/ 100", (665 + ui.text_width("100", 3.2, 4) + 14, 255), 0.7, ui.GREY)
        for i, line in enumerate(ui.wrap(formula, 580, 0.42)[:3]):
            ui.text(c, line, (665, 295 + i * 20), 0.42, ui.WHITE)
        ui.text(c, "composite tracking index, not a diagnosis", (665, 352), 0.42, ui.WARN, 2)
        ui.panel(c, 640, 380, 620, 255)
        dose = self.dose_hours
        rows = [("Hours since levodopa", "unknown" if dose is None else f"{dose:g} h"),
                ("Asymmetry flags", str(len(self.flags))),
                ("Low-confidence results", str(sum(r.low_confidence for r in self.results.values()))),
                ("Rhythm coach", ", ".join(f"{h[0]} {s.max_sustainable_rate:.2f}/s"
                                           for h, s in self.coach_results.items()) or "not run (C)"),
                ("Mode", "SIMULATION - SIM data" if (self.sim_hand or self.sim_dev) else "LIVE")]
        for i, (k, v) in enumerate(rows):
            ui.text(c, k, (665, 418 + i * 34), 0.48, ui.GREY)
            ui.text(c, v, (930, 418 + i * 34), 0.5, ui.WHITE)
        ui.text(c, "P: doctor PDF   F: FHIR JSON   SPACE: trend", (665, 612), 0.48, ui.ACCENT, 2)
        if self.export_msg:
            ui.text(c, self.export_msg[-70:], (20, 655), 0.42,
                    ui.REC if self.export_msg.startswith("Export failed") else ui.OK)

    def _draw_trend(self) -> np.ndarray:
        c = ui.blank()
        if self.trend_img is None:
            ui.centred(c, "Trend chart unavailable (see console)", 360, 1.0, ui.REC)
        else:
            h = min(self.trend_img.shape[0], hud.HINT_Y - 22)
            c[:h] = self.trend_img[:h]
        self._draw_footer(c, "SPACE back   R new session   Q quit")
        return c


# --------------------------------------------------------------------------- main
def build(args):
    from device import ArduinoDevice, MockDevice
    from tapping_tracker import CameraHand, FakeHand
    sim_dev = args.sim or args.sim_device
    if sim_dev:
        device = MockDevice(test_seconds=args.seconds)
    elif args.og:
        from device import OgArduinoDevice
        device = OgArduinoDevice(args.port)
    else:
        device = ArduinoDevice(args.port)
    hands = FakeHand(test_seconds=args.seconds) if args.sim else CameraHand(args.camera)
    if not hands.ok:
        log.warning("%s - camera tests will show a message (use --sim for a fake hand)",
                    hands.status)
    return device, hands


MIN_TEST_S, MAX_TEST_S = 9.0, 60.0   # flipping decrement compares two 4-s windows


def _seconds(value: str) -> float:
    v = float(value)
    if not MIN_TEST_S <= v <= MAX_TEST_S:
        raise argparse.ArgumentTypeError(f"--seconds must be {MIN_TEST_S:g}-{MAX_TEST_S:g}")
    return v


def warm_up_matplotlib() -> None:
    """First matplotlib import/render takes ~1 s (much longer on a fresh font cache).
    Pay that before the window opens instead of mid-demo."""
    t0 = time.perf_counter()
    try:
        ui.spectrum_image(np.linspace(0, 15, 64), np.ones(64), 5.0, 0.0, 0, 320, 240)
        history.trend_image([], 320, 240)
        ui.text(ui.blank(), "0123456789 warm", (0, 30), 0.5)    # load the mono font
    except Exception:
        log.exception("matplotlib warm-up failed (charts may be unavailable)")
    log.info("charts ready (%.1f s)", time.perf_counter() - t0)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--sim", action="store_true", help="MockDevice + FakeHand (no hardware)")
    p.add_argument("--sim-device", action="store_true", help="MockDevice + real webcam")
    p.add_argument("--og", action="store_true",
                   help="read the team's ORIGINAL tremor-monitor sketch (9600 baud, sensor D11)")
    p.add_argument("--seed-history", action="store_true",
                   help="add 14 days x 2 sessions of DEMO DATA")
    p.add_argument("--port", help="serial port, e.g. COM5 (default: auto-detect)")
    p.add_argument("--camera", type=int, default=0)
    p.add_argument("--seconds", type=_seconds, default=10.0,
                   help=f"recording length per test, {MIN_TEST_S:g}-{MAX_TEST_S:g} (default 10)")
    p.add_argument("--history", default=history.DEFAULT_PATH)
    p.add_argument("--no-boot", action="store_true", help="skip the boot self-check screen")
    p.add_argument("--record", nargs="?", const="recordings", default=None, metavar="DIR",
                   help="save raw data of every test to DIR (default: recordings/)")
    p.add_argument("--label", default="", help="label for recordings, e.g. alex_normal, "
                                               "alex_acted_tremor (used by tools/calibrate.py)")
    return p.parse_args(argv)


def run(app: App, hands, device, key_script=None, on_frame=None) -> int:
    """Main loop. key_script(now)->key|-1 injects keys (scripted GUI test);
    on_frame(app, canvas) observes each frame. Returns the number of caught errors."""
    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW, ui.W, ui.H)
    key, errors = -1, 0
    try:
        while app.running:
            now = app.clock()
            if key < 0 and key_script is not None:
                key = key_script(now)
            try:
                canvas = app.tick(key, now)
            except Exception:
                # Keep the demo alive: log, show an error screen, let R restart.
                errors += 1
                log.error("Unexpected error:\n%s", traceback.format_exc())
                canvas = ui.blank()
                ui.centred(canvas, "Something went wrong - press R to restart", 360, 1.0, ui.REC)
                if key >= 0 and chr(key & 0xFF).lower() == "r":
                    try:
                        app.restart(app.clock())
                    except Exception:
                        log.error("Restart failed:\n%s", traceback.format_exc())
            if on_frame is not None:
                on_frame(app, canvas)
            cv2.imshow(WINDOW, canvas)
            key = cv2.waitKey(1)
            if cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
                break
    finally:
        try:
            device.led("OFF")
            device.lcd("NeuroCheck", "Goodbye")
        finally:
            device.close()
            hands.close()
            cv2.destroyAllWindows()
    return errors


def main(argv=None) -> None:
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    if args.seed_history:
        history.seed_history(args.history)          # 14 days x 2 sessions of DEMO DATA
    device, hands = build(args)
    warm_up_matplotlib()
    boot = None if args.no_boot else BootChecks(device, hands).start()
    app = App(device, hands, args.seconds, args.history, boot=boot, raw_dir=args.record,
              label=args.label)
    app.restart(app.clock())
    if boot is not None:
        app.state = "boot"
    run(app, hands, device)


if __name__ == "__main__":
    main()
