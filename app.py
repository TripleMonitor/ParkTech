"""NeuroCheck - 1-minute Parkinson's motor check station (hackathon demo).

TRACKING / DECISION-SUPPORT TOOL, NOT A DIAGNOSIS. All thresholds are demo thresholds.

  python app.py                 real Arduino (auto-detect port) + webcam
  python app.py --sim           MockDevice + FakeHand: no hardware at all
  python app.py --sim-device    MockDevice + real webcam
  python app.py --seed-history  add 7 days of fake sessions first (for the trend demo)

Keys: SPACE next   R restart   Q/ESC quit   H trend   T toggle fake tremor (sim device)
"""
from __future__ import annotations

import argparse
import logging
import time
import traceback
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np

import history
import ui
from scoring import (ScoreResult, score_tapping, score_tremor, tapping_asymmetry,
                     tremor_asymmetry)
from tapping_analysis import TappingFeatures, analyze_tapping, detect_taps
from tapping_tracker import HandFrame, draw_distance_graph, draw_hand
from tremor_analysis import TremorFeatures, analyze_samples

log = logging.getLogger("neurocheck")

COUNTDOWN_S = 3
WINDOW = "NeuroCheck"
TESTS = (("tremor", "Right"), ("tremor", "Left"), ("tapping", "Right"), ("tapping", "Left"))
INSTRUCTIONS = {
    "tremor": ["Sensor on your {hand} wrist.", "Rest the forearm on the table,",
               "let the hand relax. Keep still."],
    "tapping": ["Hold your {hand} hand up to the camera.", "Tap index finger on thumb",
                "as FAST and as BIG as you can."],
}
TITLES = {"tremor": "Rest tremor (3.17)", "tapping": "Finger tapping (3.4)"}
DISCLAIMER = "Demo thresholds - tracking aid, NOT a diagnosis"
KEY_SPACE, KEY_ENTER, KEY_ESC = 32, 13, 27
LIVE_TAP_EVERY_S = 0.25


@dataclass(frozen=True)
class TestResult:
    kind: str
    hand: str
    features: object                    # TremorFeatures | TappingFeatures
    result: ScoreResult
    image: Optional[np.ndarray] = None  # spectrum for tremor


def led_for(scores: list[Optional[int]]) -> str:
    """Results LED: worst score 0-1 green, 2 yellow, 3-4 red; nothing scored -> off."""
    real = [s for s in scores if s is not None]
    if not real:
        return "OFF"
    worst = max(real)
    return "G" if worst <= 1 else "Y" if worst == 2 else "R"


class App:
    def __init__(self, device, hands, seconds: float = 10.0,
                 history_path: str = history.DEFAULT_PATH):
        self.device, self.hands, self.seconds = device, hands, seconds
        self.history_path = history_path
        self.running = True
        self.restart(0.0)

    # ---------------------------------------------------------------- state changes
    def restart(self, now: float) -> None:
        self.device.stop()
        self.state, self.idx, self.t_state = "welcome", 0, now
        self.results: dict[tuple[str, str], TestResult] = {}
        self.flags: list[str] = []
        self.samples: list = []
        self.tap_t: list[float] = []
        self.tap_d: list[Optional[float]] = []
        self.live_taps, self._live_at = 0, 0.0
        self.hf = HandFrame(None, None, None)
        self.trend_img: Optional[np.ndarray] = None
        self.save_msg = ""
        self.last_beep = 0
        self.device.led("OFF")
        self.device.lcd("NeuroCheck", "Press SPACE")

    @property
    def test(self) -> tuple[str, str]:
        return TESTS[self.idx]

    def _lcd_test(self, line2: str) -> None:
        kind, hand = self.test
        self.device.lcd(f"{hand} {kind}", line2)

    def _goto(self, state: str, now: float) -> None:
        self.state, self.t_state = state, now

    def _begin_countdown(self, now: float) -> None:
        self._goto("countdown", now)
        self.last_beep = 0
        self.samples, self.tap_t, self.tap_d = [], [], []
        self.live_taps = 0
        self.hands.reset(now + COUNTDOWN_S)

    def _begin_recording(self, now: float) -> None:
        self._goto("recording", now)
        self._lcd_test("Recording...")
        if self.test[0] == "tremor":
            self.device.start()

    def _finish_test(self, now: float) -> None:
        kind, hand = self.test
        if kind == "tremor":
            self.device.stop()
            self.samples.extend(self.device.drain())
            self.results[self.test] = self._tremor_result(hand)
        else:
            self.results[self.test] = self._tapping_result(hand)
        self.device.beep(2)
        score = self.results[self.test].result.score
        self._lcd_test(f"Score {'-' if score is None else score}")
        self._goto("done", now)

    def _tremor_result(self, hand: str) -> TestResult:
        f = analyze_samples(self.samples)
        r = score_tremor(f)
        if not f.valid and not self.device.connected:
            r = ScoreResult(None, r.reasons + [self.device.status])
        img = None
        try:
            img = ui.spectrum_image(f.freqs, f.power, f.peak_hz, f.displacement_cm,
                                    r.score, 720, 540)
        except Exception:   # a chart must never take the demo down
            log.exception("spectrum render failed")
        return TestResult("tremor", hand, f, r, img)

    def _tapping_result(self, hand: str) -> TestResult:
        f = analyze_tapping(self.tap_t, self.tap_d) if self.tap_t else \
            TappingFeatures(0, 0.0, 0.0, 0.0, 0.0, 0, 0.0, 0.0)
        r = score_tapping(f)
        if r.score is None and not self.hands.ok:
            r = ScoreResult(None, r.reasons + [self.hands.status])
        return TestResult("tapping", hand, f, r)

    def _enter_results(self, now: float) -> None:
        self._goto("results", now)
        self.flags = self._asymmetry()
        self.device.led(led_for([r.result.score for r in self.results.values()]))
        self.device.lcd(self._lcd_scores("tremor", "Tremor"), self._lcd_scores("tapping", "Tap"))
        try:
            history.save_session(self._session_row(), self.history_path)
            self.save_msg = f"Saved to {self.history_path}"
        except OSError as exc:
            log.error("Could not save session: %s", exc)
            self.save_msg = f"NOT saved: {exc.strerror} (is {self.history_path} open in Excel?)"

    def _enter_trend(self, now: float) -> None:
        self._goto("trend", now)
        try:
            self.trend_img = history.trend_image(history.load_sessions(self.history_path),
                                                 ui.W, ui.H - 36)
        except Exception:
            log.exception("trend render failed")
            self.trend_img = None

    def _lcd_scores(self, kind: str, label: str) -> str:
        def s(hand: str) -> str:
            r = self.results.get((kind, hand))
            return "-" if r is None or r.result.score is None else str(r.result.score)
        return f"{label} R{s('Right')} L{s('Left')}"

    def _asymmetry(self) -> list[str]:
        flags: list[str] = []
        for kind, fn in (("tremor", tremor_asymmetry), ("tapping", tapping_asymmetry)):
            r, l = self.results.get((kind, "Right")), self.results.get((kind, "Left"))
            if r and l:
                flags.extend(fn(r.features, l.features, r.result, l.result))
        return flags

    def _session_row(self) -> dict:
        row: dict = {"asymmetry": " ; ".join(self.flags)}
        for (kind, hand), tr in self.results.items():
            side, f = hand[0], tr.features
            if isinstance(f, TremorFeatures):
                row.update({f"tremor_{side}": tr.result.score,
                            f"tremor_{side}_cm": f.displacement_cm if f.valid else None,
                            f"tremor_{side}_hz": f.peak_hz if f.valid else None})
            elif isinstance(f, TappingFeatures):
                row.update({f"tap_{side}": tr.result.score,
                            f"tap_{side}_rate": f.taps_per_sec, f"tap_{side}_amp": f.mean_amplitude})
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
        elif ch == "t" and hasattr(self.device, "tremor_on"):
            self.device.tremor_on = not self.device.tremor_on
        elif ch == "h" and self.state in ("welcome", "results"):
            self._enter_trend(now)
        elif not go:
            return
        elif self.state == "welcome":
            self.idx = 0
            self._goto("ready", now)
            self._lcd_test("SPACE to start")
        elif self.state == "ready":
            self._begin_countdown(now)
        elif self.state == "done":
            if self.idx + 1 < len(TESTS):
                self.idx += 1
                self._begin_countdown(now)
            else:
                self._enter_results(now)
        elif self.state == "results":
            self._enter_trend(now)
        elif self.state == "trend":
            self.restart(now)

    # ---------------------------------------------------------------- update
    def update(self, now: float) -> None:
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
        if self.state == "recording" and kind == "tremor":
            self.samples.extend(self.device.drain())

        active = self.state in ("ready", "countdown", "recording")
        if kind == "tapping" and active:
            self.hf = self.hands.read(now, hand, detect=True)
            if self.state == "recording":
                self.tap_t.append(now)
                self.tap_d.append(self.hf.distance)
                self._update_live_taps(now)

        if self.state == "recording" and elapsed >= self.seconds:
            self._finish_test(now)

    def _update_live_taps(self, now: float) -> None:
        if now - self._live_at < LIVE_TAP_EVERY_S:
            return
        self._live_at = now
        valid = [(t, d) for t, d in zip(self.tap_t, self.tap_d) if d is not None]
        if len(valid) >= 10:
            t, d = np.array(valid, dtype=float).T
            self.live_taps = len(detect_taps(t, d)[0])

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
        ui.panel(c, 0, ui.H - 36, ui.W, 36)
        ui.text(c, DISCLAIMER, (20, ui.H - 12), 0.55, ui.WARN)
        hints = "SPACE next   R restart   H trend   Q quit"
        if hasattr(self.device, "tremor_on"):
            hints = "T fake tremor   " + hints
        ui.text(c, hints, (ui.W - 20 - ui.text_width(hints, 0.5), ui.H - 12), 0.5, ui.GREY)
        return c

    def _draw_header(self, c) -> None:
        ui.panel(c, 0, 0, ui.W, 56)
        ui.text(c, "NeuroCheck", (20, 39), 1.1, ui.ACCENT, 2)
        ui.text(c, "Parkinson's motor check station", (262, 37), 0.6, ui.GREY)
        dev = self.device
        status = dev.status
        if hasattr(dev, "tremor_on"):
            status += f"  |  fake tremor {'ON' if dev.tremor_on else 'off'}"
        colour = ui.OK if dev.connected else ui.REC
        if hasattr(dev, "tremor_on") and dev.tremor_on:
            colour = ui.WARN
        ui.text(c, status, (ui.W - 20 - ui.text_width(status, 0.55), 36), 0.55, colour)
        if not dev.connected:
            ui.panel(c, 0, 56, ui.W, 30, (40, 40, 150))
            ui.centred(c, f"{dev.status}  -  reconnecting automatically", 78, 0.6, ui.WHITE)

    def _draw_welcome(self, c, now) -> None:
        ui.centred(c, "1-minute motor check", 200, 1.8, ui.WHITE, 2)
        for i, (kind, hand) in enumerate(TESTS):
            ui.text(c, f"{i + 1}.  {hand} hand {kind}", (430, 290 + i * 46), 0.95, ui.WHITE)
            ui.text(c, f"{self.seconds:g} s", (900, 290 + i * 46), 0.95, ui.GREY)
        ui.centred(c, "Press SPACE to begin", 530, 1.1, ui.ACCENT, 2)
        ui.centred(c, "H: history trend", 575, 0.65, ui.GREY)

    def _test_title(self, c) -> None:
        kind, hand = self.test
        ui.text(c, f"Test {self.idx + 1}/4", (20, 112), 0.8, ui.GREY)
        ui.text(c, f"{hand.upper()} hand {kind}", (150, 112), 1.1, ui.WHITE, 2)

    def _draw_left_live(self, c, now) -> None:
        kind, hand = self.test
        if kind == "tremor":
            ui.accel_chart(c, self.samples, 20, 135, 720, 520)
            if hasattr(self.device, "tremor_on") and self.state != "recording":
                ui.text(c, "Sim: press T to toggle a fake 5 Hz tremor", (40, 630), 0.6, ui.WARN)
            return
        frame = self.hf.frame
        if frame is None:
            ui.panel(c, 20, 135, 720, 520)
            ui.centred(c, self.hands.status, 380, 0.9, ui.REC, 1, 20, 740)
            return
        frame = frame.copy()
        draw_hand(frame, self.hf, hand)
        ui.paste(c, frame, 20, 135, 693, 520)

    def _draw_right_column(self, c, big: str, sub: str, colour) -> None:
        kind, hand = self.test
        x = 770
        for i, line in enumerate(INSTRUCTIONS[kind]):
            ui.text(c, line.format(hand=hand.upper()), (x, 165 + i * 36), 0.72)
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
        self._test_title(c)
        self._draw_left_live(c, now)
        left = max(0.0, self.seconds - (now - self.t_state))
        self._draw_right_column(c, f"{left:.1f}s", "Recording...", ui.REC)
        frac = min(1.0, 1 - left / self.seconds)
        ui.panel(c, 770, 420, 490, 22)
        cv2.rectangle(c, (770, 420), (770 + int(490 * frac), 442), ui.REC, -1)
        if self.test[0] == "tapping":
            draw_distance_graph(c, self.tap_t, self.tap_d, now, 770, 465, 490, 150)
            ui.text(c, f"Taps: {self.live_taps}", (770, 650), 0.9, ui.OK, 2)

    def _draw_done(self, c, now) -> None:
        self._test_title(c)
        tr = self.results[self.test]
        if tr.kind == "tremor" and tr.image is not None:
            ui.paste(c, tr.image, 20, 135, 720, 520)
        elif tr.kind == "tapping":
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
        ui.score_card(c, 760, 135, 500, 300, TITLES[tr.kind], tr.result.score, tr.result.reasons)
        if self.idx + 1 < len(TESTS):
            kind, hand = TESTS[self.idx + 1]
            ui.text(c, f"Next: {hand.upper()} hand {kind}", (770, 485), 0.85, ui.WHITE, 1)
            for i, line in enumerate(INSTRUCTIONS[kind]):
                ui.text(c, line.format(hand=hand.upper()), (770, 520 + i * 30), 0.6, ui.GREY)
            ui.text(c, "SPACE to start", (770, 640), 0.9, ui.ACCENT, 2)
        else:
            ui.text(c, "SPACE for results", (770, 520), 1.0, ui.ACCENT, 2)

    def _draw_results(self, c, now) -> None:
        ui.text(c, "Results", (20, 105), 1.1, ui.ACCENT, 2)
        if self.save_msg:
            col = ui.REC if self.save_msg.startswith("NOT") else ui.DIM
            ui.text(c, self.save_msg, (200, 103), 0.55, col)
        for i, (kind, hand) in enumerate(TESTS):
            tr = self.results.get((kind, hand))
            res = tr.result if tr else ScoreResult(None, ["Not run"])
            row, col = (0, 0 if hand == "Right" else 1) if kind == "tremor" else \
                (1, 0 if hand == "Right" else 1)
            ui.score_card(c, 20 + col * 630, 120 + row * 235, 610, 225,
                          f"{hand} - {TITLES[kind]}", res.score, res.reasons)
        y = 610
        if self.flags:
            ui.text(c, "ASYMMETRY", (20, y + 18), 0.75, ui.WARN, 2)
            line = "  |  ".join(self.flags)
            for j, part in enumerate(ui.wrap(line, ui.W - 200, 0.52)[:3]):
                ui.text(c, part, (180, y + 16 + j * 22), 0.52, ui.WARN)
        else:
            ui.text(c, "No left/right asymmetry flagged", (20, y + 18), 0.65, ui.GREY)

    def _draw_trend(self) -> np.ndarray:
        c = ui.blank()
        if self.trend_img is None:
            ui.centred(c, "Trend chart unavailable (see console)", 360, 1.0, ui.REC)
        else:
            c[:self.trend_img.shape[0]] = self.trend_img
        ui.panel(c, 0, ui.H - 36, ui.W, 36)
        ui.text(c, DISCLAIMER, (20, ui.H - 12), 0.55, ui.WARN)
        hint = "SPACE new session   R restart   Q quit"
        ui.text(c, hint, (ui.W - 20 - ui.text_width(hint, 0.5), ui.H - 12), 0.5, ui.GREY)
        return c


# --------------------------------------------------------------------------- main
def build(args):
    from device import ArduinoDevice, MockDevice
    from tapping_tracker import CameraHand, FakeHand
    device = MockDevice() if (args.sim or args.sim_device) else ArduinoDevice(args.port)
    hands = FakeHand(test_seconds=args.seconds) if args.sim else CameraHand(args.camera)
    if not hands.ok:
        log.warning("%s - tapping tests will show a message (use --sim for a fake hand)",
                    hands.status)
    return device, hands


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--sim", action="store_true", help="MockDevice + FakeHand (no hardware)")
    p.add_argument("--sim-device", action="store_true", help="MockDevice + real webcam")
    p.add_argument("--seed-history", action="store_true", help="add 7 days of fake sessions")
    p.add_argument("--port", help="serial port, e.g. COM5 (default: auto-detect)")
    p.add_argument("--camera", type=int, default=0)
    p.add_argument("--seconds", type=float, default=10.0, help="recording length per test")
    p.add_argument("--history", default=history.DEFAULT_PATH)
    return p.parse_args(argv)


def run(app: App, hands, device, key_script=None, on_frame=None) -> int:
    """Main loop. key_script(now)->key|-1 injects keys (scripted GUI test);
    on_frame(app, canvas) observes each frame. Returns the number of caught errors."""
    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW, ui.W, ui.H)
    key, errors = -1, 0
    try:
        while app.running:
            now = time.monotonic()
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
                        app.restart(time.monotonic())
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
        history.seed_history(args.history)
        log.info("Seeded 7 days of fake sessions into %s", args.history)
    device, hands = build(args)
    app = App(device, hands, args.seconds, args.history)
    app.restart(time.monotonic())
    run(app, hands, device)


if __name__ == "__main__":
    main()
