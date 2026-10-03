"""NeuroCheck - 1-minute Parkinson's motor check station (hackathon demo).

TRACKING / DECISION-SUPPORT TOOL, NOT A DIAGNOSIS. All thresholds are demo thresholds.

Run:  python app.py            (real Arduino, auto-detects port; --port COM5 to force)
      python app.py --sim      (fake IMU; press T to toggle a 5 Hz tremor)
Keys: SPACE/ENTER start/next   S skip test   T sim tremor   H trend   R restart   Q/ESC quit
"""
from __future__ import annotations

import argparse
import logging
import textwrap
import time
from dataclasses import dataclass
from typing import Optional, Union

import cv2
import numpy as np

import history
from scoring import (ScoreResult, TappingFeatures, TremorFeatures, score_tapping,
                     score_tremor, tapping_asymmetry, tremor_asymmetry)
from tremor import TremorRecorder

W, H = 1280, 720
COUNTDOWN_S = 3
DONE_S = 1.2
WINDOW = "NeuroCheck"
TESTS = (("tremor", "Right"), ("tremor", "Left"), ("tapping", "Right"), ("tapping", "Left"))
INSTRUCTIONS = {
    "tremor": ["Strap the sensor to your {hand} wrist.",
               "Rest your forearm on the table and relax",
               "the hand completely. Keep still."],
    "tapping": ["Hold your {hand} hand up to the camera.",
                "Tap index finger on thumb as FAST and",
                "as BIG as you can until the beep."],
}
DISCLAIMER = "Demo thresholds - tracking aid, NOT a diagnosis. See a clinician."

BG, PANEL, WHITE, GREY = (30, 26, 24), (52, 46, 42), (240, 240, 240), (150, 150, 150)
ACCENT = (255, 190, 60)
SCORE_COLOURS = {0: (90, 200, 90), 1: (60, 220, 200), 2: (0, 210, 255),
                 3: (0, 140, 255), 4: (60, 60, 230), None: GREY}
FONT = cv2.FONT_HERSHEY_SIMPLEX

KEY_SPACE, KEY_ENTER, KEY_ESC = 32, 13, 27


@dataclass(frozen=True)
class TestResult:
    kind: str
    hand: str
    features: Union[TremorFeatures, TappingFeatures, None]
    result: ScoreResult


# --- drawing helpers ------------------------------------------------------------
def text(img, s, org, scale=0.7, colour=WHITE, thick=2):
    cv2.putText(img, s, org, FONT, scale, colour, thick, cv2.LINE_AA)


def centred(img, s, y, scale=1.0, colour=WHITE, thick=2):
    (tw, _), _ = cv2.getTextSize(s, FONT, scale, thick)
    text(img, s, ((img.shape[1] - tw) // 2, y), scale, colour, thick)


def paste(canvas, img, x, y, w, h):
    canvas[y:y + h, x:x + w] = cv2.resize(img, (w, h))


def trace(canvas, values, x, y, w, h, vmax):
    cv2.rectangle(canvas, (x, y), (x + w, y + h), PANEL, -1)
    if len(values) < 2:
        return
    v = np.clip(np.asarray(values) / vmax, 0, 1)
    xs = np.linspace(x, x + w, len(v)).astype(int)
    ys = (y + h - v * h).astype(int)
    cv2.polylines(canvas, [np.column_stack([xs, ys])], False, ACCENT, 2, cv2.LINE_AA)


# --- application ------------------------------------------------------------------
class App:
    def __init__(self, device, tracker=None, seconds: float = 10.0,
                 history_path: str = history.DEFAULT_PATH, sim: bool = False):
        self.device, self.tracker, self.seconds = device, tracker, seconds
        self.history_path, self.sim = history_path, sim
        self.running = True
        self.trend_img: Optional[np.ndarray] = None
        self.restart(time.monotonic())

    # state ---------------------------------------------------------------------
    def restart(self, now: float) -> None:
        self.state, self.idx, self.t_state = "welcome", 0, now
        self.results: dict[tuple[str, str], TestResult] = {}
        self.flags: tuple[str, ...] = ()
        self.save_error = ""
        self.recorder = TremorRecorder()
        self.last_beep = 0
        self.device.led("G")
        self.device.lcd("NeuroCheck", "Press SPACE")

    @property
    def test(self) -> tuple[str, str]:
        return TESTS[self.idx]

    def _goto(self, state: str, now: float) -> None:
        self.state, self.t_state = state, now

    def _begin_ready(self, now: float) -> None:
        kind, hand = self.test
        self._goto("ready", now)
        self.device.led("G")
        self.device.lcd(f"{hand} {kind}", "SPACE to start")

    def _begin_countdown(self, now: float) -> None:
        self._goto("countdown", now)
        self.last_beep = 0
        self.device.led("Y")
        self.recorder = TremorRecorder()
        if self.tracker:
            self.tracker.reset()

    def _begin_recording(self, now: float) -> None:
        self._goto("recording", now)
        kind, hand = self.test
        self.device.lcd(f"{hand} {kind}", "Recording...")
        if kind == "tremor":
            self.device.start()

    def _finish_test(self, now: float, skipped: bool = False) -> None:
        kind, hand = self.test
        if kind == "tremor":
            self.device.stop()
            self.recorder.add(self.device.drain())
        self.results[self.test] = self._score(kind, hand, skipped)
        self.device.beep(1)
        self.device.led("G")
        self._goto("done", now)

    def _score(self, kind: str, hand: str, skipped: bool) -> TestResult:
        if skipped:
            return TestResult(kind, hand, None, ScoreResult(None, ("Skipped",)))
        if kind == "tremor":
            f = self.recorder.features()
            return TestResult(kind, hand, f, score_tremor(f))
        if self.tracker is None:
            return TestResult(kind, hand, None, ScoreResult(None, ("Camera unavailable",)))
        f = self.tracker.features()
        return TestResult(kind, hand, f, score_tapping(f))

    def _enter_results(self, now: float) -> None:
        self._goto("results", now)
        self.flags = self._asymmetry()
        scores = [r.result.score for r in self.results.values() if r.result.score is not None]
        worst = max(scores, default=0)
        self.device.led("R" if worst >= 2 or self.flags else "Y" if worst == 1 else "G")
        self.device.lcd(self._lcd_line("tremor", "Tremor"), self._lcd_line("tapping", "Tap"))
        try:
            history.save_session(self._session_row(), self.history_path)
            self.save_error = ""
        except OSError as exc:
            logging.error("Could not save session to %s: %s", self.history_path, exc)
            self.save_error = f"Session NOT saved ({exc.strerror}) - close {self.history_path}?"

    def _lcd_line(self, kind: str, label: str) -> str:
        def s(hand):
            r = self.results.get((kind, hand))
            return "-" if r is None or r.result.score is None else str(r.result.score)
        return f"{label} R{s('Right')} L{s('Left')}"

    def _asymmetry(self) -> tuple[str, ...]:
        flags: list[str] = []
        for kind, fn in (("tremor", tremor_asymmetry), ("tapping", tapping_asymmetry)):
            r, l = self.results.get((kind, "Right")), self.results.get((kind, "Left"))
            if r and l and r.features is not None and l.features is not None:
                flags.extend(fn(r.features, l.features, r.result, l.result))
        return tuple(flags)

    def _session_row(self) -> dict:
        row: dict = {"asymmetry": " ; ".join(self.flags)}
        for (kind, hand), tr in self.results.items():
            side = hand[0]
            key = "tremor" if kind == "tremor" else "tap"
            row[f"{key}_{side}"] = tr.result.score
            f = tr.features
            if isinstance(f, TremorFeatures):
                row[f"tremor_{side}_cm"] = f.displacement_cm
                row[f"tremor_{side}_hz"] = f.peak_hz
            elif isinstance(f, TappingFeatures):
                row[f"tap_{side}_rate"] = f.taps_per_sec
                row[f"tap_{side}_amp"] = f.mean_amplitude
        return row

    # input ---------------------------------------------------------------------
    def handle_key(self, key: int, now: float) -> None:
        if key < 0:
            return
        ch = chr(key).lower() if key < 256 else ""
        go = key in (KEY_SPACE, KEY_ENTER)
        if key == KEY_ESC or ch == "q":
            self.running = False
        elif ch == "t" and self.sim:
            self.device.toggle_tremor()
        elif ch == "r":
            self.device.stop()
            self.restart(now)
        elif ch == "h":
            self.trend_img = history.trend_image(history.load_sessions(self.history_path), W, H)
            self._goto("trend", now)
        elif self.state == "trend" and (go or ch == "b"):
            self._goto("results" if self.results else "welcome", now)
        elif self.state == "welcome" and go:
            self.idx = 0
            self._begin_ready(now)
        elif self.state == "ready" and go:
            self._begin_countdown(now)
        elif self.state in ("ready", "countdown", "recording") and ch == "s":
            if self.test[0] == "tremor":
                self.device.stop()
            self._finish_test(now, skipped=True)
        elif self.state == "results" and go:
            self.restart(now)

    # update --------------------------------------------------------------------
    def update(self, now: float, frame: Optional[np.ndarray]) -> None:
        elapsed = now - self.t_state
        kind, hand = self.test
        if self.state == "countdown":
            n = int(elapsed) + 1
            if n <= COUNTDOWN_S and n > self.last_beep:
                self.last_beep = n
                self.device.beep(1)
                self.device.lcd(f"{hand} {kind}", f"Starting in {COUNTDOWN_S - n + 1}")
            if elapsed >= COUNTDOWN_S:
                self._begin_recording(now)
        elif self.state == "recording":
            if kind == "tremor":
                self.recorder.add(self.device.drain())
            if elapsed >= self.seconds:
                self._finish_test(now)
        elif self.state == "done" and elapsed >= DONE_S:
            if self.idx + 1 < len(TESTS):
                self.idx += 1
                self._begin_ready(now)
            else:
                self._enter_results(now)

        if frame is not None and self.tracker and kind == "tapping" \
                and self.state in ("ready", "countdown", "recording"):
            self.tracker.process(frame, now, hand, record=self.state == "recording")

    def tick(self, key: int, now: float, frame: Optional[np.ndarray]) -> np.ndarray:
        self.handle_key(key, now)
        self.update(now, frame)
        return self.draw(now, frame)

    # draw ----------------------------------------------------------------------
    def draw(self, now: float, frame: Optional[np.ndarray]) -> np.ndarray:
        if self.state == "trend" and self.trend_img is not None:
            c = self.trend_img.copy()
            text(c, "Any of SPACE / B: back    Q: quit", (20, H - 15), 0.6, (80, 80, 80), 1)
            return c
        c = np.full((H, W, 3), BG, np.uint8)
        self._draw_header(c)
        getattr(self, f"_draw_{self.state}")(c, now, frame)
        cv2.rectangle(c, (0, H - 40), (W, H), PANEL, -1)
        text(c, DISCLAIMER, (20, H - 14), 0.55, GREY, 1)
        return c

    def _draw_header(self, c) -> None:
        cv2.rectangle(c, (0, 0), (W, 56), PANEL, -1)
        text(c, "NeuroCheck", (20, 38), 1.0, ACCENT, 2)
        text(c, "Parkinson's motor check station", (220, 38), 0.65, GREY, 1)
        mode = f"SIM  tremor {'ON' if self.device.tremor_on else 'off'} (T)" if self.sim \
            else f"Arduino {getattr(self.device, 'port', '')}"
        text(c, mode, (W - 360, 38), 0.6, (0, 210, 255) if self.sim else GREY, 1)

    def _draw_welcome(self, c, now, frame) -> None:
        centred(c, "1-minute motor check", 180, 1.6, WHITE, 3)
        steps = ["1. Right hand tremor  (10 s)", "2. Left hand tremor   (10 s)",
                 "3. Right hand tapping (10 s)", "4. Left hand tapping  (10 s)"]
        for i, s in enumerate(steps):
            text(c, s.replace("10 s", f"{self.seconds:g} s"), (430, 260 + i * 45), 0.85)
        centred(c, "Press SPACE to begin", 500, 1.0, ACCENT, 2)
        centred(c, "H: history trend    Q: quit", 550, 0.6, GREY, 1)

    def _draw_test_frame(self, c, frame, big: str, sub: str, colour) -> None:
        kind, hand = self.test
        text(c, f"Test {self.idx + 1}/4  -  {hand.upper()} hand {kind}", (20, 100), 1.0)
        if kind == "tapping":
            if frame is not None:
                paste(c, frame, 20, 120, 720, 540)
            else:
                cv2.rectangle(c, (20, 120), (740, 660), PANEL, -1)
                text(c, "Camera unavailable - press S to skip", (120, 400), 0.8, GREY)
        else:
            text(c, "Live acceleration (gravity removed)", (20, 150), 0.6, GREY, 1)
            trace(c, self.recorder.recent_magnitude(), 20, 165, 720, 300, vmax=20.0)
            if frame is not None:
                paste(c, frame, 20, 480, 240, 180)
        x = 780
        for i, line in enumerate(INSTRUCTIONS[kind]):
            text(c, line.format(hand=hand.upper()), (x, 160 + i * 34), 0.65)
        text(c, big, (x, 420), 3.0 if len(big) < 4 else 1.6, colour, 5)
        text(c, sub, (x, 480), 0.75, GREY)

    def _draw_ready(self, c, now, frame) -> None:
        self._draw_test_frame(c, frame, "Ready?", "SPACE to start   S to skip", ACCENT)

    def _draw_countdown(self, c, now, frame) -> None:
        n = max(1, COUNTDOWN_S - int(now - self.t_state))
        self._draw_test_frame(c, frame, str(n), "Get ready...", ACCENT)

    def _draw_recording(self, c, now, frame) -> None:
        left = max(0.0, self.seconds - (now - self.t_state))
        self._draw_test_frame(c, frame, f"{left:4.1f}s", "Recording...", (60, 60, 230))
        frac = 1 - left / self.seconds
        cv2.rectangle(c, (780, 520), (1240, 545), PANEL, -1)
        cv2.rectangle(c, (780, 520), (780 + int(460 * frac), 545), (60, 60, 230), -1)

    def _draw_done(self, c, now, frame) -> None:
        r = self.results.get(self.test)
        self._draw_test_frame(c, frame, "Done", "", (90, 200, 90))
        if r is not None:
            col = SCORE_COLOURS[r.result.score]
            text(c, f"Score: {'-' if r.result.score is None else r.result.score}",
                 (780, 560), 1.0, col)

    def _draw_results(self, c, now, frame) -> None:
        text(c, "Results", (20, 100), 1.1, ACCENT)
        y = 130
        for kind, hand in TESTS:
            r = self.results.get((kind, hand))
            res = r.result if r else ScoreResult(None, ("Not run",))
            col = SCORE_COLOURS[res.score]
            cv2.rectangle(c, (20, y), (W - 20, y + 108), PANEL, -1)
            cv2.rectangle(c, (20, y), (28, y + 108), col, -1)
            text(c, "-" if res.score is None else str(res.score), (45, y + 75), 2.2, col, 5)
            label = "Rest tremor (3.17)" if kind == "tremor" else "Finger tapping (3.4)"
            text(c, f"{hand} - {label}", (130, y + 30), 0.75)
            for i, reason in enumerate(res.reasons[:3]):
                text(c, "- " + textwrap.shorten(reason, 110), (130, y + 58 + i * 22), 0.55, GREY, 1)
            y += 118
        if self.flags:
            text(c, "ASYMMETRY: " + textwrap.shorten(" | ".join(self.flags), 130),
                 (20, y + 22), 0.6, (0, 140, 255), 2)
        text(c, "SPACE/R: new session    H: trend    Q: quit", (20, H - 52), 0.6, GREY, 1)
        if self.save_error:
            text(c, self.save_error, (560, H - 52), 0.6, (60, 60, 230), 2)


# --- main -----------------------------------------------------------------------
def open_camera(index: int) -> Optional[cv2.VideoCapture]:
    cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap = cv2.VideoCapture(index)
    if not cap.isOpened():
        logging.warning("Camera %d unavailable - tapping tests will be skipped", index)
        return None
    return cap


def make_device(args):
    if args.sim:
        from device import MockDevice
        return MockDevice()
    from device import SerialDevice
    return SerialDevice(args.port)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--sim", action="store_true", help="fake IMU, no Arduino needed")
    p.add_argument("--port", help="serial port, e.g. COM5 (default: auto-detect)")
    p.add_argument("--camera", type=int, default=0)
    p.add_argument("--seconds", type=float, default=10.0, help="recording length per test")
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    device = make_device(args)
    cap = open_camera(args.camera)
    tracker = None
    if cap is not None:
        from tapping import TappingTracker
        tracker = TappingTracker()
    app = App(device, tracker, args.seconds, sim=args.sim)
    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW, W, H)
    key = -1
    try:
        while app.running:
            frame = None
            if cap is not None:
                ok, raw = cap.read()
                frame = cv2.flip(raw, 1) if ok else None
            cv2.imshow(WINDOW, app.tick(key, time.monotonic(), frame))
            key = cv2.waitKey(1)
            if cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
                break
    finally:
        device.led("OFF")
        device.lcd("NeuroCheck", "Goodbye")
        device.close()
        if cap is not None:
            cap.release()
        if tracker:
            tracker.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
