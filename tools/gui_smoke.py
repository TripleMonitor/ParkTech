"""Scripted GUI run: real OpenCV window, real clock, --sim devices, keys injected.

Goes through the whole flow twice-ish (full session, trend, R restart from mid-test,
second partial session) for at least 60 s, saves one screenshot per screen into
screenshots/, and exits non-zero if any exception was caught.
Usage:  python tools/gui_smoke.py
"""
from __future__ import annotations

import logging
import os
import sys
import tempfile
import time

import cv2

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app import KEY_SPACE, App, run  # noqa: E402
from boot import BootChecks  # noqa: E402
from device import MockDevice  # noqa: E402
from history import seed_history  # noqa: E402
from tapping_tracker import FakeHand  # noqa: E402

OUT = "screenshots"
MIN_SECONDS = 60.0


class Script:
    """Decides which key to press based on the app state (like a patient user)."""

    def __init__(self, app: App, dev: MockDevice):
        self.app, self.dev = app, dev
        self.t0 = app.clock()
        self.last_press = 0.0
        self.sessions_done = 0
        self.restarted_mid_test = False
        self.log: list[str] = []

    def __call__(self, now: float) -> int:
        a = self.app
        if now - self.last_press < 0.8:          # let each screen show for a moment
            return -1
        key = -1
        if a.state == "boot":
            key = KEY_SPACE if a.boot.done and now - self.t0 > 1.0 else -1
        elif a.state == "welcome":
            key = KEY_SPACE if self.sessions_done < 2 else ord("q")
        elif a.state == "ready":
            key = KEY_SPACE
        elif a.state == "done":
            key = KEY_SPACE
        elif a.state == "results":
            key = ord("h") if now - a.t_state > 1.5 else -1
        elif a.state == "trend":
            if now - a.t_state > 1.5:
                self.sessions_done += 1
                key = KEY_SPACE
        elif a.state == "recording" and self.sessions_done == 1 and not self.restarted_mid_test \
                and a.idx == 4 and now - a.t_state > 2:
            self.restarted_mid_test = True     # exercise R from the middle of a test
            self.sessions_done += 1
            key = ord("r")
        if key == ord("q") and now - self.t0 < MIN_SECONDS:
            return -1                          # keep running until the minimum time
        if key >= 0:
            self.last_press = now
            self.log.append(f"{now - self.t0:6.1f}s  {a.state:10s} -> {chr(key) if key > 32 else 'SPACE'}")
        return key


def main() -> int:
    logging.basicConfig(level=logging.WARNING)
    os.makedirs(OUT, exist_ok=True)
    tmp = tempfile.mkdtemp()
    hist = os.path.join(tmp, "sessions.csv")
    seed_history(hist)
    dev = MockDevice()
    hands = FakeHand()
    app = App(dev, hands, seconds=10.0, history_path=hist,
              boot=BootChecks(dev, hands).start())
    script = Script(app, dev)
    seen: set[str] = set()

    def on_frame(a: App, canvas) -> None:
        name = a.state if a.state in ("boot", "welcome", "results", "trend") else \
            f"{a.state}_{a.test[0]}_{a.test[1]}"
        if a.state == "recording" and a.clock() - a.t_state < 6:
            return                               # wait for a well-filled live chart
        if a.state == "boot" and not a.boot.done:
            return
        if name not in seen:
            seen.add(name)
            cv2.imwrite(os.path.join(OUT, f"gui_{len(seen):02d}_{name}.png"), canvas)

    t0 = app.clock()
    errors = run(app, hands, dev, key_script=script, on_frame=on_frame)
    elapsed = app.clock() - t0
    print("\n".join(script.log))
    print(f"\nran {elapsed:.1f}s, screens captured: {len(seen)}, caught exceptions: {errors}, "
          f"R mid-test: {script.restarted_mid_test}")
    ok = errors == 0 and elapsed >= MIN_SECONDS and script.restarted_mid_test
    print("GUI SMOKE:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
