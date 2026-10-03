"""HUD chrome: header bar, telemetry bar, boot screen, signal-quality meter.

Every number drawn here is measured; simulated sources are labeled SIM.
"""
from __future__ import annotations

import time
from datetime import datetime
from typing import Optional

import cv2
import numpy as np

import ui
from quality import MAX_DROPPED, MIN_CONFIDENCE, MIN_DETECTED, SignalQuality

VERSION = "NEUROCHECK v1.0"
HEADER_H = 50
TELEMETRY_H = 28
HINT_Y = ui.H - TELEMETRY_H - 8           # baseline of the disclaimer / key-hint line
STATUS_COLOURS = {"OK": ui.OK, "WARN": ui.WARN, "FAIL": ui.REC, "SENT": ui.ACCENT,
                  "SIM": ui.WARN, "PENDING": ui.DIM}


def header(c, session_id: str, hand: Optional[str], live: bool, status: str,
           status_ok: bool) -> None:
    cv2.rectangle(c, (0, 0), (ui.W, HEADER_H), ui.PANEL, -1)
    cv2.line(c, (0, HEADER_H), (ui.W, HEADER_H), ui.ACCENT, 1)
    ui.text(c, VERSION, (18, 33), 0.75, ui.ACCENT, 2)
    x = 18 + ui.text_width(VERSION, 0.75, 2) + 28
    for label, value in (("SESSION", session_id), ("HAND", hand or "-")):
        ui.text(c, label, (x, 21), 0.38, ui.DIM)
        ui.text(c, value, (x, 41), 0.55, ui.WHITE)
        x += max(ui.text_width(value, 0.55), ui.text_width(label, 0.38)) + 28
    clock = datetime.now().strftime("%H:%M:%S")
    xr = ui.W - 18 - ui.text_width(clock, 0.7, 2)
    ui.text(c, clock, (xr, 34), 0.7, ui.WHITE, 2)
    xr -= 12 + ui.text_width("LIVE" if live else "SIM", 0.45, 2) + 14
    ui.badge(c, "LIVE" if live else "SIM", xr, 13, ui.OK if live else ui.WARN)
    st = status[:46]
    ui.text(c, st, (xr - 16 - ui.text_width(st, 0.45), 33), 0.45, ui.OK if status_ok else ui.REC)


def banner(c, msg: str, colour) -> None:
    cv2.rectangle(c, (0, HEADER_H + 1), (ui.W, HEADER_H + 28), colour, -1)
    ui.centred(c, msg, HEADER_H + 21, 0.5, ui.WHITE, 2)


def _fmt(v: Optional[float], fmt: str) -> str:
    return "--" if v is None else format(v, fmt)


def telemetry(c, hands, device, sim_hand: bool, sim_dev: bool) -> None:
    """Bottom bar: camera FPS, inference ms, serial link + packets/s, tilt state, mode."""
    y0 = ui.H - TELEMETRY_H
    cv2.rectangle(c, (0, y0), (ui.W, ui.H), (22, 19, 16), -1)
    cv2.line(c, (0, y0), (ui.W, y0), ui.BORDER, 1)
    tel = getattr(hands, "telemetry", None)
    fps = tel.fps if tel else None
    inf = tel.inference_ms if (tel and fps is not None) else None   # only while the camera runs
    link = ("SIM" if sim_dev else ("UP" if device.connected else "DOWN"))
    items = [
        ("CAM", f"{_fmt(fps, '5.1f')} fps" + (" SIM" if sim_hand else ""), ui.WARN if sim_hand else ui.WHITE),
        ("INFER", "SIM" if sim_hand else f"{_fmt(inf, '5.1f')} ms", ui.WARN if sim_hand else ui.WHITE),
        ("SERIAL", f"{link} {device.packets_per_sec:4.0f} pkt/s",
         ui.WARN if sim_dev else (ui.OK if device.connected else ui.REC)),
        ("TILT", ("?" if device.state is None or not getattr(device, "state_fresh", True)
                  else str(device.state)) + (" SIM" if sim_dev else ""),
         ui.WARN if sim_dev else ui.WHITE),
        ("MODE", "SIMULATION" if (sim_hand or sim_dev) else "LIVE",
         ui.WARN if (sim_hand or sim_dev) else ui.OK),
    ]
    x = 14
    for label, value, colour in items:
        ui.text(c, label, (x, ui.H - 9), 0.4, ui.DIM)
        x += ui.text_width(label, 0.4) + 8
        ui.text(c, value, (x, ui.H - 9), 0.45, colour)
        x += ui.text_width(value, 0.45) + 30


def footer_line(c, hints: str) -> None:
    ui.text(c, "Demo thresholds - tracking aid, NOT a diagnosis", (18, HINT_Y), 0.45, ui.WARN)
    ui.text(c, hints, (ui.W - 18 - ui.text_width(hints, 0.42), HINT_Y), 0.42, ui.GREY)


def boot_screen(c, checks, now: float) -> None:
    ui.panel(c, 140, 90, 1000, 540)
    ui.text(c, "SYSTEM SELF-CHECK", (175, 140), 0.9, ui.ACCENT, 2)
    ui.text(c, "every line is a live check; hardware lines show SIM in simulation",
            (175, 168), 0.42, ui.DIM)
    y = 215
    for line in list(checks.lines):
        colour = STATUS_COLOURS.get(line.status, ui.WHITE)
        ui.text(c, line.name, (175, y), 0.55, ui.WHITE)
        dots = "." * max(2, 34 - len(line.name))
        ui.text(c, dots, (175 + ui.text_width(line.name, 0.55) + 6, y), 0.55, ui.DIM)
        ui.text(c, f"[{line.status}]", (640, y), 0.55, colour, 2)
        ui.text(c, line.detail[:44], (760, y), 0.45, ui.GREY)
        y += 42
    if not checks.done:
        if int(now * 3) % 2:
            ui.text(c, "_", (175, y), 0.55, ui.ACCENT, 2)
    else:
        fails = checks.failures
        msg = "ALL CHECKS COMPLETE" if not fails else f"{fails} CHECK(S) FAILED - you can still continue"
        ui.text(c, msg, (175, 600), 0.6, ui.OK if not fails else ui.REC, 2)
        ui.text(c, "SPACE to continue", (860, 600), 0.6, ui.ACCENT, 2)


def quality_meter(c, q: Optional[SignalQuality], x: int, y: int, w: int, sim: bool) -> None:
    """SIGNAL QUALITY panel: three measured bars, amber/red when out of range."""
    ui.panel(c, x, y, w, 100)
    ui.text(c, "SIGNAL QUALITY" + ("  (SIM)" if sim else ""), (x + 12, y + 22), 0.45, ui.ACCENT, 2)
    if q is None or q.frames == 0:
        ui.text(c, "waiting for frames...", (x + 12, y + 60), 0.5, ui.DIM)
        return
    rows = [("HAND FOUND", q.detected, f"{q.detected:4.0%}", q.detected >= MIN_DETECTED),
            ("HANDEDNESS", q.confidence or 0.0, "--" if q.confidence is None else f"{q.confidence:.2f}",
             True),                                   # info only (see quality.py)
            ("DROPPED", min(1.0, q.dropped / max(MAX_DROPPED * 2, 1e-9)),
             f"{q.dropped_frames} ({q.dropped:.0%})", q.dropped <= MAX_DROPPED)]
    for i, (label, frac, value, good) in enumerate(rows):
        yy = y + 36 + i * 20
        ui.text(c, label, (x + 12, yy + 10), 0.4, ui.GREY)
        ui.bar(c, x + 130, yy, w - 260, 10, frac, ui.OK if good else ui.WARN)
        ui.text(c, value, (x + w - 118, yy + 11), 0.45, ui.WHITE if good else ui.WARN)
    if q.low:
        ui.text(c, "LOW SIGNAL QUALITY", (x + w - 200, y + 22), 0.45, ui.WARN, 2)


def scan_line(c, x: int, y: int, w: int, h: int, now: float, period: float = 2.0) -> None:
    """Horizontal scan line sweeping down the camera view while recording."""
    pos = (now % period) / period
    yy = y + int(pos * h)
    overlay = c[y:y + h, x:x + w]
    for k, alpha in ((0, 0.9), (3, 0.45), (7, 0.25), (12, 0.12)):
        row = yy - y - k
        if 0 <= row < h:
            overlay[row] = (overlay[row] * (1 - alpha) + np.array(ui.ACCENT) * alpha).astype(np.uint8)


def session_id() -> str:
    return datetime.now().strftime("NC-%y%m%d-%H%M%S")
