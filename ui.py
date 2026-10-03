"""Drawing helpers for the single OpenCV window: palette, text, cards, charts."""
from __future__ import annotations

from typing import Optional, Sequence

import cv2
import numpy as np

W, H = 1280, 720
FONT = cv2.FONT_HERSHEY_DUPLEX

# BGR palette
BG = (32, 27, 24)
PANEL = (56, 49, 44)
PANEL_HI = (72, 63, 57)
WHITE = (245, 245, 245)
GREY = (165, 160, 155)
DIM = (120, 115, 110)
ACCENT = (255, 190, 60)        # blue-ish cyan
REC = (70, 70, 235)            # red
OK = (100, 205, 100)
WARN = (0, 200, 255)
AXIS_COLOURS = ((80, 80, 240), (90, 210, 90), (240, 160, 60))   # x red, y green, z blue
SCORE_COLOURS = {0: (100, 205, 100), 1: (60, 215, 190), 2: (0, 200, 255),
                 3: (0, 140, 255), 4: (70, 70, 235), None: (130, 125, 120)}


def text(img, s: str, org, scale: float = 0.7, colour=WHITE, thick: int = 1) -> None:
    cv2.putText(img, s, (int(org[0]), int(org[1])), FONT, scale, colour, thick, cv2.LINE_AA)


def text_width(s: str, scale: float, thick: int = 1) -> int:
    return cv2.getTextSize(s, FONT, scale, thick)[0][0]


def centred(img, s: str, y: int, scale: float = 1.0, colour=WHITE, thick: int = 1,
            x0: int = 0, x1: Optional[int] = None) -> None:
    x1 = img.shape[1] if x1 is None else x1
    text(img, s, ((x0 + x1 - text_width(s, scale, thick)) // 2, y), scale, colour, thick)


def wrap(s: str, width_px: int, scale: float) -> list[str]:
    """Greedy word wrap using the real rendered width of each candidate line."""
    lines: list[str] = []
    cur = ""
    for word in s.split():
        trial = f"{cur} {word}".strip()
        if cur and text_width(trial, scale) > width_px:
            lines.append(cur)
            cur = word
        else:
            cur = trial
    lines.append(cur)
    return lines


def paste(canvas, img, x: int, y: int, w: int, h: int) -> None:
    canvas[y:y + h, x:x + w] = cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)


def panel(canvas, x: int, y: int, w: int, h: int, colour=PANEL) -> None:
    cv2.rectangle(canvas, (x, y), (x + w, y + h), colour, -1)


def blank() -> np.ndarray:
    return np.full((H, W, 3), BG, np.uint8)


def accel_chart(canvas, samples: Sequence, x: int, y: int, w: int, h: int,
                seconds: float = 3.0, fs: float = 100.0) -> None:
    """Live chart of the last few seconds of x/y/z acceleration with gravity removed."""
    panel(canvas, x, y, w, h)
    mid = y + h // 2
    cv2.line(canvas, (x, mid), (x + w, mid), PANEL_HI, 1)
    label = "acceleration, gravity removed (m/s^2)"
    text(canvas, label, (x + 10, y + 22), 0.5, GREY)
    lx = x + 30 + text_width(label, 0.5)
    for i, c in enumerate(AXIS_COLOURS):
        text(canvas, "xyz"[i], (lx + i * 24, y + 22), 0.55, c, 2)
    k = int(seconds * fs)
    if len(samples) < 2:
        text(canvas, "waiting for sensor data...", (x + 10, mid - 10), 0.7, DIM)
        return
    arr = np.asarray(samples[-k:], dtype=float)[:, 1:4]
    dyn = arr - arr.mean(axis=0)
    scale = max(2.0, float(np.abs(dyn).max()) * 1.1)
    xs = x + np.linspace(0, w, k)[-len(dyn):].astype(int)
    for i, c in enumerate(AXIS_COLOURS):
        ys = (mid - dyn[:, i] / scale * (h / 2 - 30)).astype(int)
        cv2.polylines(canvas, [np.column_stack([xs, ys]).astype(np.int32)], False, c, 2, cv2.LINE_AA)
    text(canvas, f"+/-{scale:.1f}", (x + w - 90, y + h - 10), 0.5, DIM)


def spectrum_image(freqs: np.ndarray, power: np.ndarray, peak_hz: float, disp_cm: float,
                   score: Optional[int], width: int, height: int) -> np.ndarray:
    """Matplotlib spectrum: 4-6 Hz band shaded, 3-8 Hz band edges, peak marked. BGR image."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    dpi = 100
    fig, ax = plt.subplots(figsize=(width / dpi, height / dpi), dpi=dpi)
    fig.patch.set_facecolor("#181b20")
    ax.set_facecolor("#2c3138")
    if len(freqs):
        m = freqs <= 15
        amp = np.sqrt(power[m])
        ax.plot(freqs[m], amp, color="#3cbeff", lw=2)
        ax.axvspan(4, 6, color="#ff9f40", alpha=0.25, label="Parkinson's range 4-6 Hz")
        for edge in (3, 8):
            ax.axvline(edge, color="#aaaaaa", ls="--", lw=1)
        pk = np.sqrt(np.interp(peak_hz, freqs, power))
        ax.plot([peak_hz], [pk], "o", color="#ff4d4d", ms=10)
        ax.annotate(f"{peak_hz:.1f} Hz  {disp_cm:.2f} cm", (peak_hz, pk), xytext=(10, -4),
                    textcoords="offset points", color="white", fontsize=13, va="top")
        ax.set_ylim(0, max(pk, float(amp.max())) * 1.25 + 1e-9)
    ax.set_xlim(0, 15)
    ax.set_xlabel("frequency (Hz)   dashed = 3-8 Hz tremor band", color="#cccccc")
    ax.set_ylabel("amplitude (m/s^2)", color="#cccccc")
    ax.tick_params(colors="#cccccc")
    if len(freqs):
        ax.legend(loc="upper right", facecolor="#2c3138", labelcolor="white", fontsize=10)
    else:
        ax.text(7.5, 0.5, "No sensor data", color="#ff6b6b", fontsize=20, ha="center")
    ax.set_title(f"Tremor spectrum  ->  score {'-' if score is None else score}",
                 color="white", fontsize=14)
    fig.tight_layout()
    fig.canvas.draw()
    rgba = np.asarray(fig.canvas.buffer_rgba())
    plt.close(fig)
    img = np.ascontiguousarray(rgba[:, :, 2::-1])
    return cv2.resize(img, (width, height)) if img.shape[:2] != (height, width) else img


def score_card(canvas, x: int, y: int, w: int, h: int, title: str,
               score: Optional[int], reasons: Sequence[str]) -> None:
    col = SCORE_COLOURS.get(score, SCORE_COLOURS[None])
    panel(canvas, x, y, w, h)
    cv2.rectangle(canvas, (x, y), (x + 8, y + h), col, -1)
    text(canvas, "-" if score is None else str(score), (x + 28, y + 88), 2.6, col, 4)
    text(canvas, title, (x + 110, y + 36), 0.75, WHITE, 1)
    yy = y + 68
    for reason in reasons:
        for i, line in enumerate(wrap(reason, w - 130, 0.52)):
            if yy > y + h - 10:
                return
            text(canvas, ("- " if i == 0 else "  ") + line, (x + 110, yy), 0.52, GREY)
            yy += 24
