"""HUD drawing helpers for the single OpenCV window: palette, mono text, panels, charts.

Text is rendered with DejaVu Sans Mono (bundled with matplotlib) through a cached PIL
renderer, so every number is monospace and layouts don't jitter as values change.
"""
from __future__ import annotations

import functools
import os
from typing import Optional, Sequence

import cv2
import numpy as np

W, H = 1280, 720
FONT = cv2.FONT_HERSHEY_DUPLEX          # fallback only

# BGR palette: near-black, one accent (cyan), green/amber/red for status only
BG = (14, 12, 10)
PANEL = (30, 26, 22)
PANEL_HI = (48, 42, 36)
BORDER = (78, 70, 60)
WHITE = (236, 236, 232)
GREY = (160, 156, 150)
DIM = (110, 106, 100)
ACCENT = (230, 205, 40)        # cyan
REC = (70, 70, 235)            # red
OK = (110, 210, 90)            # green
WARN = (40, 180, 255)          # amber
AXIS_COLOURS = ((80, 80, 240), (110, 210, 90), (230, 205, 40))
SCORE_COLOURS = {0: OK, 1: OK, 2: WARN, 3: REC, 4: REC, None: DIM}   # same as the LED
BRACKET = 12
PX_PER_SCALE = 30              # text "scale" 1.0 ~ 30 px font


@functools.lru_cache(maxsize=8)
def _font(size: int, bold: bool):
    try:
        from PIL import ImageFont
        import matplotlib
        name = "DejaVuSansMono-Bold.ttf" if bold else "DejaVuSansMono.ttf"
        path = os.path.join(matplotlib.get_data_path(), "fonts", "ttf", name)
        return ImageFont.truetype(path, size)
    except Exception:
        return None


@functools.lru_cache(maxsize=4096)
def _render(s: str, size: int, bold: bool):
    """(alpha mask uint8, ascent) for a string, or None if no TrueType font."""
    font = _font(size, bold)
    if font is None:
        return None
    from PIL import Image, ImageDraw
    ascent, descent = font.getmetrics()
    w = max(1, int(np.ceil(font.getlength(s))))
    img = Image.new("L", (w, ascent + descent), 0)
    ImageDraw.Draw(img).text((0, 0), s, font=font, fill=255)
    return np.asarray(img), ascent


def _size(scale: float) -> int:
    return max(8, int(round(scale * PX_PER_SCALE)))


def text(img, s: str, org, scale: float = 0.7, colour=WHITE, thick: int = 1) -> None:
    """Draw text with its baseline at org (like cv2.putText). thick>=2 -> bold."""
    r = _render(str(s), _size(scale), thick >= 2)
    if r is None:
        cv2.putText(img, s, (int(org[0]), int(org[1])), FONT, scale, colour, thick, cv2.LINE_AA)
        return
    mask, ascent = r
    x0, y0 = int(org[0]), int(org[1]) - ascent
    h, w = mask.shape
    xa, ya = max(x0, 0), max(y0, 0)
    xb, yb = min(x0 + w, img.shape[1]), min(y0 + h, img.shape[0])
    if xa >= xb or ya >= yb:
        return
    a = mask[ya - y0:yb - y0, xa - x0:xb - x0, None].astype(np.float32) / 255.0
    roi = img[ya:yb, xa:xb].astype(np.float32)
    img[ya:yb, xa:xb] = (roi * (1 - a) + np.array(colour, np.float32) * a).astype(np.uint8)


def text_width(s: str, scale: float, thick: int = 1) -> int:
    font = _font(_size(scale), thick >= 2)
    if font is None:
        return cv2.getTextSize(s, FONT, scale, thick)[0][0]
    return int(np.ceil(font.getlength(str(s))))


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


def panel(canvas, x: int, y: int, w: int, h: int, colour=PANEL, brackets: bool = True,
          accent=ACCENT) -> None:
    """Filled panel with a thin border and HUD corner brackets."""
    cv2.rectangle(canvas, (x, y), (x + w, y + h), colour, -1)
    if not brackets:
        return
    cv2.rectangle(canvas, (x, y), (x + w, y + h), BORDER, 1)
    b = min(BRACKET, w // 4, h // 4)
    for cx, cy, dx, dy in ((x, y, 1, 1), (x + w, y, -1, 1), (x, y + h, 1, -1), (x + w, y + h, -1, -1)):
        cv2.line(canvas, (cx, cy), (cx + dx * b, cy), accent, 2)
        cv2.line(canvas, (cx, cy), (cx, cy + dy * b), accent, 2)


def bar(canvas, x: int, y: int, w: int, h: int, frac: float, colour) -> None:
    cv2.rectangle(canvas, (x, y), (x + w, y + h), PANEL_HI, -1)
    cv2.rectangle(canvas, (x, y), (x + int(w * max(0.0, min(1.0, frac))), y + h), colour, -1)


def badge(canvas, s: str, x: int, y: int, colour, scale: float = 0.45) -> int:
    """Outlined label box with top-left at (x, y). Returns its width."""
    w = text_width(s, scale, 2) + 14
    h = _size(scale) + 8
    cv2.rectangle(canvas, (x, y), (x + w, y + h), colour, 1)
    text(canvas, s, (x + 7, y + h - 6), scale, colour, 2)
    return w


@functools.lru_cache(maxsize=1)
def _background() -> np.ndarray:
    img = np.full((H, W, 3), BG, np.uint8)
    for x in range(0, W, 40):                      # faint HUD grid
        img[:, x] = (20, 18, 15)
    for y in range(0, H, 40):
        img[y, :] = (20, 18, 15)
    return img


def blank() -> np.ndarray:
    return _background().copy()


def fingertip_chart(canvas, times: Sequence[float], xy_cm: Sequence, now: float,
                    x: int, y: int, w: int, h: int, seconds: float = 3.0) -> None:
    """Live chart of the last few seconds of fingertip x/y movement (cm, mean removed)."""
    panel(canvas, x, y, w, h)
    mid = y + h // 2
    cv2.line(canvas, (x, mid), (x + w, mid), PANEL_HI, 1)
    label = "index fingertip movement (cm)"
    text(canvas, label, (x + 10, y + 22), 0.5, GREY)
    lx = x + 30 + text_width(label, 0.5)
    for i, c in enumerate(AXIS_COLOURS[:2]):
        text(canvas, "xy"[i], (lx + i * 24, y + 22), 0.55, c, 2)
    pts = [(t, p) for t, p in zip(times, xy_cm) if p is not None and now - t <= seconds]
    if len(pts) < 2:
        text(canvas, "waiting for the hand...", (x + 10, mid - 10), 0.7, DIM)
        return
    t = np.array([p[0] for p in pts])
    xy = np.array([p[1] for p in pts], dtype=float)
    xy = xy - xy.mean(axis=0)
    scale = max(0.5, float(np.abs(xy).max()) * 1.1)
    xs = (x + w * (1 - (now - t) / seconds)).astype(int)
    for i, c in enumerate(AXIS_COLOURS[:2]):
        ys = (mid - xy[:, i] / scale * (h / 2 - 30)).astype(int)
        cv2.polylines(canvas, [np.column_stack([xs, ys]).astype(np.int32)], False, c, 2, cv2.LINE_AA)
    text(canvas, f"+/-{scale:.2f} cm", (x + w - 130, y + h - 10), 0.5, DIM)


def flip_indicator(canvas, palm_down: Optional[bool], x: int, y: int, w: int, h: int) -> None:
    """Big PALM DOWN / PALM UP box driven by the tilt switch."""
    if palm_down is None:
        colour, label = PANEL_HI, "SWITCH ?"
    else:
        colour, label = ((150, 95, 40), "PALM DOWN") if palm_down else ((40, 140, 200), "PALM UP")
    panel(canvas, x, y, w, h, colour)
    centred(canvas, label, y + h // 2 + 20, 1.9, WHITE, 3, x, x + w)


def flip_timeline(canvas, flip_times: Sequence[float], seconds: float, now_s: Optional[float],
                  x: int, y: int, w: int, h: int) -> None:
    """Timeline 0..seconds with one tick per half-flip (taller tick = completes a full flip)."""
    panel(canvas, x, y, w, h)
    text(canvas, "flip timeline (s)", (x + 8, y + 18), 0.5, GREY)
    base = y + h - 18
    cv2.line(canvas, (x + 10, base), (x + w - 10, base), DIM, 1)
    span = w - 20
    for s in range(int(seconds) + 1):
        px = x + 10 + int(span * s / seconds)
        cv2.line(canvas, (px, base), (px, base + 5), DIM, 1)
        if s % 2 == 0:
            text(canvas, str(s), (px - 4, base + 16), 0.4, DIM)
    for i, ft in enumerate(flip_times):
        px = x + 10 + int(span * min(ft, seconds) / seconds)
        top = y + 28 if i % 2 else y + 45
        cv2.line(canvas, (px, base), (px, top), OK if i % 2 else ACCENT, 2)
    if now_s is not None:
        px = x + 10 + int(span * min(now_s, seconds) / seconds)
        cv2.line(canvas, (px, y + 24), (px, base), REC, 1)


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
    ax.set_ylabel("fingertip amplitude (cm)", color="#cccccc")
    ax.tick_params(colors="#cccccc")
    if len(freqs):
        ax.legend(loc="upper right", facecolor="#2c3138", labelcolor="white", fontsize=10)
    else:
        ax.text(7.5, 0.5, "No hand data", color="#ff6b6b", fontsize=20, ha="center")
    ax.set_title(f"Tremor spectrum  ->  score {'-' if score is None else score}",
                 color="white", fontsize=14)
    fig.tight_layout()
    fig.canvas.draw()
    rgba = np.asarray(fig.canvas.buffer_rgba())
    plt.close(fig)
    img = np.ascontiguousarray(rgba[:, :, 2::-1])
    return cv2.resize(img, (width, height)) if img.shape[:2] != (height, width) else img


def tilt_plot(canvas, events: Sequence, calib: Optional[int], now: float,
              x: int, y: int, w: int, h: int, seconds: float = 5.0) -> None:
    """Scrolling step plot of the tilt-switch state (raw reports, before Python debounce)."""
    panel(canvas, x, y, w, h)
    text(canvas, "tilt switch state (last 5 s)", (x + 10, y + 20), 0.45, GREY)
    hi, lo = y + 34, y + h - 14
    text(canvas, "UP", (x + w - 50, hi + 5), 0.4, DIM)
    text(canvas, "DOWN", (x + w - 50, lo + 5), 0.4, DIM)
    w = w - 60                                   # leave room for the labels
    pts = [(e.t, e.state) for e in events if now - e.t <= seconds]
    if not pts:
        return
    path = []
    prev = None
    for t, st in pts:
        px = x + int(w * (1 - (now - t) / seconds))
        py = hi if (calib is not None and st != calib) or (calib is None and st) else lo
        if prev is not None:
            path.append((px, prev))
        path.append((px, py))
        prev = py
    path.append((x + w, prev))
    cv2.polylines(canvas, [np.array(path, np.int32)], False, ACCENT, 2, cv2.LINE_AA)


def score_card(canvas, x: int, y: int, w: int, h: int, title: str,
               score: Optional[int], reasons: Sequence[str], compact: bool = False,
               flag: str = "") -> None:
    """Score + title + every reason (wrapped). compact=True for the 6-card results grid.
    flag (e.g. "LOW CONFIDENCE") is drawn as an amber badge in the top-right corner."""
    col = SCORE_COLOURS.get(score, SCORE_COLOURS[None])
    big, title_s, rs, lh, left = (1.9, 0.62, 0.46, 19, 78) if compact else (2.6, 0.75, 0.52, 24, 110)
    panel(canvas, x, y, w, h)
    cv2.rectangle(canvas, (x, y), (x + 8, y + h), col, -1)
    if flag:
        fw = text_width(flag, 0.4, 2) + 14
        badge(canvas, flag, x + w - fw - 8, y + 8, WARN, 0.4)
    text(canvas, "-" if score is None else str(score), (x + 22, y + (62 if compact else 88)),
         big, col, 4 if not compact else 3)
    text(canvas, title, (x + left, y + (26 if compact else 36)), title_s, WHITE, 1)
    yy = y + (50 if compact else 68)
    for reason in reasons:
        for i, line in enumerate(wrap(reason, w - left - 12 - text_width("- ", rs), rs)):
            if yy > y + h - 6:
                return
            text(canvas, ("- " if i == 0 else "  ") + line, (x + left, yy), rs, GREY)
            yy += lh


def brackets(canvas, x: int, y: int, w: int, h: int, colour=ACCENT, length: int = 18) -> None:
    """HUD corner brackets only (e.g. around the camera view)."""
    for cx, cy, dx, dy in ((x, y, 1, 1), (x + w, y, -1, 1), (x, y + h, 1, -1), (x + w, y + h, -1, -1)):
        cv2.line(canvas, (cx, cy), (cx + dx * length, cy), colour, 2)
        cv2.line(canvas, (cx, cy), (cx, cy + dy * length), colour, 2)


DIM_OK = (70, 140, 60)


def rotation_gauge(canvas, x: int, y: int, w: int, h: int, angle: Optional[float],
                   fusion, switch_half: int) -> None:
    """Half-dial 0..180 deg (palm down -> palm up) + camera/switch fusion status."""
    panel(canvas, x, y, w, h)
    text(canvas, "PALM ROTATION (camera)", (x + 12, y + 22), 0.42, ACCENT, 2)
    cx, cy, r = x + w // 2, y + 150, min(w // 2 - 30, 105)
    cv2.ellipse(canvas, (cx, cy), (r, r), 0, 180, 360, PANEL_HI, 10)
    for deg in (0, 45, 90, 135, 180):
        a = np.radians(180 + deg)
        p1 = (int(cx + (r - 14) * np.cos(a)), int(cy + (r - 14) * np.sin(a)))
        p2 = (int(cx + (r + 6) * np.cos(a)), int(cy + (r + 6) * np.sin(a)))
        cv2.line(canvas, p1, p2, DIM, 1)
    text(canvas, "DOWN", (cx - r - 22, cy + 22), 0.38, DIM)
    text(canvas, "UP", (cx + r - 6, cy + 22), 0.38, DIM)
    if angle is not None:
        cv2.ellipse(canvas, (cx, cy), (r, r), 0, 180, 180 + min(180.0, angle), ACCENT, 10)
        a = np.radians(180 + min(180.0, angle))
        cv2.line(canvas, (cx, cy), (int(cx + (r - 18) * np.cos(a)), int(cy + (r - 18) * np.sin(a))),
                 WHITE, 2, cv2.LINE_AA)
        txt = f"{angle:3.0f} deg"
        text(canvas, txt, (x + w - 16 - text_width(txt, 0.6, 2), y + 24), 0.6, WHITE, 2)
    else:
        text(canvas, "no hand", (x + w - 16 - text_width("no hand", 0.5), y + 24), 0.5, DIM)
    yy = cy + 48
    cam = "--" if fusion is None or not fusion.camera_ok else str(fusion.camera_half_flips)
    text(canvas, f"half-flips  switch {switch_half:3d}  camera {cam:>3}", (x + 12, yy), 0.42, GREY)
    if fusion is not None and fusion.camera_ok and fusion.median_amplitude_deg:
        text(canvas, f"median swing {fusion.median_amplitude_deg:4.0f} deg", (x + 12, yy + 20), 0.42, GREY)
    if fusion is None:
        status, col = "FUSION: waiting", DIM
    elif not fusion.camera_ok:
        status, col = "CAMERA LOST - switch only", WARN
    elif fusion.locked:
        status, col = "FUSION LOCKED \u2713", OK
    else:
        status, col = "FUSION MISMATCH - low confidence", WARN
    text(canvas, status, (x + 12, y + h - 12), 0.45, col, 2)


def rule_panel(canvas, x: int, y: int, w: int, h: int, title: str, explanation,
               flag: str = "") -> None:
    """WHY THIS SCORE: every rule with measured value vs threshold; fired rules lit."""
    panel(canvas, x, y, w, h)
    score = explanation.score
    col = SCORE_COLOURS.get(score, SCORE_COLOURS[None])
    text(canvas, "WHY THIS SCORE", (x + 16, y + 28), 0.55, ACCENT, 2)
    text(canvas, "demo thresholds", (x + 16, y + 48), 0.38, WARN)
    text(canvas, title, (x + 16, y + 74), 0.5, WHITE)
    text(canvas, "-" if score is None else str(score), (x + w - 70, y + 70), 2.0, col, 3)
    if flag:
        badge(canvas, flag, x + w - 200, y + 82, WARN, 0.38)
    yy = y + 118
    if not explanation.rules:
        text(canvas, "no rules evaluated:", (x + 16, yy), 0.45, GREY)
        yy += 22
    for r in explanation.rules:
        c = (REC if (score or 0) >= 3 else WARN) if r.fired else DIM_OK
        text(canvas, "\u2717" if r.fired else "\u2713", (x + 16, yy), 0.5, c, 2)
        text(canvas, r.label, (x + 40, yy), 0.45, c if r.fired else GREY)
        text(canvas, r.measured, (x + 250, yy), 0.45, WHITE if r.fired else GREY)
        text(canvas, r.threshold, (x + 340, yy), 0.42, c)
        yy += 26
    for i, line in enumerate(wrap(explanation.formula, w - 32, 0.4)[:3]):
        text(canvas, line, (x + 16, y + h - 46 + i * 17), 0.4, DIM)
