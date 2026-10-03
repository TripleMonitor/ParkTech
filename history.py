"""Session history: sessions.csv, fake seed data, trend chart (as a BGR image for OpenCV)."""
from __future__ import annotations

import csv
import logging
import os
from datetime import datetime, timedelta
from typing import Optional

import numpy as np

log = logging.getLogger(__name__)

DEFAULT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sessions.csv")
FIELDS = ["timestamp", "session_id", "seeded",
          "tremor_R", "tremor_L", "tremor_R_cm", "tremor_L_cm", "tremor_R_hz", "tremor_L_hz",
          "tap_R", "tap_L", "tap_R_rate", "tap_L_rate", "tap_R_amp", "tap_L_amp",
          "flip_R", "flip_L", "flip_R_rate", "flip_L_rate",
          "asymmetry"]


def _fmt(v) -> str:
    if v is None:
        return ""
    if isinstance(v, (float, np.floating)):
        return f"{float(v):.3f}"
    return str(v)


def save_session(row: dict, path: str = DEFAULT_PATH) -> dict:
    """Append one session (unknown keys ignored, missing ones blank). Returns the row written."""
    full = {k: "" for k in FIELDS}
    full.update({k: v for k, v in row.items() if k in FIELDS})
    if not full["timestamp"]:
        full["timestamp"] = datetime.now().isoformat(timespec="seconds")
    new_file = not os.path.exists(path) or os.path.getsize(path) == 0
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        if new_file:
            w.writeheader()
        out = {k: _fmt(v) for k, v in full.items()}
        w.writerow(out)
    return out


def load_sessions(path: str = DEFAULT_PATH) -> list[dict]:
    if not os.path.exists(path):
        return []
    try:
        # utf-8-sig: Excel re-saves CSVs with a BOM, which would corrupt the first header
        with open(path, newline="", encoding="utf-8-sig") as f:
            return list(csv.DictReader(f))
    except (OSError, csv.Error, UnicodeDecodeError) as exc:
        log.error("Could not read %s: %s", path, exc)
        return []


def seed_history(path: str = DEFAULT_PATH, days: int = 7,
                 now: Optional[datetime] = None, force: bool = False) -> list[dict]:
    """Append `days` daily fake sessions ending yesterday; the RIGHT hand slowly worsens.

    Does nothing if seeded sessions already exist (unless force), so running
    --seed-history repeatedly doesn't stack duplicate weeks.
    """
    if not force and any(r.get("seeded") == "1" for r in load_sessions(path)):
        log.info("%s already has seeded sessions - not adding more", path)
        return []
    from flipping_analysis import FlippingFeatures
    from scoring import score_flipping, score_tapping, score_tremor
    from tapping_analysis import TappingFeatures
    from tremor_analysis import TremorFeatures

    now = now or datetime.now()
    rows = []
    for i in range(days):
        k = i / max(1, days - 1)                       # 0 -> 1 over the week
        when = (now - timedelta(days=days - i)).replace(hour=10, minute=0, second=0, microsecond=0)
        sides = {"R": (0.3 + 1.5 * k, 3.4 - 1.6 * k, 0.95 - 0.4 * k, 2.4 - 1.1 * k),
                 "L": (0.05, 3.5, 0.95, 2.4)}
        row = {"timestamp": when.isoformat(timespec="seconds"), "seeded": 1}
        for side, (cm, rate, amp, flips) in sides.items():
            clear = cm >= 0.1
            tf = TremorFeatures(True, 10.0, 1.0, 0.06, 5.0, cm, clear, 10.0, 80.0)
            pf = TappingFeatures(int(rate * 10), rate, amp, 0.1 + 0.3 * k * (side == "R"),
                                 0.1, 0, 1.0, 10.0)
            ff = FlippingFeatures(int(flips * 20), int(flips * 10), flips, 0.1,
                                  0.35 * k * (side == "R"), 0, 10.0, 0.3)
            row.update({f"tremor_{side}": score_tremor(tf).score,
                        f"tremor_{side}_cm": cm if clear else 0.0, f"tremor_{side}_hz": 5.0,
                        f"tap_{side}": score_tapping(pf).score,
                        f"tap_{side}_rate": rate, f"tap_{side}_amp": amp,
                        f"flip_{side}": score_flipping(ff).score, f"flip_{side}_rate": flips})
        rows.append(save_session(row, path))
    return rows


def _num(v) -> Optional[float]:
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _when(row: dict, i: int) -> datetime:
    try:
        return datetime.fromisoformat(row.get("timestamp", ""))
    except ValueError:
        return datetime(2000, 1, 1) + timedelta(days=i)


def trend_image(sessions: list[dict], width: int = 1280, height: int = 720) -> np.ndarray:
    """2x3 trend chart (scores + raw values, R vs L). Returns BGR uint8 (height, width, 3)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt

    dpi = 100
    fig, axes = plt.subplots(2, 3, figsize=(width / dpi, height / dpi), dpi=dpi)
    panels = [("tremor_{}", "Tremor score (0-4)", (-0.3, 4.3)),
              ("tap_{}", "Finger tapping score (0-4)", (-0.3, 4.3)),
              ("flip_{}", "Hand flipping score (0-4)", (-0.3, 4.3)),
              ("tremor_{}_cm", "Tremor amplitude (cm)", None),
              ("tap_{}_rate", "Taps per second", None),
              ("flip_{}_rate", "Full flips per second", None)]
    xs = [_when(s, i) for i, s in enumerate(sessions)]
    for ax, (key, title, ylim) in zip(axes.flat, panels):
        for side, style, colour in (("R", "o-", "#d62728"), ("L", "s--", "#1f77b4")):
            pts = [(x, _num(s.get(key.format(side)))) for x, s in zip(xs, sessions)]
            pts = [(x, y) for x, y in pts if y is not None]
            if pts:
                ax.plot(*zip(*pts), style, color=colour, lw=2, ms=6,
                        label="Right" if side == "R" else "Left")
        ax.set_title(title, fontsize=12)
        if ylim:
            ax.set_ylim(*ylim)
            ax.set_yticks(range(5))
        ax.grid(alpha=0.3)
        if xs:
            ax.set_xlim(min(xs) - timedelta(hours=12), max(xs) + timedelta(hours=12))
        locator = mdates.AutoDateLocator(minticks=3, maxticks=8)
        ax.xaxis.set_major_locator(locator)
        ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))
        if ax.get_legend_handles_labels()[0]:
            ax.legend(loc="best", fontsize=9)
    if not sessions:
        fig.text(0.5, 0.5, "No sessions yet - run one, or start with --seed-history",
                 ha="center", fontsize=18)
    fig.suptitle("NeuroCheck trend  (demo thresholds - tracking aid, not a diagnosis)", fontsize=15)
    fig.tight_layout()
    fig.canvas.draw()
    rgba = np.asarray(fig.canvas.buffer_rgba())
    plt.close(fig)
    img = np.ascontiguousarray(rgba[:, :, 2::-1])
    if img.shape[:2] != (height, width):
        import cv2
        img = cv2.resize(img, (width, height))
    return img
