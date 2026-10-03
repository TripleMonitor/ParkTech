"""Save sessions to sessions.csv and render a trend chart (as a BGR image for OpenCV)."""
from __future__ import annotations

import csv
import os
from datetime import datetime
from typing import Optional

import numpy as np

FIELDS = ["timestamp",
          "tremor_R", "tremor_L", "tremor_R_cm", "tremor_L_cm", "tremor_R_hz", "tremor_L_hz",
          "tap_R", "tap_L", "tap_R_rate", "tap_L_rate", "tap_R_amp", "tap_L_amp",
          "asymmetry"]
DEFAULT_PATH = "sessions.csv"


def save_session(row: dict, path: str = DEFAULT_PATH) -> dict:
    """Append one session. Missing fields are blank. Returns the row as written."""
    full = {k: "" for k in FIELDS}
    full.update({k: v for k, v in row.items() if k in FIELDS})
    full["timestamp"] = full["timestamp"] or datetime.now().isoformat(timespec="seconds")
    new_file = not os.path.exists(path)
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        if new_file:
            w.writeheader()
        w.writerow({k: _fmt(v) for k, v in full.items()})
    return full


def _fmt(v) -> str:
    if v is None:
        return ""
    if isinstance(v, float):
        return f"{v:.3f}"
    return str(v)


def load_sessions(path: str = DEFAULT_PATH) -> list[dict]:
    if not os.path.exists(path):
        return []
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _num(v: str) -> Optional[float]:
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def trend_image(sessions: list[dict], width: int = 1280, height: int = 720) -> np.ndarray:
    """Plot scores over sessions. Returns a BGR uint8 image."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    dpi = 100
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(width / dpi, height / dpi), dpi=dpi)
    x = list(range(1, len(sessions) + 1))
    series = [(ax1, "Tremor score (0-4)", "tremor"), (ax2, "Tapping score (0-4)", "tap")]
    for ax, title, key in series:
        for side, style in (("R", "o-"), ("L", "s--")):
            ys = [_num(s.get(f"{key}_{side}")) for s in sessions]
            pts = [(xi, y) for xi, y in zip(x, ys) if y is not None]
            if pts:
                ax.plot(*zip(*pts), style, label="Right" if side == "R" else "Left", lw=2)
        ax.set_title(title)
        ax.set_ylim(-0.3, 4.3)
        ax.set_yticks(range(5))
        ax.set_xlabel("Session #")
        ax.grid(alpha=0.3)
        if ax.get_legend_handles_labels()[0]:
            ax.legend(loc="upper left")
    if not sessions:
        fig.text(0.5, 0.5, "No sessions yet", ha="center", fontsize=20)
    fig.suptitle("NeuroCheck trend (demo thresholds - not a diagnosis)", fontsize=14)
    fig.tight_layout()
    fig.canvas.draw()
    rgba = np.asarray(fig.canvas.buffer_rgba())
    plt.close(fig)
    return np.ascontiguousarray(rgba[:, :, 2::-1])  # RGBA -> BGR
