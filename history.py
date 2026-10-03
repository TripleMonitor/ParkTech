"""Session history: sessions.csv, seeded DEMO DATA, trend chart (BGR image for OpenCV).

Seeded rows have seeded=1 and are drawn hollow and labelled DEMO DATA on every chart.
"""
from __future__ import annotations

import csv
import logging
import os
from datetime import datetime, timedelta
from typing import Optional

import numpy as np

log = logging.getLogger(__name__)

DEFAULT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sessions.csv")
FIELDS = ["timestamp", "session_id", "seeded", "dose_hours", "neuroscore",
          "tremor_R", "tremor_L", "tremor_R_cm", "tremor_L_cm", "tremor_R_hz", "tremor_L_hz",
          "tap_R", "tap_L", "tap_R_rate", "tap_L_rate", "tap_R_amp", "tap_L_amp",
          "flip_R", "flip_L", "flip_R_rate", "flip_L_rate", "flip_R_amp_deg", "flip_L_amp_deg",
          "coach_R_rate", "coach_L_rate",
          "asymmetry"]
SCORE_KEYS = ("tremor", "tap", "flip")


def _fmt(v) -> str:
    if v is None:
        return ""
    if isinstance(v, (float, np.floating)):
        return f"{float(v):.3f}"
    return str(v)


def _migrate(path: str) -> None:
    """Rewrite an existing CSV whose header differs from FIELDS (older app version)."""
    with open(path, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        header = reader.fieldnames or []
        rows = list(reader)
    if header == FIELDS:
        return
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in FIELDS})
    log.info("migrated %s to the current column layout", path)


def save_session(row: dict, path: str = DEFAULT_PATH) -> dict:
    """Append one session (unknown keys ignored, missing ones blank). Returns the row written."""
    full = {k: "" for k in FIELDS}
    full.update({k: v for k, v in row.items() if k in FIELDS})
    if not full["timestamp"]:
        full["timestamp"] = datetime.now().isoformat(timespec="seconds")
    new_file = not os.path.exists(path) or os.path.getsize(path) == 0
    if not new_file:
        _migrate(path)
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


def neuroscore_of_row(row: dict) -> Optional[float]:
    v = _num(row.get("neuroscore"))
    if v is not None:
        return v
    scores = [_num(row.get(f"{k}_{s}")) for k in SCORE_KEYS for s in "RL"]
    scores = [s for s in scores if s is not None]
    return 100.0 * (1 - sum(scores) / (4 * len(scores))) if scores else None


def seed_history(path: str = DEFAULT_PATH, days: int = 14, now: Optional[datetime] = None,
                 force: bool = False) -> list[dict]:
    """DEMO DATA: `days` x 2 sessions ending yesterday. Morning session ~1.5 h after the
    levodopa dose, evening ~5 h after (wearing off: worse); right side worse and slowly
    worsening. Does nothing if seeded rows already exist (unless force)."""
    from flipping_analysis import FlippingFeatures
    from scoring import score_flipping, score_tapping, score_tremor
    from tapping_analysis import TappingFeatures
    from tremor_analysis import TremorFeatures

    if not force and any(r.get("seeded") == "1" for r in load_sessions(path)):
        log.info("%s already has seeded sessions - not adding more", path)
        return []
    now = now or datetime.now()
    rows = []
    for i in range(days):
        day = (now - timedelta(days=days - i)).replace(minute=0, second=0, microsecond=0)
        k = i / max(1, days - 1)
        for hour, dose_h in ((9, 1.5), (18, 4.5 + 1.5 * k)):
            off = min(1.0, max(0.0, (dose_h - 2.0) / 4.0))     # 0 when "on", -> 1 wearing off
            sides = {"R": (0.4 + 1.2 * off + 0.5 * k, 3.3 - 1.3 * off - 0.4 * k, 2.3 - 0.9 * off - 0.3 * k),
                     "L": (0.05 + 0.3 * off, 3.5 - 0.5 * off, 2.4 - 0.4 * off)}
            row = {"timestamp": day.replace(hour=hour).isoformat(timespec="seconds"),
                   "session_id": f"DEMO-{i:02d}{'am' if hour == 9 else 'pm'}", "seeded": 1,
                   "dose_hours": round(dose_h, 1)}
            scores = []
            for side, (cm, rate, flips) in sides.items():
                clear = cm >= 0.1
                tf = TremorFeatures(True, 10.0, 1.0, 0.06, 5.0, cm, clear, 10.0, 80.0)
                pf = TappingFeatures(int(rate * 10), rate, 0.9 - 0.3 * off * (side == "R"),
                                     0.1 + 0.35 * off * (side == "R"), 0.1, 0, 1.0, 10.0)
                ff = FlippingFeatures(int(flips * 20), int(flips * 10), flips, 0.1,
                                      0.3 * off * (side == "R"), 0, 10.0, 0.3)
                s3 = (score_tremor(tf).score, score_tapping(pf).score, score_flipping(ff).score)
                scores += list(s3)
                row.update({f"tremor_{side}": s3[0], f"tremor_{side}_cm": cm if clear else 0.0,
                            f"tremor_{side}_hz": 5.0, f"tap_{side}": s3[1],
                            f"tap_{side}_rate": rate, f"flip_{side}": s3[2],
                            f"flip_{side}_rate": flips})
            row["neuroscore"] = 100.0 * (1 - sum(scores) / (4 * len(scores)))
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


def neuroscore_series(sessions: list[dict]) -> tuple[list, list, list]:
    """(datetimes, NeuroScores, seeded flags) for rows that have a NeuroScore."""
    xs, ys, sd = [], [], []
    for i, r in enumerate(sessions):
        v = neuroscore_of_row(r)
        if v is not None:
            xs.append(_when(r, i))
            ys.append(v)
            sd.append(r.get("seeded") == "1")
    return xs, ys, sd


def trend_image(sessions: list[dict], width: int = 1280, height: int = 720) -> np.ndarray:
    """2x2: NeuroScore over time | NeuroScore vs hours since dose; sub-scores R | L.
    Seeded rows are hollow markers labelled DEMO DATA. Returns BGR uint8."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt

    dpi = 100
    fig, axes = plt.subplots(2, 2, figsize=(width / dpi, height / dpi), dpi=dpi)
    fig.patch.set_facecolor("#0e0c0a")
    for ax in axes.flat:
        ax.set_facecolor("#16130f")
        ax.tick_params(colors="#bdbab4", labelsize=8)
        for sp in ax.spines.values():
            sp.set_color("#4e463c")
        ax.grid(color="#2a2621")
    cyan, red, blue, amber = "#28cde6", "#ff5a5a", "#5aa0ff", "#ffb428"
    xs, ys, sd = neuroscore_series(sessions)
    any_demo = any(sd)
    all_t = [_when(r, i) for i, r in enumerate(sessions)]
    xlim = (min(all_t) - timedelta(hours=12), max(all_t) + timedelta(hours=12)) if all_t else None

    ax = axes[0, 0]
    for x, y, s in zip(xs, ys, sd):
        ax.plot(x, y, "o", ms=6, mfc="none" if s else cyan, mec=cyan)
    if xs:
        ax.plot(xs, ys, "-", color=cyan, lw=1, alpha=0.5)
    if xlim:
        ax.set_xlim(*xlim)
    locator = mdates.AutoDateLocator(minticks=3, maxticks=7)
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))
    ax.set_ylim(0, 105)
    ax.set_title("NeuroScore over time (100 = no problems)", color="white", fontsize=11)

    ax = axes[0, 1]
    pts = [(_num(r.get("dose_hours")), neuroscore_of_row(r), r.get("seeded") == "1") for r in sessions]
    pts = [p for p in pts if p[0] is not None and p[1] is not None]
    for h, v, s in pts:
        ax.plot(h, v, "o", ms=6, mfc="none" if s else amber, mec=amber)
    if len(pts) >= 3:
        hh, vv = np.array([p[0] for p in pts]), np.array([p[1] for p in pts])
        if np.ptp(hh) > 0:
            m, b = np.polyfit(hh, vv, 1)
            gx = np.linspace(hh.min(), hh.max(), 10)
            ax.plot(gx, m * gx + b, "--", color=amber, lw=1)
            ax.text(0.02, 0.06, f"fit: {m:+.1f} points per hour", transform=ax.transAxes,
                    color=amber, fontsize=8)
    ax.set_ylim(0, 105)
    ax.set_xlabel("hours since last levodopa dose", color="#bdbab4", fontsize=8)
    ax.set_title("NeuroScore vs dose timing", color="white", fontsize=11)
    if not pts:
        ax.text(0.5, 0.5, "no sessions with dose time", transform=ax.transAxes, ha="center",
                color="#7a7670")

    for ax, side, colour in ((axes[1, 0], "R", red), (axes[1, 1], "L", blue)):
        for key, style, name in (("tremor", "o-", "tremor"), ("tap", "s-", "tapping"),
                                 ("flip", "^-", "flipping")):
            p2 = [(_when(r, i), _num(r.get(f"{key}_{side}")), r.get("seeded") == "1")
                  for i, r in enumerate(sessions)]
            p2 = [p for p in p2 if p[1] is not None]
            if p2:
                ax.plot([p[0] for p in p2], [p[1] for p in p2], style[1], color=colour,
                        alpha=0.35 if key != "tremor" else 0.6, lw=1)
                ax.plot([p[0] for p in p2], [p[1] for p in p2], style[0], ms=4, color=colour,
                        mfc="none", label=name, alpha=0.9)
        ax.set_ylim(-0.3, 4.3)
        ax.set_yticks(range(5))
        if xlim:
            ax.set_xlim(*xlim)
        loc = mdates.AutoDateLocator(minticks=3, maxticks=7)
        ax.xaxis.set_major_locator(loc)
        ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(loc))
        ax.set_title(f"{'Right' if side == 'R' else 'Left'} hand sub-scores (0-4)", color="white",
                     fontsize=11)
        if ax.get_legend_handles_labels()[0]:
            ax.legend(loc="upper left", fontsize=8, facecolor="#16130f", labelcolor="white")
    if not sessions:
        fig.text(0.5, 0.5, "No sessions yet - run one, or start with --seed-history",
                 ha="center", color="white", fontsize=16)
    title = "NeuroCheck trend  (demo thresholds - tracking aid, not a diagnosis)"
    fig.suptitle(title, color="white", fontsize=13)
    if any_demo:
        fig.text(0.99, 0.965, "hollow markers = DEMO DATA (seeded, not measured)", ha="right",
                 color=amber, fontsize=10, weight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.canvas.draw()
    rgba = np.asarray(fig.canvas.buffer_rgba())
    plt.close(fig)
    img = np.ascontiguousarray(rgba[:, :, 2::-1])
    if img.shape[:2] != (height, width):
        import cv2
        img = cv2.resize(img, (width, height))
    return img
