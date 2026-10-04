"""Results dashboard maths + radar chart. NeuroScore is a transparent composite
tracking index, NOT a diagnosis.

    NeuroScore = 100 x (1 - sum(sub-scores) / (4 x number of scored tests))

over the 0-4 test scores that exist (tremor, finger tapping, hand flipping; both hands).
100 = no problems found by any rule. Unscored tests are left out (and listed).
Radar "Motor Fingerprint": per hand, function % per test = 100 x (1 - score/4); the
rhythm-coach axis = max sustainable rhythm / 4.0 beats/s reference (demo reference).
"""
from __future__ import annotations

from typing import Optional

import numpy as np

TEST_KEYS = ("tremor", "tapping", "flipping")
AXES = ("Tremor", "Finger tapping", "Hand flipping", "Rhythm coach")
COACH_REF_RATE = 4.0           # beats/s mapped to 100% on the radar (demo reference)


def merge_flipping(scores: dict) -> dict:
    """One hand-flipping score per hand: the wrist-sensor trial ("flipping") when it was
    scored, else the camera-only trial ("flipcam"). Keeps NeuroScore at 3 tests x 2 hands
    so running both trials doesn't double-weight flipping."""
    out = {k: v for k, v in scores.items() if k[0] != "flipcam"}
    for (kind, hand), s in scores.items():
        if kind == "flipcam" and out.get(("flipping", hand)) is None:
            out[("flipping", hand)] = s
    return out


def neuroscore(scores: dict) -> tuple[Optional[float], str]:
    """scores: {(kind, hand): score or None}. Returns (value 0-100 or None, formula text)."""
    real = [s for s in scores.values() if s is not None]
    missing = [f"{h[0]} {k}" for (k, h), s in scores.items() if s is None]
    if not real:
        return None, "NeuroScore: no scored tests yet"
    total, n = sum(real), len(real)
    value = 100.0 * (1 - total / (4 * n))
    text = f"NeuroScore = 100 x (1 - {total} / (4 x {n})) = {value:.0f}"
    if missing:
        text += f"   (not scored: {', '.join(missing)})"
    return value, text


def radar_values(scores: dict, coach_rates: dict, hand: str) -> list[Optional[float]]:
    out: list[Optional[float]] = []
    for k in TEST_KEYS:
        s = scores.get((k, hand))
        out.append(None if s is None else 100.0 * (1 - s / 4))
    r = coach_rates.get(hand)
    out.append(None if r is None else min(100.0, 100.0 * r / COACH_REF_RATE))
    return out


def radar_image(scores: dict, coach_rates: dict, width: int, height: int,
                title: str = "MOTOR FINGERPRINT") -> np.ndarray:
    """Matplotlib polar radar, R and L overlaid. Missing axes are drawn at 0 and labelled n/a."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    dpi = 100
    fig = plt.figure(figsize=(width / dpi, height / dpi), dpi=dpi)
    fig.patch.set_facecolor("#0e0c0a")
    ax = fig.add_subplot(111, polar=True)
    ax.set_facecolor("#16130f")
    ang = np.linspace(0, 2 * np.pi, len(AXES), endpoint=False).tolist()
    labels = []
    for i, name in enumerate(AXES):
        missing = [h[0] for h in ("Right", "Left") if radar_values(scores, coach_rates, h)[i] is None]
        labels.append(name if not missing else f"{name}\n(n/a: {'/'.join(missing)} = 0)")
    for hand, colour in (("Right", "#ff5a5a"), ("Left", "#2ad0e6")):
        vals = [v if v is not None else 0.0 for v in radar_values(scores, coach_rates, hand)]
        ax.plot(ang + ang[:1], vals + vals[:1], color=colour, lw=2, label=hand)
        ax.fill(ang + ang[:1], vals + vals[:1], color=colour, alpha=0.18)
    ax.set_xticks(ang)
    ax.set_xticklabels(labels, color="#dcdcd8", fontsize=9)
    ax.set_ylim(0, 100)
    ax.set_yticks([25, 50, 75, 100])
    ax.set_yticklabels(["25", "50", "75", "100%"], color="#7a7670", fontsize=7)
    ax.grid(color="#3a352f")
    ax.spines["polar"].set_color("#4e463c")
    ax.legend(loc="lower right", bbox_to_anchor=(1.15, -0.05), facecolor="#16130f",
              labelcolor="white", fontsize=8)
    ax.set_title(title, color="#28cde6", fontsize=12, pad=14)
    fig.tight_layout()
    fig.canvas.draw()
    rgba = np.asarray(fig.canvas.buffer_rgba())
    plt.close(fig)
    img = np.ascontiguousarray(rgba[:, :, 2::-1])
    if img.shape[:2] != (height, width):
        import cv2
        img = cv2.resize(img, (width, height))
    return img
