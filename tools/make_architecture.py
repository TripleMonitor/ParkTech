"""Render docs/architecture.png: sensors -> signal processing -> scoring -> dashboard/exports."""
from __future__ import annotations

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

BG, PANEL, EDGE, TXT, DIM = "#0e0c0a", "#1c1915", "#4e463c", "#ecece8", "#9a958e"
CYAN, GREEN, AMBER, RED = "#28cde6", "#5ad26e", "#ffb428", "#ff5a5a"

COLUMNS = [
    ("SENSORS", CYAN, [
        ("Laptop webcam", "MediaPipe Hands: 21 landmarks\n+ 3D world landmarks"),
        ("Arduino Uno", "SW-520D tilt switch (D2)\n30 ms debounce, S/C lines"),
        ("Arduino outputs", "buzzer, RGB LED, LCD1602\nmetronome (CUE / TEMPO)"),
    ]),
    ("SIGNAL PROCESSING", GREEN, [
        ("Camera tremor", "fingertip px -> cm (hand = 9 cm)\n30 Hz resample, FFT 3-8 Hz"),
        ("Finger tapping", "thumb-index distance\ntaps, decrement, rhythm"),
        ("Hand flipping + fusion", "switch half-flips (60 ms)\npalm-normal rotation, lock check"),
        ("Signal quality", "hand found %, confidence,\ndropped frames"),
    ]),
    ("SCORING", AMBER, [
        ("Rule-based 0-4 scores", "MDS-UPDRS-style, demo thresholds\nevery rule shown (WHY panel)"),
        ("Asymmetry", "R vs L: >= 1 point or >= 25%"),
        ("PID rhythm coach", "on-time rate -> tempo\nmax sustainable rhythm"),
    ]),
    ("DASHBOARD + DATA", RED, [
        ("Results + dashboard", "6 cards, radar fingerprint,\nNeuroScore (formula shown)"),
        ("Trend", "sessions.csv, NeuroScore vs time\nand vs levodopa timing"),
        ("Exports", "one-page doctor PDF\nFHIR R4 bundle (local codes)"),
    ]),
]


def main() -> str:
    fig = plt.figure(figsize=(16, 9), dpi=100)
    fig.patch.set_facecolor(BG)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 9)
    ax.axis("off")
    ax.text(0.5, 8.45, "ParkTech architecture", color=TXT, fontsize=24, weight="bold")
    ax.text(0.5, 8.05, "every number on screen is measured live or labelled SIM / DEMO DATA  -  "
                       "tracking & decision support, not a diagnosis", color=DIM, fontsize=11)
    col_w, gap, x0 = 3.45, 0.42, 0.5
    centres = []
    for ci, (title, colour, boxes) in enumerate(COLUMNS):
        x = x0 + ci * (col_w + gap)
        ax.text(x, 7.35, title, color=colour, fontsize=13, weight="bold")
        n = len(boxes)
        h = (6.3 - 0.25 * (n - 1)) / n
        ys = []
        for bi, (name, desc) in enumerate(boxes):
            y = 7.0 - (bi + 1) * h - bi * 0.25
            ax.add_patch(FancyBboxPatch((x, y), col_w, h, boxstyle="round,pad=0.02,rounding_size=0.08",
                                        fc=PANEL, ec=colour, lw=1.6))
            ax.text(x + 0.18, y + h - 0.38, name, color=TXT, fontsize=12, weight="bold", va="top")
            ax.text(x + 0.18, y + h - 0.85, desc, color=DIM, fontsize=9.5, va="top", linespacing=1.4)
            ys.append(y + h / 2)
        centres.append((x, x + col_w, ys))
    for (xa0, xa1, ya), (xb0, xb1, yb) in zip(centres[:-1], centres[1:]):
        ymid = sum(ya) / len(ya)
        for y in yb:
            ax.add_patch(FancyArrowPatch((xa1 + 0.03, ymid), (xb0 - 0.03, y), arrowstyle="-|>",
                                         mutation_scale=12, color=EDGE, lw=1.2,
                                         connectionstyle="arc3,rad=0.0"))
    ax.text(0.5, 0.25, "Python 3.11  |  OpenCV HUD  |  MediaPipe 0.10.14  |  numpy/scipy  |  "
                       "pyserial 115200 baud  |  matplotlib  |  346 unit tests + headless self-test",
            color=DIM, fontsize=10)
    os.makedirs("docs", exist_ok=True)
    out = os.path.join("docs", "architecture.png")
    fig.savefig(out, facecolor=BG)
    plt.close(fig)
    return out


if __name__ == "__main__":
    print("wrote", main())
