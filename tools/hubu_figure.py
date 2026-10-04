"""docs/hubu_validation.png: (1) a healthy control's tapping signal with the taps the app
detects, (2) app vs clinician MDS-UPDRS 3.4 confusion: original demo thresholds vs
cross-validated tuned thresholds. Data: HUBU-FIS (Zenodo 10.5281/zenodo.17738775, CC-BY-4.0).
Usage: python tools/hubu_figure.py [labels.csv] [cache_dir] [example_stem]
"""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tapping_analysis import detect_taps  # noqa: E402
from tools.hubu_eval import (active_segment, cross_validate, evaluate, load_clips,  # noqa: E402
                             trim)

ORIGINAL = {"TAP_SLOW": 2.0, "TAP_SMALL": 0.5, "TAP_DECREMENT": 0.30, "TAP_IRREGULAR_CV": 0.30}


def conf(ax, truth, pred, title, metrics):
    m = np.zeros((4, 5), int)
    for t, p in zip(truth, pred):
        m[t, p] += 1
    ax.imshow(m, cmap="Blues")
    for i in range(4):
        for j in range(5):
            ax.text(j, i, m[i, j], ha="center", va="center",
                    color="white" if m[i, j] > m.max() / 2 else "black", fontsize=10)
    ax.set_xticks(range(5))
    ax.set_yticks(range(4))
    ax.set_xlabel("app score")
    ax.set_ylabel("clinician MDS-UPDRS 3.4")
    ax.set_title(f"{title}\nexact {metrics['exact']:.0%}, within 1: {metrics['within1']:.0%}, "
                 f"kappa {metrics['qwk']:.2f}", fontsize=10)


def main() -> int:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = sys.argv[1] if len(sys.argv) > 1 else "datasets/hubu/labels.csv"
    cache = sys.argv[2] if len(sys.argv) > 2 else "datasets/hubu/cache"
    stem = sys.argv[3] if len(sys.argv) > 3 else "CONTROL02_DCHA"
    clips, _ = load_clips(labels, cache)

    fig = plt.figure(figsize=(14, 8.5))
    ax0 = fig.add_subplot(2, 1, 1)
    z = np.load(os.path.join(cache, f"videos_FIS__videos__{stem}.npz"))
    t, d = trim(np.asarray(z["t"], float), np.asarray(z["d"], float))
    t, d = active_segment(t, d)
    ok = ~np.isnan(d)
    tap_t, amps = detect_taps(t[ok], d[ok])
    ax0.plot(t - t[0], d, lw=1.2, color="#1f77b4", label="thumb-index distance (normalised)")
    for tt in tap_t + (t[ok][0] - t[0]):
        ax0.axvline(tt, color="#d62728", alpha=0.35, lw=1)
    ax0.set_title(f"{stem} (healthy control, clinician 3.4 = "
                  f"{next(c.rating for c in clips if c.key == stem.lower())}): "
                  f"app detects {len(tap_t)} taps in {t[-1] - t[0]:.1f} s = "
                  f"{len(tap_t) / (t[-1] - t[0]):.2f} taps/s (red lines)", fontsize=11)
    ax0.set_xlabel("time (s)")
    ax0.legend(loc="upper right")

    m0, t0, p0, _ = evaluate(clips, ORIGINAL)
    mcv, tcv, pcv, _ = cross_validate(clips)
    conf(fig.add_subplot(2, 2, 3), t0, p0, "original demo thresholds", m0)
    conf(fig.add_subplot(2, 2, 4), tcv, pcv, "tuned thresholds (5-fold CV by participant)", mcv)
    fig.suptitle(f"ParkTech finger tapping vs clinicians - HUBU-FIS dataset, {len(clips)} videos, "
                 f"{len({c.participant for c in clips})} people (CC-BY-4.0)", fontsize=13)
    fig.tight_layout()
    os.makedirs("docs", exist_ok=True)
    fig.savefig("docs/hubu_validation.png", dpi=100)
    print("wrote docs/hubu_validation.png")
    return 0


if __name__ == "__main__":
    sys.exit(main())
