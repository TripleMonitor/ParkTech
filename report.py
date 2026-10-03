"""One-page doctor PDF (matplotlib only): scores with reasons, radar, NeuroScore trend,
asymmetry, dose timing, disclaimer. Seeded history points are labelled DEMO DATA.
"""
from __future__ import annotations

import textwrap

from dashboard import radar_image

TITLES = {"tremor": "Rest tremor (3.17)", "tapping": "Finger tapping (3.4)",
          "flipping": "Hand flipping (3.6)"}


def save_pdf(snapshot: dict, sessions: list[dict], path: str) -> str:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from history import neuroscore_series

    fig = plt.figure(figsize=(8.27, 11.69))            # A4 portrait
    fig.text(0.06, 0.965, "NeuroCheck - motor check summary", fontsize=16, weight="bold")
    dose = snapshot.get("dose_hours")
    fig.text(0.06, 0.945, f"Session {snapshot['session_id']}   {snapshot['timestamp']}   "
             f"hours since levodopa: {'unknown' if dose is None else f'{dose:g}'}   "
             f"mode: {snapshot['mode']}", fontsize=8.5)
    ns = snapshot.get("neuroscore")
    fig.text(0.06, 0.92, f"NeuroScore {'-' if ns is None else f'{ns:.0f}'} / 100", fontsize=14,
             color="#0a6a7a", weight="bold")
    fig.text(0.40, 0.922, snapshot.get("neuroscore_formula", ""), fontsize=7.5)
    fig.text(0.06, 0.903, "Composite tracking index, not a diagnosis.", fontsize=7.5, style="italic")

    y = 0.875
    for t in snapshot["tests"]:
        s = "-" if t["score"] is None else str(t["score"])
        fig.text(0.06, y, f"{t['hand']:<5} {TITLES[t['kind']]}", fontsize=9, weight="bold")
        fig.text(0.38, y, f"score {s}", fontsize=9, weight="bold",
                 color="#b00020" if (t["score"] or 0) >= 3 else "#000000")
        y -= 0.014
        for r in t["reasons"][:4]:
            for line in textwrap.wrap(r, 95)[:2]:
                fig.text(0.08, y, "- " + line, fontsize=7)
                y -= 0.0115
        y -= 0.004
    if snapshot.get("asymmetry"):
        fig.text(0.06, y, "Asymmetry:", fontsize=8.5, weight="bold", color="#a05a00")
        y -= 0.013
        for f in snapshot["asymmetry"][:6]:
            fig.text(0.08, y, "- " + f, fontsize=7, color="#a05a00")
            y -= 0.0115

    rad = radar_image({(t["kind"], t["hand"]): t["score"] for t in snapshot["tests"]},
                      snapshot.get("coach_rates", {}), 520, 440, "Motor fingerprint")
    ax_r = fig.add_axes([0.52, 0.05, 0.44, 0.30])
    ax_r.imshow(rad[:, :, ::-1])
    ax_r.axis("off")

    ax_t = fig.add_axes([0.08, 0.07, 0.40, 0.22])
    xs, ys, seeded = neuroscore_series(sessions)
    for x, v, sd in zip(xs, ys, seeded):
        ax_t.plot(x, v, "o", mfc="none" if sd else "#0a6a7a", mec="#0a6a7a", ms=4)
    if any(seeded):
        ax_t.text(0.02, 0.04, "hollow = DEMO DATA (seeded)", transform=ax_t.transAxes, fontsize=7,
                  color="#a05a00")
    ax_t.set_ylim(0, 100)
    ax_t.set_title("NeuroScore over time", fontsize=9)
    ax_t.tick_params(labelsize=6)
    for lbl in ax_t.get_xticklabels():
        lbl.set_rotation(30)

    fig.text(0.06, 0.02, "Demo thresholds. TRACKING and DECISION-SUPPORT tool, NOT a diagnosis. "
             "Discuss any change with a clinician.", fontsize=7.5, color="#a05a00")
    fig.savefig(path)
    plt.close(fig)
    return path
