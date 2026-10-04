"""Raw data recording (app.py --record): one JSON file per test with everything needed
to recompute its features and score offline (tools/calibrate.py).

  tremor    frame times + 21 hand landmarks (px) per frame (None = no hand)
  tapping   frame times + normalised thumb-index distance per frame
  flipping  tilt-switch reports (time, state), palm-down reference state,
            camera frame times + 3D world landmarks, palm-down reference normal
  flipcam   camera frame times + 3D world landmarks, palm-down reference normal
  all       camera quality lists, the score and reasons the app showed, metadata
            (session, label, hand, test length, mode LIVE/SIM, time)

Times are seconds on the app clock (perf_counter). Files: <dir>/<session>_<test>_<hand>.json
"""
from __future__ import annotations

import json
import os
from typing import Optional

import numpy as np

FORMAT_VERSION = 1


def _arr(x) -> Optional[list]:
    return None if x is None else np.asarray(x, dtype=float).round(3).tolist()


def build_record(meta: dict, kind: str, data: dict, score: Optional[int], reasons: list) -> dict:
    payload: dict = {}
    if kind == "tremor":
        payload = {"t": list(map(float, data["t"])),
                   "landmarks": [_arr(lm) for lm in data["landmarks"]]}
    elif kind == "tapping":
        payload = {"t": list(map(float, data["t"])),
                   "d": [None if d is None else float(d) for d in data["d"]]}
    elif kind == "flipping":
        payload = {"events": [[float(e[0]), int(e[1])] for e in data["events"]],
                   "rec_start": float(data["rec_start"]), "calib": data["calib"],
                   "cam_t": list(map(float, data["cam_t"])),
                   "world": [_arr(w) for w in data["world"]],
                   "ref_normal": _arr(data["ref_normal"])}
    elif kind == "flipcam":
        payload = {"rec_start": float(data["rec_start"]),
                   "cam_t": list(map(float, data["cam_t"])),
                   "world": [_arr(w) for w in data["world"]],
                   "ref_normal": _arr(data["ref_normal"])}
    quality = {k: data.get(k, []) for k in ("q_t", "q_det", "q_score")}
    return {"format": FORMAT_VERSION, "meta": dict(meta, test=kind), kind: payload,
            "quality": {"t": list(map(float, quality["q_t"])),
                        "det": [bool(x) for x in quality["q_det"]],
                        "score": [None if s is None else float(s) for s in quality["q_score"]]},
            "app_result": {"score": score, "reasons": list(reasons)}}


def save_record(directory: str, record: dict) -> str:
    os.makedirs(directory, exist_ok=True)
    m = record["meta"]
    name = f"{m['session_id']}_{m['test']}_{m['hand']}.json"
    path = os.path.join(directory, name)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(record, f)
    return path


def load_record(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        rec = json.load(f)
    if rec.get("format") != FORMAT_VERSION:
        raise ValueError(f"{path}: unsupported format {rec.get('format')}")
    return rec


def reanalyse(rec: dict):
    """Recompute (features, ScoreResult[, fusion]) from a record with the CURRENT maths."""
    from flipping_analysis import analyze_flipping
    from fusion import fuse
    from scoring import score_flipping, score_tapping, score_tremor
    from tapping_analysis import analyze_tapping
    from tremor_analysis import analyze_tremor

    kind, seconds = rec["meta"]["test"], rec["meta"]["seconds"]
    p = rec[kind]
    if kind == "tremor":
        lms = [None if lm is None else np.asarray(lm) for lm in p["landmarks"]]
        f = analyze_tremor(p["t"], lms)
        return f, score_tremor(f, expected_s=seconds), None
    if kind == "tapping":
        f = analyze_tapping(p["t"], p["d"])
        return f, score_tapping(f), None
    worlds = [None if w is None else np.asarray(w) for w in p["world"]]
    ref = None if p["ref_normal"] is None else np.asarray(p["ref_normal"])
    if kind == "flipcam":
        from flipping_analysis import analyze_half_flips
        from scoring import score_flipping_camera
        fu = fuse(p["cam_t"], worlds, 0, p["rec_start"], seconds, ref)
        f = analyze_half_flips(fu.swing_times, seconds)
        return f, score_flipping_camera(f, fu), fu
    ev = p["events"]
    if rec["meta"].get("closures_only"):
        from flipping_analysis import analyze_closures
        f = analyze_closures([e[0] for e in ev], p["rec_start"], seconds)
    else:
        f = analyze_flipping([e[0] for e in ev], [e[1] for e in ev], p["rec_start"], seconds,
                             initial_state=p["calib"])
    fu = fuse(p["cam_t"], worlds, f.half_flips, p["rec_start"], seconds, ref)
    return f, score_flipping(f, fu), fu
