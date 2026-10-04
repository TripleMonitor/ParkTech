"""Stage 2: score every cached HUBU-FIS clip with the app's tapping maths and compare with the
clinicians' MDS-UPDRS 3.4 ratings; optionally re-tune the 4 tapping thresholds with
participant-level cross-validation (people in the test fold are never used for tuning).

Usage: python tools/hubu_eval.py LABELS.csv [cache_dir] [--tune]
LABELS.csv columns: video (relative path or file stem), participant, rating (0-4)
Dataset: HUBU-FIS Finger Tapping (Zenodo 10.5281/zenodo.17738775, CC-BY-4.0).
"""
from __future__ import annotations

import csv
import glob
import itertools
import os
import sys
from dataclasses import dataclass
from typing import Optional

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import scoring  # noqa: E402
from tapping_analysis import TappingFeatures, analyze_tapping  # noqa: E402

THRESHOLDS = ("TAP_SLOW", "TAP_SMALL", "TAP_DECREMENT", "TAP_IRREGULAR_CV")
GRID = {"TAP_SLOW": np.arange(1.0, 3.01, 0.25), "TAP_SMALL": np.arange(0.2, 0.81, 0.1),
        "TAP_DECREMENT": np.arange(0.10, 0.51, 0.05), "TAP_IRREGULAR_CV": np.arange(0.15, 0.51, 0.05)}


@dataclass
class Clip:
    key: str
    participant: str
    rating: int
    f: TappingFeatures


def trim(t: np.ndarray, d: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Drop leading/trailing frames without a hand (the clip around the actual test)."""
    ok = np.flatnonzero(~np.isnan(d))
    if len(ok) == 0:
        return t, d
    return t[ok[0]:ok[-1] + 1], d[ok[0]:ok[-1] + 1]


ACTIVE_PAD_S = 0.5


def active_segment(t: np.ndarray, d: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Crop to the tapping itself: first detected tap - 0.5 s .. last tap + 0.5 s.

    The dataset clips are ~20 s and include idle time before/after tapping; the app's test
    is 10 s of tapping only, so rates must be measured over the active period.
    """
    from tapping_analysis import detect_taps
    ok = ~np.isnan(d)
    if ok.sum() < 10:
        return t, d
    tap_t, _ = detect_taps(t[ok], d[ok])
    if len(tap_t) < 2:
        return t, d
    a, b = t[ok][0] + tap_t[0] - ACTIVE_PAD_S, t[ok][0] + tap_t[-1] + ACTIVE_PAD_S
    keep = (t >= a) & (t <= b)
    return t[keep], d[keep]


def features_from_cache(path: str) -> TappingFeatures:
    z = np.load(path, allow_pickle=False)
    t, d = trim(np.asarray(z["t"], float), np.asarray(z["d"], float))
    t, d = active_segment(t, d)
    return analyze_tapping(list(t), [None if np.isnan(x) else float(x) for x in d])


def score_with(f: TappingFeatures, p: dict) -> Optional[int]:
    """The app's score_tapping rule with thresholds taken from p."""
    if f.hand_visible < scoring.TAP_MIN_VISIBLE:
        return None
    if f.n_taps < scoring.TAP_MIN_TAPS:
        return 4
    problems = sum([f.taps_per_sec < p["TAP_SLOW"], f.mean_amplitude < p["TAP_SMALL"],
                    f.decrement > p["TAP_DECREMENT"], f.interval_cv > p["TAP_IRREGULAR_CV"],
                    f.hesitations >= 1])
    return min(3, problems)


def current_params() -> dict:
    return {k: getattr(scoring, k) for k in THRESHOLDS}


def qwk(a: list[int], b: list[int], k: int = 5) -> float:
    """Quadratic weighted Cohen's kappa for ordinal ratings 0..k-1."""
    a, b = np.asarray(a), np.asarray(b)
    o = np.zeros((k, k))
    for x, y in zip(a, b):
        o[x, y] += 1
    w = np.array([[(i - j) ** 2 for j in range(k)] for i in range(k)]) / (k - 1) ** 2
    e = np.outer(o.sum(1), o.sum(0)) / o.sum()
    den = (w * e).sum()
    return 1.0 - (w * o).sum() / den if den > 0 else 0.0


def spearman(a, b) -> float:
    ra, rb = np.argsort(np.argsort(a)), np.argsort(np.argsort(b))
    return float(np.corrcoef(ra, rb)[0, 1]) if len(a) > 2 else float("nan")


def metrics(truth: list[int], pred: list[int]) -> dict:
    t, p = np.asarray(truth), np.asarray(pred)
    return {"n": len(t), "exact": float(np.mean(t == p)), "within1": float(np.mean(abs(t - p) <= 1)),
            "qwk": qwk(list(t), list(p)), "spearman": spearman(t, p),
            "mae": float(np.mean(abs(t - p)))}


def confusion(truth, pred, k: int = 5) -> str:
    m = np.zeros((k, k), int)
    for x, y in zip(truth, pred):
        m[x, y] += 1
    rows = ["clinician \\ app " + " ".join(f"{j:>4d}" for j in range(k))]
    rows += [f"{i:>15d} " + " ".join(f"{m[i, j]:>4d}" for j in range(k)) for i in range(k)]
    return "\n".join(rows)


def evaluate(clips: list[Clip], p: dict) -> tuple[dict, list, list, int]:
    truth, pred, unscored = [], [], 0
    for c in clips:
        s = score_with(c.f, p)
        if s is None:
            unscored += 1
            continue
        truth.append(c.rating)
        pred.append(s)
    return metrics(truth, pred), truth, pred, unscored


def tune(clips: list[Clip]) -> dict:
    best, best_k = current_params(), -9.0
    for combo in itertools.product(*(GRID[k] for k in THRESHOLDS)):
        p = dict(zip(THRESHOLDS, (round(float(x), 3) for x in combo)))
        m, *_ = evaluate(clips, p)
        if m["n"] and m["qwk"] > best_k:
            best, best_k = p, m["qwk"]
    return best


def cross_validate(clips: list[Clip], folds: int = 5, seed: int = 0) -> tuple[dict, list, list, list]:
    """Group k-fold by participant: tune on 4/5 of the PEOPLE, test on the rest."""
    people = sorted({c.participant for c in clips})
    rng = np.random.default_rng(seed)
    rng.shuffle(people)
    groups = [set(people[i::folds]) for i in range(folds)]
    truth, pred, chosen = [], [], []
    for g in groups:
        train = [c for c in clips if c.participant not in g]
        test = [c for c in clips if c.participant in g]
        p = tune(train)
        chosen.append(p)
        _, t, pr, _ = evaluate(test, p)
        truth += t
        pred += pr
    return metrics(truth, pred), truth, pred, chosen


def load_labels(path: str) -> dict:
    """{video key: (participant, rating)}; key = file stem, lower-case."""
    out = {}
    with open(path, newline="", encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            key = os.path.splitext(os.path.basename(r["video"].strip()))[0].lower()
            out[key] = (str(r["participant"]).strip(), int(float(r["rating"])))
    return out


def load_clips(labels_csv: str, cache_dir: str) -> tuple[list[Clip], list[str]]:
    labels = load_labels(labels_csv)
    clips, missing = [], []
    for path in sorted(glob.glob(os.path.join(cache_dir, "*.npz"))):
        stem = os.path.basename(path)[:-4].split("__")[-1].lower()
        if stem not in labels:
            missing.append(stem)
            continue
        part, rating = labels[stem]
        clips.append(Clip(stem, part, rating, features_from_cache(path)))
    return clips, missing


def main() -> int:
    labels_csv = sys.argv[1]
    cache_dir = next((a for a in sys.argv[2:] if not a.startswith("--")), "datasets/hubu/cache")
    clips, missing = load_clips(labels_csv, cache_dir)
    print(f"{len(clips)} labelled clips from {len({c.participant for c in clips})} participants"
          f" ({len(missing)} cached clips without a label)")
    print("clinician rating counts:", dict(sorted(
        {r: sum(c.rating == r for c in clips) for r in range(5)}.items())))
    m, t, p, unscored = evaluate(clips, current_params())
    print("\n== CURRENT demo thresholds", current_params())
    print(f"  unscored (hand seen < 50%): {unscored}")
    print("  " + ", ".join(f"{k} {v:.3f}" if isinstance(v, float) else f"{k} {v}" for k, v in m.items()))
    print(confusion(t, p))
    if "--tune" in sys.argv:
        cvm, ct, cp, chosen = cross_validate(clips)
        print("\n== CROSS-VALIDATED tuning (5 folds by participant; honest estimate)")
        print("  " + ", ".join(f"{k} {v:.3f}" if isinstance(v, float) else f"{k} {v}"
                               for k, v in cvm.items()))
        print(confusion(ct, cp))
        print("  thresholds chosen per fold:")
        for c in chosen:
            print("   ", c)
        final = tune(clips)
        fm, *_ = evaluate(clips, final)
        print("\n== FINAL thresholds tuned on ALL clips (use these; quote the CV numbers above)")
        print("  ", final)
        print("  in-sample (optimistic): " + ", ".join(
            f"{k} {v:.3f}" if isinstance(v, float) else f"{k} {v}" for k, v in fm.items()))
    return 0


if __name__ == "__main__":
    sys.exit(main())
