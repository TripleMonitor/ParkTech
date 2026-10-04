"""Stage 1: run MediaPipe Hands over every HUBU-FIS finger-tapping video and cache, per
frame, the time and the normalised thumb-index distance (same definition as the app:
landmarks 4-8 divided by 0-9). Resumable: videos already in the cache are skipped.

Usage: python tools/hubu_extract.py [videos_dir] [cache_dir] [workers]
Dataset: HUBU-FIS Finger Tapping (Zenodo 10.5281/zenodo.17738775, CC-BY-4.0).
"""
from __future__ import annotations

import glob
import os
import sys
import time
from multiprocessing import Pool

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

VIDEO_EXT = (".mp4", ".mov", ".avi", ".m4v", ".mkv", ".MP4", ".MOV")


def find_videos(root: str) -> list[str]:
    out = []
    for ext in VIDEO_EXT:
        out += glob.glob(os.path.join(root, "**", f"*{ext}"), recursive=True)
    return sorted(set(out))


def cache_name(video: str, root: str) -> str:
    rel = os.path.relpath(video, root)
    return rel.replace(os.sep, "__").rsplit(".", 1)[0] + ".npz"


def extract(args) -> str:
    video, root, cache_dir = args
    import cv2
    import mediapipe as mp
    from tapping_analysis import normalised_distance

    out = os.path.join(cache_dir, cache_name(video, root))
    if os.path.exists(out):
        return f"skip {os.path.basename(video)}"
    cap = cv2.VideoCapture(video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    # SAME settings as the live app (tapping_tracker.CameraHand) so we evaluate the app as it runs
    hands = mp.solutions.hands.Hands(static_image_mode=False, max_num_hands=1,
                                     model_complexity=0, min_detection_confidence=0.6,
                                     min_tracking_confidence=0.5)
    t, d, score, label = [], [], [], []
    i = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        h, w = frame.shape[:2]
        if max(h, w) > 640:                        # like the app's 640x480 webcam frames
            s = 640 / max(h, w)
            frame = cv2.resize(frame, (int(w * s), int(h * s)))
            h, w = frame.shape[:2]
        res = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        ms = cap.get(cv2.CAP_PROP_POS_MSEC)
        t.append(ms / 1000.0 if ms > 0 else i / fps)
        if res.multi_hand_landmarks:
            pts = np.array([[p.x * w, p.y * h] for p in res.multi_hand_landmarks[0].landmark])
            d.append(normalised_distance(pts))
            c = res.multi_handedness[0].classification[0]
            score.append(float(c.score))
            label.append(c.label)
        else:
            d.append(np.nan)
            score.append(np.nan)
            label.append("")
        i += 1
    cap.release()
    hands.close()
    np.savez_compressed(out, t=np.array(t), d=np.array(d, dtype=float), score=np.array(score),
                        label=np.array(label), fps=fps, video=os.path.relpath(video, root))
    return f"done {os.path.basename(video)}: {i} frames, hand in {np.mean(~np.isnan(d)):.0%}"


def main() -> int:
    root = sys.argv[1] if len(sys.argv) > 1 else "datasets/hubu/extracted"
    cache = sys.argv[2] if len(sys.argv) > 2 else "datasets/hubu/cache"
    workers = int(sys.argv[3]) if len(sys.argv) > 3 else max(1, (os.cpu_count() or 4) // 2)
    os.makedirs(cache, exist_ok=True)
    videos = find_videos(root)
    print(f"{len(videos)} videos, {workers} workers", flush=True)
    t0 = time.time()
    with Pool(workers) as pool:
        for k, msg in enumerate(pool.imap_unordered(extract, [(v, root, cache) for v in videos]), 1):
            print(f"[{k}/{len(videos)} {time.time() - t0:5.0f}s] {msg}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
