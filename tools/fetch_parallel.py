"""Parallel, resumable HTTP download using Range requests (Zenodo throttles per connection).

Usage: python tools/fetch_parallel.py URL OUTFILE [parts]
Each part is written to OUTFILE.partNN and resumed from its current size; when all parts are
complete they are joined into OUTFILE and the sizes are checked.
"""
from __future__ import annotations

import os
import sys
import threading
import time
import subprocess

CHUNK = 1 << 20


def total_size(url: str) -> int:
    """Size via a 1-byte ranged GET with curl (works through TLS-inspecting proxies)."""
    out = subprocess.run(["curl", "-sSL", "-r", "0-0", "-D", "-", "-o", os.devnull, url],
                         capture_output=True, text=True, timeout=120).stdout
    for line in out.splitlines():
        if line.lower().startswith("content-range:"):
            return int(line.split("/")[-1])
    raise RuntimeError("no Content-Range in response")


def fetch_part(url: str, path: str, start: int, end: int, progress: dict, key: int) -> None:
    """Download bytes start..end into path with curl, resuming from the file's size."""
    want = end - start + 1
    for attempt in range(100):
        have = os.path.getsize(path) if os.path.exists(path) else 0
        progress[key] = have
        if have >= want:
            return
        tmp = f"{path}.tmp"
        proc = subprocess.Popen(["curl", "-sSL", "--retry", "5", "-r", f"{start + have}-{end}",
                                 "-o", tmp, url])
        while proc.poll() is None:            # live progress while curl runs
            time.sleep(2)
            if os.path.exists(tmp):
                progress[key] = have + os.path.getsize(tmp)
        if os.path.exists(tmp):
            with open(tmp, "rb") as src, open(path, "ab") as dst:
                while True:
                    buf = src.read(CHUNK * 16)
                    if not buf:
                        break
                    dst.write(buf)
            os.remove(tmp)
        if proc.returncode != 0:
            print(f"part {key}: curl exit {proc.returncode}, retry {attempt + 1}", flush=True)
            time.sleep(min(30, 2 + attempt))


def main() -> int:
    url, out = sys.argv[1], sys.argv[2]
    parts = int(sys.argv[3]) if len(sys.argv) > 3 else 16
    size = total_size(url)
    step = -(-size // parts)
    ranges = [(i * step, min(size, (i + 1) * step) - 1) for i in range(parts)]
    progress: dict = {}
    threads = []
    for i, (a, b) in enumerate(ranges):
        th = threading.Thread(target=fetch_part,
                              args=(url, f"{out}.part{i:02d}", a, b, progress, i), daemon=True)
        th.start()
        threads.append(th)
    t0, last = time.time(), 0
    while any(th.is_alive() for th in threads):
        time.sleep(15)
        got = sum(progress.values())
        rate = (got - last) / 15 / 1e6
        last = got
        eta = (size - got) / (rate * 1e6) / 60 if rate > 0 else float("inf")
        print(f"{got / 1e9:6.2f} / {size / 1e9:.2f} GB  {rate:5.1f} MB/s  ETA {eta:5.1f} min", flush=True)
    for i, (a, b) in enumerate(ranges):
        if os.path.getsize(f"{out}.part{i:02d}") != b - a + 1:
            print(f"part {i} incomplete - rerun to resume")
            return 1
    with open(out, "wb") as f:
        for i in range(parts):
            p = f"{out}.part{i:02d}"
            with open(p, "rb") as src:
                while True:
                    buf = src.read(64 * CHUNK)
                    if not buf:
                        break
                    f.write(buf)
            os.remove(p)
    ok = os.path.getsize(out) == size
    print("joined", out, "size OK" if ok else "SIZE MISMATCH", flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
