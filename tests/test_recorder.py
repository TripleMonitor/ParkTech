"""Raw-data recording: every saved test must re-score exactly as the app scored it."""
import glob
import json
import os

from app import KEY_SPACE, TESTS, App
from device import MockDevice
from recorder import load_record, reanalyse
from selftest import FakeClock
from tapping_tracker import FakeHand
from tools.calibrate import is_normal, load_all, report


def record_session(tmp_path, label):
    clock = FakeClock()
    dev = MockDevice(seed=1, clock=clock)
    hands = FakeHand(seed=1, tremor_cm={"Right": 1.5, "Left": 0.0})
    raw = str(tmp_path / "rec")
    app = App(dev, hands, seconds=10.0, history_path=str(tmp_path / "s.csv"), clock=clock,
              raw_dir=raw, label=label)
    app.restart(clock())

    def tick(k=-1):
        app.tick(k, clock())
        clock.t += 1 / 30

    tick(KEY_SPACE)
    for _ in TESTS:
        tick(KEY_SPACE)
        while app.state != "done":
            tick()
        tick(KEY_SPACE)
    return app, raw


def test_every_test_is_saved_and_rescored_identically(tmp_path):
    app, raw = record_session(tmp_path, "alex_normal")
    files = sorted(glob.glob(os.path.join(raw, "*.json")))
    assert len(files) == len(TESTS)
    for path in files:
        rec = load_record(path)
        assert rec["meta"]["label"] == "alex_normal" and rec["meta"]["mode"] == "SIM"
        f, r, fu = reanalyse(rec)
        key = (rec["meta"]["test"], rec["meta"]["hand"])
        assert r.score == app.results[key].result.score, path
        json.dumps(rec)                                     # plain JSON


def test_calibrate_report(tmp_path):
    _, raw = record_session(tmp_path, "alex_normal")
    rows = load_all(raw)
    assert len(rows) == len(TESTS) and all(r["normal"] for r in rows)
    text = report(rows)
    assert "WARNING" in text and "SIM" in text          # sim data must be called out
    assert "TREMOR" in text and "suggestions" in text
    assert is_normal("Sam_Baseline") and not is_normal("sam_acted_slow")
