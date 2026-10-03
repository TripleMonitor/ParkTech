from datetime import datetime

import pytest

from history import (FIELDS, load_sessions, neuroscore_of_row, neuroscore_series, save_session,
                     seed_history, trend_image)


def test_save_and_load_roundtrip(tmp_path):
    path = str(tmp_path / "s.csv")
    save_session({"tremor_R": 2, "tap_L": None, "tremor_R_cm": 1.23456, "bogus": 1}, path)
    save_session({"tremor_R": 1}, path)
    rows = load_sessions(path)
    assert len(rows) == 2
    assert rows[0]["tremor_R"] == "2" and rows[0]["tap_L"] == ""
    assert rows[0]["tremor_R_cm"] == "1.235"
    assert "bogus" not in rows[0] and rows[0]["timestamp"]


def test_load_missing_file(tmp_path):
    assert load_sessions(str(tmp_path / "nope.csv")) == []


def test_old_header_is_migrated(tmp_path):
    path = tmp_path / "s.csv"
    path.write_text("timestamp,tremor_R\n2026-10-01T10:00:00,2\n", encoding="utf-8")
    save_session({"tremor_R": 1, "neuroscore": 75.0}, str(path))
    rows = load_sessions(str(path))
    assert list(rows[0].keys()) == FIELDS
    assert rows[0]["tremor_R"] == "2" and rows[1]["neuroscore"] == "75.000"


def test_seed_history_wearing_off_and_right_worse(tmp_path):
    path = str(tmp_path / "s.csv")
    rows = seed_history(path, days=14, now=datetime(2026, 10, 3, 12))
    assert len(rows) == 28 and all(r["seeded"] == "1" for r in rows)
    am = [float(r["neuroscore"]) for r in rows if r["session_id"].endswith("am")]
    pm = [float(r["neuroscore"]) for r in rows if r["session_id"].endswith("pm")]
    assert sum(pm) / len(pm) < sum(am) / len(am)                    # wearing off
    right = sum(int(r["tremor_R"]) + int(r["tap_R"]) + int(r["flip_R"]) for r in rows)
    left = sum(int(r["tremor_L"]) + int(r["tap_L"]) + int(r["flip_L"]) for r in rows)
    assert right > left
    assert seed_history(path) == [] and len(load_sessions(path)) == 28   # no duplicates


def test_neuroscore_from_row():
    assert neuroscore_of_row({"neuroscore": "80"}) == 80.0
    row = {"tremor_R": "2", "tremor_L": "0", "tap_R": "", "tap_L": "", "flip_R": "", "flip_L": ""}
    assert neuroscore_of_row(row) == pytest.approx(75.0)
    assert neuroscore_of_row({}) is None


def test_trend_image_shapes(tmp_path):
    assert trend_image([], 640, 360).shape == (360, 640, 3)
    rows = seed_history(str(tmp_path / "s.csv"))
    xs, ys, sd = neuroscore_series(load_sessions(str(tmp_path / "s.csv")))
    assert len(xs) == 28 and all(sd)
    img = trend_image(rows, 1280, 720)
    assert img.shape == (720, 1280, 3) and img.std() > 10


def test_bom_from_excel_is_handled(tmp_path):
    path = tmp_path / "s.csv"
    save_session({"timestamp": "2026-10-01T10:00:00", "tremor_R": 1}, str(path))
    path.write_bytes(b"\xef\xbb\xbf" + path.read_bytes())
    assert load_sessions(str(path))[0]["timestamp"] == "2026-10-01T10:00:00"
