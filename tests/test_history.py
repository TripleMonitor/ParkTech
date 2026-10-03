from datetime import datetime

from history import load_sessions, save_session, seed_history, trend_image


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


def test_seed_history_right_hand_worsens(tmp_path):
    path = str(tmp_path / "s.csv")
    rows = seed_history(path, days=7, now=datetime(2026, 10, 3, 12))
    assert len(rows) == 7 and len(load_sessions(path)) == 7
    assert rows[0]["timestamp"].startswith("2026-09-26")
    r_tremor = [int(r["tremor_R"]) for r in rows]
    r_tap = [int(r["tap_R"]) for r in rows]
    assert r_tremor == sorted(r_tremor) and r_tremor[-1] > r_tremor[0]
    assert r_tap == sorted(r_tap) and r_tap[-1] > r_tap[0]
    assert all(r["tremor_L"] == "0" and r["tap_L"] == "0" for r in rows)


def test_trend_image_shapes(tmp_path):
    assert trend_image([], 640, 360).shape == (360, 640, 3)
    rows = seed_history(str(tmp_path / "s.csv"))
    img = trend_image(rows, 1280, 720)
    assert img.shape == (720, 1280, 3) and img.std() > 10


def test_seed_history_is_not_duplicated(tmp_path):
    path = str(tmp_path / "s.csv")
    seed_history(path)
    assert seed_history(path) == []
    assert len(load_sessions(path)) == 7


def test_bom_from_excel_is_handled(tmp_path):
    path = tmp_path / "s.csv"
    save_session({"timestamp": "2026-10-01T10:00:00", "tremor_R": 1}, str(path))
    path.write_bytes(b"\xef\xbb\xbf" + path.read_bytes())       # Excel-style BOM
    rows = load_sessions(str(path))
    assert rows[0]["timestamp"] == "2026-10-01T10:00:00"
