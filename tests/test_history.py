from history import load_sessions, save_session, trend_image


def test_save_and_load_roundtrip(tmp_path):
    path = str(tmp_path / "s.csv")
    save_session({"tremor_R": 2, "tap_L": None, "tremor_R_cm": 1.23456, "bogus": 1}, path)
    save_session({"tremor_R": 1}, path)
    rows = load_sessions(path)
    assert len(rows) == 2
    assert rows[0]["tremor_R"] == "2" and rows[0]["tap_L"] == ""
    assert rows[0]["tremor_R_cm"] == "1.235"
    assert "bogus" not in rows[0] and rows[0]["timestamp"]


def test_trend_image_shape(tmp_path):
    assert trend_image([], 640, 360).shape == (360, 640, 3)
    rows = [{"tremor_R": "1", "tremor_L": "0", "tap_R": "", "tap_L": "2"}] * 3
    assert trend_image(rows, 640, 360).shape == (360, 640, 3)
