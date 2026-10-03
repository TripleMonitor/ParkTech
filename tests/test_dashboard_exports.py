"""NeuroScore, radar, FHIR bundle structure and PDF export."""
import json

import pytest

from dashboard import neuroscore, radar_image, radar_values
from fhir import LOCAL_SYSTEM, build_bundle, validate_bundle
from report import save_pdf


def snapshot(**over):
    s = {"session_id": "NC-TEST", "timestamp": "2026-10-03T15:00:00-07:00", "mode": "LIVE",
         "dose_hours": 2.5, "neuroscore": 75.0, "neuroscore_formula": "NeuroScore = ...",
         "asymmetry": ["Tremor: Right worse (R 2 vs L 0)"], "coach_rates": {"Right": 2.3},
         "tests": [
             {"kind": "tremor", "hand": "Right", "score": 2, "reasons": ["Peak 5.0 Hz"],
              "features": {"peak_hz": 5.0, "displacement_cm": 1.5}},
             {"kind": "tapping", "hand": "Left", "score": None, "reasons": ["Hand seen 20%"],
              "features": {}},
             {"kind": "flipping", "hand": "Right", "score": 3, "reasons": ["Slow"],
              "features": {"flips_per_sec": 1.0, "interval_cv": 0.7, "median_amplitude_deg": 89.0}},
             {"kind": "tapping", "hand": "Right", "score": 1, "reasons": ["Slow"],
              "features": {"taps_per_sec": 1.4, "mean_amplitude": 0.9, "decrement": 0.36}},
         ]}
    s.update(over)
    return s


def test_neuroscore_formula():
    v, text = neuroscore({("tremor", "Right"): 2, ("tremor", "Left"): 0,
                          ("tapping", "Right"): None})
    assert v == pytest.approx(75.0)
    assert "100 x (1 - 2 / (4 x 2)) = 75" in text and "not scored: R tapping" in text
    assert neuroscore({("tremor", "Right"): None})[0] is None
    assert neuroscore({("a", "R"): 0, ("b", "L"): 0})[0] == 100.0
    assert neuroscore({("a", "R"): 4})[0] == 0.0


def test_radar_values_and_image():
    scores = {("tremor", "Right"): 2, ("tapping", "Right"): 0, ("flipping", "Right"): None}
    assert radar_values(scores, {"Right": 2.0}, "Right") == [50.0, 100.0, None, 50.0]
    img = radar_image(scores, {}, 400, 360)
    assert img.shape == (360, 400, 3)


def test_fhir_bundle_is_valid_and_uses_local_codes():
    b = build_bundle(snapshot())
    assert validate_bundle(b) == []
    json.dumps(b)                                                  # serialisable
    obs = [e["resource"] for e in b["entry"] if e["resource"]["resourceType"] == "Observation"]
    assert all(o["code"]["coding"][0]["system"] == LOCAL_SYSTEM for o in obs)
    assert "loinc" not in json.dumps(b).lower()
    unscored = [o for o in obs if o["code"]["coding"][0]["code"] == "tapping-score"
                and o["bodySite"]["text"] == "Left hand"]
    assert unscored and "dataAbsentReason" in unscored[0]
    dec = [o for o in obs if o["code"]["coding"][0]["code"] == "tapping-decrement"][0]
    assert dec["valueQuantity"]["code"] == "%" and dec["valueQuantity"]["value"] == pytest.approx(36.0)


def test_fhir_validator_catches_problems():
    b = build_bundle(snapshot())
    b["entry"][1]["resource"]["code"]["coding"][0]["system"] = "http://loinc.org"
    b["entry"][2]["fullUrl"] = b["entry"][1]["fullUrl"]
    errs = validate_bundle(b)
    assert any("local code" in e for e in errs) and any("duplicate" in e for e in errs)


def test_pdf_export(tmp_path):
    from history import load_sessions, seed_history
    path = str(tmp_path / "s.csv")
    seed_history(path, days=3)
    out = save_pdf(snapshot(), load_sessions(path), str(tmp_path / "r.pdf"))
    data = open(out, "rb").read()
    assert data.startswith(b"%PDF") and len(data) > 5000


def test_fhir_sim_session_is_tagged_preliminary():
    b = build_bundle(snapshot(mode="SIMULATION"))
    assert validate_bundle(b) == []
    assert b["meta"]["tag"][0]["code"] == "SIM"
    obs = [e["resource"] for e in b["entry"] if e["resource"]["resourceType"] == "Observation"]
    assert all(o["status"] == "preliminary" and o["meta"]["tag"][0]["code"] == "SIM" for o in obs)
    assert all(o["note"][0]["text"] == "SIMULATION - not measured" for o in obs)


def test_fhir_requires_timezone():
    b = build_bundle(snapshot(timestamp="2026-10-03T15:00:00"))
    assert any("timezone" in e for e in validate_bundle(b))
