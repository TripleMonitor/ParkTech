"""HL7 FHIR R4 export: a Bundle (type "collection") with a placeholder Patient and one
Observation per test score and per measured feature.

Codes: NeuroCheck LOCAL codes (system LOCAL_SYSTEM) with display text - no LOINC or
SNOMED codes are claimed. Units use UCUM (http://unitsofmeasure.org).
This is a demo export of a tracking tool, not a clinical record.
"""
from __future__ import annotations

import json
import uuid
from typing import Optional

LOCAL_SYSTEM = "urn:neurocheck:local-codes"
UCUM = "http://unitsofmeasure.org"
CATEGORY_SYSTEM = "http://terminology.hl7.org/CodeSystem/observation-category"

# feature key -> (display, UCUM unit code, human unit)
FEATURES = {
    "tremor": {"peak_hz": ("Tremor peak frequency", "Hz", "Hz"),
               "displacement_cm": ("Tremor fingertip displacement", "cm", "cm")},
    "tapping": {"taps_per_sec": ("Finger tap rate", "/s", "taps/s"),
                "mean_amplitude": ("Finger tap amplitude (normalised)", "1", "ratio"),
                "decrement": ("Finger tap amplitude decrement", "%", "%")},
    "flipping": {"flips_per_sec": ("Hand flip rate", "/s", "full flips/s"),
                 "interval_cv": ("Hand flip rhythm CV", "1", "ratio"),
                 "median_amplitude_deg": ("Hand flip rotation (camera)", "deg", "deg")},
}
TITLES = {"tremor": "Rest tremor score (MDS-UPDRS 3.17 style, demo)",
          "tapping": "Finger tapping score (MDS-UPDRS 3.4 style, demo)",
          "flipping": "Hand flipping score (MDS-UPDRS 3.6 style, demo)"}


def _uuid() -> str:
    return f"urn:uuid:{uuid.uuid4()}"


def _coding(code: str, display: str) -> dict:
    return {"coding": [{"system": LOCAL_SYSTEM, "code": code, "display": display}],
            "text": display}


def _observation(patient_ref: str, when: str, code: str, display: str, value: Optional[float],
                 unit: Optional[tuple] = None, note: Optional[list] = None,
                 body_site: Optional[str] = None) -> dict:
    obs = {"resourceType": "Observation", "status": "final",
           "category": [{"coding": [{"system": CATEGORY_SYSTEM, "code": "exam", "display": "Exam"}]}],
           "code": _coding(code, display), "subject": {"reference": patient_ref},
           "effectiveDateTime": when}
    if value is None:
        obs["dataAbsentReason"] = {"text": "Not scored (insufficient data)"}
    elif unit is None:
        obs["valueInteger"] = int(value)
    else:
        obs["valueQuantity"] = {"value": round(float(value), 4), "unit": unit[1],
                                "system": UCUM, "code": unit[0]}
    if body_site:
        obs["bodySite"] = {"text": f"{body_site} hand"}
    if note:
        obs["note"] = [{"text": n} for n in note]
    return obs


def build_bundle(snapshot: dict) -> dict:
    """snapshot: plain dict from App.session_snapshot()."""
    patient_url = _uuid()
    when = snapshot["timestamp"]
    entries = [{"fullUrl": patient_url, "resource": {
        "resourceType": "Patient", "active": True,
        "identifier": [{"system": "urn:neurocheck:session", "value": snapshot["session_id"]}],
        "name": [{"text": "Demo patient (placeholder)"}]}}]
    for t in snapshot["tests"]:
        kind, hand = t["kind"], t["hand"]
        entries.append({"fullUrl": _uuid(), "resource": _observation(
            patient_url, when, f"{kind}-score", TITLES[kind], t["score"], None,
            note=t["reasons"] + ["Demo thresholds - tracking aid, not a diagnosis"], body_site=hand)})
        for key, (display, ucum, human) in FEATURES[kind].items():
            v = t["features"].get(key)
            if v is None:
                continue
            if ucum == "%":
                v = 100.0 * v
            entries.append({"fullUrl": _uuid(), "resource": _observation(
                patient_url, when, f"{kind}-{key}", display, v, (ucum, human), body_site=hand)})
    if snapshot.get("neuroscore") is not None:
        entries.append({"fullUrl": _uuid(), "resource": _observation(
            patient_url, when, "neuroscore", "NeuroScore composite tracking index (0-100)",
            snapshot["neuroscore"], ("1", "score"),
            note=[snapshot["neuroscore_formula"], "Composite tracking index, not a diagnosis"])})
    if snapshot.get("dose_hours") is not None:
        entries.append({"fullUrl": _uuid(), "resource": _observation(
            patient_url, when, "hours-since-levodopa", "Hours since last levodopa dose (self-report)",
            snapshot["dose_hours"], ("h", "h"))})
    return {"resourceType": "Bundle", "type": "collection", "timestamp": when, "entry": entries}


def save_bundle(snapshot: dict, path: str) -> str:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(build_bundle(snapshot), f, indent=2)
    return path


def validate_bundle(b: dict) -> list[str]:
    """Structural checks for the subset of FHIR R4 we emit. Returns a list of problems."""
    errs = []
    if b.get("resourceType") != "Bundle" or b.get("type") != "collection":
        errs.append("not a collection Bundle")
    urls = set()
    for i, e in enumerate(b.get("entry", [])):
        url, r = e.get("fullUrl", ""), e.get("resource", {})
        if not url.startswith("urn:uuid:") or url in urls:
            errs.append(f"entry {i}: bad or duplicate fullUrl")
        urls.add(url)
        rt = r.get("resourceType")
        if rt == "Observation":
            if r.get("status") != "final":
                errs.append(f"entry {i}: status")
            cod = r.get("code", {}).get("coding", [{}])[0]
            if cod.get("system") != LOCAL_SYSTEM or not cod.get("code") or not cod.get("display"):
                errs.append(f"entry {i}: code must be a local code with display")
            if "loinc" in json.dumps(r).lower():
                errs.append(f"entry {i}: LOINC must not be claimed")
            if not r.get("subject", {}).get("reference", "").startswith("urn:uuid:"):
                errs.append(f"entry {i}: subject")
            has_val = any(k in r for k in ("valueQuantity", "valueInteger", "dataAbsentReason"))
            if not has_val:
                errs.append(f"entry {i}: no value")
            q = r.get("valueQuantity")
            if q and (q.get("system") != UCUM or "code" not in q or not isinstance(q.get("value"), (int, float))):
                errs.append(f"entry {i}: valueQuantity must be UCUM with numeric value")
            if "effectiveDateTime" not in r:
                errs.append(f"entry {i}: effectiveDateTime")
        elif rt != "Patient":
            errs.append(f"entry {i}: unexpected resourceType {rt}")
    if not any(e["resource"]["resourceType"] == "Patient" for e in b.get("entry", [])):
        errs.append("no Patient")
    return errs
