# NeuroCheck — Parkinson's Motor Check Station
A tabletop check-up with three tests per hand, each scored 0–4 with transparent rules shown on
screen: rest tremor (webcam), finger tapping (webcam), rapid hand flipping (SW-520D tilt switch +
webcam fusion). Plus a PID rhythm coach, a results dashboard and exports.
Inspired by CeMoQu (explainable SARA scoring). TRACKING and DECISION-SUPPORT tool, NOT a
diagnosis. All thresholds labeled "demo thresholds" in the UI.

## Golden rule
EVERY number on screen is real (measured live) or clearly labeled "SIM" / "DEMO DATA".
Never fake a live value.

There is NO accelerometer/IMU in this project (no MPU6050). The only motion sensor is an
SW-520D ball tilt switch, which reports ON/OFF by orientation and cannot measure tremor.

## Stack
Python 3.11, opencv-python, mediapipe 0.10.14, numpy, scipy, pyserial, matplotlib, pytest.
One OpenCV window, keyboard controlled, HUD theme (DejaVu Sans Mono from matplotlib).

## Architecture rule
All maths lives in pure functions with no hardware or UI, tested with synthetic data:
tremor_analysis, tapping_analysis, flipping_analysis, fusion, quality, scoring, pid,
rhythm_coach, dashboard, fhir.

## Hardware (Arduino Uno) — final pin table
| Part | Pin(s) | Notes |
|---|---|---|
| SW-520D tilt switch | D2 | bare 2-leg: other leg to GND (INPUT_PULLUP); 3-pin module: DO→D2, VCC 5V, GND |
| Onboard LED | D13 | mirrors the debounced switch state |
| Buzzer | D6 | active or passive (#define BUZZER_PASSIVE) |
| 3 separate LEDs | green D9, yellow D10, red D11 (long leg via 220 Ω, short leg GND) | plain on/off; boot lights G→Y→R once to check the order |
| LCD1602 parallel (optional, not fitted) | RS D12, E D8, D4 D7, D5 D5, D6 D4, D7 D3; V0 to 10k pot middle | digital use only |
| LCD1602 I2C (optional) | SDA A4, SCL A5 | address 0x27 or 0x3F, auto-detected |
| Serial | D0/D1 | USB |
#defines: SWITCH_MODULE, BUZZER_PASSIVE, LCD_MODE (NONE/I2C/PARALLEL).
tools/check_pins.py verifies no pin conflicts; firmware/compile_all.sh builds all 12 variants.
This build: bare 2-leg SW-520D on D2, no LCD. LED,G|Y|R lights one LED (B = all three).
Live LED during hand flipping = measured flip speed over the last 2 s (green >= 1.5 full
flips/s, yellow >= 1.0, red below); after the session = worst score (green 0-1, yellow 2, red 3-4).

## Session flow
Boot self-check -> Welcome (asks hours since last levodopa dose) -> Right tremor -> Left tremor
-> Right finger tapping -> Left finger tapping -> Right hand flipping -> Left hand flipping
-> Results (6 cards) -> Dashboard (radar + NeuroScore) -> Trend.
Each test: Ready screen, 3-2-1 countdown with beeps, 10 s recording, "done" beep, WHY THIS SCORE
panel. Keys: SPACE next, R restart (any screen), Q quit, H trend, C rhythm coach,
P doctor PDF / F FHIR JSON (results/dashboard), sim only: T fake tremor, F stuck sensor.

## Serial protocol (115200 baud, newline-terminated)
Arduino -> PC: READY on boot; S,millis,0|1 on every debounced switch change (only between START
and STOP) and as the reply to STATE; C,millis one per metronome beat while the cue is on.
PC -> Arduino: START, STOP, STATE, BEEP,n, LED,G|Y|R|OFF, LCD,line1|line2 (16 chars max),
CUE,ON,<ms>, CUE,OFF, TEMPO,<ms> (interval clamped 200–1500 ms; TEMPO applies from the next beat).
Firmware: 30 ms non-blocking debounce, metronome timed with millis() (beep + LED flash).

## Tremor test (webcam, MDS-UPDRS 3.17 style, demo thresholds)
Hand held still, palm to camera, 10 s. Index fingertip (8) in pixels -> cm using wrist(0) to
middle-MCP(9) = 9 cm (median over the recording). Cubic resample to 30 Hz, detrend + 1 Hz
high-pass, Hann FFT on x and y, power summed, peak in 3–8 Hz; clear peak = >= 3x median power
in band AND a local maximum. Amplitude = vector amplitude from band-limited RMS (±0.75 Hz).
0: <0.1 cm or no clear peak, 1: <1 cm, 2: 1–3 cm, 3: 3–10 cm, 4: ≥10 cm.
UI note: "Camera tremor: may miss very small tremors (< ~0.5 cm)".

## Finger tapping (webcam, MDS-UPDRS 3.4 style, demo thresholds)
Distance = thumb-tip(4) to index-tip(8) / wrist(0)-middle-MCP(9). Taps = openings (prominence
>= 0.15). Problems: slow (<1.5 taps/s), small amplitude (<0.8), decrement >30% (first vs last 3 s),
irregular (CV >0.5), any hesitation (interval > 2x median).
Thresholds tuned on the HUBU-FIS dataset (234 clinician-rated videos, 118 people, Zenodo
10.5281/zenodo.17738775, CC-BY-4.0) with participant-level 5-fold CV (tools/hubu_eval.py):
agreement with clinicians exact 45%, within 1 point 85%, weighted kappa 0.49 (original
demo thresholds: 42% / 85% / 0.47). Dataset lives in datasets/ (git-ignored). Tracking gaps > 0.15 s are excluded
from rhythm. Score: problems count capped at 3; 4 if fewer than 5 taps.

## Hand flipping (SW-520D + camera fusion, MDS-UPDRS 3.6 style, demo thresholds)
Each debounced switch change = one half-flip. Calibration: Ready screen state = palm-down.
Python ignores changes not held >= 60 ms. Features: full flips/s, CV, decrement (rate first
4 s vs last 4 s), hesitations. Fusion: palm normal from MediaPipe world landmarks 0, 5, 17;
rotation vs palm-down reference; camera swings >= 45 deg counted; counts within 15% =
FUSION LOCKED else FUSION MISMATCH (low confidence). Problems: slow (<1.5 full flips/s),
irregular (CV >0.35), decrement >25%, any hesitation, small rotation (<120 deg median),
rotation decrement >25% (camera rows only when the camera tracked the hand).
Score: problems capped at 3; 4 if < 5 full flips; no flips -> 4 + sensor-check reason.

## Signal quality
Camera tests: hand-detected %, MediaPipe confidence, dropped frames (failed reads + gaps >2x
median). Out of range (<80%, <0.70, >10%) -> "LOW CONFIDENCE" on the result.

## Asymmetry
Flag if left vs right differ by ≥1 point, or ≥25% on taps/sec, flips/sec or tremor
displacement (% relative to the larger side; displacement only from clear peaks ≥ 0.1 cm).

## Rhythm coach (C)
10 s uncued + 35 s cued. Beat = one half-flip. Graded on-time credit (1 within ±75 ms, 0 at
±150 ms) over the last 8 answered beats; set point 0.85; PID (reverse acting) outputs a tempo
change, limited ±0.15 beats/s per beat. Warm start 1.15x uncued rate; saturation probing
(+0.02/beat while 8/8 on time and tight). Random misses after an on-time beat are excluded;
misses after a late beat count as late. Gains from tools/tune_pid.py on a SimulatedPatient
(docs/pid_tuning.png): only P-only settings passed all criteria (kp 0.01, ki 0, kd 0).
Result: max sustainable rhythm, mean asynchrony, cued vs uncued rate.

## Dashboard, data, exports
NeuroScore = 100 x (1 - sum(scores) / (4 x number of scored tests)), formula on screen,
"composite tracking index, not a diagnosis". Radar "Motor Fingerprint" (function % per test,
coach axis = rate / 4.0 beats/s demo reference). sessions.csv stores every session incl.
dose_hours and neuroscore. --seed-history: 14 days x 2 sessions, wearing-off pattern, right
worse; seeded points drawn hollow and labeled DEMO DATA. Exports: P one-page PDF, F FHIR R4
Bundle (Patient placeholder + Observations, LOCAL codes only, UCUM units, no LOINC).

## Raw data + calibration
`app.py --record [DIR] --label NAME` saves every test's raw signals (recorder.py, JSON).
`tools/calibrate.py [DIR]` re-scores them with the current maths and suggests thresholds
from normal (label contains normal/healthy/baseline) vs acted runs. Healthy-volunteer
calibration only - not clinical validation. recordings/ and exports/ are git-ignored.

## Sim mode
--sim: MockDevice (flip events; right impaired, left normal; metronome + SimulatedPatient for
the coach) + FakeHand (tapping profiles; 5 Hz tremor of set size; palm rotation following the
fake switch). Every sim value is labeled SIM in the header/telemetry. --sim must always work.

## Rules
Working > pretty. --sim must always work. Commit after each working step.
Never claim something works without running it.

## Environment notes (this machine)
- System Python is 3.14; mediapipe 0.10.14 needs <=3.12, so use the 3.11 venv: `.venv\Scripts\python`.
- `uv` needs `--native-tls` here (Cloudflare WARP intercepts TLS).
- arduino-cli is a portable exe in `~/.local/bin` (winget MSI needs admin).
- Python edits via scripts: always open files with encoding="utf-8" (Windows default is cp1252).
