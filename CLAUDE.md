# NeuroCheck — Parkinson's Motor Check Station
A tabletop check-up with three tests per hand, each scored 0–4 with transparent rules shown on
screen: rest tremor (webcam), finger tapping (webcam), rapid hand flipping (SW-520D tilt switch).
Inspired by CeMoQu (explainable SARA scoring). TRACKING and DECISION-SUPPORT tool, NOT a
diagnosis. All thresholds labeled "demo thresholds" in the UI.

There is NO accelerometer/IMU in this project (no MPU6050). The only motion sensor is an
SW-520D ball tilt switch, which reports ON/OFF by orientation and cannot measure tremor.

## Stack
Python 3.11, opencv-python, mediapipe, numpy, scipy, pyserial, matplotlib, pytest.
One OpenCV window, keyboard controlled.

## Architecture rule
All maths lives in pure functions (tremor_analysis, tapping_analysis, flipping_analysis, scoring)
with no hardware or UI, so it can be tested with synthetic data.

## Hardware (Arduino Uno) — final pin table
| Part | Pin(s) | Notes |
|---|---|---|
| SW-520D tilt switch | D2 | bare 2-leg: other leg to GND (INPUT_PULLUP); 3-pin module: DO→D2, VCC 5V, GND |
| Onboard LED | D13 | mirrors the debounced switch state |
| Buzzer | D8 | active or passive (#define) |
| RGB LED | R D9, G D10, B D5 (PWM, via 220 Ω) | common cathode or anode (#define). Not D3/D11: tone() uses Timer2, which breaks PWM there |
| LCD1602 parallel (optional) | RS D12, E D11, D4 D7, D5 D6, D6 D4, D7 D3; V0 to 10k pot middle | digital use only, no conflict with Timer2 |
| LCD1602 I2C (optional) | SDA A4, SCL A5 | address 0x27 or 0x3F, auto-detected |
| Serial | D0/D1 | USB |

## Session flow
Welcome -> Right tremor (10s) -> Left tremor (10s) -> Right finger tapping (10s)
-> Left finger tapping (10s) -> Right hand flipping (10s) -> Left hand flipping (10s)
-> Results -> Trend. Each test: Ready screen, 3-2-1 countdown with beeps, recording, "done" beep.
Spacebar advances. R restarts. Q quits.

## Serial protocol (115200 baud, newline-terminated)
Arduino -> PC: READY on boot; S,millis,0|1 on every debounced switch change (only between START
and STOP); reply to STATE: S,millis,0|1
PC -> Arduino: START, STOP, STATE, BEEP,n, LED,G|Y|R|OFF, LCD,line1|line2 (16 chars max per line)

## Tremor test (webcam, MDS-UPDRS 3.17 style, demo thresholds)
Patient holds the hand still in front of the camera, palm facing it, for 10 s. Track index
fingertip (8) and wrist (0) in pixels with frame timestamps. Pixels -> cm: wrist(0) to
middle-MCP(9) = 9 cm (adult average), median over the recording. Remove drift (detrend /
high-pass), resample to uniform 30 Hz, FFT on x and y, sum power, peak in 3–8 Hz, peak
displacement amplitude in cm.
0: <0.1 cm or no clear peak (peak < 3× median power in band), 1: <1 cm, 2: 1–3 cm, 3: 3–10 cm,
4: ≥10 cm. Reasons mention the frequency and whether it's in the Parkinson's range (4–6 Hz).
Live fingertip trace while recording, then spectrum with 4–6 Hz band shaded.
UI note: "Camera tremor: may miss very small tremors (< ~0.5 cm)".

## Finger tapping (webcam, MDS-UPDRS 3.4 style, demo thresholds)
Distance = thumb-tip(4) to index-tip(8), normalised by wrist(0) to middle-MCP(9) length.
Features: taps/sec, mean amplitude, decrement (mean amplitude first 3s vs last 3s),
interval CV, hesitations (interval > 2× median).
Problems: slow (<2 taps/s), small amplitude, decrement >30%, irregular (CV >0.3), any hesitation.
Score: 0 no problems, 1 one, 2 two, 3 three or more, 4 fewer than 5 taps.

## Rapid hand flipping (SW-520D, MDS-UPDRS 3.6 style, demo thresholds)
Tilt switch taped to the back of the hand; flip palm-down / palm-up as fast and fully as
possible for 10 s, once per hand. Each debounced switch change = one half-flip; two = one full flip.
Calibration on the Ready screen: "Hold your hand palm-down", PC sends STATE. If no change in
the first 3 s of the test, show "Sensor not flipping — check it's taped on and upright".
Python also ignores any change < 60 ms after the previous one (ball bounce).
Features: full flips/sec, total flips, interval CV, decrement (rate first 4 s vs last 4 s),
hesitations (half-flip interval > 2× median).
Problems: slow (< 1.5 full flips/s), irregular (CV > 0.35), decrement > 25%, any hesitation.
Score: 0 no problems, 1 one, 2 two, 3 three or more, 4 fewer than 5 full flips.
Live: flip counter, big PALM DOWN / PALM UP indicator, timeline of flips.

## Asymmetry
Flag if left vs right differ by ≥1 point, or ≥25% on taps/sec, flips/sec or tremor displacement.

## Sim mode
--sim: MockDevice (flip events: left normal, right slower + decrementing with a pause) +
FakeHand (tapping profiles; 5 Hz tremor of a set size). --sim must always work.

## Rules
Working > pretty. --sim must always work. Commit after each working step.
Never claim something works without running it.

## Environment notes (this machine)
- System Python is 3.14; mediapipe 0.10.14 needs <=3.12, so use the 3.11 venv: `.venv\Scripts\python`.
- `uv` needs `--native-tls` here (Cloudflare WARP intercepts TLS).
- arduino-cli is a portable exe in `~/.local/bin` (winget MSI needs admin).
- Python edits via scripts: always open files with encoding="utf-8" (Windows default is cp1252).
