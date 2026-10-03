# NeuroCheck — Parkinson's Motor Check Station
A 1-minute tabletop check-up: wrist tremor test (MPU6050) + finger tapping test (webcam/MediaPipe),
each scored 0–4 with transparent rules shown on screen. Inspired by CeMoQu (explainable SARA
scoring). TRACKING and DECISION-SUPPORT tool, NOT a diagnosis. All thresholds labeled
"demo thresholds" in the UI.

## Stack
Python 3.11, opencv-python, mediapipe, numpy, scipy, pyserial, matplotlib, pytest.
One OpenCV window, keyboard controlled.

## Architecture rule
All maths lives in pure functions (tremor_analysis, tapping_analysis, scoring) with no hardware or
UI, so it can be tested with synthetic data.

## Hardware (Arduino Uno)
MPU6050: VCC 5V, GND, SDA A4, SCL A5. Buzzer: pin 8. RGB LED: pins 9 R, 10 G, 11 B via 220Ω.
LCD1602 (optional): parallel RS 7, E 6, D4 5, D5 4, D6 3, D7 2, V0 to 10k pot middle — or I2C on A4/A5.

## Session flow
Welcome -> Right tremor (10s) -> Left tremor (10s) -> Right tapping (10s) -> Left tapping (10s)
-> Results -> Trend. Each test: 3-2-1 countdown with beeps, recording, "done" beep.
Spacebar advances. R restarts. Q quits.

## Serial protocol (115200 baud, newline-terminated)
Arduino -> PC: READY on boot; T,millis,ax,ay,az (m/s², 100 Hz, only while recording)
PC -> Arduino: START, STOP, BEEP,n, LED,G|Y|R|OFF, LCD,line1|line2 (16 chars max per line)

## Tremor scoring (MDS-UPDRS 3.17 style, demo thresholds)
Subtract each axis's mean (removes gravity), detrend, FFT each axis, sum power spectra.
Peak frequency = max in 3–8 Hz. Amplitude = peak sinusoid accel amplitude at that frequency on
the strongest axis. Displacement cm = accel / (2πf)² × 100.
0: <0.1 cm or no clear peak (peak < 3× median power in band), 1: <1 cm, 2: 1–3 cm, 3: 3–10 cm, 4: ≥10 cm.
Reason strings mention frequency and whether it's in the Parkinson's range (4–6 Hz).

## Tapping scoring (MDS-UPDRS 3.4 style, demo thresholds)
Distance = thumb-tip(4) to index-tip(8), normalised by wrist(0) to middle-MCP(9) length.
Features: taps/sec, mean amplitude, decrement (mean amplitude first 3s vs last 3s),
interval CV, hesitations (interval > 2× median).
Problems: slow (<2 taps/s), small amplitude, decrement >30%, irregular (CV >0.3), any hesitation.
Score: 0 no problems, 1 one mild problem, 2 two problems, 3 three or more, 4 fewer than 5 taps.

## Asymmetry
Flag if left vs right differ by ≥1 point, or ≥25% on taps/sec or tremor displacement.

## Rules
Working > pretty. --sim must always work. Commit after each working step.
Never claim something works without running it.

## Environment notes (this machine)
- System Python is 3.14; mediapipe 0.10.14 needs <=3.12, so use the 3.11 venv: `.venv\Scripts\python`.
- `uv` needs `--native-tls` here (Cloudflare WARP intercepts TLS).
- arduino-cli is a portable exe in `~/.local/bin` (winget MSI needs admin).
