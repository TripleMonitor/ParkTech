# NeuroCheck - Parkinson's Motor Check Station

A 1-minute tabletop check-up: wrist tremor (MPU6050) + finger tapping (webcam/MediaPipe),
each scored 0-4 with the rules that fired shown on screen. Inspired by CeMoQu.

> **Tracking and decision-support tool, NOT a diagnosis. All thresholds are demo thresholds.**

## Setup (Windows)
MediaPipe 0.10.14 needs Python 3.11:
```
uv venv --python 3.11 .venv          # add --native-tls behind a TLS-inspecting proxy/VPN
uv pip install --python .venv\Scripts\python.exe -r requirements.txt
```

## Run
```
.venv\Scripts\python app.py --sim              # fake IMU, press T to toggle a 5 Hz tremor
.venv\Scripts\python app.py                    # real Arduino (auto-detects port)
.venv\Scripts\python app.py --port COM5 --seconds 10 --camera 0
.venv\Scripts\python -m pytest                 # tests
```
Keys: `SPACE` start/next, `S` skip test, `T` sim tremor, `H` trend chart, `R` restart, `Q`/`ESC` quit.

## Flow
Welcome -> R tremor -> L tremor -> R tapping -> L tapping -> Results -> (H) Trend.
Each test: Ready screen, 3-2-1 countdown with beeps, recording, done beep.
Sessions are appended to `sessions.csv`.

## Scoring (see `scoring.py` - all constants at the top)
- **Tremor**: per-axis mean removed (gravity), Hann-windowed FFT per axis, spectra combined.
  Peak in 3-8 Hz must be >= 6x the median 1-20 Hz amplitude to count as "clear".
  Displacement = a / (2*pi*f)^2. 0: none/<0.1 cm, 1: <1, 2: 1-3, 3: 3-10, 4: >=10 cm.
- **Tapping**: thumb-tip/index-tip distance / wrist-to-middle-MCP length. Each opening is a tap.
  Problems: slow (<2 taps/s; <1 counts double), small (<0.5), decrement (>25% first vs last 3 s),
  irregular (interval CV >0.25), hesitations (gap >2x median; >=3 counts double).
  0 problems -> 0, 1 -> 1, 2 -> 2, 3+ -> 3, <=3 taps total -> 4. Hand visible <50% -> not scored.
- **Asymmetry**: flagged if R/L scores differ by >=1 or a key feature differs by >=25%.

## Hardware (`firmware/neurocheck/neurocheck.ino`)
Uno/Nano, MPU6050 + 16x2 I2C LCD on A4/A5, buzzer D8, common-cathode RGB LED D9/D10/D11.
Needs the "LiquidCrystal I2C" library (Frank de Brabander).
