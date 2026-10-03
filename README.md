# NeuroCheck — Parkinson's Motor Check Station

A 1-minute tabletop check-up: wrist tremor (MPU6050 on an Arduino Uno) + finger tapping
(webcam + MediaPipe). Each test is scored 0–4 with the rules that fired shown on screen,
plus a left/right asymmetry flag and a trend chart across sessions.

> **Tracking and decision-support tool, NOT a diagnosis. All thresholds are demo thresholds.**

## 1. Install (Windows, one time)
MediaPipe 0.10.14 needs Python 3.11 (3.13/3.14 have no wheels).
```powershell
uv venv --python 3.11 .venv                       # add --native-tls behind Cloudflare WARP / proxies
uv pip install --python .venv\Scripts\python.exe -r requirements.txt
```
Arduino tooling (portable, no admin needed):
```powershell
# put arduino-cli.exe on PATH (https://github.com/arduino/arduino-cli/releases), then:
arduino-cli core update-index
arduino-cli core install arduino:avr
arduino-cli lib install "LiquidCrystal" "LiquidCrystal I2C" "Adafruit MPU6050"
```
Always run Python as `.venv\Scripts\python` (plain `python` may be a different version).

## 2. Wire (UNPLUG USB FIRST)
| Part | Part pin | Arduino Uno |
|---|---|---|
| MPU6050 | VCC / GND / SDA / SCL | 5V / GND / A4 / A5 |
| Buzzer | + (long leg / red) / − | D8 / GND |
| RGB LED | R / G / B (each via 220 Ω) | D9 / D10 / D11 |
| RGB LED | common (longest leg) | GND if common-cathode, 5V if common-anode |
| LCD1602 I2C *(optional)* | VCC / GND / SDA / SCL | 5V / GND / A4 / A5 |
| LCD1602 parallel *(optional)* | RS / E / D4 / D5 / D6 / D7 | D7 / D6 / D5 / D4 / D3 / D2 |
| LCD1602 parallel | VSS / VDD / RW / K | GND / 5V / GND / GND |
| LCD1602 parallel | A (backlight +) | 5V via 220 Ω |
| LCD1602 parallel | V0 (contrast) | middle pin of 10 kΩ pot (outer pins 5V / GND) |

The MPU6050 and an I2C LCD share A4/A5 — that's fine (different I2C addresses).

## 3. Configure + upload firmware
Edit the three `#define`s at the top of `firmware/neurocheck/neurocheck.ino`:
`BUZZER_PASSIVE` (0 active: sticker, sealed bottom / 1 passive: green board visible),
`LED_COMMON_ANODE` (0/1), `LCD_MODE` (`LCD_NONE` / `LCD_I2C` / `LCD_PARALLEL`).
Everything works with no LCD connected.
```powershell
arduino-cli board list                                   # find the COM port
arduino-cli compile --fqbn arduino:avr:uno firmware\neurocheck
arduino-cli upload  --fqbn arduino:avr:uno -p COM5 firmware\neurocheck
bash firmware/compile_all.sh                             # compile all 12 #define combinations
```
Close the app (or any serial monitor) before uploading — only one program can own the port.

## 4. Run
```powershell
.venv\Scripts\python app.py                    # real Arduino (auto-detect) + webcam
.venv\Scripts\python app.py --port COM5        # force a port
.venv\Scripts\python app.py --sim              # no hardware: fake IMU + fake hand
.venv\Scripts\python app.py --sim-device       # fake IMU + real webcam
.venv\Scripts\python app.py --sim --seed-history   # add 7 days of fake history first
.venv\Scripts\python app.py --seconds 5        # shorter tests
.venv\Scripts\python -m pytest                 # unit + app tests
.venv\Scripts\python selftest.py --screens screenshots   # headless full session, PASS/FAIL table
```
Keys: **SPACE** next · **R** restart (any screen) · **Q/ESC** quit · **H** trend (welcome/results) ·
**T** toggle fake 5 Hz tremor (sim device only).

If the Arduino is unplugged the header turns red and it reconnects automatically; a tremor test
recorded while disconnected shows "-" with the reason instead of a score.

## 5. Demo script (~3 min)
1. Before the audience arrives: `app.py --seed-history` once (or `--sim --seed-history` as backup).
2. **Welcome** — "one-minute check, four 10-second tests, every score explained".
3. **Right tremor** — sensor on wrist, forearm resting, shake ~5 Hz. Point at the live x/y/z chart.
   Done screen: spectrum with the 4–6 Hz band shaded and the peak marked → score + reasons.
4. **Left tremor** — hold still → score 0 "no clear tremor".
5. **Right tapping** — tap slowly / small to show problems; live skeleton, distance graph, tap count.
6. **Left tapping** — tap fast and big → score 0.
7. **Results** — four cards with every rule that fired, the ASYMMETRY line, LED colour + LCD.
8. **Trend** (SPACE) — the seeded week shows the right hand slowly worsening.
9. Close: "tracking tool for between clinic visits, not a diagnosis".
Backup: `app.py --sim`, press **T** during the right tremor test.

## Scoring (all in `scoring.py`, constants at the top)
**Tremor** (MDS-UPDRS 3.17 style): subtract each axis' mean (gravity), detrend, Hann FFT per axis,
sum power. Peak = max in 3–8 Hz; clear if ≥ 3× the median power in the band. Amplitude = sinusoid
amplitude at the peak on the strongest axis; displacement = a/(2πf)² × 100 cm.
0: < 0.1 cm or no clear peak · 1: < 1 cm · 2: 1–3 cm · 3: 3–10 cm · 4: ≥ 10 cm.
Also reports the % of 1-second windows showing tremor.

**Tapping** (MDS-UPDRS 3.4 style): thumb-tip–index-tip distance / wrist–middle-MCP length; each
opening is a tap. Problems: slow (< 2 taps/s), small (amplitude < 0.5), decrement > 30 % (first vs
last 3 s), irregular (interval CV > 0.3), any hesitation (gap > 2× median).
Score = number of problems (max 3); 4 if fewer than 5 taps. Hand visible < 50 % → not scored.

**Asymmetry**: flagged if R/L differ by ≥ 1 point, or ≥ 25 % in taps/s or tremor displacement
(displacement only compared once either side is ≥ 0.1 cm, so noise isn't flagged).

## Files
| File | What |
|---|---|
| `app.py` | OpenCV window, state machine, keyboard, main |
| `ui.py` | drawing helpers, live accel chart, spectrum chart, score cards |
| `device.py` | `ArduinoDevice` (auto-detect, READY, reconnect) + `MockDevice` |
| `tremor_analysis.py` / `tapping_analysis.py` / `scoring.py` | pure maths, fully unit-tested |
| `tapping_tracker.py` | `CameraHand` (MediaPipe), `FakeHand`, skeleton + graph drawing |
| `history.py` | `sessions.csv`, `seed_history()`, trend chart |
| `selftest.py` | headless full session with PASS/FAIL table |
| `firmware/neurocheck/neurocheck.ino` | Uno firmware; `firmware/compile_all.sh` builds all variants |
