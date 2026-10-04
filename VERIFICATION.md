# ParkTech — Verification

Run on Windows 11, Python 3.11.15 (.venv), arduino-cli 1.5.1 / arduino:avr 1.8.8, 2026-10-03.
Raw outputs are in `verification/`. Screenshots are in `screenshots/`.

**Bottom line:** everything that can be verified without the physical Arduino passes. All
hardware-dependent behaviour is listed under *Not verifiable without hardware* below.

## Summary table

| # | Check | Result | Evidence |
|---|---|---|---|
| 1 | Unit + app tests (`pytest`) | **PASS** — 358 passed, 0 failed | `verification/pytest.txt` |
| 2 | Headless full session (`selftest.py`): boot, 6 tests, coach, results, exports, dashboard, trend | **PASS** — 20/20 checks | `verification/selftest.txt`, `screenshots/selftest_*.png` |
| 3 | Scripted real-window GUI run (`tools/gui_smoke.py`): coach, 2 sessions, R mid-test, every screen, ≥ 60 s | **PASS** — 0 exceptions | `verification/gui_smoke.txt`, `screenshots/gui_*.png` |
| 4 | Firmware compiles in all 24 `#define` combinations | **PASS** — 24/24, 0 sketch warnings | `verification/firmware.txt` |
| 5 | Firmware size vs Uno | **PASS** — max flash 24 %, max RAM 30 % (limit 75 %) | `verification/firmware.txt` |
| 6 | Pin conflicts (static check, all LCD modes, Timer2 vs PWM) | **PASS** | `tools/check_pins.py`, `tests/test_pins.py` |
| 7 | PID rhythm coach tuning on a simulated patient | **PASS** — 7/7 scenarios (see table below) | `verification/tune_pid.txt`, `docs/pid_tuning.png` |
| 8 | Independent code review #1 (IMU version) | all findings fixed | git history `a5d7bb1` |
| 9 | Independent code review #2 (final, against CLAUDE.md) | 1 CRITICAL, 3 HIGH, 13 MEDIUM, 8 LOW — **all CRITICAL/HIGH fixed**, MEDIUM fixed except as noted | see *Review #2* below |
| 10 | Every screenshot looked at; layout fixes applied (overlaps, cut-off text, stale data) | **PASS** | `screenshots/` |

## Maths checks (synthetic signals with known answers)

| Area | Test | Result |
|---|---|---|
| Camera tremor | 5 Hz fingertip movement 0.5 / 2 / 5 cm, hand scaled to 9 cm, ±2 px jitter | 5.0 Hz; 0.512 / 2.008 / 5.000 cm → scores 1 / 2 / 3 ✔ |
| | Jitter only (±2 px) | 0.048 cm → score 0 "no clear tremor" ✔ |
| | Dropped frames (every 3rd/4th), 22/24 fps, ±8 ms timing jitter | 2 cm read as 1.96–1.99 cm, 5.0 Hz ✔ |
| | Slow drift (80 px), 1 Hz sway, diagonal motion 0/45/90° | removed / outside band / not under-read ✔ |
| Finger tapping | steady 3 Hz; 50 % shrink; two 1-s pauses; flat; tracking dropouts | score 0; decrement flagged; 2 hesitations, ≥ 2; score 4; dropouts not hesitations ✔ |
| Hand flipping | steady 2 full flips/s; 40 % slow-down; two 1-s pauses; 3-change bounce bursts in 20 ms; no flips | score 0; decrement 40 % flagged; 2 hesitations; counted once; score 4 + sensor-check reason ✔ |
| | Boundary (documented): 40 % slow-down spread linearly over 10 s | first-4-s vs last-4-s = 25 % (borderline) — test `test_linear_40pct_over_whole_test_is_borderline` |
| Fusion | synthetic hand rotations (two forearm axes), matching / mismatched counts, small + shrinking rotation | LOCKED / MISMATCH, 160° ± 8°, decrement > 25 % ✔ |
| Scoring | WHY-panel rules recomputed vs scorer over 200+ parameter combinations | identical in every case ✔ |
| Asymmetry | R3/L0 flagged, R1/L1 not, 25 % rate rule, noise-level tremor ignored | ✔ |
| Every score has a reason | all score levels 0–4 and unscored | ✔ |
| Serial parser | good, split, garbage, empty, NaN/negative lines; `S`, `C` lines | never crash ✔ |
| Clock mapping | Arduino 0.2 % slow, 10 min | maps within 50 ms ✔ |
| PID | limits, anti-windup, derivative on measurement, low-pass, reverse acting, reset | ✔ |
| Exports | FHIR R4 bundle structure, local codes only, UCUM, timezone, SIM tagging; PDF renders | ✔ |

## PID tuning (simulated patient, true max 3.0 half-flips/s)
Criteria: within 10 % of true max within 20 s and stays there, overshoot < 15 %, late std < 5 %,
no windup after a 5 s stop. Result: **kp 0.01, ki 0, kd 0, probe +0.02/beat** — the only
settings that passed all scenarios were P-only (I and D made it oscillate; the output is
already a tempo *change*, so P acts integrally). The PID class supports and tests I and D.

| Scenario | Result |
|---|---|
| nominal / 20 % missed (undetected) beats / 5 s stop / 60 ms noise / fast self-pacer / fatigue 15 % + 20 % missed | PASS |
| slow self-pacer (warm start 30 % too low) | PASS — converges at 16.8 s |
| cold start 1.5 beats/s (info only) | converges at 33.8 s — slower than 20 s; mitigated by the warm start |

Design deviations made to pass (documented in `rhythm_coach.py`, CLAUDE.md): graded on-time
credit (full within ±75 ms, 0 at ±150 ms), warm start at 1.15× the uncued rate, saturation
probing, random misses excluded / misses after a late beat counted.

## Review #2 findings and what was done
| Severity | Finding | Fix |
|---|---|---|
| CRITICAL | stale tilt state → wrong flipping calibration (phantom flips, score 0→2) | PC polls STATE every 0.2 s on Ready/countdown; palm-down reference = first report after START; stale state shows "?"; regression test with a stale-state device |
| HIGH | Arduino clock drift breaks event timing over long sessions | sliding-window clock offset + reset at every START; drift test |
| HIGH | coach saved defaults as results; no disconnect handling; cue not restored | result "NOT MEASURED" unless ≥ 8 answered beats & on-time beats; abort on disconnect; cue re-sent after reconnect; `CUE,OFF` on quit |
| HIGH | SIM sessions saved/exported unlabeled | `mode` column; SIM rows hollow + labeled; FHIR `meta.tag` SIM + `preliminary` + note; PDF banner |
| MEDIUM | "confidence" was MediaPipe's handedness score | relabeled HANDEDNESS, information only, no longer triggers LOW CONFIDENCE |
| MEDIUM | partial serial lines dropped | receive buffer re-joins lines split by a read timeout |
| MEDIUM | trend hollow markers / regression mixed demo + real | only DEMO/SIM hollow; fit uses measured rows when ≥ 3 exist and says which |
| MEDIUM | PDF dropped LOW CONFIDENCE | always printed on its own line |
| MEDIUM | Trend was a dead end | SPACE returns to the previous screen; R = new session |
| MEDIUM | FHIR times without timezone | timezone-aware times + validator check |
| MEDIUM | frame times included inference | capture timestamp carried in each frame |
| MEDIUM | tremor amplitude meaning | screen says "(half peak-to-peak)"; circular tremor reads √2 high — documented, not changed |
| MEDIUM | defaults saved/shown as measurements | gauge shows "--" before recording; empty rotation/tremor/tap values saved blank |
| MEDIUM | stale TILT / INFER telemetry | state cleared on disconnect; INFER only while the camera runs |
| MEDIUM | camera not reopened after unplug | reopen attempt after 15 failed reads, every 2 s |
| MEDIUM | corrupt CSV lost the session silently | any save error shown on screen + logged |
| LOW | sim stuck-sensor toggle invisible; spectrum annotated without a clear peak; docstring 7 days; R ignored on boot; MediaPipe exceptions | indicator added; "no clear tremor peak"; fixed; R works on boot; exceptions caught |
| LOW (kept) | coach run started from Results adds a second, partial CSV row (same session ID); matplotlib renders (~0.3–1 s) on the UI thread for dashboard/trend/PDF; I2C LCD write blocks the loop for a few ms | accepted for the demo |

## Validation on real patients (finger tapping, HUBU-FIS)
234 clinician-rated videos, 118 people, MDS-UPDRS 3.4 per hand (Zenodo 10.5281/zenodo.17738775,
CC-BY-4.0). App's MediaPipe settings + tapping maths, participant-level 5-fold CV.
| Thresholds | exact | within 1 | weighted kappa | clinician-0 clips scored 0 |
|---|---|---|---|---|
| original demo | 42 % | 85 % | 0.47 | 17 / 71 |
| tuned (CV) | 45 % | 85 % | 0.49 | 34 / 71 |
Applied to the app: slow < 1.5 taps/s, small < 0.8, CV > 0.5 (decrement unchanged 30 %).
Tap detection checked visually: `docs/hubu_validation.png`. Full output: `verification/hubu_eval.txt`.
Limitations: phone videos (20 s, 30 fps) vs the app's 10 s webcam test; one dataset, one site;
tremor and flipping thresholds NOT validated on patients.

## Not verifiable without hardware (do these in bring-up)
- Real SW-520D behaviour: bounce pattern, orientation of "closed", 30 ms firmware debounce
  in practice, whether D13 mirrors every flip.
- Buzzer audibility (active vs passive), LED colours/brightness (common anode/cathode), LCD text.
- Serial timing on a real Uno: READY handshake, STATE round-trip latency, clock offset quality.
- Metronome timing accuracy on the Uno and whether real patients follow it (coach tuning used a
  *simulated* patient only — thresholds and gains need a real run).
- Camera: real MediaPipe tracking quality at the booth's lighting/distance, actual FPS, whether
  fast flips lose the hand (fusion lock rate), tremor sensitivity (< ~0.5 cm likely missed).
- All thresholds are demo thresholds, not clinically validated. NeuroScore is a composite
  tracking index, not a diagnosis.
