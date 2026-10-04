// NeuroCheck firmware - Arduino Uno
// SW-520D tilt switch (hand flipping), buzzer, 3 separate LEDs (green/yellow/red),
// optional LCD1602. No IMU.
//
// Serial 115200, newline-terminated:
//   Arduino -> PC:  READY (boot)
//                   S,millis,0|1   on every debounced switch change (only between START/STOP)
//                                  and as the reply to STATE (any time)
//                   C,millis       one per metronome beat while the cue is on
//   PC -> Arduino:  START | STOP | STATE | BEEP,n | LED,G|Y|R|B|OFF | LCD,line1|line2
//                   CUE,ON,<ms> | CUE,OFF | TEMPO,<ms>   (beat interval clamped 200-1500 ms;
//                   a TEMPO change applies from the next beat)
//
// Pin table (no conflicts; see CLAUDE.md / README):
//   D2  SW-520D (bare 2-leg: other leg to GND, INPUT_PULLUP | 3-pin module: DO, VCC 5V, GND)
//   D13 onboard LED mirrors the debounced switch state
//   D6  buzzer
//   D9 green, D10 yellow, D11 red   separate LEDs, long leg -> pin via 220 ohm, short leg -> GND
//                       (plain on/off, so tone() on Timer2 can't interfere)
//   LCD parallel: RS D12, E D8, D4 D7, D5 D5, D6 D4, D7 D3 (digital only) | LCD I2C: A4/A5
//   On boot the LEDs light green -> yellow -> red once, so you can check the pin order.
//
// ---- configuration: edit here, or override with -D flags ----------------------------
#ifndef SWITCH_MODULE
#define SWITCH_MODULE 0         // 0 = bare 2-leg SW-520D (to GND), 1 = 3-pin module (VCC/GND/DO)
#endif
#ifndef BUZZER_PASSIVE
#define BUZZER_PASSIVE 0        // 0 = active buzzer (sticker, sealed bottom), 1 = passive (board visible)
#endif
#define LCD_NONE 0
#define LCD_I2C 1
#define LCD_PARALLEL 2
#ifndef LCD_MODE
#define LCD_MODE LCD_NONE       // LCD_NONE, LCD_I2C or LCD_PARALLEL
#endif
// ---------------------------------------------------------------------------------------

#if LCD_MODE == LCD_I2C
#include <Wire.h>
#include <LiquidCrystal_I2C.h>
#elif LCD_MODE == LCD_PARALLEL
#include <LiquidCrystal.h>
#endif

const uint8_t PIN_SWITCH = 2;
const uint8_t PIN_MIRROR = 13;
const uint8_t PIN_BUZZER = 6;
const uint8_t PIN_LED_GREEN = 9, PIN_LED_YELLOW = 10, PIN_LED_RED = 11;
const unsigned long DEBOUNCE_MS = 30;
const unsigned int BEEP_HZ = 2400, BEEP_ON_MS = 90, BEEP_OFF_MS = 90;
const uint8_t CMD_MAX = 48;                // "LCD," + 16 + "|" + 16 fits

#if LCD_MODE == LCD_I2C
LiquidCrystal_I2C *lcd = nullptr;
#elif LCD_MODE == LCD_PARALLEL
LiquidCrystal lcdPar(12, 8, 7, 5, 4, 3);   // RS, E, D4, D5, D6, D7
#endif
bool lcdOk = false;

bool streaming = false;
uint8_t rawState = 0, stableState = 0;
unsigned long rawChangedMs = 0;

char cmd[CMD_MAX + 1];
uint8_t cmdLen = 0;
bool cmdOverflow = false;

uint8_t beepsLeft = 0;
bool beepOn = false;
unsigned long beepNextMs = 0;

// rhythm cue (metronome) - timed by the Arduino itself with millis()
const unsigned long CUE_MIN_MS = 200, CUE_MAX_MS = 1500, CUE_CLICK_MS = 40;
bool cueOn = false;
unsigned long cueIntervalMs = 600, cuePendingMs = 600, cueNextMs = 0, cueClickOffMs = 0;
bool cueClicking = false;
char ledColour[4] = "OFF";

// ---------------------------------------------------------------- tilt switch
uint8_t readSwitch() {
#if SWITCH_MODULE
  return digitalRead(PIN_SWITCH) == HIGH ? 1 : 0;   // module DO level as-is
#else
  return digitalRead(PIN_SWITCH) == LOW ? 1 : 0;    // bare switch: closed pulls the pin LOW
#endif
}

void reportState() {
  Serial.print(F("S,"));
  Serial.print(millis());
  Serial.print(',');
  Serial.println(stableState);
}

void updateSwitch() {
  // Non-blocking debounce: accept a new level once it has been steady for DEBOUNCE_MS.
  uint8_t r = readSwitch();
  unsigned long now = millis();
  if (r != rawState) {
    rawState = r;
    rawChangedMs = now;
  }
  if (rawState != stableState && now - rawChangedMs >= DEBOUNCE_MS) {
    stableState = rawState;
    digitalWrite(PIN_MIRROR, stableState ? HIGH : LOW);
    if (streaming) reportState();
  }
}

// ---------------------------------------------------------------- LEDs (3 separate)
void ledsRaw(bool g, bool y, bool r) {
  digitalWrite(PIN_LED_GREEN, g ? HIGH : LOW);
  digitalWrite(PIN_LED_YELLOW, y ? HIGH : LOW);
  digitalWrite(PIN_LED_RED, r ? HIGH : LOW);
}

void setLed(const char *c) {
  // G / Y / R light one LED; B (legacy "blue") lights all three; anything else = off
  strncpy(ledColour, c, sizeof(ledColour) - 1);
  bool all = strcmp(c, "B") == 0;
  ledsRaw(all || strcmp(c, "G") == 0, all || strcmp(c, "Y") == 0, all || strcmp(c, "R") == 0);
}

// ---------------------------------------------------------------- buzzer (non-blocking)
void buzzer(bool on) {
#if BUZZER_PASSIVE
  if (on) tone(PIN_BUZZER, BEEP_HZ); else noTone(PIN_BUZZER);
#else
  digitalWrite(PIN_BUZZER, on ? HIGH : LOW);
#endif
}

void startBeeps(int n) {
  beepsLeft = (uint8_t)constrain(n, 0, 10);
  if (beepOn) { buzzer(false); beepOn = false; }
  beepNextMs = millis();
}

void updateBeeper() {
  if (!beepsLeft && !beepOn) return;
  unsigned long now = millis();
  if ((long)(now - beepNextMs) < 0) return;
  if (beepOn) {
    buzzer(false);
    beepOn = false;
    beepNextMs = now + BEEP_OFF_MS;
  } else if (beepsLeft) {
    buzzer(true);
    beepOn = true;
    beepsLeft--;
    beepNextMs = now + BEEP_ON_MS;
  }
}

// ---------------------------------------------------------------- rhythm cue
unsigned long clampCue(long ms) {
  if (ms < (long)CUE_MIN_MS) return CUE_MIN_MS;
  if (ms > (long)CUE_MAX_MS) return CUE_MAX_MS;
  return (unsigned long)ms;
}

void updateCue() {
  unsigned long now = millis();
  if (cueClicking && (long)(now - cueClickOffMs) >= 0) {     // end of click + flash
    cueClicking = false;
    if (!beepOn) buzzer(false);
    char saved[4];
    strncpy(saved, ledColour, sizeof(saved));
    setLed(saved);                                           // restore the result colour
  }
  if (!cueOn || (long)(now - cueNextMs) < 0) return;
  unsigned long beat = cueNextMs;
  cueIntervalMs = cuePendingMs;                              // TEMPO applies from this beat on
  cueNextMs = beat + cueIntervalMs;
  if ((long)(now - cueNextMs) >= 0) cueNextMs = now + cueIntervalMs;   // never burst-catch-up
  buzzer(true);
  ledsRaw(true, true, true);                                 // beat flash: all three LEDs
  cueClicking = true;
  cueClickOffMs = now + CUE_CLICK_MS;
  Serial.print(F("C,"));
  Serial.println(beat);
}

// ---------------------------------------------------------------- LCD (optional)
void lcdInit() {
#if LCD_MODE == LCD_I2C
  Wire.begin();
  Wire.setWireTimeout(3000, true);           // a missing/flaky I2C device must not hang the loop
  const uint8_t addrs[] = {0x27, 0x3F};
  for (uint8_t i = 0; i < 2 && !lcdOk; i++) {
    Wire.beginTransmission(addrs[i]);
    if (Wire.endTransmission() == 0) {
      static LiquidCrystal_I2C dev(addrs[i], 16, 2);   // constructed once, first match
      lcd = &dev;
      lcd->init();
      lcd->backlight();
      lcdOk = true;
    }
  }
#elif LCD_MODE == LCD_PARALLEL
  lcdPar.begin(16, 2);
  lcdOk = true;          // cannot detect a parallel LCD; writes are harmless if absent
#endif
}

void lcdLine(uint8_t row, const char *s, uint8_t len) {
#if LCD_MODE != LCD_NONE
  char buf[17];
  uint8_t n = len > 16 ? 16 : len;
  memcpy(buf, s, n);
  for (uint8_t i = n; i < 16; i++) buf[i] = ' ';   // pad instead of clear(): no flicker
  buf[16] = '\0';
#if LCD_MODE == LCD_I2C
  lcd->setCursor(0, row);
  lcd->print(buf);
#else
  lcdPar.setCursor(0, row);
  lcdPar.print(buf);
#endif
#else
  (void)row; (void)s; (void)len;
#endif
}

void lcdShow(const char *payload) {
  if (!lcdOk) return;
  const char *bar = strchr(payload, '|');
  uint8_t len1 = bar ? (uint8_t)(bar - payload) : (uint8_t)strlen(payload);
  lcdLine(0, payload, len1);
  if (bar) lcdLine(1, bar + 1, (uint8_t)strlen(bar + 1));
  else lcdLine(1, "", 0);
}

// ---------------------------------------------------------------- commands
void handleCommand(const char *c) {
  if (strcmp(c, "START") == 0) {
    streaming = true;
  } else if (strcmp(c, "STOP") == 0) {
    streaming = false;
  } else if (strcmp(c, "STATE") == 0) {
    reportState();
  } else if (strncmp(c, "BEEP,", 5) == 0) {
    startBeeps(atoi(c + 5));
  } else if (strncmp(c, "LED,", 4) == 0) {
    setLed(c + 4);
  } else if (strncmp(c, "LCD,", 4) == 0) {
    lcdShow(c + 4);
  } else if (strncmp(c, "CUE,ON,", 7) == 0) {
    cuePendingMs = cueIntervalMs = clampCue(atol(c + 7));
    cueOn = true;
    cueNextMs = millis();                                    // first beat now
  } else if (strcmp(c, "CUE,OFF") == 0) {
    cueOn = false;
  } else if (strncmp(c, "TEMPO,", 6) == 0) {
    cuePendingMs = clampCue(atol(c + 6));
  }
}

void readSerial() {
  // Non-blocking: consume what's available, act on complete lines only.
  while (Serial.available() > 0) {
    char ch = (char)Serial.read();
    if (ch == '\r') continue;
    if (ch == '\n') {
      cmd[cmdLen] = '\0';
      if (!cmdOverflow && cmdLen > 0) handleCommand(cmd);
      cmdLen = 0;
      cmdOverflow = false;
    } else if (cmdLen < CMD_MAX) {
      cmd[cmdLen++] = ch;
    } else {
      cmdOverflow = true;       // drop over-long lines entirely
    }
  }
}

void setup() {
  Serial.begin(115200);
#if SWITCH_MODULE
  pinMode(PIN_SWITCH, INPUT);
#else
  pinMode(PIN_SWITCH, INPUT_PULLUP);
#endif
  pinMode(PIN_MIRROR, OUTPUT);
  pinMode(PIN_BUZZER, OUTPUT);
  pinMode(PIN_LED_GREEN, OUTPUT);
  pinMode(PIN_LED_YELLOW, OUTPUT);
  pinMode(PIN_LED_RED, OUTPUT);
  buzzer(false);
  ledsRaw(true, false, false); delay(300);   // boot check: green -> yellow -> red
  ledsRaw(false, true, false); delay(300);   // (setup only; the main loop never blocks)
  ledsRaw(false, false, true); delay(300);
  setLed("OFF");
  rawState = stableState = readSwitch();
  rawChangedMs = millis();
  digitalWrite(PIN_MIRROR, stableState ? HIGH : LOW);
  lcdInit();
  lcdShow("NeuroCheck|Ready");
  Serial.println(F("READY"));
}

void loop() {
  readSerial();
  updateSwitch();
  updateBeeper();
  updateCue();
}
