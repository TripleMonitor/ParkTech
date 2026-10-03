// NeuroCheck firmware - Arduino Uno
//
// Serial 115200, newline-terminated:
//   Arduino -> PC:  READY (boot)   T,millis,ax,ay,az  (m/s^2, 100 Hz, only between START/STOP)
//                   ERROR ...      (diagnostics; the PC logs and ignores these)
//   PC -> Arduino:  START | STOP | BEEP,n | LED,G|Y|R|B|OFF | LCD,line1|line2
//
// Wiring (see README):
//   MPU6050  VCC 5V, GND, SDA A4, SCL A5          Buzzer  pin 8 (+), GND (-)
//   RGB LED  R 9, G 10, B 11 via 220 ohm; common pin to GND (cathode) or 5V (anode)
//   LCD1602  parallel RS 7, E 6, D4 5, D5 4, D6 3, D7 2, V0 to 10k pot middle
//            or I2C backpack on A4/A5 (address 0x27 or 0x3F, auto-detected)
//
// ---- configuration: edit here, or override with -D flags ----------------------------
#ifndef BUZZER_PASSIVE
#define BUZZER_PASSIVE 0        // 0 = active buzzer (sticker, sealed bottom), 1 = passive (board visible)
#endif
#ifndef LED_COMMON_ANODE
#define LED_COMMON_ANODE 0      // 0 = common cathode (long leg to GND), 1 = common anode (long leg to 5V)
#endif
#define LCD_NONE 0
#define LCD_I2C 1
#define LCD_PARALLEL 2
#ifndef LCD_MODE
#define LCD_MODE LCD_NONE       // LCD_NONE, LCD_I2C or LCD_PARALLEL
#endif
// ---------------------------------------------------------------------------------------

#include <Wire.h>
#if LCD_MODE == LCD_I2C
#include <LiquidCrystal_I2C.h>
#elif LCD_MODE == LCD_PARALLEL
#include <LiquidCrystal.h>
#endif

const uint8_t MPU_ADDR = 0x68;
const uint8_t PIN_BUZZER = 8;
const uint8_t PIN_R = 9, PIN_G = 10, PIN_B = 11;
const unsigned long SAMPLE_US = 10000UL;   // 100 Hz
const float ACCEL_LSB_PER_G = 4096.0;      // +-8 g
const float G_MS2 = 9.80665;
const unsigned int BEEP_HZ = 2400, BEEP_ON_MS = 90, BEEP_OFF_MS = 90;
const uint8_t CMD_MAX = 48;                // "LCD," + 16 + "|" + 16 fits

#if LCD_MODE == LCD_I2C
LiquidCrystal_I2C *lcd = nullptr;
#elif LCD_MODE == LCD_PARALLEL
LiquidCrystal lcdPar(7, 6, 5, 4, 3, 2);    // RS, E, D4, D5, D6, D7
#endif
bool lcdOk = false;

bool mpuOk = false;
uint8_t mpuFailures = 0;                   // consecutive failed reads
unsigned long mpuRetryMs = 0;
const uint8_t MPU_MAX_FAILURES = 10;
const unsigned long MPU_RETRY_MS = 1000;
bool streaming = false;
unsigned long nextSampleUs = 0;

char cmd[CMD_MAX + 1];
uint8_t cmdLen = 0;
bool cmdOverflow = false;

uint8_t beepsLeft = 0;
bool beepOn = false;
unsigned long beepNextMs = 0;

// ---------------------------------------------------------------- MPU6050 (raw registers)
bool mpuWrite(uint8_t reg, uint8_t val) {
  Wire.beginTransmission(MPU_ADDR);
  Wire.write(reg);
  Wire.write(val);
  return Wire.endTransmission() == 0;
}

bool mpuInit() {
  // Many clones report WHO_AM_I 0x70/0x72/0x98, so we only check that the chip ACKs.
  return mpuWrite(0x6B, 0x00)      // wake up, internal clock
      && mpuWrite(0x1A, 0x03)      // DLPF ~44 Hz: keeps the 3-8 Hz tremor band
      && mpuWrite(0x1C, 0x10);     // accel full scale +-8 g
}

bool mpuReadAccel(float &ax, float &ay, float &az) {
  Wire.beginTransmission(MPU_ADDR);
  Wire.write(0x3B);
  if (Wire.endTransmission(false) != 0) return false;
  if (Wire.requestFrom(MPU_ADDR, (uint8_t)6) != 6) return false;
  int16_t raw[3];
  for (uint8_t i = 0; i < 3; i++) {
    uint8_t hi = Wire.read();
    uint8_t lo = Wire.read();
    raw[i] = (int16_t)((hi << 8) | lo);
  }
  ax = raw[0] / ACCEL_LSB_PER_G * G_MS2;
  ay = raw[1] / ACCEL_LSB_PER_G * G_MS2;
  az = raw[2] / ACCEL_LSB_PER_G * G_MS2;
  return true;
}

// ---------------------------------------------------------------- LED
void ledPin(uint8_t pin, bool on) {
  digitalWrite(pin, (on != (bool)LED_COMMON_ANODE) ? HIGH : LOW);
}

void setLed(const char *c) {
  // digitalWrite (not PWM): tone() uses Timer2, which would break PWM on pin 11.
  bool r = false, g = false, b = false;
  if (strcmp(c, "R") == 0) r = true;
  else if (strcmp(c, "G") == 0) g = true;
  else if (strcmp(c, "Y") == 0) { r = true; g = true; }
  else if (strcmp(c, "B") == 0) b = true;
  ledPin(PIN_R, r);
  ledPin(PIN_G, g);
  ledPin(PIN_B, b);
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

// ---------------------------------------------------------------- LCD (optional)
#if LCD_MODE == LCD_I2C
bool i2cPresent(uint8_t addr) {
  Wire.beginTransmission(addr);
  return Wire.endTransmission() == 0;
}
#endif

void lcdInit() {
#if LCD_MODE == LCD_I2C
  const uint8_t addrs[] = {0x27, 0x3F};
  for (uint8_t i = 0; i < 2 && !lcdOk; i++) {
    if (i2cPresent(addrs[i])) {
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
    if (!mpuOk) mpuOk = mpuInit();
    if (!mpuOk) Serial.println(F("ERROR MPU6050 not responding (check SDA A4 / SCL A5 / 5V)"));
    streaming = true;
    nextSampleUs = micros();
  } else if (strcmp(c, "STOP") == 0) {
    streaming = false;
  } else if (strncmp(c, "BEEP,", 5) == 0) {
    startBeeps(atoi(c + 5));
  } else if (strncmp(c, "LED,", 4) == 0) {
    setLed(c + 4);
  } else if (strncmp(c, "LCD,", 4) == 0) {
    lcdShow(c + 4);
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

// ---------------------------------------------------------------- streaming
void mpuLost() {
  mpuOk = false;
  mpuRetryMs = millis() + MPU_RETRY_MS;
  Serial.println(F("ERROR MPU6050 stopped responding (check wires)"));
}

void streamSample() {
  unsigned long nowUs = micros();
  if ((long)(nowUs - nextSampleUs) < 0) return;
  nextSampleUs += SAMPLE_US;
  if ((long)(nowUs - nextSampleUs) > (long)SAMPLE_US) nextSampleUs = nowUs + SAMPLE_US;  // fell behind
  if (!mpuOk) {                                       // try to bring the sensor back once a second
    if ((long)(millis() - mpuRetryMs) < 0) return;
    mpuRetryMs = millis() + MPU_RETRY_MS;
    mpuOk = mpuInit();
    mpuFailures = 0;
    if (!mpuOk) return;
  }
  unsigned long tMs = millis();                       // stamp at read time, not print time
  float ax, ay, az;
  if (!mpuReadAccel(ax, ay, az)) {
    if (++mpuFailures >= MPU_MAX_FAILURES) mpuLost();
    return;
  }
  mpuFailures = 0;
  Serial.print(F("T,"));
  Serial.print(tMs);
  Serial.print(',');
  Serial.print(ax, 3);
  Serial.print(',');
  Serial.print(ay, 3);
  Serial.print(',');
  Serial.println(az, 3);
}

void setup() {
  Serial.begin(115200);
  pinMode(PIN_BUZZER, OUTPUT);
  pinMode(PIN_R, OUTPUT);
  pinMode(PIN_G, OUTPUT);
  pinMode(PIN_B, OUTPUT);
  buzzer(false);
  setLed("OFF");
  Wire.begin();
  Wire.setWireTimeout(3000, true);   // a missing/flaky I2C device must not hang the loop
  lcdInit();                         // LiquidCrystal_I2C::init() calls Wire.begin() again...
  Wire.setClock(400000);             // ...so set the fast clock after it
  mpuOk = mpuInit();
  lcdShow(mpuOk ? "NeuroCheck|Ready" : "NeuroCheck|MPU6050 ERROR");
  if (!mpuOk) Serial.println(F("ERROR MPU6050 not responding (check SDA A4 / SCL A5 / 5V)"));
  Serial.println(F("READY"));
}

void loop() {
  readSerial();
  updateBeeper();
  if (streaming) streamSample();
}
