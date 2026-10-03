// NeuroCheck firmware - Arduino Uno/Nano
// MPU6050 (I2C, raw registers - no library), 16x2 I2C LCD, buzzer, RGB LED.
//
// Serial 115200, newline-terminated:
//   Arduino -> PC:  T,millis,ax,ay,az   (m/s^2, 100 Hz, only while streaming)
//   PC -> Arduino:  START | STOP | BEEP,n | LED,G|Y|R|OFF | LCD,line1|line2
//
// Wiring (change below if yours differs):
//   MPU6050 SDA->A4 SCL->A5 VCC->5V GND->GND  (addr 0x68, AD0 low)
//   LCD I2C backpack SDA->A4 SCL->A5           (addr 0x27, try 0x3F)
//   Buzzer (active or passive) + -> D8
//   RGB LED common-cathode: R->D9 G->D10 B->D11 via 220 ohm resistors
//
// Library needed: "LiquidCrystal I2C" by Frank de Brabander (Library Manager).

#include <Wire.h>
#include <LiquidCrystal_I2C.h>

const uint8_t MPU_ADDR = 0x68;
const uint8_t LCD_ADDR = 0x27;
const uint8_t PIN_BUZZER = 8;
const uint8_t PIN_R = 9, PIN_G = 10, PIN_B = 11;
const bool LED_COMMON_ANODE = false;

const unsigned long SAMPLE_US = 10000;       // 100 Hz
const float ACCEL_LSB_PER_G = 4096.0;        // +-8 g range
const float G = 9.80665;
const unsigned int BEEP_HZ = 2000, BEEP_ON_MS = 80, BEEP_OFF_MS = 70;

LiquidCrystal_I2C lcd(LCD_ADDR, 16, 2);

bool streaming = false;
unsigned long nextSampleUs = 0;
char cmdBuf[48];
uint8_t cmdLen = 0;

// non-blocking beeper
uint8_t beepsLeft = 0;
bool beepOn = false;
unsigned long beepNextMs = 0;

void mpuWrite(uint8_t reg, uint8_t val) {
  Wire.beginTransmission(MPU_ADDR);
  Wire.write(reg);
  Wire.write(val);
  Wire.endTransmission();
}

bool mpuInit() {
  mpuWrite(0x6B, 0x00);   // wake up
  mpuWrite(0x1A, 0x03);   // DLPF ~44 Hz (keeps 3-8 Hz tremor, cuts noise)
  mpuWrite(0x1C, 0x10);   // accel +-8 g
  Wire.beginTransmission(MPU_ADDR);
  Wire.write(0x75);       // WHO_AM_I
  Wire.endTransmission(false);
  Wire.requestFrom(MPU_ADDR, (uint8_t)1);
  return Wire.available() && Wire.read() == 0x68;
}

bool mpuReadAccel(float &ax, float &ay, float &az) {
  Wire.beginTransmission(MPU_ADDR);
  Wire.write(0x3B);
  if (Wire.endTransmission(false) != 0) return false;
  if (Wire.requestFrom(MPU_ADDR, (uint8_t)6) != 6) return false;
  int16_t raw[3];
  for (uint8_t i = 0; i < 3; i++) raw[i] = (Wire.read() << 8) | Wire.read();
  ax = raw[0] / ACCEL_LSB_PER_G * G;
  ay = raw[1] / ACCEL_LSB_PER_G * G;
  az = raw[2] / ACCEL_LSB_PER_G * G;
  return true;
}

void setLed(char c) {
  uint8_t r = 0, g = 0, b = 0;
  if (c == 'R') r = 255;
  else if (c == 'G') g = 255;
  else if (c == 'Y') { r = 255; g = 120; }
  if (LED_COMMON_ANODE) { r = 255 - r; g = 255 - g; b = 255 - b; }
  analogWrite(PIN_R, r);
  analogWrite(PIN_G, g);
  analogWrite(PIN_B, b);
}

void lcdShow(const char *payload) {
  // payload = "line1|line2"
  const char *bar = strchr(payload, '|');
  lcd.clear();
  lcd.setCursor(0, 0);
  for (const char *p = payload; *p && p != bar && p - payload < 16; p++) lcd.print(*p);
  if (bar) {
    lcd.setCursor(0, 1);
    const char *s = bar + 1;
    for (const char *p = s; *p && p - s < 16; p++) lcd.print(*p);
  }
}

void handleCommand(char *cmd) {
  if (strcmp(cmd, "START") == 0) {
    streaming = true;
    nextSampleUs = micros();
  } else if (strcmp(cmd, "STOP") == 0) {
    streaming = false;
  } else if (strncmp(cmd, "BEEP,", 5) == 0) {
    int n = atoi(cmd + 5);
    beepsLeft = constrain(n, 0, 10);
    beepOn = false;
    beepNextMs = millis();
  } else if (strncmp(cmd, "LED,", 4) == 0) {
    setLed(strcmp(cmd + 4, "OFF") == 0 ? 'O' : cmd[4]);
  } else if (strncmp(cmd, "LCD,", 4) == 0) {
    lcdShow(cmd + 4);
  }
}

void readSerial() {
  while (Serial.available()) {
    char c = Serial.read();
    if (c == '\r') continue;
    if (c == '\n') {
      cmdBuf[cmdLen] = '\0';
      if (cmdLen) handleCommand(cmdBuf);
      cmdLen = 0;
    } else if (cmdLen < sizeof(cmdBuf) - 1) {
      cmdBuf[cmdLen++] = c;
    }
  }
}

void updateBeeper() {
  if (!beepsLeft && !beepOn) return;
  unsigned long now = millis();
  if ((long)(now - beepNextMs) < 0) return;
  if (beepOn) {
    noTone(PIN_BUZZER);
    digitalWrite(PIN_BUZZER, LOW);
    beepOn = false;
    beepNextMs = now + BEEP_OFF_MS;
  } else if (beepsLeft) {
    tone(PIN_BUZZER, BEEP_HZ);   // works for passive; active buzzer also sounds
    beepOn = true;
    beepsLeft--;
    beepNextMs = now + BEEP_ON_MS;
  }
}

void streamSample() {
  unsigned long nowUs = micros();
  if ((long)(nowUs - nextSampleUs) < 0) return;
  nextSampleUs += SAMPLE_US;
  if ((long)(nowUs - nextSampleUs) > (long)SAMPLE_US) nextSampleUs = nowUs;  // fell behind
  float ax, ay, az;
  if (!mpuReadAccel(ax, ay, az)) return;
  Serial.print(F("T,"));
  Serial.print(millis());
  Serial.print(',');
  Serial.print(ax, 3);
  Serial.print(',');
  Serial.print(ay, 3);
  Serial.print(',');
  Serial.println(az, 3);
}

void setup() {
  Serial.begin(115200);
  Wire.begin();
  Wire.setClock(400000);
  pinMode(PIN_BUZZER, OUTPUT);
  pinMode(PIN_R, OUTPUT);
  pinMode(PIN_G, OUTPUT);
  pinMode(PIN_B, OUTPUT);
  lcd.init();
  lcd.backlight();
  bool ok = mpuInit();
  lcdShow(ok ? "NeuroCheck|Ready" : "NeuroCheck|MPU6050 ERROR");
  setLed(ok ? 'G' : 'R');
  Serial.println(ok ? F("READY") : F("ERROR MPU6050 not found"));
}

void loop() {
  readSerial();
  updateBeeper();
  if (streaming) streamSample();
}
