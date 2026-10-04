// Diagnostic: find which pin the tilt switch is wired to.
// Every digital pin D2..D12 and A0..A5 is an input with the internal pull-up; any change
// (debounced 20 ms) is reported as  P,<pin name>,<0|1>.  Prints SCAN READY on boot.
// LEDs on scanned pins may glow faintly (pull-up current) - harmless.

const uint8_t PINS[] = {2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, A0, A1, A2, A3, A4, A5};
const char *NAMES[] = {"D2", "D3", "D4", "D5", "D6", "D7", "D8", "D9", "D10", "D11", "D12",
                       "A0", "A1", "A2", "A3", "A4", "A5"};
const uint8_t N = sizeof(PINS);
uint8_t stable[N], raw[N];
unsigned long changed[N];

void setup() {
  Serial.begin(115200);
  pinMode(13, OUTPUT);
  for (uint8_t i = 0; i < N; i++) {
#ifdef NO_PULLUP
    pinMode(PINS[i], INPUT);         // finds switches wired 5V -> switch -> pin (+ pull-down)
#else
    pinMode(PINS[i], INPUT_PULLUP);  // finds switches wired pin -> switch -> GND
#endif
    stable[i] = raw[i] = digitalRead(PINS[i]);
    changed[i] = 0;
  }
  Serial.println(F("SCAN READY"));
  for (uint8_t i = 0; i < N; i++) {          // initial levels
    Serial.print(F("I,")); Serial.print(NAMES[i]); Serial.print(','); Serial.println(stable[i]);
  }
}

void loop() {
  unsigned long now = millis();
  for (uint8_t i = 0; i < N; i++) {
    uint8_t r = digitalRead(PINS[i]);
    if (r != raw[i]) { raw[i] = r; changed[i] = now; }
    if (raw[i] != stable[i] && now - changed[i] >= 20) {
      stable[i] = raw[i];
      digitalWrite(13, stable[i] ? LOW : HIGH);
      Serial.print(F("P,")); Serial.print(NAMES[i]); Serial.print(','); Serial.println(stable[i]);
    }
  }
}
