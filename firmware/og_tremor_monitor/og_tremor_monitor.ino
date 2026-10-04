// Define Pins
#define SENSOR_PIN 11
#define TREMOR_FAST_LED 6   // Rapid tremor (Parkinson's frequency range)
#define MEDIUM_LED 9        // Moderate hand movement / slow shake
#define CALM_REST_LED 10    // Stable hand / resting state

int lastState = HIGH;
int currentState;

unsigned long lastTiltTime = 0;
unsigned long timeDifference = 0;
unsigned long inactivityTimeout = 800; // Resets to Calm quickly (0.8 seconds of no movement)

void setup() {
  Serial.begin(9600);

  pinMode(SENSOR_PIN, INPUT_PULLUP);
  pinMode(TREMOR_FAST_LED, OUTPUT);
  pinMode(MEDIUM_LED, OUTPUT);
  pinMode(CALM_REST_LED, OUTPUT);

  // Default state is calm/resting
  digitalWrite(CALM_REST_LED, HIGH);
  digitalWrite(MEDIUM_LED, LOW);
  digitalWrite(TREMOR_FAST_LED, LOW);

  Serial.println("Parkinson's Tremor Monitor Active.");
}

void loop() {
  currentState = digitalRead(SENSOR_PIN);
  unsigned long currentTime = millis();

  if (currentState != lastState) {
    if (currentState == LOW) {
      timeDifference = currentTime - lastTiltTime;
      lastTiltTime = currentTime;

      Serial.print("Interval: ");
      Serial.print(timeDifference);
      Serial.println(" ms");

      // --- Parkinson's Relative Speed Tuning ---

      // FAST TREMOR: Shaking 4-6+ times a second (Interval under 220ms)
      if (timeDifference < 220) {
        Serial.println("[!] Tremor Detected (Fast)");
        digitalWrite(TREMOR_FAST_LED, HIGH);
        digitalWrite(MEDIUM_LED, LOW);
        digitalWrite(CALM_REST_LED, LOW);
      }
      // MEDIUM MOVEMENT: Regular hand movement or slow tremor (220ms to 500ms)
      else if (timeDifference >= 220 && timeDifference < 500) {
        Serial.println("[.] Moderate Movement");
        digitalWrite(TREMOR_FAST_LED, LOW);
        digitalWrite(MEDIUM_LED, HIGH);
        digitalWrite(CALM_REST_LED, LOW);
      }
      // SLOW MOVEMENT: Very deliberate or slow hand shifts (Over 500ms)
      else {
        Serial.println("[ ] Slow / Intentional Movement");
        digitalWrite(TREMOR_FAST_LED, LOW);
        digitalWrite(MEDIUM_LED, LOW);
        digitalWrite(CALM_REST_LED, HIGH);
      }
    }

    lastState = currentState;
    // Lowered debounce to 15ms to ensure rapid tremor pulses aren't ignored
    delay(15);
  }

  // AUTO-RESET TO CALM: If the hand stops shaking for 800 milliseconds, reset immediately
  if (currentTime - lastTiltTime > inactivityTimeout) {
    digitalWrite(TREMOR_FAST_LED, LOW);
    digitalWrite(MEDIUM_LED, LOW);
    digitalWrite(CALM_REST_LED, HIGH);
  }
}
