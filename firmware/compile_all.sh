#!/usr/bin/env bash
# Compile neurocheck.ino for arduino:avr:uno in every #define combination.
cd "$(dirname "$0")"
fail=0
for buz in 0 1; do for led in 0 1; do for lcd in 0 1 2; do
  out=$(arduino-cli compile --fqbn arduino:avr:uno --warnings all \
    --build-property "compiler.cpp.extra_flags=-DBUZZER_PASSIVE=$buz -DLED_COMMON_ANODE=$led -DLCD_MODE=$lcd" \
    neurocheck 2>&1)
  if [ $? -eq 0 ]; then
    flash=$(echo "$out" | grep -o 'uses [0-9]* bytes ([0-9]*%) of program' | grep -o '[0-9]*%')
    ram=$(echo "$out" | grep -o 'use [0-9]* bytes ([0-9]*%) of dynamic' | grep -o '[0-9]* bytes ([0-9]*%)')
    warn=$(echo "$out" | grep -ci "warning")
    echo "BUZZER_PASSIVE=$buz LED_COMMON_ANODE=$led LCD_MODE=$lcd  OK  flash $flash  RAM $ram  warnings $warn"
  else
    echo "BUZZER_PASSIVE=$buz LED_COMMON_ANODE=$led LCD_MODE=$lcd  FAIL"; echo "$out" | tail -15; fail=1
  fi
done; done; done
exit $fail
