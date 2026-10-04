#!/usr/bin/env bash
# Compile neurocheck.ino for arduino:avr:uno in every #define combination (12),
# then run the static pin-conflict check.
cd "$(dirname "$0")"
fail=0
for sw in 0 1; do for buz in 0 1; do for lcd in 0 1 2; do
  flags="-DSWITCH_MODULE=$sw -DBUZZER_PASSIVE=$buz -DLCD_MODE=$lcd"
  out=$(arduino-cli compile --fqbn arduino:avr:uno --warnings all \
    --build-property "compiler.cpp.extra_flags=$flags" neurocheck 2>&1)
  if [ $? -eq 0 ]; then
    flash=$(echo "$out" | grep -o 'uses [0-9]* bytes ([0-9]*%) of program' | grep -o '[0-9]* bytes ([0-9]*%)')
    ram=$(echo "$out" | grep -o 'use [0-9]* bytes ([0-9]*%) of dynamic' | grep -o '[0-9]* bytes ([0-9]*%)')
    own=$(echo "$out" | grep -i "warning" | grep -c "neurocheck.ino")
    lib=$(echo "$out" | grep -i "warning" | grep -vc "neurocheck.ino")
    echo "SWITCH_MODULE=$sw BUZZER_PASSIVE=$buz LCD_MODE=$lcd  OK  flash $flash  RAM $ram  warnings: sketch $own, libraries $lib"
  else
    echo "SWITCH_MODULE=$sw BUZZER_PASSIVE=$buz LCD_MODE=$lcd  FAIL"; echo "$out" | tail -15; fail=1
  fi
done; done; done
../.venv/Scripts/python.exe ../tools/check_pins.py || fail=1
exit $fail
