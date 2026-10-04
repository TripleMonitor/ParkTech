"""Static pin-conflict check for firmware/neurocheck/neurocheck.ino.

Reads the pin constants and the parallel-LCD constructor from the sketch and checks,
for every LCD mode, that no pin is used twice and nothing uses D0/D1 (USB serial).
The three LEDs are plain on/off (digitalWrite), so Timer2/tone() can't interfere.
Exit code 0 = no conflicts.
"""
from __future__ import annotations

import pathlib
import re
import sys

INO = pathlib.Path(__file__).resolve().parent.parent / "firmware" / "neurocheck" / "neurocheck.ino"
TIMER2_PWM = {3, 11}
UNO_PWM = {3, 5, 6, 9, 10, 11}


def pins_from_sketch(text: str) -> tuple[dict[str, int], list[int]]:
    consts: dict[str, int] = {}
    for line in text.splitlines():
        if line.lstrip().startswith("//"):
            continue
        for name, val in re.findall(r"\b(PIN_\w+)\s*=\s*(\d+)", line):
            consts[name] = int(val)
    m = re.search(r"LiquidCrystal\s+lcdPar\(([^)]*)\)", text)
    lcd = [int(x) for x in m.group(1).split(",")] if m else []
    return consts, lcd


def check(text: str) -> list[str]:
    consts, lcd_par = pins_from_sketch(text)
    problems = []
    for k in ("PIN_SWITCH", "PIN_BUZZER", "PIN_LED_GREEN", "PIN_LED_YELLOW", "PIN_LED_RED"):
        if k not in consts:
            problems.append(f"{k} not found in the sketch")
    for mode, extra in (("none", []), ("parallel", lcd_par), ("i2c", ["A4", "A5"])):
        used = list(consts.values()) + extra
        dupes = sorted({str(p) for p in used if used.count(p) > 1})
        if dupes:
            problems.append(f"LCD {mode}: pin(s) used twice: {', '.join(dupes)}")
        if {0, 1} & {p for p in used if isinstance(p, int)}:
            problems.append(f"LCD {mode}: uses D0/D1 (USB serial)")
    return problems


def main() -> int:
    text = INO.read_text(encoding="utf-8")
    consts, lcd = pins_from_sketch(text)
    print("pins:", ", ".join(f"{k}=D{v}" for k, v in consts.items()))
    print("parallel LCD (RS,E,D4..D7):", ", ".join(f"D{p}" for p in lcd))
    problems = check(text)
    for p in problems:
        print("CONFLICT:", p)
    print("PIN CHECK:", "PASS" if not problems else "FAIL")
    return 0 if not problems else 1


if __name__ == "__main__":
    sys.exit(main())
