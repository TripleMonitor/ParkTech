from tools.check_pins import INO, check


def test_firmware_has_no_pin_conflicts():
    assert check(INO.read_text(encoding="utf-8")) == []


def test_checker_catches_conflicts():
    text = INO.read_text(encoding="utf-8")
    assert check(text.replace("PIN_B = 5", "PIN_B = 11"))            # Timer2 PWM pin
    assert check(text.replace("lcdPar(12, 11, 7, 6, 4, 3)", "lcdPar(12, 11, 7, 6, 4, 2)"))  # D2 twice
