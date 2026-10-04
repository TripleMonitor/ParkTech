from tools.check_pins import INO, check


def test_firmware_has_no_pin_conflicts():
    assert check(INO.read_text(encoding="utf-8")) == []


def test_checker_catches_conflicts():
    text = INO.read_text(encoding="utf-8")
    assert check(text.replace("PIN_LED_RED = 11", "PIN_LED_RED = 6"))     # same pin as buzzer
    assert check(text.replace("lcdPar(12, 8, 7, 5, 4, 3)", "lcdPar(12, 8, 7, 5, 4, 2)"))  # D2 twice
    assert check(text.replace("PIN_LED_GREEN = 9", "PIN_LED_GRN = 9"))      # missing pin
