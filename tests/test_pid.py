import pytest

from pid import PID


def test_p_only():
    c = PID(2.0, 0, 0, setpoint=1.0, out_min=-10, out_max=10)
    assert c.update(0.5) == pytest.approx(1.0)


def test_output_limits():
    c = PID(100.0, 0, 0, setpoint=1.0, out_min=-0.15, out_max=0.15)
    assert c.update(0.0) == 0.15 and c.update(2.0) == -0.15


def test_integral_drives_to_setpoint_on_simple_plant():
    c = PID(0.2, 0.3, 0.0, setpoint=5.0, out_min=-10, out_max=10)
    y = 0.0
    for _ in range(200):
        u = c.update(y, dt=0.1)
        y += (u - 0.2 * y) * 0.1 * 5        # first-order plant
    assert y == pytest.approx(5.0, abs=0.05)


def test_anti_windup_limits_integrator_during_saturation():
    c = PID(0.0, 1.0, 0.0, setpoint=10.0, out_min=-1, out_max=1)
    for _ in range(1000):                   # long saturation: big constant error
        c.update(0.0)
    assert c.integral <= 1.0                # clamped, not 10000
    # once the error flips, output leaves saturation immediately (no long unwinding)
    assert c.update(20.0) < 1.0


def test_derivative_on_measurement_no_kick_on_setpoint_change():
    c = PID(0.0, 0.0, 1.0, setpoint=0.0, out_min=-10, out_max=10)
    c.update(1.0)
    c.setpoint = 100.0                      # set point jump, measurement unchanged
    assert c.update(1.0) == pytest.approx(0.0)


def test_derivative_low_pass():
    raw = PID(0, 0, 1.0, out_min=-10, out_max=10)
    filt = PID(0, 0, 1.0, out_min=-10, out_max=10, d_tau=4.0)
    raw.update(0.0)
    filt.update(0.0)
    assert abs(filt.update(1.0)) < abs(raw.update(1.0))


def test_reverse_acting():
    c = PID(1.0, 0, 0, setpoint=0.85, out_min=-1, out_max=1, reverse=True)
    assert c.update(0.95) > 0 and c.update(0.5) < 0


def test_reset():
    c = PID(1.0, 1.0, 1.0, setpoint=1.0, out_min=-5, out_max=5)
    c.update(0.0)
    c.update(0.3)
    c.reset()
    assert c.integral == 0.0 and c.update(1.0) == 0.0


def test_bad_args():
    with pytest.raises(ValueError):
        PID(1, 0, 0, out_min=1, out_max=0)
    with pytest.raises(ValueError):
        PID(1, 0, 0).update(0.0, dt=0)
