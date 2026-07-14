import math

import pytest

from venjix.signal import EwmaPredictionError


def test_starts_at_zero():
    assert EwmaPredictionError(0.3).value == 0.0


def test_exact_arithmetic():
    signal = EwmaPredictionError(0.5)
    assert signal.update(True) == 0.5
    assert signal.update(True) == 0.75
    assert signal.update(False) == 0.375
    assert math.isclose(signal.update(True), 0.6875)


def test_converges_up_then_decays():
    signal = EwmaPredictionError(0.3)
    for _ in range(20):
        signal.update(True)
    assert signal.value > 0.99
    for _ in range(20):
        signal.update(False)
    assert signal.value < 0.01


def test_alpha_one_tracks_latest():
    signal = EwmaPredictionError(1.0)
    assert signal.update(True) == 1.0
    assert signal.update(False) == 0.0


def test_alpha_validation():
    for bad in (0.0, -0.1, 1.5):
        with pytest.raises(ValueError):
            EwmaPredictionError(bad)
