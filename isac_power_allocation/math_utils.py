"""Shared numeric helpers."""

from __future__ import annotations

import math


LIGHT_SPEED_MPS = 299_792_458.0


def db_to_linear(value_db: float) -> float:
    return 10.0 ** (value_db / 10.0)


def linear_to_db(value_linear: float, floor: float = 1e-18) -> float:
    return 10.0 * math.log10(max(value_linear, floor))


def fspl_at_1m_db(carrier_frequency_hz: float) -> float:
    wavelength = LIGHT_SPEED_MPS / carrier_frequency_hz
    return 20.0 * math.log10(4.0 * math.pi / wavelength)
