from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Signal:
    action: str
    probability: float
    expected_edge: float


def evaluate_signal(probability: float, commission: float, target_edge: float = 0.002) -> Signal:
    expected_edge = (probability - 0.5) * 2 * target_edge - commission / 10000
    if expected_edge > 0:
        return Signal(action="TRADE", probability=probability, expected_edge=expected_edge)
    return Signal(action="HOLD", probability=probability, expected_edge=expected_edge)
