"""Problem definition and objective functions."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .constraints import project_power_allocation
from .math_utils import linear_to_db


@dataclass
class AllocationMetrics:
    communication_rate_bps_hz: float
    sensing_snr_linear: float
    sensing_snr_db: float
    weighted_objective: float


@dataclass
class OptimizationResult:
    solver_name: str
    power_allocation: np.ndarray
    metrics: AllocationMetrics
    alpha: float | None = None
    history: list[float] = field(default_factory=list)
    metadata: dict[str, float | int | str] = field(default_factory=dict)


@dataclass(frozen=True)
class ISACSnapshotProblem:
    communication_gain: np.ndarray
    sensing_gain: np.ndarray
    total_power_w: float
    noise_power_w: float
    gamma: float
    per_subcarrier_max_power_w: float | None = None

    @property
    def dimension(self) -> int:
        return int(self.communication_gain.size)

    def repair(self, power_allocation: np.ndarray) -> np.ndarray:
        return project_power_allocation(
            power_allocation,
            total_power_w=self.total_power_w,
            per_subcarrier_max_power_w=self.per_subcarrier_max_power_w,
        )

    def communication_rate(self, power_allocation: np.ndarray) -> float:
        snr_per_subcarrier = power_allocation * self.communication_gain / self.noise_power_w
        return float(np.sum(np.log2(1.0 + snr_per_subcarrier)))

    def sensing_snr_linear(self, power_allocation: np.ndarray) -> float:
        numerator = float(np.dot(power_allocation, self.sensing_gain))
        denominator = self.dimension * self.noise_power_w
        return numerator / denominator

    def metrics(self, power_allocation: np.ndarray, alpha: float) -> AllocationMetrics:
        repaired = self.repair(power_allocation)
        communication_rate = self.communication_rate(repaired)
        sensing_snr_linear = self.sensing_snr_linear(repaired)
        weighted_objective = (
            alpha * communication_rate
            + (1.0 - alpha) * self.gamma * np.log10(max(sensing_snr_linear, 1e-12))
        )
        return AllocationMetrics(
            communication_rate_bps_hz=communication_rate,
            sensing_snr_linear=sensing_snr_linear,
            sensing_snr_db=linear_to_db(sensing_snr_linear),
            weighted_objective=float(weighted_objective),
        )

    def scalar_objective(self, power_allocation: np.ndarray, alpha: float) -> float:
        return self.metrics(power_allocation, alpha).weighted_objective

    def multi_objective(self, power_allocation: np.ndarray) -> tuple[float, float]:
        repaired = self.repair(power_allocation)
        return self.communication_rate(repaired), self.sensing_snr_linear(repaired)
