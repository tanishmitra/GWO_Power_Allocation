"""Problem definition and objective functions."""

from __future__ import annotations

from dataclasses import dataclass, field
from statistics import NormalDist

import numpy as np

from .constraints import project_power_allocation
from .math_utils import linear_to_db


@dataclass
class AllocationMetrics:
    communication_rate_bps_hz: float
    sensing_snr_linear: float
    sensing_snr_db: float
    sensing_detection_probability: float
    sensing_utility: float
    sensing_metric_name: str
    weighted_objective: float


@dataclass
class OptimizationResult:
    solver_name: str
    power_allocation: np.ndarray
    metrics: AllocationMetrics
    waveform_profile: np.ndarray | None = None
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
    sensing_metric: str = "snr"
    detection_false_alarm_probability: float = 1e-3
    detection_integration_gain: float = 8.0
    waveform_co_optimization: bool = False
    waveform_min_value: float = 1e-6
    waveform_comm_exponent: float = 0.15
    waveform_sensing_exponent: float = 1.0

    @property
    def dimension(self) -> int:
        return int(self.communication_gain.size)

    @property
    def decision_dimension(self) -> int:
        multiplier = 2 if self.waveform_co_optimization else 1
        return multiplier * self.dimension

    def equal_power_decision(self) -> np.ndarray:
        power = np.full(self.dimension, self.total_power_w / max(self.dimension, 1))
        if not self.waveform_co_optimization:
            return power
        waveform = np.ones(self.dimension)
        return np.concatenate([power, waveform])

    def random_decision(self, rng: np.random.Generator) -> np.ndarray:
        power = rng.dirichlet(np.ones(self.dimension)) * self.total_power_w
        if not self.waveform_co_optimization:
            return power
        waveform = self._repair_waveform(rng.gamma(shape=1.0, scale=1.0, size=self.dimension))
        return np.concatenate([power, waveform])

    def repair(self, decision: np.ndarray) -> np.ndarray:
        power, waveform = self._split_and_repair(decision)
        if not self.waveform_co_optimization:
            return power
        return np.concatenate([power, waveform])

    def decompose_decision(self, decision: np.ndarray) -> tuple[np.ndarray, np.ndarray | None]:
        repaired = self.repair(decision)
        power, waveform = self._split_and_repair(repaired)
        if self.waveform_co_optimization:
            return power, waveform
        return power, None

    def communication_rate(
        self,
        decision_or_power: np.ndarray,
        waveform_profile: np.ndarray | None = None,
    ) -> float:
        power, waveform = self._coerce_inputs(decision_or_power, waveform_profile)
        communication_gain, _ = self._effective_gains(waveform)
        snr_per_subcarrier = power * communication_gain / self.noise_power_w
        return float(np.sum(np.log2(1.0 + snr_per_subcarrier)))

    def sensing_snr_linear(
        self,
        decision_or_power: np.ndarray,
        waveform_profile: np.ndarray | None = None,
    ) -> float:
        power, waveform = self._coerce_inputs(decision_or_power, waveform_profile)
        _, sensing_gain = self._effective_gains(waveform)
        numerator = float(np.dot(power, sensing_gain))
        denominator = self.dimension * self.noise_power_w
        return numerator / denominator

    def sensing_detection_probability(self, sensing_snr_linear: float) -> float:
        pfa = float(np.clip(self.detection_false_alarm_probability, 1e-12, 1.0 - 1e-12))
        threshold = NormalDist().inv_cdf(1.0 - pfa)
        integrated_snr = max(float(sensing_snr_linear), 0.0) * max(self.detection_integration_gain, 1e-12)
        z_value = threshold - np.sqrt(2.0 * integrated_snr)
        return float(1.0 - NormalDist().cdf(z_value))

    def sensing_utility(self, sensing_snr_linear: float) -> tuple[float, str, float]:
        mode = self.sensing_metric.strip().lower()
        detection_probability = self.sensing_detection_probability(sensing_snr_linear)
        if mode in {"snr", "sensing_snr"}:
            return float(np.log10(max(sensing_snr_linear, 1e-12))), "snr_db", detection_probability
        if mode in {"pd", "detection_probability", "probabilistic_detection"}:
            return detection_probability, "detection_probability", detection_probability
        raise ValueError(f"Unsupported sensing metric mode: {self.sensing_metric}")

    def metrics(self, decision: np.ndarray, alpha: float) -> AllocationMetrics:
        repaired = self.repair(decision)
        communication_rate = self.communication_rate(repaired)
        sensing_snr_linear = self.sensing_snr_linear(repaired)
        sensing_utility, sensing_metric_name, detection_probability = self.sensing_utility(
            sensing_snr_linear
        )
        weighted_objective = alpha * communication_rate + (1.0 - alpha) * self.gamma * sensing_utility
        return AllocationMetrics(
            communication_rate_bps_hz=communication_rate,
            sensing_snr_linear=sensing_snr_linear,
            sensing_snr_db=linear_to_db(sensing_snr_linear),
            sensing_detection_probability=detection_probability,
            sensing_utility=sensing_utility,
            sensing_metric_name=sensing_metric_name,
            weighted_objective=float(weighted_objective),
        )

    def scalar_objective(self, decision: np.ndarray, alpha: float) -> float:
        return self.metrics(decision, alpha).weighted_objective

    def multi_objective(self, decision: np.ndarray) -> tuple[float, float]:
        repaired = self.repair(decision)
        communication_rate = self.communication_rate(repaired)
        sensing_snr_linear = self.sensing_snr_linear(repaired)
        sensing_utility, _, _ = self.sensing_utility(sensing_snr_linear)
        return communication_rate, sensing_utility

    def _coerce_inputs(
        self,
        decision_or_power: np.ndarray,
        waveform_profile: np.ndarray | None,
    ) -> tuple[np.ndarray, np.ndarray]:
        if waveform_profile is None:
            repaired = self.repair(decision_or_power)
            return self._split_and_repair(repaired)

        power = project_power_allocation(
            np.asarray(decision_or_power, dtype=float).reshape(-1),
            total_power_w=self.total_power_w,
            per_subcarrier_max_power_w=self.per_subcarrier_max_power_w,
        )
        if self.waveform_co_optimization:
            waveform = self._repair_waveform(waveform_profile)
        else:
            waveform = np.ones(self.dimension)
        return power, waveform

    def _split_and_repair(self, decision: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        vector = np.asarray(decision, dtype=float).reshape(-1)
        if self.waveform_co_optimization:
            if vector.size == self.dimension:
                power_raw = vector
                waveform_raw = np.ones(self.dimension)
            elif vector.size == 2 * self.dimension:
                power_raw = vector[: self.dimension]
                waveform_raw = vector[self.dimension :]
            else:
                raise ValueError(
                    f"Expected decision size {self.dimension} or {2 * self.dimension}, got {vector.size}."
                )
        else:
            if vector.size != self.dimension:
                raise ValueError(f"Expected power-allocation size {self.dimension}, got {vector.size}.")
            power_raw = vector
            waveform_raw = np.ones(self.dimension)

        power = project_power_allocation(
            power_raw,
            total_power_w=self.total_power_w,
            per_subcarrier_max_power_w=self.per_subcarrier_max_power_w,
        )
        waveform = self._repair_waveform(waveform_raw)
        return power, waveform

    def _repair_waveform(self, waveform_raw: np.ndarray) -> np.ndarray:
        waveform = np.asarray(waveform_raw, dtype=float).reshape(-1)
        if waveform.size != self.dimension:
            raise ValueError(f"Expected waveform size {self.dimension}, got {waveform.size}.")
        waveform = np.maximum(waveform, self.waveform_min_value)
        normalization = float(np.sum(waveform))
        if normalization <= 0.0:
            return np.ones(self.dimension)
        return waveform * (self.dimension / normalization)

    def _effective_gains(self, waveform_profile: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        communication_shape = np.power(np.maximum(waveform_profile, self.waveform_min_value), self.waveform_comm_exponent)
        sensing_shape = np.power(np.maximum(waveform_profile, self.waveform_min_value), self.waveform_sensing_exponent)
        return self.communication_gain * communication_shape, self.sensing_gain * sensing_shape
