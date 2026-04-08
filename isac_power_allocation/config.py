"""Experiment and system configuration objects."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True)
class OFDMConfig:
    num_subcarriers: int = 32
    subcarrier_spacing_hz: float = 30_000.0
    carrier_frequency_hz: float = 3.5e9

    @property
    def bandwidth_hz(self) -> float:
        return self.num_subcarriers * self.subcarrier_spacing_hz

    def subcarrier_offsets_hz(self) -> np.ndarray:
        centered = np.arange(self.num_subcarriers) - (self.num_subcarriers - 1) / 2.0
        return centered * self.subcarrier_spacing_hz


@dataclass(frozen=True)
class LinkBudgetConfig:
    total_power_w: float = 10.0
    noise_power_w: float = 1e-8
    per_subcarrier_max_power_w: float | None = 1.25
    user_distance_m: float = 120.0
    target_distance_m: float = 75.0
    user_speed_mps: float = 8.33
    target_speed_mps: float = 12.0
    communication_processing_gain_db: float = 30.0
    radar_cross_section_dbsm: float = 10.0
    sensing_processing_gain_db: float = 115.0


@dataclass(frozen=True)
class SimulationConfig:
    num_time_steps: int = 20
    csi_update_interval_s: float = 1e-3
    num_taps: int = 12
    random_seed: int = 7


@dataclass(frozen=True)
class ObjectiveConfig:
    gamma: float = 10.0
    default_alpha: float = 0.5
    sensing_metric: str = "snr"
    detection_false_alarm_probability: float = 1e-3
    detection_integration_gain: float = 8.0
    waveform_co_optimization: bool = False
    waveform_min_value: float = 1e-6
    waveform_comm_exponent: float = 0.15
    waveform_sensing_exponent: float = 1.0


@dataclass(frozen=True)
class GWOHyperparameters:
    population_size: int = 30
    iterations: int = 50
    mutation_sigma: float = 0.02
    seed: int = 11


@dataclass(frozen=True)
class NSGA2Hyperparameters:
    population_size: int = 48
    generations: int = 60
    crossover_rate: float = 0.9
    mutation_rate: float = 0.12
    mutation_sigma: float = 0.04
    seed: int = 29


@dataclass(frozen=True)
class SLSQPHyperparameters:
    restarts: int = 1
    max_iterations: int = 120
    seed: int = 5


@dataclass(frozen=True)
class PSOHyperparameters:
    population_size: int = 30
    iterations: int = 50
    inertia_weight: float = 0.7
    cognitive_weight: float = 1.5
    social_weight: float = 1.5
    seed: int = 17


@dataclass(frozen=True)
class DEHyperparameters:
    population_size: int = 30
    iterations: int = 50
    differential_weight: float = 0.8
    crossover_rate: float = 0.9
    seed: int = 19


@dataclass(frozen=True)
class SAHyperparameters:
    iterations: int = 1000
    initial_temperature: float = 100.0
    final_temperature: float = 0.1
    proposal_sigma: float = 0.08
    seed: int = 23


@dataclass(frozen=True)
class SFOHyperparameters:
    population_size: int = 30
    iterations: int = 50
    regeneration_probability: float = 0.1
    seed: int = 31


@dataclass(frozen=True)
class POAHyperparameters:
    population_size: int = 30
    iterations: int = 50
    surface_attack_probability: float = 0.2
    seed: int = 37


@dataclass(frozen=True)
class ParetoConfig:
    alpha_grid: tuple[float, ...] = field(
        default_factory=lambda: tuple(float(x) for x in np.linspace(0.0, 1.0, 11))
    )


@dataclass(frozen=True)
class ExperimentConfig:
    scenario_name: str = "UMi_NLOS"
    ofdm: OFDMConfig = field(default_factory=OFDMConfig)
    link_budget: LinkBudgetConfig = field(default_factory=LinkBudgetConfig)
    simulation: SimulationConfig = field(default_factory=SimulationConfig)
    objective: ObjectiveConfig = field(default_factory=ObjectiveConfig)
    slsqp: SLSQPHyperparameters = field(default_factory=SLSQPHyperparameters)
    pso: PSOHyperparameters = field(default_factory=PSOHyperparameters)
    de: DEHyperparameters = field(default_factory=DEHyperparameters)
    sa: SAHyperparameters = field(default_factory=SAHyperparameters)
    sfo: SFOHyperparameters = field(default_factory=SFOHyperparameters)
    poa: POAHyperparameters = field(default_factory=POAHyperparameters)
    gwo: GWOHyperparameters = field(default_factory=GWOHyperparameters)
    nsga2: NSGA2Hyperparameters = field(default_factory=NSGA2Hyperparameters)
    pareto: ParetoConfig = field(default_factory=ParetoConfig)


def build_default_experiment_config(scenario_name: str = "UMi_NLOS") -> ExperimentConfig:
    return ExperimentConfig(scenario_name=scenario_name)
