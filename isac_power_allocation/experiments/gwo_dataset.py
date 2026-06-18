"""CSV dataset export for GWO-based surrogate modeling."""

from __future__ import annotations

import csv
from dataclasses import replace
from pathlib import Path
import time
from typing import Iterable

import numpy as np

from ..channels import TimeVaryingOFDMChannel
from ..channels.scenarios import STANDARD_SCENARIOS, get_scenario
from ..config import ExperimentConfig, build_default_experiment_config
from ..objectives import ISACSnapshotProblem
from ..optimizers.gwo import GreyWolfOptimizer


def export_gwo_dataset(
    output_path: str | Path,
    config: ExperimentConfig | None = None,
    scenario_names: Iterable[str] | None = None,
) -> Path:
    """Run GWO on channel snapshots and save supervised-learning rows as CSV."""
    base_config = config or build_default_experiment_config()
    scenarios = list(scenario_names) if scenario_names is not None else sorted(STANDARD_SCENARIOS)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, float | int | str | bool]] = []
    for scenario_name in scenarios:
        scenario_config = replace(base_config, scenario_name=scenario_name)
        rows.extend(_build_rows_for_scenario(scenario_config))

    fieldnames = _fieldnames(base_config.ofdm.num_subcarriers)
    with output_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    return output_path


def _build_rows_for_scenario(config: ExperimentConfig) -> list[dict[str, float | int | str | bool]]:
    scenario = get_scenario(config.scenario_name)
    channel = TimeVaryingOFDMChannel(
        ofdm_config=config.ofdm,
        link_budget=config.link_budget,
        simulation_config=config.simulation,
        scenario=scenario,
    )
    sequence = channel.generate_sequence()
    solver = GreyWolfOptimizer(config.gwo)
    alpha = config.objective.default_alpha

    rows: list[dict[str, float | int | str | bool]] = []
    for snapshot in sequence:
        problem = ISACSnapshotProblem(
            communication_gain=snapshot.communication_gain,
            sensing_gain=snapshot.sensing_gain,
            total_power_w=config.link_budget.total_power_w,
            noise_power_w=config.link_budget.noise_power_w,
            gamma=config.objective.gamma,
            per_subcarrier_max_power_w=config.link_budget.per_subcarrier_max_power_w,
            sensing_metric=config.objective.sensing_metric,
            detection_false_alarm_probability=config.objective.detection_false_alarm_probability,
            detection_integration_gain=config.objective.detection_integration_gain,
            waveform_co_optimization=config.objective.waveform_co_optimization,
            waveform_min_value=config.objective.waveform_min_value,
            waveform_comm_exponent=config.objective.waveform_comm_exponent,
            waveform_sensing_exponent=config.objective.waveform_sensing_exponent,
        )

        start = time.perf_counter()
        result = solver.solve(problem, alpha)
        runtime_ms = (time.perf_counter() - start) * 1000.0

        row = _base_row(config, snapshot.time_index, snapshot.time_s)
        row.update(_gain_features("communication", snapshot.communication_gain))
        row.update(_gain_features("sensing", snapshot.sensing_gain))
        row.update(_result_features(result, runtime_ms))
        rows.append(row)

    return rows


def _base_row(
    config: ExperimentConfig,
    time_index: int,
    time_s: float,
) -> dict[str, float | int | str | bool]:
    scenario = get_scenario(config.scenario_name)
    return {
        "scenario_name": config.scenario_name,
        "time_index": int(time_index),
        "time_s": float(time_s),
        "alpha": float(config.objective.default_alpha),
        "gamma": float(config.objective.gamma),
        "sensing_metric": config.objective.sensing_metric,
        "waveform_co_optimization": bool(config.objective.waveform_co_optimization),
        "num_subcarriers": int(config.ofdm.num_subcarriers),
        "subcarrier_spacing_hz": float(config.ofdm.subcarrier_spacing_hz),
        "carrier_frequency_hz": float(config.ofdm.carrier_frequency_hz),
        "bandwidth_hz": float(config.ofdm.bandwidth_hz),
        "total_power_w": float(config.link_budget.total_power_w),
        "noise_power_w": float(config.link_budget.noise_power_w),
        "per_subcarrier_max_power_w": _optional_float(config.link_budget.per_subcarrier_max_power_w),
        "user_distance_m": float(config.link_budget.user_distance_m),
        "target_distance_m": float(config.link_budget.target_distance_m),
        "user_speed_mps": float(config.link_budget.user_speed_mps),
        "target_speed_mps": float(config.link_budget.target_speed_mps),
        "path_loss_exponent": float(scenario.path_loss_exponent),
        "shadowing_std_db": float(scenario.shadowing_std_db),
        "rms_delay_spread_s": float(scenario.rms_delay_spread_s),
        "num_clusters": int(scenario.num_clusters),
        "k_factor_db": _optional_float(scenario.k_factor_db),
        "gwo_population_size": int(config.gwo.population_size),
        "gwo_iterations": int(config.gwo.iterations),
        "gwo_mutation_sigma": float(config.gwo.mutation_sigma),
        "gwo_seed": int(config.gwo.seed),
    }


def _gain_features(prefix: str, values: np.ndarray) -> dict[str, float]:
    values = np.asarray(values, dtype=float).reshape(-1)
    features = {
        f"{prefix}_gain_mean": float(np.mean(values)),
        f"{prefix}_gain_std": float(np.std(values)),
        f"{prefix}_gain_min": float(np.min(values)),
        f"{prefix}_gain_max": float(np.max(values)),
        f"{prefix}_gain_sum": float(np.sum(values)),
    }
    features.update({f"{prefix}_gain_sc{index:02d}": float(value) for index, value in enumerate(values)})
    return features


def _result_features(result, runtime_ms: float) -> dict[str, float | int | str]:
    features: dict[str, float | int | str] = {
        "solver_name": result.solver_name,
        "runtime_ms": float(runtime_ms),
        "communication_rate_bps_hz": float(result.metrics.communication_rate_bps_hz),
        "sensing_snr_linear": float(result.metrics.sensing_snr_linear),
        "sensing_snr_db": float(result.metrics.sensing_snr_db),
        "sensing_detection_probability": float(result.metrics.sensing_detection_probability),
        "sensing_utility": float(result.metrics.sensing_utility),
        "sensing_metric_name": result.metrics.sensing_metric_name,
        "weighted_objective": float(result.metrics.weighted_objective),
        "history_start_objective": float(result.history[0]) if result.history else "",
        "history_end_objective": float(result.history[-1]) if result.history else "",
        "history_best_objective": float(max(result.history)) if result.history else "",
        "history_improvement": float(result.history[-1] - result.history[0]) if len(result.history) >= 2 else "",
    }
    features.update(
        {f"optimal_power_sc{index:02d}": float(value) for index, value in enumerate(result.power_allocation)}
    )
    if result.waveform_profile is not None:
        features.update(
            {f"waveform_sc{index:02d}": float(value) for index, value in enumerate(result.waveform_profile)}
        )
    return features


def _fieldnames(num_subcarriers: int) -> list[str]:
    base_fields = [
        "scenario_name",
        "time_index",
        "time_s",
        "alpha",
        "gamma",
        "sensing_metric",
        "waveform_co_optimization",
        "num_subcarriers",
        "subcarrier_spacing_hz",
        "carrier_frequency_hz",
        "bandwidth_hz",
        "total_power_w",
        "noise_power_w",
        "per_subcarrier_max_power_w",
        "user_distance_m",
        "target_distance_m",
        "user_speed_mps",
        "target_speed_mps",
        "path_loss_exponent",
        "shadowing_std_db",
        "rms_delay_spread_s",
        "num_clusters",
        "k_factor_db",
        "gwo_population_size",
        "gwo_iterations",
        "gwo_mutation_sigma",
        "gwo_seed",
    ]
    gain_fields = []
    for prefix in ("communication", "sensing"):
        gain_fields.extend(
            [
                f"{prefix}_gain_mean",
                f"{prefix}_gain_std",
                f"{prefix}_gain_min",
                f"{prefix}_gain_max",
                f"{prefix}_gain_sum",
            ]
        )
        gain_fields.extend(f"{prefix}_gain_sc{index:02d}" for index in range(num_subcarriers))

    result_fields = [
        "solver_name",
        "runtime_ms",
        "communication_rate_bps_hz",
        "sensing_snr_linear",
        "sensing_snr_db",
        "sensing_detection_probability",
        "sensing_utility",
        "sensing_metric_name",
        "weighted_objective",
        "history_start_objective",
        "history_end_objective",
        "history_best_objective",
        "history_improvement",
    ]
    result_fields.extend(f"optimal_power_sc{index:02d}" for index in range(num_subcarriers))
    result_fields.extend(f"waveform_sc{index:02d}" for index in range(num_subcarriers))
    return base_fields + gain_fields + result_fields


def _optional_float(value: float | None) -> float | str:
    return "" if value is None else float(value)
