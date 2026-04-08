"""High-level experiment assembly and result serialization."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
import time

import numpy as np

from ..channels import TimeVaryingOFDMChannel, get_scenario
from ..config import ExperimentConfig
from ..objectives import ISACSnapshotProblem, OptimizationResult
from ..optimizers.baselines import equal_power_result, water_filling_result
from ..optimizers.de import DifferentialEvolutionOptimizer
from ..optimizers.gwo import GreyWolfOptimizer
from ..optimizers.nsga2 import NSGA2Optimizer
from ..optimizers.poa import PelicanOptimizer
from ..optimizers.pso import ParticleSwarmOptimizer
from ..optimizers.sa import SimulatedAnnealingOptimizer
from ..optimizers.sfo import StarfishOptimizer
from ..optimizers.slsqp import SLSQPOptimizer
from ..pareto import extract_non_dominated_results, run_weighted_sum_sweep


@dataclass
class ParetoStudySummary:
    scenario_name: str
    snapshot_index: int
    weighted_results: list[OptimizationResult]
    nsga2_results: list[OptimizationResult]
    baseline_results: list[OptimizationResult]
    problem: ISACSnapshotProblem
    channel_sequence_length: int

    def to_dict(self) -> dict:
        def serialize_result(result: OptimizationResult) -> dict:
            return {
                "solver_name": result.solver_name,
                "alpha": result.alpha,
                "metrics": asdict(result.metrics),
                "power_allocation": [float(value) for value in result.power_allocation],
                "waveform_profile": (
                    None
                    if result.waveform_profile is None
                    else [float(value) for value in result.waveform_profile]
                ),
                "metadata": dict(result.metadata),
            }

        return {
            "scenario_name": self.scenario_name,
            "snapshot_index": self.snapshot_index,
            "channel_sequence_length": self.channel_sequence_length,
            "weighted_results": [serialize_result(result) for result in self.weighted_results],
            "nsga2_results": [serialize_result(result) for result in self.nsga2_results],
            "baseline_results": [serialize_result(result) for result in self.baseline_results],
            "problem": {
                "dimension": self.problem.dimension,
                "decision_dimension": self.problem.decision_dimension,
                "total_power_w": self.problem.total_power_w,
                "noise_power_w": self.problem.noise_power_w,
                "gamma": self.problem.gamma,
                "sensing_metric": self.problem.sensing_metric,
                "waveform_co_optimization": self.problem.waveform_co_optimization,
            },
        }

    def save_json(self, output_path: str | Path) -> None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")


@dataclass
class AlgorithmComparisonSummary:
    scenario_name: str
    snapshot_index: int
    alpha: float
    results: list[OptimizationResult]
    channel_sequence_length: int

    def to_dict(self) -> dict:
        return {
            "scenario_name": self.scenario_name,
            "snapshot_index": self.snapshot_index,
            "channel_sequence_length": self.channel_sequence_length,
            "alpha": self.alpha,
            "results": [
                {
                    "solver_name": result.solver_name,
                    "alpha": result.alpha,
                    "metrics": asdict(result.metrics),
                    "power_allocation": [float(value) for value in result.power_allocation],
                    "waveform_profile": (
                        None
                        if result.waveform_profile is None
                        else [float(value) for value in result.waveform_profile]
                    ),
                    "metadata": dict(result.metadata),
                }
                for result in self.results
            ],
        }

    def save_json(self, output_path: str | Path) -> None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")


@dataclass
class DynamicAlgorithmAggregate:
    solver_name: str
    mean_rate_bps_hz: float
    std_rate_bps_hz: float
    mean_sensing_snr_db: float
    std_sensing_snr_db: float
    mean_sensing_detection_probability: float
    std_sensing_detection_probability: float
    mean_objective: float
    std_objective: float
    mean_runtime_ms: float
    total_runtime_ms: float
    per_step_rate_bps_hz: list[float]
    per_step_sensing_snr_db: list[float]
    per_step_sensing_detection_probability: list[float]
    per_step_objective: list[float]
    per_step_runtime_ms: list[float]


@dataclass
class DynamicComparisonSummary:
    scenario_name: str
    alpha: float
    num_time_steps: int
    aggregates: list[DynamicAlgorithmAggregate]

    def to_dict(self) -> dict:
        return {
            "scenario_name": self.scenario_name,
            "alpha": self.alpha,
            "num_time_steps": self.num_time_steps,
            "aggregates": [asdict(item) for item in self.aggregates],
        }

    def save_json(self, output_path: str | Path) -> None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")


def run_pareto_study(config: ExperimentConfig) -> tuple[ParetoStudySummary, object]:
    scenario = get_scenario(config.scenario_name)
    channel = TimeVaryingOFDMChannel(
        ofdm_config=config.ofdm,
        link_budget=config.link_budget,
        simulation_config=config.simulation,
        scenario=scenario,
    )
    sequence = channel.generate_sequence()
    snapshot_index = len(sequence) // 2
    snapshot = sequence[snapshot_index]

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

    gwo_solver = GreyWolfOptimizer(config.gwo)
    weighted_results = run_weighted_sum_sweep(problem, gwo_solver, config.pareto.alpha_grid)

    nsga2_solver = NSGA2Optimizer(config.nsga2)
    nsga2_results = extract_non_dominated_results(nsga2_solver.solve(problem, config.objective.default_alpha))

    baseline_results = [
        equal_power_result(problem, config.objective.default_alpha),
        water_filling_result(problem, 1.0),
    ]

    summary = ParetoStudySummary(
        scenario_name=config.scenario_name,
        snapshot_index=snapshot_index,
        weighted_results=weighted_results,
        nsga2_results=nsga2_results,
        baseline_results=baseline_results,
        problem=problem,
        channel_sequence_length=len(sequence),
    )
    return summary, snapshot


def build_problem_from_config(config: ExperimentConfig) -> tuple[ISACSnapshotProblem, object, int, int]:
    scenario = get_scenario(config.scenario_name)
    channel = TimeVaryingOFDMChannel(
        ofdm_config=config.ofdm,
        link_budget=config.link_budget,
        simulation_config=config.simulation,
        scenario=scenario,
    )
    sequence = channel.generate_sequence()
    snapshot_index = len(sequence) // 2
    snapshot = sequence[snapshot_index]
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
    return problem, snapshot, snapshot_index, len(sequence)


def build_problem_sequence_from_config(config: ExperimentConfig) -> list[ISACSnapshotProblem]:
    scenario = get_scenario(config.scenario_name)
    channel = TimeVaryingOFDMChannel(
        ofdm_config=config.ofdm,
        link_budget=config.link_budget,
        simulation_config=config.simulation,
        scenario=scenario,
    )
    sequence = channel.generate_sequence()
    return [
        ISACSnapshotProblem(
            communication_gain=state.communication_gain,
            sensing_gain=state.sensing_gain,
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
        for state in sequence
    ]


def run_algorithm_comparison(config: ExperimentConfig) -> tuple[AlgorithmComparisonSummary, object]:
    problem, snapshot, snapshot_index, channel_sequence_length = build_problem_from_config(config)
    alpha = config.objective.default_alpha

    solvers = [
        SLSQPOptimizer(config.slsqp),
        ParticleSwarmOptimizer(config.pso),
        DifferentialEvolutionOptimizer(config.de),
        SimulatedAnnealingOptimizer(config.sa),
        StarfishOptimizer(config.sfo),
        PelicanOptimizer(config.poa),
        GreyWolfOptimizer(config.gwo),
    ]

    results: list[OptimizationResult] = []
    for solver in solvers:
        start = time.perf_counter()
        result = solver.solve(problem, alpha)
        runtime_ms = (time.perf_counter() - start) * 1000.0
        result.metadata["runtime_ms"] = round(runtime_ms, 4)
        results.append(result)

    results.sort(key=lambda item: item.metrics.weighted_objective, reverse=True)
    summary = AlgorithmComparisonSummary(
        scenario_name=config.scenario_name,
        snapshot_index=snapshot_index,
        alpha=alpha,
        results=results,
        channel_sequence_length=channel_sequence_length,
    )
    return summary, snapshot


def run_dynamic_algorithm_comparison(config: ExperimentConfig) -> DynamicComparisonSummary:
    problems = build_problem_sequence_from_config(config)
    alpha = config.objective.default_alpha

    solver_builders = [
        ("SLSQP", lambda: SLSQPOptimizer(config.slsqp)),
        ("PSO", lambda: ParticleSwarmOptimizer(config.pso)),
        ("DE", lambda: DifferentialEvolutionOptimizer(config.de)),
        ("SA", lambda: SimulatedAnnealingOptimizer(config.sa)),
        ("SFO", lambda: StarfishOptimizer(config.sfo)),
        ("POA", lambda: PelicanOptimizer(config.poa)),
        ("GWO", lambda: GreyWolfOptimizer(config.gwo)),
    ]

    aggregates: list[DynamicAlgorithmAggregate] = []
    for solver_name, solver_factory in solver_builders:
        rates: list[float] = []
        snrs_db: list[float] = []
        detection_probabilities: list[float] = []
        objectives: list[float] = []
        runtimes: list[float] = []

        for problem in problems:
            solver = solver_factory()
            start = time.perf_counter()
            result = solver.solve(problem, alpha)
            runtime_ms = (time.perf_counter() - start) * 1000.0

            rates.append(result.metrics.communication_rate_bps_hz)
            snrs_db.append(result.metrics.sensing_snr_db)
            detection_probabilities.append(result.metrics.sensing_detection_probability)
            objectives.append(result.metrics.weighted_objective)
            runtimes.append(runtime_ms)

        aggregates.append(
            DynamicAlgorithmAggregate(
                solver_name=solver_name,
                mean_rate_bps_hz=float(np.mean(rates)),
                std_rate_bps_hz=float(np.std(rates)),
                mean_sensing_snr_db=float(np.mean(snrs_db)),
                std_sensing_snr_db=float(np.std(snrs_db)),
                mean_sensing_detection_probability=float(np.mean(detection_probabilities)),
                std_sensing_detection_probability=float(np.std(detection_probabilities)),
                mean_objective=float(np.mean(objectives)),
                std_objective=float(np.std(objectives)),
                mean_runtime_ms=float(np.mean(runtimes)),
                total_runtime_ms=float(np.sum(runtimes)),
                per_step_rate_bps_hz=[float(x) for x in rates],
                per_step_sensing_snr_db=[float(x) for x in snrs_db],
                per_step_sensing_detection_probability=[float(x) for x in detection_probabilities],
                per_step_objective=[float(x) for x in objectives],
                per_step_runtime_ms=[float(x) for x in runtimes],
            )
        )

    aggregates.sort(key=lambda item: item.mean_objective, reverse=True)
    return DynamicComparisonSummary(
        scenario_name=config.scenario_name,
        alpha=alpha,
        num_time_steps=len(problems),
        aggregates=aggregates,
    )
