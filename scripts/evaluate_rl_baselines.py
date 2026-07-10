"""Evaluate simple RL baselines on dynamic ISAC channel episodes."""

from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import json
from pathlib import Path
import sys
import time

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from isac_power_allocation.config import (
    ExperimentConfig,
    build_default_experiment_config,
)
from isac_power_allocation.objectives import ISACSnapshotProblem, OptimizationResult
from isac_power_allocation.optimizers.baselines import equal_power_result
from isac_power_allocation.optimizers.gwo import GreyWolfOptimizer
from isac_power_allocation.optimizers.pso import ParticleSwarmOptimizer
from isac_power_allocation.rl import ISACPowerWaveformEnv


def random_policy_result(
    problem: ISACSnapshotProblem,
    alpha: float,
    rng: np.random.Generator,
) -> OptimizationResult:
    decision = problem.repair(problem.random_decision(rng))
    power, waveform = problem.decompose_decision(decision)
    return OptimizationResult(
        solver_name="RandomPolicy",
        power_allocation=power,
        waveform_profile=waveform,
        metrics=problem.metrics(decision, alpha),
        alpha=alpha,
    )


def summarize_results(results: list[OptimizationResult], runtimes_ms: list[float]) -> dict[str, float]:
    return {
        "mean_weighted_objective": float(np.mean([x.metrics.weighted_objective for x in results])),
        "mean_communication_rate_bps_hz": float(
            np.mean([x.metrics.communication_rate_bps_hz for x in results])
        ),
        "mean_detection_probability": float(
            np.mean([x.metrics.sensing_detection_probability for x in results])
        ),
        "mean_sensing_snr_db": float(np.mean([x.metrics.sensing_snr_db for x in results])),
        "mean_runtime_ms_per_step": float(np.mean(runtimes_ms)),
        "total_runtime_ms": float(np.sum(runtimes_ms)),
    }


def evaluate_baselines(config: ExperimentConfig, include_pso: bool, seed: int) -> dict:
    env = ISACPowerWaveformEnv(config, seed=seed, randomize_channel_seed=False)
    env.reset()
    rng = np.random.default_rng(seed)
    alpha = config.objective.default_alpha

    evaluators = {
        "random_policy": lambda problem: random_policy_result(problem, alpha, rng),
        "equal_power": lambda problem: equal_power_result(problem, alpha),
        "gwo_per_snapshot": lambda problem: GreyWolfOptimizer(config.gwo).solve(problem, alpha),
    }
    if include_pso:
        evaluators["pso_per_snapshot"] = lambda problem: ParticleSwarmOptimizer(config.pso).solve(
            problem,
            alpha,
        )

    summaries: dict[str, dict[str, float]] = {}
    per_step: dict[str, list[dict]] = {}
    for name, evaluator in evaluators.items():
        results: list[OptimizationResult] = []
        runtimes_ms: list[float] = []
        per_step[name] = []

        for time_index, problem in enumerate(env.problems):
            start = time.perf_counter()
            result = evaluator(problem)
            runtime_ms = (time.perf_counter() - start) * 1000.0
            results.append(result)
            runtimes_ms.append(runtime_ms)
            per_step[name].append(
                {
                    "time_index": time_index,
                    "metrics": asdict(result.metrics),
                    "runtime_ms": runtime_ms,
                }
            )

        summaries[name] = summarize_results(results, runtimes_ms)

    return {
        "scenario_name": config.scenario_name,
        "num_subcarriers": config.ofdm.num_subcarriers,
        "num_time_steps": config.simulation.num_time_steps,
        "alpha": alpha,
        "seed": seed,
        "summaries": summaries,
        "per_step": per_step,
    }


def build_config(args: argparse.Namespace) -> ExperimentConfig:
    config = build_default_experiment_config(args.scenario)
    config = replace(
        config,
        ofdm=replace(config.ofdm, num_subcarriers=args.subcarriers),
        link_budget=replace(
            config.link_budget,
            total_power_w=args.total_power,
            per_subcarrier_max_power_w=args.per_subcarrier_max_power,
        ),
        simulation=replace(
            config.simulation,
            num_time_steps=args.horizon,
            random_seed=args.seed,
        ),
        gwo=replace(config.gwo, population_size=args.gwo_population, iterations=args.gwo_iterations),
        pso=replace(config.pso, population_size=args.pso_population, iterations=args.pso_iterations),
    )
    return config


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario", default="UMi_NLOS")
    parser.add_argument("--subcarriers", type=int, default=32)
    parser.add_argument("--horizon", type=int, default=20)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--total-power", type=float, default=10.0)
    parser.add_argument("--per-subcarrier-max-power", type=float, default=1.25)
    parser.add_argument("--gwo-population", type=int, default=30)
    parser.add_argument("--gwo-iterations", type=int, default=50)
    parser.add_argument("--include-pso", action="store_true")
    parser.add_argument("--pso-population", type=int, default=30)
    parser.add_argument("--pso-iterations", type=int, default=50)
    parser.add_argument(
        "--output",
        default="outputs/rl/rl_baseline_summary.json",
        help="Path for the JSON summary.",
    )
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    config = build_config(args)
    summary = evaluate_baselines(config, include_pso=args.include_pso, seed=args.seed)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary["summaries"], indent=2))
    print(f"Saved summary to {output_path}")


if __name__ == "__main__":
    main()
