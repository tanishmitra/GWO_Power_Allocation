"""Simple benchmark allocators."""

from __future__ import annotations

import numpy as np

from ..objectives import ISACSnapshotProblem, OptimizationResult


def equal_power_result(problem: ISACSnapshotProblem, alpha: float) -> OptimizationResult:
    repaired = problem.repair(problem.equal_power_decision())
    power, waveform = problem.decompose_decision(repaired)
    return OptimizationResult(
        solver_name="EqualPower",
        power_allocation=power,
        waveform_profile=waveform,
        metrics=problem.metrics(repaired, alpha),
        alpha=alpha,
    )


def water_filling_result(problem: ISACSnapshotProblem, alpha: float = 1.0) -> OptimizationResult:
    gains = np.maximum(problem.communication_gain, 1e-12)
    noise_terms = problem.noise_power_w / gains

    low = float(np.min(noise_terms))
    high = float(np.max(noise_terms) + problem.total_power_w)

    for _ in range(80):
        water_level = 0.5 * (low + high)
        candidate = np.maximum(0.0, water_level - noise_terms)
        if candidate.sum() > problem.total_power_w:
            high = water_level
        else:
            low = water_level

    allocation = np.maximum(0.0, low - noise_terms)
    repaired = problem.repair(allocation)
    power, waveform = problem.decompose_decision(repaired)
    return OptimizationResult(
        solver_name="WaterFilling",
        power_allocation=power,
        waveform_profile=waveform,
        metrics=problem.metrics(repaired, alpha),
        alpha=alpha,
    )
