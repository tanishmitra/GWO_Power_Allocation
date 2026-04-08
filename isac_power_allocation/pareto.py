"""Pareto utilities for weighted sweeps and non-dominated filtering."""

from __future__ import annotations

from typing import Iterable

from .objectives import ISACSnapshotProblem, OptimizationResult
from .optimizers.gwo import GreyWolfOptimizer


def dominates(left: OptimizationResult, right: OptimizationResult) -> bool:
    left_pair = (
        left.metrics.communication_rate_bps_hz,
        left.metrics.sensing_utility,
    )
    right_pair = (
        right.metrics.communication_rate_bps_hz,
        right.metrics.sensing_utility,
    )
    return (
        left_pair[0] >= right_pair[0]
        and left_pair[1] >= right_pair[1]
        and (left_pair[0] > right_pair[0] or left_pair[1] > right_pair[1])
    )


def extract_non_dominated_results(results: Iterable[OptimizationResult]) -> list[OptimizationResult]:
    candidates = list(results)
    front: list[OptimizationResult] = []
    for index, candidate in enumerate(candidates):
        if any(
            dominates(other, candidate) for other_index, other in enumerate(candidates) if other_index != index
        ):
            continue
        front.append(candidate)
    front.sort(key=lambda result: result.metrics.communication_rate_bps_hz)
    return front


def run_weighted_sum_sweep(
    problem: ISACSnapshotProblem,
    solver: GreyWolfOptimizer,
    alphas: Iterable[float],
) -> list[OptimizationResult]:
    results = [solver.solve(problem, alpha) for alpha in alphas]
    return extract_non_dominated_results(results)
