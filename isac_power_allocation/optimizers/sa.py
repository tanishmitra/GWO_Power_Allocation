"""Simulated Annealing for scalarized ISAC power allocation."""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from ..config import SAHyperparameters
from ..objectives import ISACSnapshotProblem, OptimizationResult


@dataclass(frozen=True)
class SimulatedAnnealingOptimizer:
    hyperparameters: SAHyperparameters

    def solve(self, problem: ISACSnapshotProblem, alpha: float) -> OptimizationResult:
        rng = np.random.default_rng(self.hyperparameters.seed)
        current = np.full(problem.dimension, problem.total_power_w / problem.dimension)
        current = problem.repair(current)
        current_score = problem.scalar_objective(current, alpha)
        best = current.copy()
        best_score = current_score
        history: list[float] = []

        temperature_ratio = self.hyperparameters.final_temperature / self.hyperparameters.initial_temperature
        for iteration in range(self.hyperparameters.iterations):
            fraction = iteration / max(self.hyperparameters.iterations - 1, 1)
            temperature = self.hyperparameters.initial_temperature * (temperature_ratio ** fraction)

            sigma = self.hyperparameters.proposal_sigma * problem.total_power_w / max(problem.dimension, 1)
            proposal = problem.repair(current + rng.normal(0.0, sigma, size=problem.dimension))
            proposal_score = problem.scalar_objective(proposal, alpha)
            delta = proposal_score - current_score

            accept = delta >= 0.0 or rng.random() < math.exp(delta / max(temperature, 1e-12))
            if accept:
                current = proposal
                current_score = proposal_score
            if current_score > best_score:
                best = current.copy()
                best_score = current_score
            history.append(best_score)

        return OptimizationResult(
            solver_name="SA",
            power_allocation=best,
            metrics=problem.metrics(best, alpha),
            alpha=alpha,
            history=history,
            metadata={"iterations": self.hyperparameters.iterations},
        )
