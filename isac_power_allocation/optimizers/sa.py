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
        decision_dim = problem.decision_dimension
        current = problem.repair(problem.equal_power_decision())
        current_score = problem.scalar_objective(current, alpha)
        best = current.copy()
        best_score = current_score
        history: list[float] = []

        temperature_ratio = self.hyperparameters.final_temperature / self.hyperparameters.initial_temperature
        for iteration in range(self.hyperparameters.iterations):
            fraction = iteration / max(self.hyperparameters.iterations - 1, 1)
            temperature = self.hyperparameters.initial_temperature * (temperature_ratio ** fraction)

            sigma = self.hyperparameters.proposal_sigma * problem.total_power_w / max(decision_dim, 1)
            proposal = problem.repair(current + rng.normal(0.0, sigma, size=decision_dim))
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

        best_power, best_waveform = problem.decompose_decision(best)
        return OptimizationResult(
            solver_name="SA",
            power_allocation=best_power,
            waveform_profile=best_waveform,
            metrics=problem.metrics(best, alpha),
            alpha=alpha,
            history=history,
            metadata={"iterations": self.hyperparameters.iterations},
        )
