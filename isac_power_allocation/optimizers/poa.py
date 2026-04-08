"""Pelican Optimization Algorithm for scalarized ISAC power allocation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..config import POAHyperparameters
from ..objectives import ISACSnapshotProblem, OptimizationResult


@dataclass(frozen=True)
class PelicanOptimizer:
    hyperparameters: POAHyperparameters

    def solve(self, problem: ISACSnapshotProblem, alpha: float) -> OptimizationResult:
        rng = np.random.default_rng(self.hyperparameters.seed)
        decision_dim = problem.decision_dimension
        population = np.vstack(
            [
                problem.repair(problem.random_decision(rng))
                for _ in range(self.hyperparameters.population_size)
            ]
        )
        scores = np.array([problem.scalar_objective(x, alpha) for x in population])
        history: list[float] = []

        for _ in range(self.hyperparameters.iterations):
            best_index = int(np.argmax(scores))
            best = population[best_index].copy()

            for i in range(self.hyperparameters.population_size):
                candidate = population[i].copy()
                if rng.random() < 0.5:
                    j = int(rng.integers(0, self.hyperparameters.population_size))
                    interaction = int(rng.integers(1, 3))
                    candidate = candidate + rng.random(decision_dim) * (population[j] - interaction * candidate)
                else:
                    candidate = candidate + rng.random(decision_dim) * (best - candidate)

                if rng.random() < self.hyperparameters.surface_attack_probability:
                    sigma = 0.03 * problem.total_power_w / max(decision_dim, 1)
                    candidate = candidate + rng.normal(0.0, sigma, size=decision_dim)

                candidate = problem.repair(candidate)
                candidate_score = problem.scalar_objective(candidate, alpha)
                if candidate_score > scores[i]:
                    population[i] = candidate
                    scores[i] = candidate_score
            history.append(float(scores.max()))

        best_index = int(np.argmax(scores))
        best_decision = population[best_index]
        best_power, best_waveform = problem.decompose_decision(best_decision)
        return OptimizationResult(
            solver_name="POA",
            power_allocation=best_power,
            waveform_profile=best_waveform,
            metrics=problem.metrics(best_decision, alpha),
            alpha=alpha,
            history=history,
            metadata={
                "population_size": self.hyperparameters.population_size,
                "iterations": self.hyperparameters.iterations,
            },
        )
