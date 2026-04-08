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
        population = np.vstack(
            [
                problem.repair(rng.dirichlet(np.ones(problem.dimension)) * problem.total_power_w)
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
                    candidate = candidate + rng.random(problem.dimension) * (population[j] - interaction * candidate)
                else:
                    candidate = candidate + rng.random(problem.dimension) * (best - candidate)

                if rng.random() < self.hyperparameters.surface_attack_probability:
                    sigma = 0.03 * problem.total_power_w / max(problem.dimension, 1)
                    candidate = candidate + rng.normal(0.0, sigma, size=problem.dimension)

                candidate = problem.repair(candidate)
                candidate_score = problem.scalar_objective(candidate, alpha)
                if candidate_score > scores[i]:
                    population[i] = candidate
                    scores[i] = candidate_score
            history.append(float(scores.max()))

        best_index = int(np.argmax(scores))
        best_allocation = population[best_index]
        return OptimizationResult(
            solver_name="POA",
            power_allocation=best_allocation,
            metrics=problem.metrics(best_allocation, alpha),
            alpha=alpha,
            history=history,
            metadata={
                "population_size": self.hyperparameters.population_size,
                "iterations": self.hyperparameters.iterations,
            },
        )
