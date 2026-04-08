"""Starfish Optimization Algorithm for scalarized ISAC power allocation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..config import SFOHyperparameters
from ..objectives import ISACSnapshotProblem, OptimizationResult


@dataclass(frozen=True)
class StarfishOptimizer:
    hyperparameters: SFOHyperparameters

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

        for iteration in range(self.hyperparameters.iterations):
            best_index = int(np.argmax(scores))
            best = population[best_index].copy()
            adaptive_a = 2.0 - 2.0 * iteration / max(self.hyperparameters.iterations - 1, 1)

            for i in range(self.hyperparameters.population_size):
                candidate = population[i].copy()
                if rng.random() < 0.5:
                    random_direction = rng.dirichlet(np.ones(problem.dimension)) * problem.total_power_w
                    candidate = candidate + adaptive_a * rng.random(problem.dimension) * (random_direction - candidate)
                else:
                    candidate = candidate + adaptive_a * rng.random(problem.dimension) * (best - candidate)

                if rng.random() < self.hyperparameters.regeneration_probability:
                    regenerate_mask = rng.random(problem.dimension) < 0.1
                    if np.any(regenerate_mask):
                        candidate[regenerate_mask] = rng.random(int(regenerate_mask.sum()))

                candidate = problem.repair(candidate)
                candidate_score = problem.scalar_objective(candidate, alpha)
                if candidate_score > scores[i]:
                    population[i] = candidate
                    scores[i] = candidate_score
            history.append(float(scores.max()))

        best_index = int(np.argmax(scores))
        best_allocation = population[best_index]
        return OptimizationResult(
            solver_name="SFO",
            power_allocation=best_allocation,
            metrics=problem.metrics(best_allocation, alpha),
            alpha=alpha,
            history=history,
            metadata={
                "population_size": self.hyperparameters.population_size,
                "iterations": self.hyperparameters.iterations,
            },
        )
