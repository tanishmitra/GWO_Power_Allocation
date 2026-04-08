"""Differential Evolution for scalarized ISAC power allocation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..config import DEHyperparameters
from ..objectives import ISACSnapshotProblem, OptimizationResult


@dataclass(frozen=True)
class DifferentialEvolutionOptimizer:
    hyperparameters: DEHyperparameters

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
            for i in range(self.hyperparameters.population_size):
                indices = [idx for idx in range(self.hyperparameters.population_size) if idx != i]
                a, b, c = rng.choice(indices, size=3, replace=False)
                mutant = population[a] + self.hyperparameters.differential_weight * (population[b] - population[c])

                crossover_mask = rng.random(problem.dimension) < self.hyperparameters.crossover_rate
                crossover_mask[int(rng.integers(0, problem.dimension))] = True
                trial = np.where(crossover_mask, mutant, population[i])
                trial = problem.repair(trial)
                trial_score = problem.scalar_objective(trial, alpha)

                if trial_score > scores[i]:
                    population[i] = trial
                    scores[i] = trial_score
            history.append(float(scores.max()))

        best_index = int(np.argmax(scores))
        best_allocation = population[best_index]
        return OptimizationResult(
            solver_name="DE",
            power_allocation=best_allocation,
            metrics=problem.metrics(best_allocation, alpha),
            alpha=alpha,
            history=history,
            metadata={
                "population_size": self.hyperparameters.population_size,
                "iterations": self.hyperparameters.iterations,
            },
        )
