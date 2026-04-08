"""Grey Wolf Optimizer for scalarized ISAC power allocation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..config import GWOHyperparameters
from ..objectives import ISACSnapshotProblem, OptimizationResult


@dataclass(frozen=True)
class GreyWolfOptimizer:
    hyperparameters: GWOHyperparameters

    def solve(self, problem: ISACSnapshotProblem, alpha: float) -> OptimizationResult:
        rng = np.random.default_rng(self.hyperparameters.seed)
        population = self._initialize_population(problem, rng)
        scores = np.array([problem.scalar_objective(candidate, alpha) for candidate in population])
        history: list[float] = []

        for iteration in range(self.hyperparameters.iterations):
            order = np.argsort(scores)[::-1]
            alpha_wolf, beta_wolf, delta_wolf = population[order[:3]]
            exploration = 2.0 - 2.0 * iteration / max(self.hyperparameters.iterations - 1, 1)

            next_population = np.empty_like(population)
            for index, wolf in enumerate(population):
                leaders = []
                for leader in (alpha_wolf, beta_wolf, delta_wolf):
                    r1 = rng.random(problem.dimension)
                    r2 = rng.random(problem.dimension)
                    a_vec = 2.0 * exploration * r1 - exploration
                    c_vec = 2.0 * r2
                    distance = np.abs(c_vec * leader - wolf)
                    leaders.append(leader - a_vec * distance)

                candidate = np.mean(leaders, axis=0)
                noise_scale = (
                    self.hyperparameters.mutation_sigma * problem.total_power_w / max(problem.dimension, 1)
                )
                candidate += rng.normal(0.0, noise_scale, size=problem.dimension)
                next_population[index] = problem.repair(candidate)

            population = next_population
            scores = np.array([problem.scalar_objective(candidate, alpha) for candidate in population])
            history.append(float(scores.max()))

        best_index = int(np.argmax(scores))
        best_allocation = problem.repair(population[best_index])
        return OptimizationResult(
            solver_name="GWO",
            power_allocation=best_allocation,
            metrics=problem.metrics(best_allocation, alpha),
            alpha=alpha,
            history=history,
            metadata={
                "population_size": self.hyperparameters.population_size,
                "iterations": self.hyperparameters.iterations,
            },
        )

    def _initialize_population(
        self,
        problem: ISACSnapshotProblem,
        rng: np.random.Generator,
    ) -> np.ndarray:
        population = []
        equal_power = np.full(problem.dimension, problem.total_power_w / problem.dimension)
        population.append(problem.repair(equal_power))

        for _ in range(self.hyperparameters.population_size - 1):
            sample = rng.dirichlet(np.ones(problem.dimension)) * problem.total_power_w
            population.append(problem.repair(sample))
        return np.vstack(population)
