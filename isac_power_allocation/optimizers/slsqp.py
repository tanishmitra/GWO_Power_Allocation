"""SLSQP wrapper for constrained scalarized ISAC optimization."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize

from ..config import SLSQPHyperparameters
from ..objectives import ISACSnapshotProblem, OptimizationResult


@dataclass(frozen=True)
class SLSQPOptimizer:
    hyperparameters: SLSQPHyperparameters

    def solve(self, problem: ISACSnapshotProblem, alpha: float) -> OptimizationResult:
        rng = np.random.default_rng(self.hyperparameters.seed)
        upper_bound = (
            problem.total_power_w if problem.per_subcarrier_max_power_w is None else problem.per_subcarrier_max_power_w
        )
        bounds = [(0.0, upper_bound) for _ in range(problem.dimension)]
        constraint = {"type": "ineq", "fun": lambda x: problem.total_power_w - float(np.sum(x))}

        initial_points = [np.full(problem.dimension, problem.total_power_w / problem.dimension)]
        for _ in range(max(self.hyperparameters.restarts - 1, 0)):
            initial_points.append(rng.dirichlet(np.ones(problem.dimension)) * problem.total_power_w)

        best_allocation = problem.repair(initial_points[0])
        best_score = problem.scalar_objective(best_allocation, alpha)

        for initial in initial_points:
            result = minimize(
                fun=lambda x: -problem.scalar_objective(problem.repair(x), alpha),
                x0=problem.repair(initial),
                method="SLSQP",
                bounds=bounds,
                constraints=[constraint],
                options={"maxiter": self.hyperparameters.max_iterations, "disp": False},
            )
            candidate = problem.repair(result.x)
            score = problem.scalar_objective(candidate, alpha)
            if score > best_score:
                best_score = score
                best_allocation = candidate

        return OptimizationResult(
            solver_name="SLSQP",
            power_allocation=best_allocation,
            metrics=problem.metrics(best_allocation, alpha),
            alpha=alpha,
            metadata={
                "restarts": self.hyperparameters.restarts,
                "max_iterations": self.hyperparameters.max_iterations,
            },
        )
