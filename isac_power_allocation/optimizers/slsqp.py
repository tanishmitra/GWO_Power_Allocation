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
        num_subcarriers = problem.dimension
        power_upper_bound = (
            problem.total_power_w if problem.per_subcarrier_max_power_w is None else problem.per_subcarrier_max_power_w
        )
        if problem.waveform_co_optimization:
            bounds = [(0.0, power_upper_bound) for _ in range(num_subcarriers)] + [
                (problem.waveform_min_value, None) for _ in range(num_subcarriers)
            ]
            constraints = [
                {
                    "type": "ineq",
                    "fun": lambda x: problem.total_power_w - float(np.sum(x[:num_subcarriers])),
                },
                {
                    "type": "eq",
                    "fun": lambda x: float(np.sum(x[num_subcarriers:])) - float(num_subcarriers),
                },
            ]
        else:
            bounds = [(0.0, power_upper_bound) for _ in range(num_subcarriers)]
            constraints = [{"type": "ineq", "fun": lambda x: problem.total_power_w - float(np.sum(x))}]

        initial_points = [problem.equal_power_decision()]
        for _ in range(max(self.hyperparameters.restarts - 1, 0)):
            initial_points.append(problem.random_decision(rng))

        best_decision = problem.repair(initial_points[0])
        best_score = problem.scalar_objective(best_decision, alpha)

        for initial in initial_points:
            result = minimize(
                fun=lambda x: -problem.scalar_objective(problem.repair(x), alpha),
                x0=problem.repair(initial),
                method="SLSQP",
                bounds=bounds,
                constraints=constraints,
                options={"maxiter": self.hyperparameters.max_iterations, "disp": False},
            )
            candidate = problem.repair(result.x)
            score = problem.scalar_objective(candidate, alpha)
            if score > best_score:
                best_score = score
                best_decision = candidate

        best_power, best_waveform = problem.decompose_decision(best_decision)
        return OptimizationResult(
            solver_name="SLSQP",
            power_allocation=best_power,
            waveform_profile=best_waveform,
            metrics=problem.metrics(best_decision, alpha),
            alpha=alpha,
            metadata={
                "restarts": self.hyperparameters.restarts,
                "max_iterations": self.hyperparameters.max_iterations,
            },
        )
