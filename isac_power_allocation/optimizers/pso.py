"""Particle Swarm Optimization for scalarized ISAC power allocation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..config import PSOHyperparameters
from ..objectives import ISACSnapshotProblem, OptimizationResult


@dataclass(frozen=True)
class ParticleSwarmOptimizer:
    hyperparameters: PSOHyperparameters

    def solve(self, problem: ISACSnapshotProblem, alpha: float) -> OptimizationResult:
        rng = np.random.default_rng(self.hyperparameters.seed)
        positions = np.vstack(
            [
                problem.repair(problem.random_decision(rng))
                for _ in range(self.hyperparameters.population_size)
            ]
        )
        velocities = np.zeros_like(positions)

        personal_best_positions = positions.copy()
        personal_best_scores = np.array([problem.scalar_objective(x, alpha) for x in positions])
        global_best_index = int(np.argmax(personal_best_scores))
        global_best_position = personal_best_positions[global_best_index].copy()
        global_best_score = float(personal_best_scores[global_best_index])
        history: list[float] = []

        for _ in range(self.hyperparameters.iterations):
            r1 = rng.random(size=positions.shape)
            r2 = rng.random(size=positions.shape)
            velocities = (
                self.hyperparameters.inertia_weight * velocities
                + self.hyperparameters.cognitive_weight * r1 * (personal_best_positions - positions)
                + self.hyperparameters.social_weight * r2 * (global_best_position - positions)
            )
            positions = np.vstack([problem.repair(position + velocity) for position, velocity in zip(positions, velocities)])

            scores = np.array([problem.scalar_objective(x, alpha) for x in positions])
            improved = scores > personal_best_scores
            personal_best_positions[improved] = positions[improved]
            personal_best_scores[improved] = scores[improved]

            iteration_best_index = int(np.argmax(personal_best_scores))
            if personal_best_scores[iteration_best_index] > global_best_score:
                global_best_score = float(personal_best_scores[iteration_best_index])
                global_best_position = personal_best_positions[iteration_best_index].copy()
            history.append(global_best_score)

        best_power, best_waveform = problem.decompose_decision(global_best_position)
        return OptimizationResult(
            solver_name="PSO",
            power_allocation=best_power,
            waveform_profile=best_waveform,
            metrics=problem.metrics(global_best_position, alpha),
            alpha=alpha,
            history=history,
            metadata={
                "population_size": self.hyperparameters.population_size,
                "iterations": self.hyperparameters.iterations,
            },
        )
