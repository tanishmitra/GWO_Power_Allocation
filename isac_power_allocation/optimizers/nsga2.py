"""A compact NSGA-II implementation for Pareto analysis."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..config import NSGA2Hyperparameters
from ..objectives import ISACSnapshotProblem, OptimizationResult


@dataclass(frozen=True)
class NSGA2Optimizer:
    hyperparameters: NSGA2Hyperparameters

    def solve(self, problem: ISACSnapshotProblem, alpha_for_reporting: float = 0.5) -> list[OptimizationResult]:
        rng = np.random.default_rng(self.hyperparameters.seed)
        population = self._initialize_population(problem, rng)

        for _ in range(self.hyperparameters.generations):
            objectives = np.array([problem.multi_objective(candidate) for candidate in population])
            fronts = self._fast_non_dominated_sort(objectives)
            crowding = self._crowding_distances(objectives, fronts)

            offspring = []
            while len(offspring) < self.hyperparameters.population_size:
                parent_a = population[self._tournament_select(fronts, crowding, rng)]
                parent_b = population[self._tournament_select(fronts, crowding, rng)]
                child_a, child_b = self._crossover(parent_a, parent_b, problem, rng)
                offspring.append(self._mutate(child_a, problem, rng))
                if len(offspring) < self.hyperparameters.population_size:
                    offspring.append(self._mutate(child_b, problem, rng))

            combined = np.vstack([population, np.vstack(offspring)])
            combined_objectives = np.array([problem.multi_objective(candidate) for candidate in combined])
            combined_fronts = self._fast_non_dominated_sort(combined_objectives)
            combined_crowding = self._crowding_distances(combined_objectives, combined_fronts)
            population = self._environmental_selection(
                combined,
                combined_fronts,
                combined_crowding,
                self.hyperparameters.population_size,
            )

        final_objectives = np.array([problem.multi_objective(candidate) for candidate in population])
        final_front = self._fast_non_dominated_sort(final_objectives)[0]
        results: list[OptimizationResult] = []
        for index in final_front:
            allocation = problem.repair(population[index])
            results.append(
                OptimizationResult(
                    solver_name="NSGA2",
                    power_allocation=allocation,
                    metrics=problem.metrics(allocation, alpha_for_reporting),
                    alpha=None,
                )
            )
        return results

    def _initialize_population(
        self,
        problem: ISACSnapshotProblem,
        rng: np.random.Generator,
    ) -> np.ndarray:
        return np.vstack(
            [
                problem.repair(rng.dirichlet(np.ones(problem.dimension)) * problem.total_power_w)
                for _ in range(self.hyperparameters.population_size)
            ]
        )

    @staticmethod
    def _dominates(left: np.ndarray, right: np.ndarray) -> bool:
        return bool(np.all(left >= right) and np.any(left > right))

    def _fast_non_dominated_sort(self, objectives: np.ndarray) -> list[list[int]]:
        domination_sets = [set() for _ in range(len(objectives))]
        domination_counts = np.zeros(len(objectives), dtype=int)
        fronts: list[list[int]] = [[]]

        for i in range(len(objectives)):
            for j in range(len(objectives)):
                if i == j:
                    continue
                if self._dominates(objectives[i], objectives[j]):
                    domination_sets[i].add(j)
                elif self._dominates(objectives[j], objectives[i]):
                    domination_counts[i] += 1
            if domination_counts[i] == 0:
                fronts[0].append(i)

        front_index = 0
        while front_index < len(fronts) and fronts[front_index]:
            next_front: list[int] = []
            for i in fronts[front_index]:
                for dominated in domination_sets[i]:
                    domination_counts[dominated] -= 1
                    if domination_counts[dominated] == 0:
                        next_front.append(dominated)
            if next_front:
                fronts.append(next_front)
            front_index += 1

        return fronts

    def _crowding_distances(self, objectives: np.ndarray, fronts: list[list[int]]) -> np.ndarray:
        crowding = np.zeros(len(objectives), dtype=float)
        for front in fronts:
            if len(front) <= 2:
                crowding[front] = np.inf
                continue

            front_objectives = objectives[front]
            for objective_index in range(front_objectives.shape[1]):
                order = np.argsort(front_objectives[:, objective_index])
                sorted_front = [front[i] for i in order]
                crowding[sorted_front[0]] = np.inf
                crowding[sorted_front[-1]] = np.inf

                min_value = front_objectives[order[0], objective_index]
                max_value = front_objectives[order[-1], objective_index]
                scale = max(max_value - min_value, 1e-12)

                for local_index in range(1, len(sorted_front) - 1):
                    prev_value = front_objectives[order[local_index - 1], objective_index]
                    next_value = front_objectives[order[local_index + 1], objective_index]
                    crowding[sorted_front[local_index]] += (next_value - prev_value) / scale
        return crowding

    def _tournament_select(
        self,
        fronts: list[list[int]],
        crowding: np.ndarray,
        rng: np.random.Generator,
    ) -> int:
        left = int(rng.integers(0, len(crowding)))
        right = int(rng.integers(0, len(crowding)))
        left_rank = self._rank_of(left, fronts)
        right_rank = self._rank_of(right, fronts)

        if left_rank < right_rank:
            return left
        if right_rank < left_rank:
            return right
        return left if crowding[left] >= crowding[right] else right

    @staticmethod
    def _rank_of(index: int, fronts: list[list[int]]) -> int:
        for rank, front in enumerate(fronts):
            if index in front:
                return rank
        return len(fronts) + 1

    def _crossover(
        self,
        parent_a: np.ndarray,
        parent_b: np.ndarray,
        problem: ISACSnapshotProblem,
        rng: np.random.Generator,
    ) -> tuple[np.ndarray, np.ndarray]:
        if rng.random() > self.hyperparameters.crossover_rate:
            return parent_a.copy(), parent_b.copy()

        blend = rng.random(problem.dimension)
        child_a = blend * parent_a + (1.0 - blend) * parent_b
        child_b = blend * parent_b + (1.0 - blend) * parent_a
        return problem.repair(child_a), problem.repair(child_b)

    def _mutate(
        self,
        candidate: np.ndarray,
        problem: ISACSnapshotProblem,
        rng: np.random.Generator,
    ) -> np.ndarray:
        mutation_mask = rng.random(problem.dimension) < self.hyperparameters.mutation_rate
        if not np.any(mutation_mask):
            return problem.repair(candidate)

        sigma = self.hyperparameters.mutation_sigma * problem.total_power_w / max(problem.dimension, 1)
        updated = candidate.copy()
        updated[mutation_mask] += rng.normal(0.0, sigma, size=int(mutation_mask.sum()))
        return problem.repair(updated)

    @staticmethod
    def _environmental_selection(
        population: np.ndarray,
        fronts: list[list[int]],
        crowding: np.ndarray,
        target_size: int,
    ) -> np.ndarray:
        selected_indices: list[int] = []
        for front in fronts:
            if len(selected_indices) + len(front) <= target_size:
                selected_indices.extend(front)
                continue

            ranked = sorted(front, key=lambda index: crowding[index], reverse=True)
            remaining = target_size - len(selected_indices)
            selected_indices.extend(ranked[:remaining])
            break
        return population[selected_indices]
