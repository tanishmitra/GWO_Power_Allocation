from __future__ import annotations

import unittest

import numpy as np

from isac_power_allocation.config import GWOHyperparameters, NSGA2Hyperparameters
from isac_power_allocation.config import (
    DEHyperparameters,
    ExperimentConfig,
    LinkBudgetConfig,
    ObjectiveConfig,
    OFDMConfig,
    POAHyperparameters,
    PSOHyperparameters,
    SAHyperparameters,
    SFOHyperparameters,
    SimulationConfig,
    SLSQPHyperparameters,
)
from isac_power_allocation.experiments.runner import run_dynamic_algorithm_comparison
from isac_power_allocation.objectives import ISACSnapshotProblem
from isac_power_allocation.optimizers.de import DifferentialEvolutionOptimizer
from isac_power_allocation.optimizers.gwo import GreyWolfOptimizer
from isac_power_allocation.optimizers.nsga2 import NSGA2Optimizer
from isac_power_allocation.optimizers.poa import PelicanOptimizer
from isac_power_allocation.optimizers.pso import ParticleSwarmOptimizer
from isac_power_allocation.optimizers.sa import SimulatedAnnealingOptimizer
from isac_power_allocation.optimizers.sfo import StarfishOptimizer
from isac_power_allocation.optimizers.slsqp import SLSQPOptimizer
from isac_power_allocation.pareto import extract_non_dominated_results


class OptimizerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.problem = ISACSnapshotProblem(
            communication_gain=np.array([1.5, 1.0, 0.8, 0.4]),
            sensing_gain=np.array([0.5, 1.0, 1.4, 1.8]),
            total_power_w=2.0,
            noise_power_w=0.1,
            gamma=10.0,
            per_subcarrier_max_power_w=1.0,
        )

    def test_gwo_returns_feasible_solution(self) -> None:
        optimizer = GreyWolfOptimizer(GWOHyperparameters(population_size=10, iterations=12, seed=3))
        result = optimizer.solve(self.problem, alpha=0.5)
        self.assertAlmostEqual(float(result.power_allocation.sum()), 2.0, places=8)
        self.assertTrue(np.all(result.power_allocation >= 0.0))
        self.assertGreater(result.metrics.communication_rate_bps_hz, 0.0)

    def test_nsga2_front_is_non_dominated(self) -> None:
        optimizer = NSGA2Optimizer(
            NSGA2Hyperparameters(
                population_size=16,
                generations=10,
                crossover_rate=0.9,
                mutation_rate=0.2,
                seed=4,
            )
        )
        results = extract_non_dominated_results(optimizer.solve(self.problem))
        self.assertGreaterEqual(len(results), 1)
        for result in results:
            self.assertLessEqual(float(result.power_allocation.sum()), 2.0 + 1e-8)

    def test_additional_solvers_return_feasible_solutions(self) -> None:
        solvers = [
            SLSQPOptimizer(SLSQPHyperparameters(restarts=2, max_iterations=40, seed=1)),
            ParticleSwarmOptimizer(PSOHyperparameters(population_size=10, iterations=8, seed=2)),
            DifferentialEvolutionOptimizer(DEHyperparameters(population_size=10, iterations=8, seed=3)),
            SimulatedAnnealingOptimizer(SAHyperparameters(iterations=60, seed=4)),
            StarfishOptimizer(SFOHyperparameters(population_size=10, iterations=8, seed=5)),
            PelicanOptimizer(POAHyperparameters(population_size=10, iterations=8, seed=6)),
        ]
        for solver in solvers:
            result = solver.solve(self.problem, alpha=0.5)
            self.assertTrue(np.all(result.power_allocation >= 0.0), msg=result.solver_name)
            self.assertLessEqual(float(result.power_allocation.sum()), 2.0 + 1e-8, msg=result.solver_name)
            self.assertGreater(result.metrics.communication_rate_bps_hz, 0.0, msg=result.solver_name)

    def test_dynamic_comparison_smoke(self) -> None:
        config = ExperimentConfig(
            scenario_name="ITU_PedB",
            ofdm=OFDMConfig(num_subcarriers=8),
            link_budget=LinkBudgetConfig(total_power_w=2.0, per_subcarrier_max_power_w=0.5),
            simulation=SimulationConfig(num_time_steps=3, random_seed=9),
            objective=ObjectiveConfig(default_alpha=0.5),
            slsqp=SLSQPHyperparameters(restarts=1, max_iterations=20, seed=1),
            pso=PSOHyperparameters(population_size=8, iterations=4, seed=2),
            de=DEHyperparameters(population_size=8, iterations=4, seed=3),
            sa=SAHyperparameters(iterations=20, seed=4),
            sfo=SFOHyperparameters(population_size=8, iterations=4, seed=5),
            poa=POAHyperparameters(population_size=8, iterations=4, seed=6),
            gwo=GWOHyperparameters(population_size=8, iterations=4, seed=7),
        )
        summary = run_dynamic_algorithm_comparison(config)
        self.assertEqual(summary.num_time_steps, 3)
        self.assertEqual(len(summary.aggregates), 7)

    def test_probabilistic_detection_with_waveform_cooptimization(self) -> None:
        problem = ISACSnapshotProblem(
            communication_gain=np.array([1.2, 0.8, 0.7, 1.0]),
            sensing_gain=np.array([0.4, 1.3, 1.5, 0.9]),
            total_power_w=2.0,
            noise_power_w=0.1,
            gamma=10.0,
            per_subcarrier_max_power_w=1.0,
            sensing_metric="detection_probability",
            waveform_co_optimization=True,
        )
        optimizer = GreyWolfOptimizer(GWOHyperparameters(population_size=10, iterations=10, seed=12))
        result = optimizer.solve(problem, alpha=0.4)

        self.assertIsNotNone(result.waveform_profile)
        assert result.waveform_profile is not None
        self.assertEqual(result.waveform_profile.size, problem.dimension)
        self.assertAlmostEqual(float(np.sum(result.waveform_profile)), float(problem.dimension), places=6)
        self.assertGreaterEqual(result.metrics.sensing_detection_probability, 0.0)
        self.assertLessEqual(result.metrics.sensing_detection_probability, 1.0)


if __name__ == "__main__":
    unittest.main()
