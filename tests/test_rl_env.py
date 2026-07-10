from __future__ import annotations

import unittest

import numpy as np

from isac_power_allocation.config import (
    ExperimentConfig,
    LinkBudgetConfig,
    ObjectiveConfig,
    OFDMConfig,
    SimulationConfig,
    build_default_experiment_config,
)
from isac_power_allocation.rl import ISACPowerWaveformEnv


class RLEnvironmentTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = ExperimentConfig(
            scenario_name="ITU_PedB",
            ofdm=OFDMConfig(num_subcarriers=8),
            link_budget=LinkBudgetConfig(total_power_w=2.0, per_subcarrier_max_power_w=0.5),
            simulation=SimulationConfig(num_time_steps=5, random_seed=21),
            objective=ObjectiveConfig(default_alpha=0.5, waveform_co_optimization=True),
        )

    def test_reset_returns_correct_observation_shape(self) -> None:
        env = ISACPowerWaveformEnv(self.config, randomize_channel_seed=False)
        observation = env.reset()

        self.assertEqual(observation.shape, env.observation_shape)
        self.assertEqual(env.action_shape, (2 * self.config.ofdm.num_subcarriers,))

    def test_step_accepts_random_action_and_returns_finite_reward(self) -> None:
        env = ISACPowerWaveformEnv(self.config, randomize_channel_seed=False)
        env.reset()

        observation, reward, done, info = env.step(env.sample_random_action())

        self.assertEqual(observation.shape, env.observation_shape)
        self.assertTrue(np.isfinite(reward))
        self.assertFalse(done)
        self.assertIn("metrics", info)

    def test_episode_ends_after_configured_horizon(self) -> None:
        env = ISACPowerWaveformEnv(self.config, randomize_channel_seed=False)
        env.reset()

        done = False
        steps = 0
        while not done:
            _, _, done, _ = env.step(env.sample_random_action())
            steps += 1

        self.assertEqual(steps, self.config.simulation.num_time_steps)

    def test_default_episode_ends_after_twenty_steps(self) -> None:
        env = ISACPowerWaveformEnv(
            build_default_experiment_config(),
            randomize_channel_seed=False,
        )
        env.reset()

        done = False
        steps = 0
        while not done:
            _, _, done, _ = env.step(env.sample_random_action())
            steps += 1

        self.assertEqual(steps, 20)

    def test_repaired_action_satisfies_power_and_waveform_constraints(self) -> None:
        env = ISACPowerWaveformEnv(self.config, randomize_channel_seed=False)
        env.reset()

        _, _, _, info = env.step(np.full(env.action_shape, -2.0))
        power = info["power_allocation"]
        waveform = info["waveform_profile"]

        self.assertAlmostEqual(float(power.sum()), self.config.link_budget.total_power_w, places=8)
        self.assertTrue(np.all(power >= 0.0))
        self.assertTrue(np.all(power <= self.config.link_budget.per_subcarrier_max_power_w + 1e-10))
        self.assertIsNotNone(waveform)
        self.assertAlmostEqual(float(waveform.sum()), self.config.ofdm.num_subcarriers, places=8)
        self.assertTrue(np.all(waveform >= self.config.objective.waveform_min_value))

    def test_reward_matches_existing_snapshot_problem_objective(self) -> None:
        env = ISACPowerWaveformEnv(self.config, randomize_channel_seed=False)
        env.reset()
        action = np.linspace(-1.0, 1.0, env.action_shape[0])

        _, reward, _, _ = env.step(action)
        expected = env.problems[0].metrics(action, self.config.objective.default_alpha).weighted_objective

        self.assertAlmostEqual(reward, expected)

    def test_same_seed_gives_reproducible_behavior(self) -> None:
        env_a = ISACPowerWaveformEnv(self.config, seed=99)
        env_b = ISACPowerWaveformEnv(self.config, seed=99)

        obs_a = env_a.reset()
        obs_b = env_b.reset()

        np.testing.assert_allclose(obs_a, obs_b)

        action = np.linspace(0.0, 1.0, env_a.action_shape[0])
        next_a, reward_a, done_a, info_a = env_a.step(action)
        next_b, reward_b, done_b, info_b = env_b.step(action)

        np.testing.assert_allclose(next_a, next_b)
        self.assertAlmostEqual(reward_a, reward_b)
        self.assertEqual(done_a, done_b)
        np.testing.assert_allclose(info_a["repaired_action"], info_b["repaired_action"])


if __name__ == "__main__":
    unittest.main()
