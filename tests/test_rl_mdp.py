from __future__ import annotations

import unittest

import numpy as np

from isac_power_allocation.config import (
    ExperimentConfig,
    LinkBudgetConfig,
    ObjectiveConfig,
    OFDMConfig,
    SimulationConfig,
)
from isac_power_allocation.rl import ISACAllocationMDP


class RLMDPTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = ExperimentConfig(
            scenario_name="UMi_NLOS",
            ofdm=OFDMConfig(num_subcarriers=8),
            link_budget=LinkBudgetConfig(total_power_w=2.0, per_subcarrier_max_power_w=0.5),
            simulation=SimulationConfig(num_time_steps=5, random_seed=13),
            objective=ObjectiveConfig(default_alpha=0.4, waveform_co_optimization=True),
        )

    def test_mdp_specs_match_power_waveform_formulation(self) -> None:
        mdp = ISACAllocationMDP(self.config)
        self.assertEqual(mdp.action_spec.dimension, 16)
        self.assertEqual(mdp.action_spec.power_slice, slice(0, 8))
        self.assertEqual(mdp.action_spec.waveform_slice, slice(8, 16))
        self.assertEqual(mdp.observation_spec.dimension, 2 * 8 + 4 + len(mdp.scenario_names))

    def test_episode_states_have_expected_observation_shape(self) -> None:
        mdp = ISACAllocationMDP(self.config)
        states, problems = mdp.build_episode()

        self.assertEqual(len(states), 5)
        self.assertEqual(len(problems), 5)
        observation = states[0].to_observation()
        self.assertEqual(observation.shape, (mdp.observation_spec.dimension,))
        self.assertAlmostEqual(observation[mdp.observation_spec.scalar_slice][0], 0.0)
        self.assertAlmostEqual(states[-1].to_observation()[mdp.observation_spec.scalar_slice][0], 1.0)
        self.assertAlmostEqual(float(observation[mdp.observation_spec.scenario_slice].sum()), 1.0)

    def test_reward_matches_snapshot_problem_objective(self) -> None:
        mdp = ISACAllocationMDP(self.config)
        states, problems = mdp.build_episode()
        action = np.linspace(-1.0, 1.0, mdp.action_spec.dimension)

        transition = mdp.transition(states, problems, time_index=0, action=action)
        expected = problems[0].metrics(action, self.config.objective.default_alpha).weighted_objective

        self.assertAlmostEqual(transition.reward, expected)
        self.assertFalse(transition.done)
        self.assertIsNotNone(transition.next_state)

    def test_transition_repairs_power_and_waveform_constraints(self) -> None:
        mdp = ISACAllocationMDP(self.config)
        states, problems = mdp.build_episode()
        action = np.full(mdp.action_spec.dimension, -3.0)

        transition = mdp.transition(states, problems, time_index=0, action=action)
        power = transition.repaired_action[mdp.action_spec.power_slice]
        waveform = transition.repaired_action[mdp.action_spec.waveform_slice]

        self.assertAlmostEqual(float(power.sum()), self.config.link_budget.total_power_w, places=8)
        self.assertTrue(np.all(power >= 0.0))
        self.assertTrue(np.all(power <= self.config.link_budget.per_subcarrier_max_power_w + 1e-10))
        self.assertAlmostEqual(float(waveform.sum()), self.config.ofdm.num_subcarriers, places=8)
        self.assertTrue(np.all(waveform >= self.config.objective.waveform_min_value))

    def test_final_transition_is_terminal(self) -> None:
        mdp = ISACAllocationMDP(self.config)
        states, problems = mdp.build_episode()

        transition = mdp.transition(
            states,
            problems,
            time_index=len(states) - 1,
            action=problems[-1].equal_power_decision(),
        )

        self.assertTrue(transition.done)
        self.assertIsNone(transition.next_state)

    def test_same_seed_reproduces_initial_state(self) -> None:
        mdp_a = ISACAllocationMDP(self.config)
        mdp_b = ISACAllocationMDP(self.config)

        states_a, _ = mdp_a.build_episode()
        states_b, _ = mdp_b.build_episode()

        np.testing.assert_allclose(states_a[0].to_observation(), states_b[0].to_observation())


if __name__ == "__main__":
    unittest.main()
