"""Gym-style environment for sequential ISAC power-waveform allocation."""

from __future__ import annotations

from dataclasses import replace
from typing import Any

import numpy as np

from ..channels.scenarios import STANDARD_SCENARIOS
from ..config import ExperimentConfig
from .mdp import ISACAllocationMDP, MDPState


class ISACPowerWaveformEnv:
    """Dependency-free Gym-style ISAC allocation environment.

    The environment intentionally avoids importing Gymnasium until the training
    phase.  Its API mirrors the classic Gym shape:

    reset() -> observation
    step(action) -> observation, reward, done, info
    """

    def __init__(
        self,
        config: ExperimentConfig,
        scenario_names: tuple[str, ...] | None = None,
        seed: int | None = None,
        randomize_scenario: bool = False,
        randomize_channel_seed: bool = True,
    ) -> None:
        self.base_config = config
        self.scenario_names = scenario_names or tuple(sorted(STANDARD_SCENARIOS))
        if config.scenario_name not in self.scenario_names:
            self.scenario_names = tuple([*self.scenario_names, config.scenario_name])

        self.randomize_scenario = randomize_scenario
        self.randomize_channel_seed = randomize_channel_seed
        initial_seed = config.simulation.random_seed if seed is None else seed
        self.rng = np.random.default_rng(initial_seed)

        self.config = config
        self.mdp = ISACAllocationMDP(config, scenario_names=self.scenario_names)
        self.states: list[MDPState] = []
        self.problems = []
        self.time_index = 0
        self._last_observation: np.ndarray | None = None

    @property
    def observation_shape(self) -> tuple[int]:
        return (self.mdp.observation_spec.dimension,)

    @property
    def action_shape(self) -> tuple[int]:
        return (self.mdp.action_spec.dimension,)

    def reset(self, seed: int | None = None) -> np.ndarray:
        """Start a new channel sequence and return the first observation."""

        if seed is not None:
            self.rng = np.random.default_rng(seed)

        self.config = self._episode_config()
        self.mdp = ISACAllocationMDP(self.config, scenario_names=self.scenario_names)
        self.states, self.problems = self.mdp.build_episode()
        self.time_index = 0
        self._last_observation = self.states[0].to_observation()
        return self._last_observation.copy()

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, dict[str, Any]]:
        """Apply a raw action, repair it, score it, and advance one step."""

        if not self.states or not self.problems:
            self.reset()

        transition = self.mdp.transition(self.states, self.problems, self.time_index, action)
        power, waveform = self.problems[self.time_index].decompose_decision(
            transition.repaired_action
        )

        info: dict[str, Any] = {
            "time_index": self.time_index,
            "scenario_name": transition.state.scenario_name,
            "raw_action": transition.action.copy(),
            "repaired_action": transition.repaired_action.copy(),
            "power_allocation": power.copy(),
            "waveform_profile": None if waveform is None else waveform.copy(),
            "metrics": transition.metrics,
            "communication_rate_bps_hz": transition.metrics.communication_rate_bps_hz,
            "sensing_snr_linear": transition.metrics.sensing_snr_linear,
            "sensing_snr_db": transition.metrics.sensing_snr_db,
            "sensing_detection_probability": transition.metrics.sensing_detection_probability,
            "weighted_objective": transition.metrics.weighted_objective,
        }

        self.time_index += 1
        if transition.done:
            observation = transition.state.to_observation()
            info["terminal_observation"] = observation.copy()
        else:
            observation = transition.next_state.to_observation()

        self._last_observation = observation.copy()
        return observation.copy(), float(transition.reward), transition.done, info

    def sample_random_action(self) -> np.ndarray:
        """Draw an unconstrained action suitable for smoke tests and baselines."""

        return self.rng.normal(0.0, 1.0, size=self.mdp.action_spec.dimension)

    def _episode_config(self) -> ExperimentConfig:
        scenario_name = self.base_config.scenario_name
        if self.randomize_scenario:
            scenario_name = str(self.rng.choice(self.scenario_names))

        simulation = self.base_config.simulation
        if self.randomize_channel_seed:
            simulation = replace(
                simulation,
                random_seed=int(self.rng.integers(0, np.iinfo(np.int32).max)),
            )

        return replace(
            self.base_config,
            scenario_name=scenario_name,
            simulation=simulation,
        )
