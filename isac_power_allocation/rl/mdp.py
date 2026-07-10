"""Markov Decision Process formulation for dynamic ISAC allocation.

This module defines the first RL layer without depending on Gymnasium or a
training library.  It gives the future environment a stable contract for
observations, continuous actions, rewards, and episode transitions.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..channels import ChannelState, get_scenario
from ..channels.models import TimeVaryingOFDMChannel
from ..channels.scenarios import STANDARD_SCENARIOS
from ..config import ExperimentConfig
from ..objectives import AllocationMetrics, ISACSnapshotProblem


@dataclass(frozen=True)
class MDPActionSpec:
    """Continuous action layout for one ISAC allocation decision."""

    num_subcarriers: int
    includes_waveform: bool = True

    @property
    def power_slice(self) -> slice:
        return slice(0, self.num_subcarriers)

    @property
    def waveform_slice(self) -> slice:
        if not self.includes_waveform:
            return slice(self.num_subcarriers, self.num_subcarriers)
        return slice(self.num_subcarriers, 2 * self.num_subcarriers)

    @property
    def dimension(self) -> int:
        return self.num_subcarriers * (2 if self.includes_waveform else 1)


@dataclass(frozen=True)
class MDPObservationSpec:
    """Flat observation layout used by RL agents.

    The observation contains communication gains, sensing gains, normalized
    time, alpha, total power, noise power, and a scenario one-hot indicator.
    """

    num_subcarriers: int
    scenario_names: tuple[str, ...]

    @property
    def communication_gain_slice(self) -> slice:
        return slice(0, self.num_subcarriers)

    @property
    def sensing_gain_slice(self) -> slice:
        return slice(self.num_subcarriers, 2 * self.num_subcarriers)

    @property
    def scalar_slice(self) -> slice:
        return slice(2 * self.num_subcarriers, 2 * self.num_subcarriers + 4)

    @property
    def scenario_slice(self) -> slice:
        start = self.scalar_slice.stop
        return slice(start, start + len(self.scenario_names))

    @property
    def dimension(self) -> int:
        return self.scenario_slice.stop


@dataclass(frozen=True)
class MDPState:
    """State for one time step in the ISAC allocation MDP."""

    communication_gain: np.ndarray
    sensing_gain: np.ndarray
    time_index: int
    horizon: int
    alpha: float
    total_power_w: float
    noise_power_w: float
    scenario_name: str
    scenario_names: tuple[str, ...]

    def to_observation(self) -> np.ndarray:
        scenario_one_hot = np.zeros(len(self.scenario_names), dtype=float)
        scenario_one_hot[self.scenario_names.index(self.scenario_name)] = 1.0

        normalized_time = 0.0
        if self.horizon > 1:
            normalized_time = self.time_index / float(self.horizon - 1)

        scalars = np.array(
            [
                normalized_time,
                self.alpha,
                self.total_power_w,
                self.noise_power_w,
            ],
            dtype=float,
        )
        return np.concatenate(
            [
                np.asarray(self.communication_gain, dtype=float).reshape(-1),
                np.asarray(self.sensing_gain, dtype=float).reshape(-1),
                scalars,
                scenario_one_hot,
            ]
        )


@dataclass(frozen=True)
class MDPTransition:
    """Result of applying one action in the formulated MDP."""

    state: MDPState
    action: np.ndarray
    repaired_action: np.ndarray
    reward: float
    metrics: AllocationMetrics
    next_state: MDPState | None
    done: bool


class ISACAllocationMDP:
    """MDP formulation for sequential ISAC power-waveform allocation.

    State:
        Communication gains, sensing gains, time index, objective alpha, power
        and noise budget scalars, and scenario identity.

    Action:
        A continuous vector.  With waveform co-optimization enabled, the first
        N values are power allocation and the next N values are waveform profile.

    Reward:
        The existing weighted ISAC objective:
        alpha * communication_rate + (1 - alpha) * gamma * sensing_utility.

    Episode:
        One dynamic channel sequence generated from the configured scenario.
    """

    def __init__(
        self,
        config: ExperimentConfig,
        scenario_names: tuple[str, ...] | None = None,
    ) -> None:
        self.config = config
        self.scenario_names = scenario_names or tuple(sorted(STANDARD_SCENARIOS))
        if config.scenario_name not in self.scenario_names:
            self.scenario_names = tuple([*self.scenario_names, config.scenario_name])

        self.action_spec = MDPActionSpec(
            num_subcarriers=config.ofdm.num_subcarriers,
            includes_waveform=config.objective.waveform_co_optimization,
        )
        self.observation_spec = MDPObservationSpec(
            num_subcarriers=config.ofdm.num_subcarriers,
            scenario_names=self.scenario_names,
        )

    @property
    def horizon(self) -> int:
        return self.config.simulation.num_time_steps

    @property
    def alpha(self) -> float:
        return self.config.objective.default_alpha

    def build_episode(self) -> tuple[list[MDPState], list[ISACSnapshotProblem]]:
        channel_sequence = self._generate_channel_sequence()
        states = [self.state_from_channel_state(state) for state in channel_sequence]
        problems = [self.problem_from_channel_state(state) for state in channel_sequence]
        return states, problems

    def state_from_channel_state(self, channel_state: ChannelState) -> MDPState:
        return MDPState(
            communication_gain=channel_state.communication_gain,
            sensing_gain=channel_state.sensing_gain,
            time_index=channel_state.time_index,
            horizon=self.horizon,
            alpha=self.alpha,
            total_power_w=self.config.link_budget.total_power_w,
            noise_power_w=self.config.link_budget.noise_power_w,
            scenario_name=self.config.scenario_name,
            scenario_names=self.scenario_names,
        )

    def problem_from_channel_state(self, channel_state: ChannelState) -> ISACSnapshotProblem:
        return ISACSnapshotProblem(
            communication_gain=channel_state.communication_gain,
            sensing_gain=channel_state.sensing_gain,
            total_power_w=self.config.link_budget.total_power_w,
            noise_power_w=self.config.link_budget.noise_power_w,
            gamma=self.config.objective.gamma,
            per_subcarrier_max_power_w=self.config.link_budget.per_subcarrier_max_power_w,
            sensing_metric=self.config.objective.sensing_metric,
            detection_false_alarm_probability=self.config.objective.detection_false_alarm_probability,
            detection_integration_gain=self.config.objective.detection_integration_gain,
            waveform_co_optimization=self.config.objective.waveform_co_optimization,
            waveform_min_value=self.config.objective.waveform_min_value,
            waveform_comm_exponent=self.config.objective.waveform_comm_exponent,
            waveform_sensing_exponent=self.config.objective.waveform_sensing_exponent,
        )

    def reward(self, problem: ISACSnapshotProblem, action: np.ndarray) -> tuple[float, AllocationMetrics]:
        metrics = problem.metrics(action, self.alpha)
        return metrics.weighted_objective, metrics

    def transition(
        self,
        states: list[MDPState],
        problems: list[ISACSnapshotProblem],
        time_index: int,
        action: np.ndarray,
    ) -> MDPTransition:
        if time_index < 0 or time_index >= len(states):
            raise IndexError(f"time_index {time_index} is outside the episode length {len(states)}.")

        problem = problems[time_index]
        repaired_action = problem.repair(action)
        reward, metrics = self.reward(problem, repaired_action)
        next_index = time_index + 1
        done = next_index >= len(states)
        next_state = None if done else states[next_index]
        return MDPTransition(
            state=states[time_index],
            action=np.asarray(action, dtype=float).reshape(-1),
            repaired_action=repaired_action,
            reward=reward,
            metrics=metrics,
            next_state=next_state,
            done=done,
        )

    def _generate_channel_sequence(self) -> list[ChannelState]:
        channel = TimeVaryingOFDMChannel(
            ofdm_config=self.config.ofdm,
            link_budget=self.config.link_budget,
            simulation_config=self.config.simulation,
            scenario=get_scenario(self.config.scenario_name),
        )
        return channel.generate_sequence()
