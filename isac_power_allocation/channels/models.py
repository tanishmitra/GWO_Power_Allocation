"""Time-varying frequency-selective OFDM channel generation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..config import LinkBudgetConfig, OFDMConfig, SimulationConfig
from ..math_utils import LIGHT_SPEED_MPS, db_to_linear, fspl_at_1m_db
from .scenarios import ChannelScenario


@dataclass(frozen=True)
class ChannelState:
    time_index: int
    time_s: float
    communication_response: np.ndarray
    sensing_response: np.ndarray

    @property
    def communication_gain(self) -> np.ndarray:
        return np.abs(self.communication_response) ** 2

    @property
    def sensing_gain(self) -> np.ndarray:
        return np.abs(self.sensing_response) ** 2


@dataclass(frozen=True)
class _ClusterState:
    delays_s: np.ndarray
    powers: np.ndarray
    phases_rad: np.ndarray
    dopplers_hz: np.ndarray
    los_phase_rad: float
    los_doppler_hz: float
    large_scale_power_gain: float
    k_factor_linear: float | None


class TimeVaryingOFDMChannel:
    """Generates communication and sensing channel sequences for one scenario."""

    def __init__(
        self,
        ofdm_config: OFDMConfig,
        link_budget: LinkBudgetConfig,
        simulation_config: SimulationConfig,
        scenario: ChannelScenario,
    ) -> None:
        self.ofdm_config = ofdm_config
        self.link_budget = link_budget
        self.simulation_config = simulation_config
        self.scenario = scenario
        self.rng = np.random.default_rng(simulation_config.random_seed)
        self.subcarrier_offsets_hz = ofdm_config.subcarrier_offsets_hz()

        self._comm_cluster_state = self._sample_cluster_state(
            distance_m=link_budget.user_distance_m,
            speed_mps=link_budget.user_speed_mps,
            round_trip=False,
        )
        self._sense_cluster_state = self._sample_cluster_state(
            distance_m=link_budget.target_distance_m,
            speed_mps=link_budget.target_speed_mps,
            round_trip=True,
        )

    def generate_sequence(self) -> list[ChannelState]:
        sequence: list[ChannelState] = []
        for time_index in range(self.simulation_config.num_time_steps):
            time_s = time_index * self.simulation_config.csi_update_interval_s
            communication_response = self._frequency_response(self._comm_cluster_state, time_s)
            sensing_response = self._frequency_response(self._sense_cluster_state, time_s)
            sequence.append(
                ChannelState(
                    time_index=time_index,
                    time_s=time_s,
                    communication_response=communication_response,
                    sensing_response=sensing_response,
                )
            )
        return sequence

    def _sample_cluster_state(
        self,
        distance_m: float,
        speed_mps: float,
        round_trip: bool,
    ) -> _ClusterState:
        num_taps = max(self.scenario.num_clusters, self.simulation_config.num_taps)
        rms_delay = max(self.scenario.rms_delay_spread_s, 1e-9)

        delays_s = np.sort(self.rng.exponential(scale=rms_delay, size=num_taps))
        decay = np.exp(-delays_s / rms_delay)
        powers = decay / decay.sum()
        phases_rad = self.rng.uniform(0.0, 2.0 * np.pi, size=num_taps)

        carrier_frequency_hz = self.ofdm_config.carrier_frequency_hz
        max_doppler_hz = speed_mps * carrier_frequency_hz / LIGHT_SPEED_MPS
        if round_trip:
            max_doppler_hz *= 2.0

        arrival_angles = self.rng.uniform(0.0, 2.0 * np.pi, size=num_taps)
        dopplers_hz = max_doppler_hz * np.cos(arrival_angles)

        path_loss_db = self._close_in_path_loss_db(distance_m=distance_m)
        shadowing_db = self.rng.normal(0.0, self.scenario.shadowing_std_db)
        large_scale_power_gain = 10.0 ** (-(path_loss_db + shadowing_db) / 10.0)

        if not round_trip:
            large_scale_power_gain *= db_to_linear(self.link_budget.communication_processing_gain_db)

        if round_trip:
            processing_gain = db_to_linear(
                self.link_budget.radar_cross_section_dbsm + self.link_budget.sensing_processing_gain_db
            )
            large_scale_power_gain = (large_scale_power_gain ** 2) * processing_gain

        los_phase_rad = float(self.rng.uniform(0.0, 2.0 * np.pi))
        los_doppler_hz = float(self.rng.uniform(-max_doppler_hz, max_doppler_hz))
        k_factor_linear = (
            None if self.scenario.k_factor_db is None else db_to_linear(self.scenario.k_factor_db)
        )

        return _ClusterState(
            delays_s=delays_s,
            powers=powers,
            phases_rad=phases_rad,
            dopplers_hz=dopplers_hz,
            los_phase_rad=los_phase_rad,
            los_doppler_hz=los_doppler_hz,
            large_scale_power_gain=large_scale_power_gain,
            k_factor_linear=k_factor_linear,
        )

    def _frequency_response(self, cluster_state: _ClusterState, time_s: float) -> np.ndarray:
        phase_time = cluster_state.phases_rad + 2.0 * np.pi * cluster_state.dopplers_hz * time_s
        time_vector = np.sqrt(cluster_state.powers) * np.exp(1j * phase_time)
        frequency_phase = np.exp(
            -1j * 2.0 * np.pi * np.outer(self.subcarrier_offsets_hz, cluster_state.delays_s)
        )
        diffuse_response = frequency_phase @ time_vector

        if cluster_state.k_factor_linear is None:
            return np.sqrt(cluster_state.large_scale_power_gain) * diffuse_response

        k_linear = cluster_state.k_factor_linear
        los_scale = np.sqrt(cluster_state.large_scale_power_gain * k_linear / (k_linear + 1.0))
        diffuse_scale = np.sqrt(cluster_state.large_scale_power_gain / (k_linear + 1.0))
        los_component = los_scale * np.exp(
            1j * (cluster_state.los_phase_rad + 2.0 * np.pi * cluster_state.los_doppler_hz * time_s)
        )
        return diffuse_scale * diffuse_response + los_component

    def _close_in_path_loss_db(self, distance_m: float) -> float:
        distance_m = max(distance_m, 1.0)
        reference_loss_db = fspl_at_1m_db(self.ofdm_config.carrier_frequency_hz)
        return reference_loss_db + 10.0 * self.scenario.path_loss_exponent * np.log10(distance_m)
