from __future__ import annotations

import unittest

import numpy as np

from isac_power_allocation.channels import TimeVaryingOFDMChannel, get_scenario
from isac_power_allocation.config import LinkBudgetConfig, OFDMConfig, SimulationConfig


class ChannelTests(unittest.TestCase):
    def test_channel_sequence_varies_over_time(self) -> None:
        channel = TimeVaryingOFDMChannel(
            ofdm_config=OFDMConfig(num_subcarriers=8),
            link_budget=LinkBudgetConfig(total_power_w=5.0),
            simulation_config=SimulationConfig(num_time_steps=4, random_seed=5),
            scenario=get_scenario("UMi_NLOS"),
        )
        sequence = channel.generate_sequence()
        self.assertEqual(len(sequence), 4)
        delta = np.linalg.norm(sequence[0].communication_response - sequence[-1].communication_response)
        self.assertGreater(float(delta), 0.0)
        self.assertTrue(np.all(sequence[0].communication_gain >= 0.0))
        self.assertTrue(np.all(sequence[0].sensing_gain >= 0.0))


if __name__ == "__main__":
    unittest.main()
