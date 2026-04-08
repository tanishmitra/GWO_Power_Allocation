"""Scenario presets inspired by standardized ISAC-relevant channel families."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ChannelScenario:
    name: str
    description: str
    path_loss_exponent: float
    shadowing_std_db: float
    rms_delay_spread_s: float
    num_clusters: int
    k_factor_db: float | None = None


STANDARD_SCENARIOS: dict[str, ChannelScenario] = {
    "UMi_NLOS": ChannelScenario(
        name="UMi_NLOS",
        description="Urban microcell non-line-of-sight with rich multipath and moderate-to-high Doppler.",
        path_loss_exponent=3.1,
        shadowing_std_db=6.0,
        rms_delay_spread_s=250e-9,
        num_clusters=16,
        k_factor_db=None,
    ),
    "UMa_LOS": ChannelScenario(
        name="UMa_LOS",
        description="Urban macro LOS with stronger direct path and larger delay spread support.",
        path_loss_exponent=2.2,
        shadowing_std_db=4.0,
        rms_delay_spread_s=400e-9,
        num_clusters=12,
        k_factor_db=9.0,
    ),
    "WINNER_A1": ChannelScenario(
        name="WINNER_A1",
        description="Indoor office style channel with lower delay spread and mild LOS structure.",
        path_loss_exponent=1.8,
        shadowing_std_db=3.0,
        rms_delay_spread_s=100e-9,
        num_clusters=10,
        k_factor_db=6.0,
    ),
    "ITU_PedB": ChannelScenario(
        name="ITU_PedB",
        description="Pedestrian legacy profile with low Doppler and softer outdoor/indoor transitions.",
        path_loss_exponent=2.5,
        shadowing_std_db=4.0,
        rms_delay_spread_s=750e-9,
        num_clusters=8,
        k_factor_db=None,
    ),
}


def get_scenario(name: str) -> ChannelScenario:
    try:
        return STANDARD_SCENARIOS[name]
    except KeyError as exc:
        available = ", ".join(sorted(STANDARD_SCENARIOS))
        raise ValueError(f"Unknown scenario '{name}'. Available scenarios: {available}") from exc
